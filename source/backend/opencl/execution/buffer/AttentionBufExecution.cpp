//
//  SoftmaxBufExecution.cpp
//  MNN
//
//  Created by MNN on 2024/04/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "backend/opencl/execution/buffer/AttentionBufExecution.hpp"

namespace MNN {
namespace OpenCL {

AttentionBufImpl::AttentionBufImpl(const MNN::Op *op, Backend *backend, bool kv_cahce)
    : mKVCache(kv_cahce){
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("softmax_buf", "softmax_channel", {"-DSOFTMAX_LOCAL_SIZE=512"});
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
}

void AttentionBufImpl::allocKVCache() {
    if (!mKVCache || mPastLength < mMaxLength) {
        return;
    }
    mMaxLength = mPastLength + mExpandChunk;
    size_t buffer_size = UP_DIV(mMaxLength, 4) * mKvNumHead * mHeadDim * 4 * mByte;
    // past_key: [1, numhead, headdim, maxlen]
    mPastKey.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
    // past_value: [1, numhead, maxlen, headdim]
    mPastValue.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
}

void AttentionBufImpl::reallocKVCache() {
    if (!mKVCache || mPastLength < mMaxLength) {
        return;
    }
    
    size_t old_size = mKvNumHead * UP_DIV(mMaxLength, 4) * mHeadDim * 4 * mByte;
    mMaxLength = mPastLength + mExpandChunk;
    size_t buffer_size = UP_DIV(mMaxLength, 4) * mKvNumHead * mHeadDim * 4 * mByte;
    // past_key: [1, numhead, headdim, maxlen]
    auto new_key = new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    // past_value: [1, numhead, maxlen, headdim]
    auto new_value = new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    // copy
    cl_int res;
    auto new_key_ptr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*new_key, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
    auto key_ptr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*mPastKey.get(), true, CL_MAP_READ, 0, old_size, nullptr, nullptr, &res);
    if(new_key_ptr != nullptr && key_ptr != nullptr && res == CL_SUCCESS){
        ::memcpy(new_key_ptr, key_ptr, old_size);
    }else{
        MNN_ERROR("Map error key_ptr == nullptr \n");
        MNN_ASSERT(false);
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*new_key, new_key_ptr);
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mPastKey.get(), key_ptr);
    
    auto new_value_ptr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*new_value, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
    auto value_ptr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*mPastValue.get(), true, CL_MAP_READ, 0, old_size, nullptr, nullptr, &res);
    if(new_value_ptr != nullptr && value_ptr != nullptr && res == CL_SUCCESS){
        ::memcpy(new_value_ptr, value_ptr, old_size);
    }else{
        MNN_ERROR("Map error value_ptr == nullptr \n");
        MNN_ASSERT(false);
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*new_value, new_value_ptr);
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mPastValue.get(), value_ptr);
    
    mPastKey.reset(new_key);
    mPastValue.reset(new_value);
    mTempQK.reset(Tensor::createDevice<float>({UP_DIV(mMaxLength, 4) * mNumHead * 4}));
    mTempSoftMax.reset(Tensor::createDevice<float>({UP_DIV(mMaxLength, 4) * mNumHead * 4}));
    mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::STATIC);
    mOpenCLBackend->onAcquireBuffer(mTempSoftMax.get(), Backend::STATIC);
    // reset memory for args
    if(mOpenCLBackend->isUseRecordQueue()){
        mQkUpdateInfo.update_kernel_args[1].arg_value = &openCLBuffer(mTempQK.get())();
        mQkUpdateInfo.update_kernel_args[2].arg_value = &(*(mPastKey.get()))();
        mSoftMaxUpdateInfo.update_kernel_args[0].arg_value = &openCLBuffer(mTempQK.get())();
        mSoftMaxUpdateInfo.update_kernel_args[1].arg_value = &openCLBuffer(mTempSoftMax.get())();
        mQkvUpdateInfo.update_kernel_args[0].arg_value = &openCLBuffer(mTempSoftMax.get())();
        mQkvUpdateInfo.update_kernel_args[1].arg_value = &(*(mPastValue.get()))();
    }else{
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_qk->get().setArg(5, openCLBuffer(mTempQK.get()));
        ret |= mKernel_qk->get().setArg(6, *mPastKey.get());
        ret |= mKernel_softmax->get().setArg(3, openCLBuffer(mTempQK.get()));
        ret |= mKernel_softmax->get().setArg(4, openCLBuffer(mTempSoftMax.get()));
        ret |= mKernel_qkv->get().setArg(3, openCLBuffer(mTempSoftMax.get()));
        ret |= mKernel_qkv->get().setArg(6, *mPastValue.get());
        MNN_CHECK_CL_SUCCESS(ret, "reset memory arg for AttentionBufExecution");
    }
}

int AttentionBufImpl::getLocalSize(int size, int maxGroupSize){
    int local_size = 1;
    while(local_size * 2 <= maxGroupSize && local_size * 2 <= size){
        local_size *= 2;
    }
    return local_size;
}

ErrorCode AttentionBufImpl::onResize(Backend *backend, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    mOpenCLBackend->startRecord(mRecording);
    //clear update arg vector, if prefill and decode use the same one
    mOpRecordUpdateInfo.clear();
    mQkUpdateInfo.update_kernel_args.clear();
    mQkUpdateInfo.update_global_size.clear();
    mQkUpdateInfo.update_local_size.clear();
    mSoftMaxUpdateInfo.update_kernel_args.clear();
    mSoftMaxUpdateInfo.update_global_size.clear();
    mSoftMaxUpdateInfo.update_local_size.clear();
    mQkvUpdateInfo.update_kernel_args.clear();
    mQkvUpdateInfo.update_global_size.clear();
    mQkvUpdateInfo.update_local_size.clear();
    
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto mask = inputs[3];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto shape = query->shape();
    
    int seq_len = shape[1];
    mNumHead = shape[2];
    mKvNumHead = key->shape()[2];
    int group_size = mNumHead / mKvNumHead;
    mHeadDim = shape[3];
    mScale = 1.0 / sqrt(mHeadDim);
    mIsDecode = seq_len == 1;
    mIsFirstDecode = true;
    if (mPastLength == 0 || seq_len > 1) {
        mPastLength = seq_len;
    }
    mKv_seq_len = mPastLength;
    if(mIsDecode){
        mKv_seq_len = mPastLength + 1;
    }
    
    if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()){
        mByte = 2;
    }
    allocKVCache();
    if (mIsDecode) {
        mTempQK.reset(Tensor::createDevice<float>({UP_DIV(mMaxLength, 4) * mNumHead * 4}));
        mTempSoftMax.reset(Tensor::createDevice<float>({UP_DIV(mMaxLength, 4) * mNumHead * 4}));
    } else {
        mTempQK.reset(Tensor::createDevice<float>({UP_DIV(mPastLength, 4) * mPastLength * mNumHead * 4}));
        mTempSoftMax.reset(Tensor::createDevice<float>({UP_DIV(mPastLength, 4) * mPastLength * mNumHead * 4}));
    }
    mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempSoftMax.get(), Backend::DYNAMIC);
    
    // query * key -> div -> select
    {
        std::set<std::string> buildOption;
        if(!mIsDecode){
            buildOption.emplace("-DOPENCL_PREFILL_ATTENTION");
        }
        if((mHeadDim % 4) != 0){
            buildOption.emplace("-DHEADDIM_LEAVE");
        }
        if(mask->getType() == halide_type_of<float>()){
            buildOption.emplace("-DADD_MASK");
        }
        buildOption.emplace("-DNUMHEAD_GROUP_SIZE=" + std::to_string(group_size));
        mKernel_qk = runtime->buildKernel("attention_buf", "matmul_qk_div_mask", buildOption, inputs[0], outputs[0]);
        mGlobalWorkSizeQk =  {static_cast<uint32_t>(UP_DIV(seq_len, 4)), static_cast<uint32_t>(mNumHead), static_cast<uint32_t>(UP_DIV(mKv_seq_len, 4))};
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_qk));
        mGlobalWorkSizeQk2 = UP_DIV(mKv_seq_len, 4);
        
        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_qk->get().setArg(index++, mGlobalWorkSizeQk[0]);
        ret |= mKernel_qk->get().setArg(index++, mGlobalWorkSizeQk[1]);
        ret |= mKernel_qk->get().setArg(index++, mGlobalWorkSizeQk2);
        ret |= mKernel_qk->get().setArg(index++, openCLBuffer(query));
        ret |= mKernel_qk->get().setArg(index++, openCLBuffer(key));
        ret |= mKernel_qk->get().setArg(index++, openCLBuffer(mTempQK.get()));
        ret |= mKernel_qk->get().setArg(index++, *mPastKey.get());
        ret |= mKernel_qk->get().setArg(index++, openCLBuffer(mask));
        ret |= mKernel_qk->get().setArg(index++, mScale);
        ret |= mKernel_qk->get().setArg(index++, seq_len);
        ret |= mKernel_qk->get().setArg(index++, mKv_seq_len);
        ret |= mKernel_qk->get().setArg(index++, mNumHead);
        ret |= mKernel_qk->get().setArg(index++, mKvNumHead);
        ret |= mKernel_qk->get().setArg(index++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qk_div_mask");

        mLocalWorkSizeQk = localWS3DDefault(mGlobalWorkSizeQk, maxWorkGroupSize, runtime, "matmul_qk_div_mask", mKernel_qk).first;
        mGlobalWorkSizeQk[0] = ROUND_UP(mGlobalWorkSizeQk[0], std::max((uint32_t)1, mLocalWorkSizeQk[0]));
        mGlobalWorkSizeQk[1] = ROUND_UP(mGlobalWorkSizeQk[1], std::max((uint32_t)1, mLocalWorkSizeQk[1]));
        mGlobalWorkSizeQk[2] = ROUND_UP(mGlobalWorkSizeQk[2], std::max((uint32_t)1, mLocalWorkSizeQk[2]));
        mQkUpdateInfo.update_kernel_args.push_back({0, 2, sizeof(mGlobalWorkSizeQk2), &mGlobalWorkSizeQk2});
        mQkUpdateInfo.update_kernel_args.push_back({0, 5, sizeof(cl_mem), &openCLBuffer(mTempQK.get())()});
        mQkUpdateInfo.update_kernel_args.push_back({0, 6, sizeof(cl_mem), &(*(mPastKey.get()))()});
        mQkUpdateInfo.update_kernel_args.push_back({0, 10, sizeof(mKv_seq_len), &mKv_seq_len});
        mQkGlobal_size[0] = mGlobalWorkSizeQk[0];
        mQkGlobal_size[1] = mGlobalWorkSizeQk[1];
        mQkGlobal_size[2] = mGlobalWorkSizeQk[2];
        mQkUpdateInfo.update_global_size.push_back({0, mQkGlobal_size});
        mOpRecordUpdateInfo.emplace_back(&mQkUpdateInfo);
        mOpenCLBackend->recordKernel3d(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk, &mQkUpdateInfo);
    }
    
    // softmax
    {
        auto MaxLocalSize = std::min(std::min(runtime->getMaxWorkItemSizes()[0], mMaxWorkGroupSize), static_cast<uint32_t>(512));
        int localSize = getLocalSize(mKv_seq_len, MaxLocalSize);
        if(localSize < 4){
            localSize = 1;
        }
        int past_len4 = UP_DIV(mKv_seq_len, 4);
        mSoftMaxRemainChannels = past_len4 * 4 - mKv_seq_len;
        mSoftmaxShape[0] = mNumHead;
        mSoftmaxShape[1] = past_len4;
        mSoftmaxShape[2] = 1;
        mSoftmaxShape[3] = mPastLength;
        std::set<std::string> buildOption;
        buildOption.emplace("-DSOFTMAX_LOCAL_SIZE=" + std::to_string(localSize));
        if(!mIsDecode){
            mKernel_softmax = runtime->buildKernel("softmax_buf", "softmax_width", buildOption, inputs[0], outputs[0]);
            mGlobalWorkSizeSoftMax =  {static_cast<uint32_t>(localSize), static_cast<uint32_t>(past_len4), static_cast<uint32_t>(mNumHead)};
        } else{
            mKernel_softmax = runtime->buildKernel("softmax_buf", "softmax_channel", buildOption, inputs[0], outputs[0]);
            mSoftmaxShape[3] = 1;
            mGlobalWorkSizeSoftMax =  {static_cast<uint32_t>(localSize), static_cast<uint32_t>(1), static_cast<uint32_t>(mNumHead)};
        }
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_softmax));
        
        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[0]);
        ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[1]);
        ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[2]);
        ret |= mKernel_softmax->get().setArg(index++, openCLBuffer(mTempQK.get()));
        ret |= mKernel_softmax->get().setArg(index++, openCLBuffer(mTempSoftMax.get()));
        ret |= mKernel_softmax->get().setArg(index++, mSoftMaxRemainChannels);
        ret |= mKernel_softmax->get().setArg(index++, mSoftmaxShape);
        MNN_CHECK_CL_SUCCESS(ret, "setArg softmax");

        mLocalWorkSizeSoftMax = {static_cast<uint32_t>(localSize), 1, 1};
        if(localSize == 1){
            mLocalWorkSizeSoftMax = localWS3DDefault(mGlobalWorkSizeSoftMax, maxWorkGroupSize, runtime, "softmax", mKernel_softmax).first;
        }
        mGlobalWorkSizeSoftMax[0] = ROUND_UP(mGlobalWorkSizeSoftMax[0], std::max((uint32_t)1, mLocalWorkSizeSoftMax[0]));
        mGlobalWorkSizeSoftMax[1] = ROUND_UP(mGlobalWorkSizeSoftMax[1], std::max((uint32_t)1, mLocalWorkSizeSoftMax[1]));
        mGlobalWorkSizeSoftMax[2] = ROUND_UP(mGlobalWorkSizeSoftMax[2], std::max((uint32_t)1, mLocalWorkSizeSoftMax[2]));
        mSoftMaxUpdateInfo.update_kernel_args.push_back({0, 3, sizeof(cl_mem), &openCLBuffer(mTempQK.get())()});
        mSoftMaxUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &openCLBuffer(mTempSoftMax.get())()});
        mSoftMaxUpdateInfo.update_kernel_args.push_back({0, 5, sizeof(mSoftMaxRemainChannels), &mSoftMaxRemainChannels});
        mSoftMaxUpdateInfo.update_kernel_args.push_back({0, 6, sizeof(mSoftmaxShape), &mSoftmaxShape});
        mOpRecordUpdateInfo.emplace_back(&mSoftMaxUpdateInfo);
        mOpenCLBackend->recordKernel3d(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax, &mSoftMaxUpdateInfo);
    }
    
    // qk * value
    {
        std::set<std::string> buildOption;
        if(!mIsDecode){
            buildOption.emplace("-DOPENCL_PREFILL_ATTENTION");
        }
        if((mHeadDim % 4) != 0){
            buildOption.emplace("-DHEADDIM_LEAVE");
        }
        buildOption.emplace("-DNUMHEAD_GROUP_SIZE=" + std::to_string(group_size));
        mKernel_qkv = runtime->buildKernel("attention_buf", "matmul_qkv", buildOption, inputs[0], outputs[0]);
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_qkv));
        mGlobalWorkSizeQkv =  {static_cast<uint32_t>(UP_DIV(seq_len, 4)), static_cast<uint32_t>(mNumHead), static_cast<uint32_t>(UP_DIV(mHeadDim, 4))};
        
        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_qkv->get().setArg(index++, mGlobalWorkSizeQkv[0]);
        ret |= mKernel_qkv->get().setArg(index++, mGlobalWorkSizeQkv[1]);
        ret |= mKernel_qkv->get().setArg(index++, mGlobalWorkSizeQkv[2]);
        ret |= mKernel_qkv->get().setArg(index++, openCLBuffer(mTempSoftMax.get()));
        ret |= mKernel_qkv->get().setArg(index++, openCLBuffer(value));
        ret |= mKernel_qkv->get().setArg(index++, openCLBuffer(outputs[0]));
        ret |= mKernel_qkv->get().setArg(index++, *mPastValue.get());
        ret |= mKernel_qkv->get().setArg(index++, seq_len);
        ret |= mKernel_qkv->get().setArg(index++, mKv_seq_len);
        ret |= mKernel_qkv->get().setArg(index++, mNumHead);
        ret |= mKernel_qkv->get().setArg(index++, mKvNumHead);
        ret |= mKernel_qkv->get().setArg(index++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qkv");

        mLocalWorkSizeQkv = localWS3DDefault(mGlobalWorkSizeQkv, maxWorkGroupSize, runtime, "matmul_qkv", mKernel_qkv).first;
        mGlobalWorkSizeQkv[0] = ROUND_UP(mGlobalWorkSizeQkv[0], std::max((uint32_t)1, mLocalWorkSizeQkv[0]));
        mGlobalWorkSizeQkv[1] = ROUND_UP(mGlobalWorkSizeQkv[1], std::max((uint32_t)1, mLocalWorkSizeQkv[1]));
        mGlobalWorkSizeQkv[2] = ROUND_UP(mGlobalWorkSizeQkv[2], std::max((uint32_t)1, mLocalWorkSizeQkv[2]));
        
        mQkvUpdateInfo.update_kernel_args.push_back({0, 3, sizeof(cl_mem), &openCLBuffer(mTempSoftMax.get())()});
        mQkvUpdateInfo.update_kernel_args.push_back({0, 6, sizeof(cl_mem), &(*(mPastValue.get()))()});
        mQkvUpdateInfo.update_kernel_args.push_back({0, 8, sizeof(mKv_seq_len), &mKv_seq_len});
        mOpRecordUpdateInfo.emplace_back(&mQkvUpdateInfo);
        mOpenCLBackend->recordKernel3d(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv, &mQkvUpdateInfo);
    }
    
    mOpenCLBackend->endRecord(mRecording);
    
    mOpenCLBackend->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mTempSoftMax.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode AttentionBufImpl::onExecute(Backend *backend, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start AttentionBufExecution onExecute !\n");
#endif
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    reallocKVCache();
#ifdef ENABLE_OPENCL_TIME_PROFILER
    {
        cl::Event event;
        run3DKernelDefault(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk,
                           mOpenCLBackend->getOpenCLRuntime(), &event);
        
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"matmul_qk_div_mask", event});
    }
    {
        cl::Event event;
        run3DKernelDefault(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax,
                           mOpenCLBackend->getOpenCLRuntime(), &event);
        
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"softmax", event});
    }
    {
        cl::Event event;
        run3DKernelDefault(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv,
                           mOpenCLBackend->getOpenCLRuntime(), &event);
        
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"matmul_qkv", event});
    }
#else
    if(mOpenCLBackend->isUseRecordQueue()){
        mOpenCLBackend->addRecord(mRecording, mOpRecordUpdateInfo);
        if(mIsDecode){
            if(mIsFirstDecode){
                mIsFirstDecode = false;
            }else{
                mPastLength += 1;
                mKv_seq_len = mPastLength + 1;
                int past_len4 = UP_DIV(mKv_seq_len, 4);
                mSoftMaxRemainChannels = past_len4 * 4 - mKv_seq_len;
                mSoftmaxShape[1] = past_len4;
                mGlobalWorkSizeQk2 = past_len4;
                mQkGlobal_size[2] = ROUND_UP(mGlobalWorkSizeQk2, std::max((uint32_t)1, mLocalWorkSizeQk[2]));
            }
        }
#ifdef LOG_VERBOSE
        MNN_PRINT("End AttentionBufExecution onExecute... \n");
#endif
        return NO_ERROR;
    }
    run3DKernelDefault(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk, mOpenCLBackend->getOpenCLRuntime());
    run3DKernelDefault(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax, mOpenCLBackend->getOpenCLRuntime());
    run3DKernelDefault(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv, mOpenCLBackend->getOpenCLRuntime());
#endif
    
    // decode
    if(mIsDecode){
        mPastLength += 1;
        mKv_seq_len = mPastLength + 1;
        int past_len4 = UP_DIV(mKv_seq_len, 4);
        mSoftMaxRemainChannels = past_len4 * 4 - mKv_seq_len;
        mSoftmaxShape[1] = past_len4;
        cl_int ret = CL_SUCCESS;
        mGlobalWorkSizeQk2 = past_len4;
        mGlobalWorkSizeQk[2] = ROUND_UP(mGlobalWorkSizeQk2, std::max((uint32_t)1, mLocalWorkSizeQk[2]));
        ret |= mKernel_qk->get().setArg(2, mGlobalWorkSizeQk2);
        ret |= mKernel_qk->get().setArg(10, mKv_seq_len);
        ret |= mKernel_softmax->get().setArg(5, mSoftMaxRemainChannels);
        ret |= mKernel_softmax->get().setArg(6, mSoftmaxShape);
        ret |= mKernel_qkv->get().setArg(8, mKv_seq_len);
        MNN_CHECK_CL_SUCCESS(ret, "reset arg for AttentionBufExecution");
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end AttentionBufExecution onExecute !\n");
#endif

    return NO_ERROR;
}

AttentionBufExecution::AttentionBufExecution(const MNN::Op *op, Backend* backend, bool kv_cahce) : CommonExecution(backend, op) {
    mImpl.reset(new AttentionBufImpl(op, backend, kv_cahce));
}

AttentionBufExecution::AttentionBufExecution(std::shared_ptr<AttentionBufImpl> impl, const MNN::Op *op, Backend *backend) : CommonExecution(backend, op), mImpl(impl) {}

ErrorCode AttentionBufExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return mImpl->onResize(backend(), inputs, outputs);
}

ErrorCode AttentionBufExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return mImpl->onExecute(backend(), inputs, outputs);
}

bool AttentionBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    *dst = new AttentionBufExecution(mImpl, op, bn);
    return true;
}

class AttentionBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        auto param = op->main_as_AttentionParam();
        return new AttentionBufExecution(op, backend, param->kv_cache());
    }
};
REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(AttentionBufCreator, OpType_Attention, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
