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

KVCacheCLManager::KVCacheCLManager(Backend *backend, bool kv_cahce) : mKVCache(kv_cahce){
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
}

void KVCacheCLManager::allocKVCache() {
    if (!mKVCache) {
        return;
    }
    if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()){
        mByte = 2;
    }
    mMaxLength = mPastLength + mExpandChunk;
    size_t buffer_size = UP_DIV(mMaxLength, 4) * mKvNumHead * mHeadDim * 4 * mByte;
    // past_key: [1, numhead, headdim, maxlen]
    mPastKey.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
    // past_value: [1, numhead, maxlen, headdim]
    mPastValue.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
}

bool KVCacheCLManager::reallocKVCache() {
    if (!mKVCache || mPastLength < mMaxLength) {
        return false;
    }
    size_t old_size = mKvNumHead * UP_DIV(mMaxLength, 4) * mHeadDim * 4 * mByte;
    size_t old_maxlen = ROUND_UP(mMaxLength, 4);
    mMaxLength = mPastLength + mExpandChunk;
    size_t new_maxlen = ROUND_UP(mMaxLength, 4);
    size_t buffer_size = UP_DIV(mMaxLength, 4) * mKvNumHead * mHeadDim * 4 * mByte;
    // past_key: [1, numhead, headdim, maxlen]
    auto new_key = new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    // past_value: [1, numhead, maxlen, headdim]
    auto new_value = new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    cl_int res;
    // copy key
    {
        size_t old_maxlen_size = old_maxlen * mByte;
        size_t new_maxlen_size = new_maxlen * mByte;
        char *new_key_ptr = (char*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*new_key, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
        char *key_ptr = (char*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*mPastKey.get(), true, CL_MAP_READ, 0, old_size, nullptr, nullptr, &res);
        if(new_key_ptr != nullptr && key_ptr != nullptr && res == CL_SUCCESS){
            for(int i = 0; i < mKvNumHead * mHeadDim; ++i){
                ::memcpy(new_key_ptr + i * new_maxlen_size, key_ptr + i * old_maxlen_size, old_maxlen_size);
            }
        }else{
            MNN_ERROR("Map error key_ptr == nullptr \n");
            MNN_ASSERT(false);
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*new_key, new_key_ptr);
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mPastKey.get(), key_ptr);
    }
    
    // copy value
    {
        char *new_value_ptr = (char*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*new_value, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
        char *value_ptr = (char*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*mPastValue.get(), true, CL_MAP_READ, 0, old_size, nullptr, nullptr, &res);
        if(new_value_ptr != nullptr && value_ptr != nullptr && res == CL_SUCCESS){
            for(int i = 0; i < mKvNumHead; ++i){
                for(int j = 0; j < old_maxlen; ++j){
                    ::memcpy(new_value_ptr + (i * new_maxlen + j) * mHeadDim * mByte, value_ptr + (i * old_maxlen + j) * mHeadDim * mByte, mHeadDim * mByte);
                }
            }
        }else{
            MNN_ERROR("Map error value_ptr == nullptr \n");
            MNN_ASSERT(false);
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*new_value, new_value_ptr);
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mPastValue.get(), value_ptr);
    }
    mPastKey.reset(new_key);
    mPastValue.reset(new_value);
    return true;
}

int AttentionBufExecution::getLocalSize(int size, int maxGroupSize){
    int local_size = 1;
    while(local_size * 2 <= maxGroupSize && local_size * 2 <= size){
        local_size *= 2;
    }
    return local_size;
}

ErrorCode AttentionBufExecution::longPrefillResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs){
    
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto mask = inputs[3];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto shape = query->shape();
       
    int batch = shape[0];
    int seq_len = shape[1];
    int numHead = shape[2];
    int kvNumHead = key->shape()[2];
    int headDim = shape[3];
    int group_size = numHead / kvNumHead;
    float scale = 1.0 / sqrt(headDim);
    mKVCacheCLManager->setArgs(seq_len, numHead, kvNumHead, headDim);
    mKVCacheCLManager->allocKVCache();
    mKv_seq_len = mKVCacheCLManager->kvLength();
    mKeyValueMaxlen = ROUND_UP(mKVCacheCLManager->maxLength(), 4);
    mAlignQ = 128;
    mAlignKV = 128;
    mAlignHDK = 4;
    mAlignHDN = 128;
    
    mTempQ.reset(Tensor::createDevice<float>({ROUND_UP(seq_len, mAlignQ) * ROUND_UP(headDim, mAlignHDK) * batch * numHead}));
    mTempK.reset(Tensor::createDevice<float>({ROUND_UP(seq_len, mAlignKV) * ROUND_UP(headDim, mAlignHDK) * batch * numHead}));
    mTempV.reset(Tensor::createDevice<float>({ROUND_UP(seq_len, mAlignKV) * ROUND_UP(headDim, mAlignHDN) * batch * numHead}));
    if(mIsAddMask) {
        mTempMask.reset(Tensor::createDevice<float>({ROUND_UP(seq_len, mAlignQ) * ROUND_UP(seq_len, mAlignKV) * batch}));
    } else {
        mTempMask.reset(Tensor::createDevice<uint32_t>({ROUND_UP(seq_len, mAlignQ) * ROUND_UP(seq_len, mAlignKV) * batch}));
    }
    mTempQK.reset(Tensor::createDevice<float>({ROUND_UP(seq_len, mAlignQ) * ROUND_UP(seq_len, mAlignKV) * batch * numHead}));
    mTempSoftMax.reset(Tensor::createDevice<float>({ROUND_UP(seq_len, mAlignQ) * ROUND_UP(seq_len, mAlignKV) * batch * numHead}));
    mTempQKV.reset(Tensor::createDevice<float>({ROUND_UP(seq_len, mAlignQ) * ROUND_UP(headDim, mAlignHDN) * batch * numHead}));
    
    
    mOpenCLBackend->onAcquireBuffer(mTempQ.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempK.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempV.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempMask.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC);

    mOpenCLBackend->onReleaseBuffer(mTempQ.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mTempK.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mTempMask.get(), Backend::DYNAMIC);

    mOpenCLBackend->onAcquireBuffer(mTempSoftMax.get(), Backend::DYNAMIC);
        
    mOpenCLBackend->onReleaseBuffer(mTempSoftMax.get(), Backend::DYNAMIC);
        
    mOpenCLBackend->onAcquireBuffer(mTempQKV.get(), Backend::DYNAMIC);

    mOpenCLBackend->onReleaseBuffer(mTempV.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mTempQKV.get(), Backend::DYNAMIC);
    
    // query: [batch, seqLenQ, headNum, headDim] -> mTempQ: [batch*headNum, ROUND_UP(headDim, mAlignHDK), ROUND_UP(seqLenQ, mAlignQ)]
    // key: [batch, seqLenKV/4, headNum/group, headDim, seqLenKV_4] -> mTempK: [batch*headNum/group, ROUND_UP(headDim, mAlignHDK), ROUND_UP(seqLenKV, mAlignKV)]
    // value: [batch, seqLenKV/4, headNum/group, headDim, seqLenKV_4] -> mTempV: [batch*headNum/group, ROUND_UP(seqLenKV, mAlignKV), ROUND_UP(headDim, mAlignHDK]
    // key & value -> pastKey & pastValue (copy)
    {
        std::set<std::string> buildOption;
        if((headDim % 4) != 0){
            buildOption.emplace("-DHEADDIM_LEAVE");
        }
        if((seq_len % 4) != 0){
            buildOption.emplace("-DSEQLEN_LEAVE");
        }
        
        int seq_len_pack_q = ROUND_UP(seq_len, mAlignQ);
        int seq_len_pack_kv = ROUND_UP(mKv_seq_len, mAlignKV);

        int head_dim_pack_qk = ROUND_UP(headDim, mAlignHDK);
        int head_dim_pack_v = ROUND_UP(headDim, mAlignHDN);

        int tile[4] = {mAlignQ, mAlignKV, mAlignHDK, mAlignHDN};
        int shape[4] = {seq_len, mKv_seq_len, numHead, headDim};
        int param[4] = {group_size, batch, 0, 0};
        mKernel_rearrange = runtime->buildKernel("attention_buf", "rearrange_qkv", buildOption, inputs[0], outputs[0]);
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_rearrange));
        
        mGlobalWorkSizeRearrg = {static_cast<uint32_t>(ALIMAX(UP_DIV(seq_len_pack_q, 4), UP_DIV(seq_len_pack_kv, 4))), \
                                static_cast<uint32_t>(ALIMAX(UP_DIV(head_dim_pack_qk, 4), UP_DIV(head_dim_pack_v, 4))), \
                                static_cast<uint32_t>(batch*numHead)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[0]);
        ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[1]);
        ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[2]);
        ret |= mKernel_rearrange->get().setArg(index++, openCLBuffer(query));
        ret |= mKernel_rearrange->get().setArg(index++, openCLBuffer(key));
        ret |= mKernel_rearrange->get().setArg(index++, openCLBuffer(value));
        ret |= mKernel_rearrange->get().setArg(index++, openCLBuffer(mTempQ.get()));
        ret |= mKernel_rearrange->get().setArg(index++, openCLBuffer(mTempK.get()));
        ret |= mKernel_rearrange->get().setArg(index++, openCLBuffer(mTempV.get()));
        ret |= mKernel_rearrange->get().setArg(index++, *mKVCacheCLManager->key());
        ret |= mKernel_rearrange->get().setArg(index++, *mKVCacheCLManager->value());
        ret |= mKernel_rearrange->get().setArg(index++, tile);
        ret |= mKernel_rearrange->get().setArg(index++, shape);
        ret |= mKernel_rearrange->get().setArg(index++, param);
        ret |= mKernel_rearrange->get().setArg(index++, mKeyValueMaxlen);
        
        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_qkv");
        mLocalWorkSizeRearrg = localWS3DDefault(mGlobalWorkSizeRearrg, maxWorkGroupSize, runtime, "rearrange_qkv", mKernel_rearrange).first;
        mGlobalWorkSizeRearrg[0] = ROUND_UP(mGlobalWorkSizeRearrg[0], std::max((uint32_t)1, mLocalWorkSizeRearrg[0]));
        mGlobalWorkSizeRearrg[1] = ROUND_UP(mGlobalWorkSizeRearrg[1], std::max((uint32_t)1, mLocalWorkSizeRearrg[1]));
        mGlobalWorkSizeRearrg[2] = ROUND_UP(mGlobalWorkSizeRearrg[2], std::max((uint32_t)1, mLocalWorkSizeRearrg[2]));
        mRgUpdateInfo.update_kernel_args.push_back({0, 9, sizeof(cl_mem), &(*(mKVCacheCLManager->key()))()});
        mRgUpdateInfo.update_kernel_args.push_back({0, 10, sizeof(cl_mem), &(*(mKVCacheCLManager->value()))()});
        mRgUpdateInfo.update_kernel_args.push_back({0, 14, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
        mOpRecordUpdateInfo.emplace_back(&mRgUpdateInfo);
        mOpenCLBackend->recordKernel3d(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg, &mRgUpdateInfo);
    }
    
    // mask rearaange
    {
        std::set<std::string> buildOption;

        int seq_len_pack_q = ROUND_UP(seq_len, mAlignQ);
        int seq_len_pack_kv = ROUND_UP(mKv_seq_len, mAlignKV);
        int shape[4] = {seq_len, mKv_seq_len, mAlignQ, mAlignKV};

        mKernel_mask = runtime->buildKernel("attention_buf", "rearrange_mask", buildOption, inputs[0], outputs[0]);
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_mask));
        
        mGlobalWorkSizeMask = {static_cast<uint32_t>(UP_DIV(seq_len_pack_q, 4)), \
                                static_cast<uint32_t>(UP_DIV(seq_len_pack_kv, 4)), \
                                static_cast<uint32_t>(batch)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_mask->get().setArg(index++, mGlobalWorkSizeMask[0]);
        ret |= mKernel_mask->get().setArg(index++, mGlobalWorkSizeMask[1]);
        ret |= mKernel_mask->get().setArg(index++, mGlobalWorkSizeMask[2]);
        ret |= mKernel_mask->get().setArg(index++, openCLBuffer(mask));
        ret |= mKernel_mask->get().setArg(index++, openCLBuffer(mTempMask.get()));
        ret |= mKernel_mask->get().setArg(index++, shape);
        
        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_mask");
        mLocalWorkSizeMask = localWS3DDefault(mGlobalWorkSizeMask, maxWorkGroupSize, runtime, "rearrange_mask", mKernel_mask).first;
        mGlobalWorkSizeMask[0] = ROUND_UP(mGlobalWorkSizeMask[0], std::max((uint32_t)1, mLocalWorkSizeMask[0]));
        mGlobalWorkSizeMask[1] = ROUND_UP(mGlobalWorkSizeMask[1], std::max((uint32_t)1, mLocalWorkSizeMask[1]));
        mGlobalWorkSizeMask[2] = ROUND_UP(mGlobalWorkSizeMask[2], std::max((uint32_t)1, mLocalWorkSizeMask[2]));
        mOpenCLBackend->recordKernel3d(mKernel_mask, mGlobalWorkSizeMask, mLocalWorkSizeMask);
    }
    
    {
        // Q : [batch*headNum, ROUND_UP(headDim, mAlignHDK), ROUND_UP(seqLenQ, mAlignQ)] -> [B, K, M]
        // K : [batch*headNum/group, ROUND_UP(headDim, mAlignHDK), ROUND_UP(seqLenKV, mAlignKV)] -> [B, K, N]
        // QV: [Batch * numHead, ROUND_UP(seqLenQ, mAlignQ), ROUND_UP(seqLenKV, mAlignKV)]   -> [B, M, N]
        int loop = batch * numHead;
        int e_pack = ROUND_UP(seq_len, mAlignQ);
        int h_pack = ROUND_UP(mKv_seq_len, mAlignKV);
        int l_pack = ROUND_UP(headDim, mAlignHDK);
        
        std::set<std::string> buildOptions;

        int biasType = 5;// int value mask
        if(mIsAddMask) {
            biasType = 2;
        }
        uint32_t layout = 14; // 10 means mix-precision, 4 means layput
        auto param = getGemmParams({(uint32_t)e_pack, (uint32_t)h_pack, (uint32_t)l_pack, layout, (uint32_t)loop, (uint32_t)(biasType + 10*(group_size-1))}, {openCLBuffer(mTempQ.get()), openCLBuffer(mTempK.get()), openCLBuffer(mTempQK.get()), openCLBuffer(mTempMask.get())}, mOpenCLBackend->getOpenCLRuntime());
        
        int KWG=param[0], KWI=param[1], MDIMA=param[2], MDIMC=param[3], MWG=param[4], NDIMB=param[5], NDIMC=param[6], NWG=param[7], SA=param[8], SB=param[9], STRM=param[10], STRN=param[11], VWM=param[12], VWN=param[13];
        buildOptions.emplace("-DKWG=" + std::to_string(KWG));
        buildOptions.emplace("-DKWI=" + std::to_string(KWI));
        buildOptions.emplace("-DMDIMA=" + std::to_string(MDIMA));
        buildOptions.emplace("-DMDIMC=" + std::to_string(MDIMC));
        buildOptions.emplace("-DMWG=" + std::to_string(MWG));
        buildOptions.emplace("-DNDIMB=" + std::to_string(NDIMB));
        buildOptions.emplace("-DNDIMC=" + std::to_string(NDIMC));
        buildOptions.emplace("-DNWG=" + std::to_string(NWG));
        buildOptions.emplace("-DSA=" + std::to_string(SA));
        buildOptions.emplace("-DSB=" + std::to_string(SB));
        buildOptions.emplace("-DSTRM=" + std::to_string(STRM));
        buildOptions.emplace("-DSTRN=" + std::to_string(STRN));
        buildOptions.emplace("-DVWM=" + std::to_string(VWM));
        buildOptions.emplace("-DVWN=" + std::to_string(VWN));
        if(layout >= 4) {
            buildOptions.emplace("-DOUTPUTMN");
        }
        
        int tileM = MWG;
        int tileN = NWG;
        int localM = MDIMC;
        int localN = NDIMC;
        
        if(mOpenCLBackend->getOpenCLRuntime()->getGpuType() == GpuType::ADRENO) {
            buildOptions.emplace("-DUSE_CL_MAD=1");
            buildOptions.emplace("-DRELAX_WORKGROUP_SIZE=1");
        }
        buildOptions.emplace("-DONLY_HAVE_ALPHA");
        buildOptions.emplace("-DBIAS_TYPE=" + std::to_string(biasType));
        
        buildOptions.emplace("-DPRECISION_COMPUTE=float -DCONVERT_PRECISION_COMPUTE=convert_float");
        buildOptions.emplace("-DPRECISION_COMPUTE2=float2 -DCONVERT_PRECISION_COMPUTE2=convert_float2");
        buildOptions.emplace("-DPRECISION_COMPUTE4=float4 -DCONVERT_PRECISION_COMPUTE4=convert_float4");
        buildOptions.emplace("-DPRECISION_COMPUTE8=float8 -DCONVERT_PRECISION_COMPUTE8=convert_float8");
        buildOptions.emplace("-DPRECISION_COMPUTE16=float16 -DCONVERT_PRECISION_COMPUTE16=convert_float16");

        mKernel_qk = mOpenCLBackend->getOpenCLRuntime()->buildKernel("matmul_params_buf", "XgemmBatched", buildOptions);
        
        int out_per_thread_m = tileM / localM;
        int out_per_thread_n = tileN / localN;
        
        mGlobalWorkSizeQk = {static_cast<uint32_t>(e_pack/out_per_thread_m), static_cast<uint32_t>(h_pack/out_per_thread_n), static_cast<uint32_t>(loop)};
        mLocalWorkSizeQk = {static_cast<uint32_t>(localM), static_cast<uint32_t>(localN), 1};
        
        float alpha = scale;
        float beta = 0.0f;
        int batch_offset_a = e_pack * l_pack;
        int batch_offset_b = h_pack * l_pack;
        int batch_offset_c = e_pack * h_pack;
        
        int batch_offset[4] = {batch_offset_a, batch_offset_b, batch_offset_c, 0};
        int base_ptr_offset[4] = {0, 0, 0, 0};
        int stride[4] = {e_pack, h_pack, h_pack, h_pack};
        int group[4] = {1, group_size, 1, numHead};
        
        int idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_qk->get().setArg(idx++, static_cast<int>(e_pack));
        ret |= mKernel_qk->get().setArg(idx++, static_cast<int>(h_pack));
        ret |= mKernel_qk->get().setArg(idx++, static_cast<int>(l_pack));
        ret |= mKernel_qk->get().setArg(idx++, alpha);
        ret |= mKernel_qk->get().setArg(idx++, beta);
        ret |= mKernel_qk->get().setArg(idx++, openCLBuffer(mTempQ.get()));
        ret |= mKernel_qk->get().setArg(idx++, openCLBuffer(mTempK.get()));
        ret |= mKernel_qk->get().setArg(idx++, openCLBuffer(mTempMask.get()));
        ret |= mKernel_qk->get().setArg(idx++, openCLBuffer(mTempQK.get()));
        ret |= mKernel_qk->get().setArg(idx++, batch_offset);
        ret |= mKernel_qk->get().setArg(idx++, base_ptr_offset);
        ret |= mKernel_qk->get().setArg(idx++, stride);
        ret |= mKernel_qk->get().setArg(idx++, group);
        MNN_CHECK_CL_SUCCESS(ret, "setArg Self-Attention batchmatmul qk Kernel");
        mOpenCLBackend->recordKernel3d(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk);
    }
    
    // softmax
    {
        // QV:     [Batch * numHead, ROUND_UP(seqLenQ, mAlignQ), ROUND_UP(seqLenKV, mAlignKV)]
        // Sotmax: [Batch * numHead, ROUND_UP(seqLenQ, mAlignQ), ROUND_UP(seqLenKV, mAlignKV)]
        // axis  : 2 (last dim)
        int softmaxShape[4];
        softmaxShape[0] = batch*numHead;
        softmaxShape[1] = ROUND_UP(seq_len, mAlignQ);
        softmaxShape[2] = ROUND_UP(mKv_seq_len, mAlignKV);
        
        auto MaxLocalSize = std::min(std::min(runtime->getMaxWorkItemSizes()[0], mMaxWorkGroupSize), static_cast<uint32_t>(256));
        int localSize = getLocalSize(softmaxShape[2], MaxLocalSize);
        if(localSize < 4){
            localSize = 1;
        }
        
        std::set<std::string> buildOption;
        buildOption.emplace("-DSOFTMAX_LOCAL_SIZE=" + std::to_string(localSize));
        
        mKernel_softmax = runtime->buildKernel("self_attention_buf", "softmax_inside", buildOption, inputs[0], outputs[0]);
        mGlobalWorkSizeSoftMax =  {static_cast<uint32_t>(localSize), static_cast<uint32_t>(softmaxShape[1]), static_cast<uint32_t>(softmaxShape[0])};
        
        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[0]);
        ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[1]);
        ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[2]);
        ret |= mKernel_softmax->get().setArg(index++, openCLBuffer(mTempQK.get()));
        ret |= mKernel_softmax->get().setArg(index++, openCLBuffer(mTempSoftMax.get()));
        ret |= mKernel_softmax->get().setArg(index++, mKv_seq_len);
        ret |= mKernel_softmax->get().setArg(index++, softmaxShape);
        MNN_CHECK_CL_SUCCESS(ret, "setArg Attention softmax");
        
        mLocalWorkSizeSoftMax = {static_cast<uint32_t>(localSize), 1, 1};
        mOpenCLBackend->recordKernel3d(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax);
    }
    {
        // Sotmax: [Batch * numHead, ROUND_UP(seqLenQ, mAlignQ), ROUND_UP(seqLenKV, mAlignKV)]
        // Trans:  [Batch * numHead, ROUND_UP(seqLenKV, mAlignKV), ROUND_UP(seqLenQ, mAlignQ)]
        int loop = batch * numHead;
        int transDimW = ROUND_UP(seq_len, mAlignQ);
        int transDimH = ROUND_UP(mKv_seq_len, mAlignKV);
        
        std::set<std::string> buildOptions;
        mKernel_trans = runtime->buildKernel("self_attention_buf", "trans_3d_buf", buildOptions, inputs[0], outputs[0]);
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel_trans));

        mGlobalWorkSizeTrans = {(uint32_t)transDimW/8, (uint32_t)transDimH/8, (uint32_t)(loop)};
        
        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_trans->get().setArg(index++, mGlobalWorkSizeTrans[0]);
        ret |= mKernel_trans->get().setArg(index++, mGlobalWorkSizeTrans[1]);
        ret |= mKernel_trans->get().setArg(index++, mGlobalWorkSizeTrans[2]);
        ret |= mKernel_trans->get().setArg(index++, openCLBuffer(mTempSoftMax.get()));
        ret |= mKernel_trans->get().setArg(index++, openCLBuffer(mTempQK.get()));
        ret |= mKernel_trans->get().setArg(index++, loop);
        ret |= mKernel_trans->get().setArg(index++, transDimW);
        ret |= mKernel_trans->get().setArg(index++, transDimH);
        MNN_CHECK_CL_SUCCESS(ret, "setArg Attention transpose");
        mLocalWorkSizeTrans = localWS3DDefault(mGlobalWorkSizeTrans, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "trans_3d_buf", mKernel_trans).first;
        
        mGlobalWorkSizeTrans[0] = ROUND_UP(mGlobalWorkSizeTrans[0], std::max((uint32_t)1, mLocalWorkSizeTrans[0]));
        mGlobalWorkSizeTrans[1] = ROUND_UP(mGlobalWorkSizeTrans[1], std::max((uint32_t)1, mLocalWorkSizeTrans[1]));
        mGlobalWorkSizeTrans[2] = ROUND_UP(mGlobalWorkSizeTrans[2], std::max((uint32_t)1, mLocalWorkSizeTrans[2]));
        
        mOpenCLBackend->recordKernel3d(mKernel_trans, mGlobalWorkSizeTrans, mLocalWorkSizeTrans);
    }

    // qk * value
    {
        // Trans: [Batch * numHead, ROUND_UP(seqLenKV, mAlignKV), ROUND_UP(seqLenQ, mAlignQ)]   -> [B, K, M]
        // V :     [Batch * numHead / group, ROUND_UP(seqLenKV, mAlignKV), ROUND_UP(headDim, mAlignHDN)] -> [B, K, N]
        // QKV :   [Batch * numHead, ROUND_UP(headDim, mAlignHDN), ROUND_UP(seqLenQ, mAlignQ)] -> [B, N, M]
        
        int loop = batch * numHead;
        int e_pack = ROUND_UP(seq_len, mAlignQ);
        int l_pack = ROUND_UP(mKv_seq_len, mAlignKV);
        int h_pack = ROUND_UP(headDim, mAlignHDN);
        
        std::set<std::string> buildOptions;

        uint32_t layout = 0;
        auto param = getGemmParams({(uint32_t)e_pack, (uint32_t)h_pack, (uint32_t)l_pack, layout, (uint32_t)loop, (uint32_t)0}, {openCLBuffer(mTempQK.get()), openCLBuffer(mTempV.get()), openCLBuffer(mTempQKV.get())}, mOpenCLBackend->getOpenCLRuntime());

        int KWG=param[0], KWI=param[1], MDIMA=param[2], MDIMC=param[3], MWG=param[4], NDIMB=param[5], NDIMC=param[6], NWG=param[7], SA=param[8], SB=param[9], STRM=param[10], STRN=param[11], VWM=param[12], VWN=param[13];
        buildOptions.emplace("-DKWG=" + std::to_string(KWG));
        buildOptions.emplace("-DKWI=" + std::to_string(KWI));
        buildOptions.emplace("-DMDIMA=" + std::to_string(MDIMA));
        buildOptions.emplace("-DMDIMC=" + std::to_string(MDIMC));
        buildOptions.emplace("-DMWG=" + std::to_string(MWG));
        buildOptions.emplace("-DNDIMB=" + std::to_string(NDIMB));
        buildOptions.emplace("-DNDIMC=" + std::to_string(NDIMC));
        buildOptions.emplace("-DNWG=" + std::to_string(NWG));
        buildOptions.emplace("-DSA=" + std::to_string(SA));
        buildOptions.emplace("-DSB=" + std::to_string(SB));
        buildOptions.emplace("-DSTRM=" + std::to_string(STRM));
        buildOptions.emplace("-DSTRN=" + std::to_string(STRN));
        buildOptions.emplace("-DVWM=" + std::to_string(VWM));
        buildOptions.emplace("-DVWN=" + std::to_string(VWN));
        if(layout >= 4) {
            buildOptions.emplace("-DOUTPUTMN");
        }
        
        int tileM = MWG;
        int tileN = NWG;
        int localM = MDIMC;
        int localN = NDIMC;
        
        if(mOpenCLBackend->getOpenCLRuntime()->getGpuType() == GpuType::ADRENO) {
            buildOptions.emplace("-DUSE_CL_MAD=1");
            buildOptions.emplace("-DRELAX_WORKGROUP_SIZE=1");
        }

        mKernel_qkv = mOpenCLBackend->getOpenCLRuntime()->buildKernel("matmul_params_buf", "XgemmBatched", buildOptions);
        
        int out_per_thread_m = tileM / localM;
        int out_per_thread_n = tileN / localN;
        
        mGlobalWorkSizeQkv = {static_cast<uint32_t>(e_pack/out_per_thread_m), static_cast<uint32_t>(h_pack/out_per_thread_n), static_cast<uint32_t>(loop)};
        mLocalWorkSizeQkv = {static_cast<uint32_t>(localM), static_cast<uint32_t>(localN), 1};
        
        float alpha = 1.0f;
        float beta = 0.0f;
        int batch_offset_a = e_pack * l_pack;
        int batch_offset_b = h_pack * l_pack;
        int batch_offset_c = e_pack * h_pack;
        int batch_offset[4] = {batch_offset_a, batch_offset_b, batch_offset_c, 0};
        int base_ptr_offset[4] = {0, 0, 0, 0};
        int stride[4] = {e_pack, h_pack, e_pack, h_pack};
        int group[4] = {1, group_size, 1, numHead};
        
        int idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_qkv->get().setArg(idx++, static_cast<int>(e_pack));
        ret |= mKernel_qkv->get().setArg(idx++, static_cast<int>(h_pack));
        ret |= mKernel_qkv->get().setArg(idx++, static_cast<int>(l_pack));
        ret |= mKernel_qkv->get().setArg(idx++, alpha);
        ret |= mKernel_qkv->get().setArg(idx++, beta);
        ret |= mKernel_qkv->get().setArg(idx++, openCLBuffer(mTempQK.get()));
        ret |= mKernel_qkv->get().setArg(idx++, openCLBuffer(mTempV.get()));
        ret |= mKernel_qkv->get().setArg(idx++, openCLBuffer(mTempQKV.get()));
        ret |= mKernel_qkv->get().setArg(idx++, batch_offset);
        ret |= mKernel_qkv->get().setArg(idx++, base_ptr_offset);
        ret |= mKernel_qkv->get().setArg(idx++, stride);
        ret |= mKernel_qkv->get().setArg(idx++, group);
        MNN_CHECK_CL_SUCCESS(ret, "setArg Self-Attention batchmatmul qkv Kernel");
        mOpenCLBackend->recordKernel3d(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv);
    }
    
    // transpose to output
    {
        // QKV :   [Batch * numHead, ROUND_UP(headDim, mAlignHDN), ROUND_UP(seqLenQ, mAlignQ)] -> [B, N, M]
        // output: [batch, seqLenQ/4, headNum, headDim, seqLenQ_4]
        std::set<std::string> buildOption;
        
        mKernel_clip = runtime->buildKernel("attention_buf", "qkv_transpose_output", buildOption, inputs[0], outputs[0]);
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_clip));
                    
        mGlobalWorkSizeClip = {static_cast<uint32_t>(UP_DIV(seq_len, 4)), static_cast<uint32_t>(UP_DIV(headDim, 4)), static_cast<uint32_t>(batch*numHead)};
        
        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_clip->get().setArg(index++, mGlobalWorkSizeClip[0]);
        ret |= mKernel_clip->get().setArg(index++, mGlobalWorkSizeClip[1]);
        ret |= mKernel_clip->get().setArg(index++, mGlobalWorkSizeClip[2]);
        ret |= mKernel_clip->get().setArg(index++, openCLBuffer(mTempQKV.get()));
        ret |= mKernel_clip->get().setArg(index++, openCLBuffer(outputs[0]));
        ret |= mKernel_clip->get().setArg(index++, mAlignQ);
        ret |= mKernel_clip->get().setArg(index++, mAlignHDN);
        ret |= mKernel_clip->get().setArg(index++, seq_len);
        ret |= mKernel_clip->get().setArg(index++, numHead);
        ret |= mKernel_clip->get().setArg(index++, headDim);

        mLocalWorkSizeClip = localWS3DDefault(mGlobalWorkSizeClip, maxWorkGroupSize, runtime, "qkv_transpose_output", mKernel_clip).first;
        mGlobalWorkSizeClip[0] = ROUND_UP(mGlobalWorkSizeClip[0], std::max((uint32_t)1, mLocalWorkSizeClip[0]));
        mGlobalWorkSizeClip[1] = ROUND_UP(mGlobalWorkSizeClip[1], std::max((uint32_t)1, mLocalWorkSizeClip[1]));
        mGlobalWorkSizeClip[2] = ROUND_UP(mGlobalWorkSizeClip[2], std::max((uint32_t)1, mLocalWorkSizeClip[2]));

        MNN_CHECK_CL_SUCCESS(ret, "setArg qkv_transpose_output");
        mOpenCLBackend->recordKernel3d(mKernel_clip, mGlobalWorkSizeClip, mLocalWorkSizeClip);
    }
    mOpenCLBackend->endRecord(mRecording);

    return NO_ERROR;
}
ErrorCode AttentionBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mOpenCLBackend->startRecord(mRecording);
    //clear update arg vector, if prefill and decode use the same one
    mOpRecordUpdateInfo.clear();
    mRgUpdateInfo.update_kernel_args.clear();
    mRgUpdateInfo.update_global_size.clear();
    mRgUpdateInfo.update_local_size.clear();
    mQkUpdateInfo.update_kernel_args.clear();
    mQkUpdateInfo.update_global_size.clear();
    mQkUpdateInfo.update_local_size.clear();
    mSoftMaxUpdateInfo.update_kernel_args.clear();
    mSoftMaxUpdateInfo.update_global_size.clear();
    mSoftMaxUpdateInfo.update_local_size.clear();
    mRgVUpdateInfo.update_kernel_args.clear();
    mRgVUpdateInfo.update_global_size.clear();
    mRgVUpdateInfo.update_local_size.clear();
    mQkvUpdateInfo.update_kernel_args.clear();
    mQkvUpdateInfo.update_global_size.clear();
    mQkvUpdateInfo.update_local_size.clear();
    
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto mask = inputs[3];
    auto mask_shape = mask->shape();
    int mask_seqlen = mask_shape[2];
    int mask_kvlen  = mask_shape[3];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto shape = query->shape();
    
    int batch = shape[0];
    int seq_len = shape[1];
    int numHead = shape[2];
    int kvNumHead = key->shape()[2];
    int headDim = shape[3];
    int group_size = numHead / kvNumHead;
    float scale = 1.0 / sqrt(headDim);
    mIsDecode = seq_len == 1;
    mIsFirstPrefill = (!mIsDecode) && (mask_kvlen == mask_seqlen);
    
    mIsAddMask = (mask->getType() == halide_type_of<float>());
    mLongPrefill = false;
    if(seq_len > 512 && mIsFirstPrefill) {
        mLongPrefill = true;
        return longPrefillResize(inputs, outputs);
    }
    if(false == mIsDecode){
        mKVCacheCLManager->setArgs(mask_kvlen, numHead, kvNumHead, headDim);
        if(mIsFirstPrefill){
            mKVCacheCLManager->allocKVCache();
        } else{
            mKVCacheCLManager->reallocKVCache();
        }
        mTempQ.reset(Tensor::createDevice<float>({ROUND_UP(seq_len, 4) * ROUND_UP(headDim, 4) * numHead}));
        mTempQK.reset(Tensor::createDevice<float>({ROUND_UP(seq_len, 4) * mask_kvlen * numHead}));
        mTempSoftMax.reset(Tensor::createDevice<float>({ROUND_UP(seq_len, 4) * mask_kvlen * numHead}));
        
        mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC);
        mOpenCLBackend->onAcquireBuffer(mTempSoftMax.get(), Backend::DYNAMIC);
        mOpenCLBackend->onAcquireBuffer(mTempQ.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mTempQ.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mTempSoftMax.get(), Backend::DYNAMIC);
        
        int past_len = mIsFirstPrefill ? 0 : mask_kvlen - mask_seqlen;
        mKeyValueMaxlen = ROUND_UP(mKVCacheCLManager->maxLength(), 4);
        {
            // rearrange query
            std::set<std::string> buildOption;

            mKernel_rearrangeQ = runtime->buildKernel("attention_buf", "rearrange_q", buildOption, inputs[0], outputs[0]);
            auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_rearrangeQ));
            
            mGlobalWorkSizeRearrgQ = {static_cast<uint32_t>(UP_DIV(seq_len, 4)), \
                                    static_cast<uint32_t>(UP_DIV(headDim, 4)), \
                                    static_cast<uint32_t>(numHead)};
  
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_rearrangeQ->get().setArg(index++, mGlobalWorkSizeRearrgQ[0]);
            ret |= mKernel_rearrangeQ->get().setArg(index++, mGlobalWorkSizeRearrgQ[1]);
            ret |= mKernel_rearrangeQ->get().setArg(index++, mGlobalWorkSizeRearrgQ[2]);
            ret |= mKernel_rearrangeQ->get().setArg(index++, openCLBuffer(query));
            ret |= mKernel_rearrangeQ->get().setArg(index++, openCLBuffer(mTempQ.get()));
            ret |= mKernel_rearrangeQ->get().setArg(index++, seq_len);
            ret |= mKernel_rearrangeQ->get().setArg(index++, headDim);
            ret |= mKernel_rearrangeQ->get().setArg(index++, numHead);
            
            MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_q");
            mLocalWorkSizeRearrgQ = localWS3DDefault(mGlobalWorkSizeRearrgQ, maxWorkGroupSize, runtime, "rearrange_q", mKernel_rearrangeQ).first;
            mGlobalWorkSizeRearrgQ[0] = ROUND_UP(mGlobalWorkSizeRearrgQ[0], std::max((uint32_t)1, mLocalWorkSizeRearrgQ[0]));
            mGlobalWorkSizeRearrgQ[1] = ROUND_UP(mGlobalWorkSizeRearrgQ[1], std::max((uint32_t)1, mLocalWorkSizeRearrgQ[1]));
            mGlobalWorkSizeRearrgQ[2] = ROUND_UP(mGlobalWorkSizeRearrgQ[2], std::max((uint32_t)1, mLocalWorkSizeRearrgQ[2]));
            mOpenCLBackend->recordKernel3d(mKernel_rearrangeQ, mGlobalWorkSizeRearrgQ, mLocalWorkSizeRearrgQ);
        }
        {
            // rearrange key
            std::set<std::string> buildOption;
            
            buildOption.emplace("-DOPENCL_PREFILL_ATTENTION");
            mKernel_rearrange = runtime->buildKernel("attention_buf", "rearrange_k", buildOption, inputs[0], outputs[0]);
            auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_rearrange));
            
            mGlobalWorkSizeRearrg = {static_cast<uint32_t>(UP_DIV(seq_len, 4)), \
                                    static_cast<uint32_t>(UP_DIV(headDim, 4)), \
                                    static_cast<uint32_t>(kvNumHead)};
  
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[0]);
            ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[1]);
            ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[2]);
            ret |= mKernel_rearrange->get().setArg(index++, openCLBuffer(key));
            ret |= mKernel_rearrange->get().setArg(index++, *mKVCacheCLManager->key());
            ret |= mKernel_rearrange->get().setArg(index++, past_len);
            ret |= mKernel_rearrange->get().setArg(index++, mKeyValueMaxlen);
            ret |= mKernel_rearrange->get().setArg(index++, seq_len);
            ret |= mKernel_rearrange->get().setArg(index++, kvNumHead);
            ret |= mKernel_rearrange->get().setArg(index++, numHead);
            ret |= mKernel_rearrange->get().setArg(index++, headDim);
            
            MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_k");
            mLocalWorkSizeRearrg = localWS3DDefault(mGlobalWorkSizeRearrg, maxWorkGroupSize, runtime, "rearrange_k", mKernel_rearrange).first;
            mGlobalWorkSizeRearrg[0] = ROUND_UP(mGlobalWorkSizeRearrg[0], std::max((uint32_t)1, mLocalWorkSizeRearrg[0]));
            mGlobalWorkSizeRearrg[1] = ROUND_UP(mGlobalWorkSizeRearrg[1], std::max((uint32_t)1, mLocalWorkSizeRearrg[1]));
            mGlobalWorkSizeRearrg[2] = ROUND_UP(mGlobalWorkSizeRearrg[2], std::max((uint32_t)1, mLocalWorkSizeRearrg[2]));
            mRgUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &(*(mKVCacheCLManager->key()))()});
            mRgUpdateInfo.update_kernel_args.push_back({0, 6, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
            mOpRecordUpdateInfo.emplace_back(&mRgUpdateInfo);
            mOpenCLBackend->recordKernel3d(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg, &mRgUpdateInfo);
        }
        {
            // matmul qk
            std::set<std::string> buildOption;
            if(mask->getType() == halide_type_of<float>()){
                buildOption.emplace("-DADD_MASK");
            }
            buildOption.emplace("-DNUMHEAD_GROUP_SIZE=" + std::to_string(group_size));
            mKernel_qk = runtime->buildKernel("attention_buf", "matmul_qk_div_mask_prefill", buildOption, inputs[0], outputs[0]);
            mGlobalWorkSizeQk =  {static_cast<uint32_t>(UP_DIV(seq_len, 4)), static_cast<uint32_t>(UP_DIV(mask_kvlen, 4)), static_cast<uint32_t>(numHead)};
            auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_qk));
            
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_qk->get().setArg(index++, mGlobalWorkSizeQk[0]);
            ret |= mKernel_qk->get().setArg(index++, mGlobalWorkSizeQk[1]);
            ret |= mKernel_qk->get().setArg(index++, mGlobalWorkSizeQk[2]);
            ret |= mKernel_qk->get().setArg(index++, openCLBuffer(mTempQ.get()));
            ret |= mKernel_qk->get().setArg(index++, *mKVCacheCLManager->key());
            ret |= mKernel_qk->get().setArg(index++, openCLBuffer(mask));
            ret |= mKernel_qk->get().setArg(index++, openCLBuffer(mTempQK.get()));
            ret |= mKernel_qk->get().setArg(index++, scale);
            ret |= mKernel_qk->get().setArg(index++, seq_len);
            ret |= mKernel_qk->get().setArg(index++, mask_kvlen);
            ret |= mKernel_qk->get().setArg(index++, mKeyValueMaxlen);
            ret |= mKernel_qk->get().setArg(index++, numHead);
            ret |= mKernel_qk->get().setArg(index++, headDim);
            MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qk_div_mask_prefill");
            
            mLocalWorkSizeQk = localWS3DDefault(mGlobalWorkSizeQk, maxWorkGroupSize, runtime, "matmul_qk_div_mask_prefill", mKernel_qk).first;
            mGlobalWorkSizeQk[0] = ROUND_UP(mGlobalWorkSizeQk[0], std::max((uint32_t)1, mLocalWorkSizeQk[0]));
            mGlobalWorkSizeQk[1] = ROUND_UP(mGlobalWorkSizeQk[1], std::max((uint32_t)1, mLocalWorkSizeQk[1]));
            mGlobalWorkSizeQk[2] = ROUND_UP(mGlobalWorkSizeQk[2], std::max((uint32_t)1, mLocalWorkSizeQk[2]));
            mQkUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &(*(mKVCacheCLManager->key()))()});
            mQkUpdateInfo.update_kernel_args.push_back({0, 10, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
            mOpRecordUpdateInfo.emplace_back(&mQkUpdateInfo);
            mOpenCLBackend->recordKernel3d(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk, &mQkUpdateInfo);
        }
        {
            // softmax
            int inside  = ROUND_UP(seq_len, 4);
            int outside = numHead;
            int localSize = getLocalSize(mask_kvlen, 128);
            if(localSize < 4){
                localSize = 1;
            }
            
            std::set<std::string> buildOption;
            buildOption.emplace("-DSOFTMAX_LOCAL_SIZE=" + std::to_string(localSize));
            mKernel_softmax = runtime->buildKernel("softmax_buf", "softmax_v4_buf", buildOption);
            mGlobalWorkSizeSoftMax = {static_cast<uint32_t>(localSize), static_cast<uint32_t>(UP_DIV(inside, 4)), static_cast<uint32_t>(outside)};
            auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_softmax));
            
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[0]);
            ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[1]);
            ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[2]);
            ret |= mKernel_softmax->get().setArg(index++, openCLBuffer(mTempQK.get()));
            ret |= mKernel_softmax->get().setArg(index++, openCLBuffer(mTempSoftMax.get()));
            ret |= mKernel_softmax->get().setArg(index++, inside);
            ret |= mKernel_softmax->get().setArg(index++, outside);
            ret |= mKernel_softmax->get().setArg(index++, mask_kvlen);
            MNN_CHECK_CL_SUCCESS(ret, "setArg softmax");
            
            mLocalWorkSizeSoftMax = {static_cast<uint32_t>(localSize), 1, 1};
            if(localSize == 1){
                mLocalWorkSizeSoftMax = localWS3DDefault(mGlobalWorkSizeSoftMax, maxWorkGroupSize, runtime, "softmax", mKernel_softmax).first;
            }
            mGlobalWorkSizeSoftMax[0] = ROUND_UP(mGlobalWorkSizeSoftMax[0], std::max((uint32_t)1, mLocalWorkSizeSoftMax[0]));
            mGlobalWorkSizeSoftMax[1] = ROUND_UP(mGlobalWorkSizeSoftMax[1], std::max((uint32_t)1, mLocalWorkSizeSoftMax[1]));
            mGlobalWorkSizeSoftMax[2] = ROUND_UP(mGlobalWorkSizeSoftMax[2], std::max((uint32_t)1, mLocalWorkSizeSoftMax[2]));
            mOpenCLBackend->recordKernel3d(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax);
        }
        {
            // rearrange value
            std::set<std::string> buildOption;
            
            buildOption.emplace("-DOPENCL_PREFILL_ATTENTION");
            mKernel_rearrangeV = runtime->buildKernel("attention_buf", "rearrange_v", buildOption, inputs[0], outputs[0]);
            auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_rearrangeV));
            
            mGlobalWorkSizeRearrgV = {static_cast<uint32_t>(UP_DIV(headDim, 4)), \
                                    static_cast<uint32_t>(UP_DIV(seq_len, 4)), \
                                    static_cast<uint32_t>(kvNumHead)};
  
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_rearrangeV->get().setArg(index++, mGlobalWorkSizeRearrgV[0]);
            ret |= mKernel_rearrangeV->get().setArg(index++, mGlobalWorkSizeRearrgV[1]);
            ret |= mKernel_rearrangeV->get().setArg(index++, mGlobalWorkSizeRearrgV[2]);
            ret |= mKernel_rearrangeV->get().setArg(index++, openCLBuffer(value));
            ret |= mKernel_rearrangeV->get().setArg(index++, *mKVCacheCLManager->value());
            ret |= mKernel_rearrangeV->get().setArg(index++, past_len);
            ret |= mKernel_rearrangeV->get().setArg(index++, mKeyValueMaxlen);
            ret |= mKernel_rearrangeV->get().setArg(index++, seq_len);
            ret |= mKernel_rearrangeV->get().setArg(index++, kvNumHead);
            ret |= mKernel_rearrangeV->get().setArg(index++, headDim);
            
            MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_v");
            mLocalWorkSizeRearrgV = localWS3DDefault(mGlobalWorkSizeRearrgV, maxWorkGroupSize, runtime, "rearrange_v", mKernel_rearrangeV).first;
            mGlobalWorkSizeRearrgV[0] = ROUND_UP(mGlobalWorkSizeRearrgV[0], std::max((uint32_t)1, mLocalWorkSizeRearrgV[0]));
            mGlobalWorkSizeRearrgV[1] = ROUND_UP(mGlobalWorkSizeRearrgV[1], std::max((uint32_t)1, mLocalWorkSizeRearrgV[1]));
            mGlobalWorkSizeRearrgV[2] = ROUND_UP(mGlobalWorkSizeRearrgV[2], std::max((uint32_t)1, mLocalWorkSizeRearrgV[2]));
            mRgVUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &(*(mKVCacheCLManager->value()))()});
            mRgVUpdateInfo.update_kernel_args.push_back({0, 6, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
            mOpRecordUpdateInfo.emplace_back(&mRgVUpdateInfo);
            mOpenCLBackend->recordKernel3d(mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV, &mRgVUpdateInfo);
        }
        // qk * value
        {
            std::set<std::string> buildOption;
            buildOption.emplace("-DNUMHEAD_GROUP_SIZE=" + std::to_string(group_size));
            mKernel_qkv = runtime->buildKernel("attention_buf", "matmul_qkv_prefill", buildOption, inputs[0], outputs[0]);
            auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_qkv));
            mGlobalWorkSizeQkv =  {static_cast<uint32_t>(UP_DIV(headDim, 8)), static_cast<uint32_t>(UP_DIV(seq_len, 4)), static_cast<uint32_t>(numHead)};
            
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_qkv->get().setArg(index++, mGlobalWorkSizeQkv[0]);
            ret |= mKernel_qkv->get().setArg(index++, mGlobalWorkSizeQkv[1]);
            ret |= mKernel_qkv->get().setArg(index++, mGlobalWorkSizeQkv[2]);
            ret |= mKernel_qkv->get().setArg(index++, openCLBuffer(mTempSoftMax.get()));
            ret |= mKernel_qkv->get().setArg(index++, *mKVCacheCLManager->value());
            ret |= mKernel_qkv->get().setArg(index++, openCLBuffer(outputs[0]));
            ret |= mKernel_qkv->get().setArg(index++, seq_len);
            ret |= mKernel_qkv->get().setArg(index++, mask_kvlen);
            ret |= mKernel_qkv->get().setArg(index++, mKeyValueMaxlen);
            ret |= mKernel_qkv->get().setArg(index++, numHead);
            ret |= mKernel_qkv->get().setArg(index++, kvNumHead);
            ret |= mKernel_qkv->get().setArg(index++, headDim);
            MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qkv_prefill");
            
            mLocalWorkSizeQkv = localWS3DDefault(mGlobalWorkSizeQkv, maxWorkGroupSize, runtime, "matmul_qkv_prefill", mKernel_qkv).first;
            mGlobalWorkSizeQkv[0] = ROUND_UP(mGlobalWorkSizeQkv[0], std::max((uint32_t)1, mLocalWorkSizeQkv[0]));
            mGlobalWorkSizeQkv[1] = ROUND_UP(mGlobalWorkSizeQkv[1], std::max((uint32_t)1, mLocalWorkSizeQkv[1]));
            mGlobalWorkSizeQkv[2] = ROUND_UP(mGlobalWorkSizeQkv[2], std::max((uint32_t)1, mLocalWorkSizeQkv[2]));
            mQkvUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &(*(mKVCacheCLManager->value()))()});
            mQkvUpdateInfo.update_kernel_args.push_back({0, 8, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
            mOpRecordUpdateInfo.emplace_back(&mQkvUpdateInfo);
            mOpenCLBackend->recordKernel3d(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv, &mQkvUpdateInfo);
        }
    } else {
        mKv_seq_len = mKVCacheCLManager->kvLength() + 1;
        mKeyValueMaxlen = ROUND_UP(mKVCacheCLManager->maxLength(), 4);
        mDecodeTmpMaxlen = mKeyValueMaxlen;
        mTempQK.reset(Tensor::createDevice<float>({mDecodeTmpMaxlen * numHead}));
        mTempSoftMax.reset(Tensor::createDevice<float>({mDecodeTmpMaxlen * numHead}));
        mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION);
        mOpenCLBackend->onAcquireBuffer(mTempSoftMax.get(), Backend::DYNAMIC_IN_EXECUTION);
        mOpenCLBackend->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION);
        mOpenCLBackend->onReleaseBuffer(mTempSoftMax.get(), Backend::DYNAMIC_IN_EXECUTION);
        {
            // rearrange key
            std::set<std::string> buildOption;
            
            mKernel_rearrange = runtime->buildKernel("attention_buf", "rearrange_k", buildOption, inputs[0], outputs[0]);
            auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_rearrange));
            
            mGlobalWorkSizeRearrg = {static_cast<uint32_t>(1), \
                                    static_cast<uint32_t>(UP_DIV(headDim, 4)), \
                                    static_cast<uint32_t>(kvNumHead)};
  
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[0]);
            ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[1]);
            ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[2]);
            ret |= mKernel_rearrange->get().setArg(index++, openCLBuffer(key));
            ret |= mKernel_rearrange->get().setArg(index++, *mKVCacheCLManager->key());
            ret |= mKernel_rearrange->get().setArg(index++, mKv_seq_len);
            ret |= mKernel_rearrange->get().setArg(index++, mKeyValueMaxlen);
            ret |= mKernel_rearrange->get().setArg(index++, seq_len);
            ret |= mKernel_rearrange->get().setArg(index++, kvNumHead);
            ret |= mKernel_rearrange->get().setArg(index++, numHead);
            ret |= mKernel_rearrange->get().setArg(index++, headDim);
            
            MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_k");
            mLocalWorkSizeRearrg = localWS3DDefault(mGlobalWorkSizeRearrg, maxWorkGroupSize, runtime, "rearrange_k", mKernel_rearrange).first;
            mGlobalWorkSizeRearrg[0] = ROUND_UP(mGlobalWorkSizeRearrg[0], std::max((uint32_t)1, mLocalWorkSizeRearrg[0]));
            mGlobalWorkSizeRearrg[1] = ROUND_UP(mGlobalWorkSizeRearrg[1], std::max((uint32_t)1, mLocalWorkSizeRearrg[1]));
            mGlobalWorkSizeRearrg[2] = ROUND_UP(mGlobalWorkSizeRearrg[2], std::max((uint32_t)1, mLocalWorkSizeRearrg[2]));
            mRgUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &(*(mKVCacheCLManager->key()))()});
            mRgUpdateInfo.update_kernel_args.push_back({0, 5, sizeof(mKv_seq_len), &mKv_seq_len});
            mRgUpdateInfo.update_kernel_args.push_back({0, 6, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
            mOpRecordUpdateInfo.emplace_back(&mRgUpdateInfo);
            mOpenCLBackend->recordKernel3d(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg, &mRgUpdateInfo);
        }
        {
            // matmul qk
            std::set<std::string> buildOption;
            if(mask->getType() == halide_type_of<float>()){
                buildOption.emplace("-DADD_MASK");
            }
            buildOption.emplace("-DNUMHEAD_GROUP_SIZE=" + std::to_string(group_size));
            mKernel_qk = runtime->buildKernel("attention_buf", "matmul_qk_decode", buildOption, inputs[0], outputs[0]);
            mGlobalWorkSizeQk =  {static_cast<uint32_t>(UP_DIV(mKv_seq_len, 4)), static_cast<uint32_t>(numHead)};
            auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_qk));
            
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_qk->get().setArg(index++, mGlobalWorkSizeQk[0]);
            ret |= mKernel_qk->get().setArg(index++, mGlobalWorkSizeQk[1]);
            ret |= mKernel_qk->get().setArg(index++, openCLBuffer(query));
            ret |= mKernel_qk->get().setArg(index++, *mKVCacheCLManager->key());
            ret |= mKernel_qk->get().setArg(index++, openCLDeferBuffer(mTempQK.get()));
            ret |= mKernel_qk->get().setArg(index++, scale);
            ret |= mKernel_qk->get().setArg(index++, mKv_seq_len);
            ret |= mKernel_qk->get().setArg(index++, mKeyValueMaxlen);
            ret |= mKernel_qk->get().setArg(index++, numHead);
            ret |= mKernel_qk->get().setArg(index++, headDim);
            MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qk_decode");
            
            mLocalWorkSizeQk = localWS2DDefault(mGlobalWorkSizeQk, maxWorkGroupSize, runtime, "matmul_qk_decode", mKernel_qk).first;
            mGlobalWorkSizeQk[0] = ROUND_UP(mGlobalWorkSizeQk[0], std::max((uint32_t)1, mLocalWorkSizeQk[0]));
            mGlobalWorkSizeQk[1] = ROUND_UP(mGlobalWorkSizeQk[1], std::max((uint32_t)1, mLocalWorkSizeQk[1]));
            mQkUpdateInfo.update_kernel_args.push_back({0, 0, sizeof(mGlobalWorkSizeQk0), &mGlobalWorkSizeQk0});
            mQkUpdateInfo.update_kernel_args.push_back({0, 3, sizeof(cl_mem), &(*(mKVCacheCLManager->key()))()});
            mQkUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &openCLDeferBuffer(mTempQK.get())()});
            mQkUpdateInfo.update_kernel_args.push_back({0, 6, sizeof(mKv_seq_len), &mKv_seq_len});
            mQkUpdateInfo.update_kernel_args.push_back({0, 7, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
            mQkGlobal_size[0] = mGlobalWorkSizeQk[0];
            mQkGlobal_size[1] = mGlobalWorkSizeQk[1];
            mQkUpdateInfo.update_global_size.push_back({0, mQkGlobal_size});
            mOpRecordUpdateInfo.emplace_back(&mQkUpdateInfo);
            mOpenCLBackend->recordKernel2d(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk, &mQkUpdateInfo);
        }
        {
            // softmax
            int inside  = 1;
            int outside = numHead;
            int localSize = getLocalSize(UP_DIV(mKv_seq_len, 4), 128);
            if(localSize < 4){
                localSize = 1;
            }
            
            std::set<std::string> buildOption;
            buildOption.emplace("-DSOFTMAX_LOCAL_SIZE=" + std::to_string(localSize));
            mKernel_softmax = runtime->buildKernel("softmax_buf", "softmax_in1_buf", buildOption);
            mGlobalWorkSizeSoftMax = {static_cast<uint32_t>(localSize), static_cast<uint32_t>(inside), static_cast<uint32_t>(outside)};
            auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_softmax));
            
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[0]);
            ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[1]);
            ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[2]);
            ret |= mKernel_softmax->get().setArg(index++, openCLDeferBuffer(mTempQK.get()));
            ret |= mKernel_softmax->get().setArg(index++, openCLDeferBuffer(mTempSoftMax.get()));
            ret |= mKernel_softmax->get().setArg(index++, inside);
            ret |= mKernel_softmax->get().setArg(index++, outside);
            ret |= mKernel_softmax->get().setArg(index++, mKv_seq_len);
            MNN_CHECK_CL_SUCCESS(ret, "setArg softmax");
            
            mLocalWorkSizeSoftMax = {static_cast<uint32_t>(localSize), 1, 1};
            if(localSize == 1){
                mLocalWorkSizeSoftMax = localWS3DDefault(mGlobalWorkSizeSoftMax, maxWorkGroupSize, runtime, "softmax", mKernel_softmax).first;
            }
            mGlobalWorkSizeSoftMax[0] = ROUND_UP(mGlobalWorkSizeSoftMax[0], std::max((uint32_t)1, mLocalWorkSizeSoftMax[0]));
            mGlobalWorkSizeSoftMax[1] = ROUND_UP(mGlobalWorkSizeSoftMax[1], std::max((uint32_t)1, mLocalWorkSizeSoftMax[1]));
            mGlobalWorkSizeSoftMax[2] = ROUND_UP(mGlobalWorkSizeSoftMax[2], std::max((uint32_t)1, mLocalWorkSizeSoftMax[2]));
            mSoftMaxUpdateInfo.update_kernel_args.push_back({0, 3, sizeof(cl_mem), &openCLDeferBuffer(mTempQK.get())()});
            mSoftMaxUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &openCLDeferBuffer(mTempSoftMax.get())()});
            mSoftMaxUpdateInfo.update_kernel_args.push_back({0, 7, sizeof(mKv_seq_len), &mKv_seq_len});
            mOpRecordUpdateInfo.emplace_back(&mSoftMaxUpdateInfo);
            mOpenCLBackend->recordKernel3d(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax, &mSoftMaxUpdateInfo);
        }
        {
            // rearrange value
            std::set<std::string> buildOption;
            
            mKernel_rearrangeV = runtime->buildKernel("attention_buf", "rearrange_v", buildOption, inputs[0], outputs[0]);
            auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_rearrangeV));
            
            mGlobalWorkSizeRearrgV = {static_cast<uint32_t>(UP_DIV(headDim, 4)), \
                                    static_cast<uint32_t>(1), \
                                    static_cast<uint32_t>(kvNumHead)};
  
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_rearrangeV->get().setArg(index++, mGlobalWorkSizeRearrgV[0]);
            ret |= mKernel_rearrangeV->get().setArg(index++, mGlobalWorkSizeRearrgV[1]);
            ret |= mKernel_rearrangeV->get().setArg(index++, mGlobalWorkSizeRearrgV[2]);
            ret |= mKernel_rearrangeV->get().setArg(index++, openCLBuffer(value));
            ret |= mKernel_rearrangeV->get().setArg(index++, *mKVCacheCLManager->value());
            ret |= mKernel_rearrangeV->get().setArg(index++, mKv_seq_len);
            ret |= mKernel_rearrangeV->get().setArg(index++, mKeyValueMaxlen);
            ret |= mKernel_rearrangeV->get().setArg(index++, seq_len);
            ret |= mKernel_rearrangeV->get().setArg(index++, kvNumHead);
            ret |= mKernel_rearrangeV->get().setArg(index++, headDim);
            
            MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_v");
            mLocalWorkSizeRearrgV = localWS3DDefault(mGlobalWorkSizeRearrgV, maxWorkGroupSize, runtime, "rearrange_v", mKernel_rearrangeV).first;
            mGlobalWorkSizeRearrgV[0] = ROUND_UP(mGlobalWorkSizeRearrgV[0], std::max((uint32_t)1, mLocalWorkSizeRearrgV[0]));
            mGlobalWorkSizeRearrgV[1] = ROUND_UP(mGlobalWorkSizeRearrgV[1], std::max((uint32_t)1, mLocalWorkSizeRearrgV[1]));
            mGlobalWorkSizeRearrgV[2] = ROUND_UP(mGlobalWorkSizeRearrgV[2], std::max((uint32_t)1, mLocalWorkSizeRearrgV[2]));
            mRgVUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &(*(mKVCacheCLManager->value()))()});
            mRgVUpdateInfo.update_kernel_args.push_back({0, 5, sizeof(mKv_seq_len), &mKv_seq_len});
            mRgVUpdateInfo.update_kernel_args.push_back({0, 6, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
            mOpRecordUpdateInfo.emplace_back(&mRgVUpdateInfo);
            mOpenCLBackend->recordKernel3d(mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV, &mRgVUpdateInfo);
        }
        // qk * value
        {
            std::set<std::string> buildOption;
            buildOption.emplace("-DNUMHEAD_GROUP_SIZE=" + std::to_string(group_size));
            const int total_kernel = 2;
            std::string kernelName[total_kernel] = {"matmul_qkv_decode_b4", "matmul_qkv_decode_b8"};
            std::string unroll[total_kernel] = {"-DLOOP_UNROLL_4", "-DLOOP_UNROLL_8"};
            int itemC[total_kernel] = {4, 8};
            int actual_kernel = 2;
            std::shared_ptr<KernelWrap> kernel[total_kernel * total_kernel];
            std::vector<uint32_t> globalWorkSize[total_kernel * total_kernel];
            std::vector<uint32_t> localWorkSize[total_kernel * total_kernel];
            std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
            
            for (int i = 0; i < actual_kernel; i++) {
                for(int j = 0; j < actual_kernel; j++){
                    int knl_idx = i * total_kernel + j;
                    auto option = buildOption;
                    option.emplace(unroll[j]);
                    kernel[knl_idx] = mOpenCLBackend->getOpenCLRuntime()->buildKernel("attention_buf", kernelName[i], option);
                    uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
                    globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(headDim, itemC[i])), static_cast<uint32_t>(numHead)};
                    uint32_t index = 0;
                    cl_int ret = CL_SUCCESS;
                    ret |= kernel[knl_idx]->get().setArg(index++, globalWorkSize[knl_idx][0]);
                    ret |= kernel[knl_idx]->get().setArg(index++, globalWorkSize[knl_idx][1]);
                    ret |= kernel[knl_idx]->get().setArg(index++, openCLDeferBuffer(mTempSoftMax.get()));
                    ret |= kernel[knl_idx]->get().setArg(index++, *mKVCacheCLManager->value());
                    ret |= kernel[knl_idx]->get().setArg(index++, openCLBuffer(outputs[0]));
                    ret |= kernel[knl_idx]->get().setArg(index++, mKv_seq_len);
                    ret |= kernel[knl_idx]->get().setArg(index++, mKeyValueMaxlen);
                    ret |= kernel[knl_idx]->get().setArg(index++, numHead);
                    ret |= kernel[knl_idx]->get().setArg(index++, kvNumHead);
                    ret |= kernel[knl_idx]->get().setArg(index++, headDim);
                    MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qkv_decode");
                    std::pair<std::vector<uint32_t>, int> retTune;
                    retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[i] + unroll[j], kernel[knl_idx]);
                    if(min_cost.first > retTune.second) {
                        min_cost.first = retTune.second;
                        min_cost.second = knl_idx;
                        mLocalWorkSizeQkv = {retTune.first[0], retTune.first[1]};
                    }
                }
            }
            int min_index  = min_cost.second / 2;
            int min_index_unroll  = min_cost.second % 2;
            mGlobalWorkSizeQkv = {globalWorkSize[min_cost.second][0], globalWorkSize[min_cost.second][1]};
            buildOption.emplace(unroll[min_index_unroll]);
            mKernel_qkv = runtime->buildKernel("attention_buf", kernelName[min_index], buildOption, inputs[0], outputs[0]);
            
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_qkv->get().setArg(index++, mGlobalWorkSizeQkv[0]);
            ret |= mKernel_qkv->get().setArg(index++, mGlobalWorkSizeQkv[1]);
            ret |= mKernel_qkv->get().setArg(index++, openCLDeferBuffer(mTempSoftMax.get()));
            ret |= mKernel_qkv->get().setArg(index++, *mKVCacheCLManager->value());
            ret |= mKernel_qkv->get().setArg(index++, openCLBuffer(outputs[0]));
            ret |= mKernel_qkv->get().setArg(index++, mKv_seq_len);
            ret |= mKernel_qkv->get().setArg(index++, mKeyValueMaxlen);
            ret |= mKernel_qkv->get().setArg(index++, numHead);
            ret |= mKernel_qkv->get().setArg(index++, kvNumHead);
            ret |= mKernel_qkv->get().setArg(index++, headDim);
            MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qkv_decode");
            
            mGlobalWorkSizeQkv[0] = ROUND_UP(mGlobalWorkSizeQkv[0], std::max((uint32_t)1, mLocalWorkSizeQkv[0]));
            mGlobalWorkSizeQkv[1] = ROUND_UP(mGlobalWorkSizeQkv[1], std::max((uint32_t)1, mLocalWorkSizeQkv[1]));
            mQkvUpdateInfo.update_kernel_args.push_back({0, 2, sizeof(cl_mem), &openCLDeferBuffer(mTempSoftMax.get())()});
            mQkvUpdateInfo.update_kernel_args.push_back({0, 3, sizeof(cl_mem), &(*(mKVCacheCLManager->value()))()});
            mQkvUpdateInfo.update_kernel_args.push_back({0, 5, sizeof(mKv_seq_len), &mKv_seq_len});
            mQkvUpdateInfo.update_kernel_args.push_back({0, 6, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
            mOpRecordUpdateInfo.emplace_back(&mQkvUpdateInfo);
            mOpenCLBackend->recordKernel2d(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv, &mQkvUpdateInfo);
        }
    }
    mOpenCLBackend->endRecord(mRecording);

    return NO_ERROR;
}

ErrorCode AttentionBufExecution::UpdateArgs(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs){
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto mask = inputs[3];
    auto shape = query->shape();
    
    int batch = shape[0];
    int seq_len = shape[1];
    int numHead = shape[2];
    int kvNumHead = key->shape()[2];
    int headDim = shape[3];
    int group_size = numHead / kvNumHead;
    float scale = 1.0 / sqrt(headDim);
    auto mask_shape = mask->shape();
    int mask_seqlen = mask_shape[2];
    int mask_kvlen  = mask_shape[3];
    // prefill
    if(mIsDecode == false){
        // reset mPastLength
        mKVCacheCLManager->setArgs(mask_kvlen, numHead, kvNumHead, headDim);
        // key value static memory has been changed, need reset args
        if(mKeyValueMaxlen != ROUND_UP(mKVCacheCLManager->maxLength(), 4)){
            mKeyValueMaxlen = ROUND_UP(mKVCacheCLManager->maxLength(), 4);
            #ifndef ENABLE_OPENCL_TIME_PROFILER
            if(mOpenCLBackend->isUseRecordQueue()){
                if(mLongPrefill){
                    mRgUpdateInfo.update_kernel_args[0].arg_value = &(*(mKVCacheCLManager->key()))();
                    mQkUpdateInfo.update_kernel_args[0].arg_value = &(*(mKVCacheCLManager->key()))();
                    mRgVUpdateInfo.update_kernel_args[0].arg_value = &(*(mKVCacheCLManager->value()))();
                    mQkvUpdateInfo.update_kernel_args[0].arg_value = &(*(mKVCacheCLManager->value()))();
                }else{
                    mRgUpdateInfo.update_kernel_args[0].arg_value = &(*(mKVCacheCLManager->key()))();
                }
            } else {
            #endif
                if(mLongPrefill){
                    // rearrange key value
                    cl_int ret = CL_SUCCESS;
                    ret |= mKernel_rearrange->get().setArg(9, *mKVCacheCLManager->key());
                    ret |= mKernel_rearrange->get().setArg(10, *mKVCacheCLManager->value());
                    ret |= mKernel_rearrange->get().setArg(14, mKeyValueMaxlen);
                    MNN_CHECK_CL_SUCCESS(ret, "reSetArg rearrange_k");
                }else{
                    {
                        // rearrange key
                        cl_int ret = CL_SUCCESS;
                        ret |= mKernel_rearrange->get().setArg(4, *mKVCacheCLManager->key());
                        ret |= mKernel_rearrange->get().setArg(6, mKeyValueMaxlen);
                        MNN_CHECK_CL_SUCCESS(ret, "reSetArg rearrange_k");
                    }
                    {
                        // matmul qk
                        cl_int ret = CL_SUCCESS;
                        ret |= mKernel_qk->get().setArg(4, *mKVCacheCLManager->key());
                        ret |= mKernel_qk->get().setArg(10, mKeyValueMaxlen);
                        MNN_CHECK_CL_SUCCESS(ret, "reSetArg matmul_qk_decode");
                    }
                    {
                        cl_int ret = CL_SUCCESS;
                        ret |= mKernel_rearrangeV->get().setArg(4, *mKVCacheCLManager->value());
                        ret |= mKernel_rearrangeV->get().setArg(6, mKeyValueMaxlen);
                        MNN_CHECK_CL_SUCCESS(ret, "reSetArg rearrange_v");
                    }
                    // qk * value
                    {
                        cl_int ret = CL_SUCCESS;
                        ret |= mKernel_qkv->get().setArg(4, *mKVCacheCLManager->value());
                        ret |= mKernel_qkv->get().setArg(8, mKeyValueMaxlen);
                        MNN_CHECK_CL_SUCCESS(ret, "reSetArg matmul_qkv_decode");
                    }
                }
            #ifndef ENABLE_OPENCL_TIME_PROFILER
            }
            #endif
        }
        return NO_ERROR;
    }
    
    // Decode
    mKv_seq_len = mKVCacheCLManager->kvLength() + 1;
    mKeyValueMaxlen = ROUND_UP(mKVCacheCLManager->maxLength(), 4);
    mKVCacheCLManager->addKvLength();
    if(mKv_seq_len > mDecodeTmpMaxlen){
        mDecodeTmpMaxlen = mKeyValueMaxlen;
        mTempQK.reset(Tensor::createDevice<float>({mDecodeTmpMaxlen * numHead}));
        mTempSoftMax.reset(Tensor::createDevice<float>({mDecodeTmpMaxlen * numHead}));
        mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION);
        mOpenCLBackend->onAcquireBuffer(mTempSoftMax.get(), Backend::DYNAMIC_IN_EXECUTION);
        mOpenCLBackend->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION);
        mOpenCLBackend->onReleaseBuffer(mTempSoftMax.get(), Backend::DYNAMIC_IN_EXECUTION);
    }
    mGlobalWorkSizeQk0 = UP_DIV(mKv_seq_len, 4);
    mQkGlobal_size[0] = ROUND_UP(mGlobalWorkSizeQk0, std::max((uint32_t)1, mLocalWorkSizeQk[0]));
    mGlobalWorkSizeQk[0] = mQkGlobal_size[0];
    
#ifndef ENABLE_OPENCL_TIME_PROFILER
    if(mOpenCLBackend->isUseRecordQueue()){
        mRgUpdateInfo.update_kernel_args[0].arg_value = &(*(mKVCacheCLManager->key()))();
        mQkUpdateInfo.update_kernel_args[1].arg_value = &(*(mKVCacheCLManager->key()))();
        mQkUpdateInfo.update_kernel_args[2].arg_value = &openCLDeferBuffer(mTempQK.get())();
        mSoftMaxUpdateInfo.update_kernel_args[0].arg_value = &openCLDeferBuffer(mTempQK.get())();
        mSoftMaxUpdateInfo.update_kernel_args[1].arg_value = &openCLDeferBuffer(mTempSoftMax.get())();
        mRgVUpdateInfo.update_kernel_args[0].arg_value = &(*(mKVCacheCLManager->value()))();
        mQkvUpdateInfo.update_kernel_args[0].arg_value = &openCLDeferBuffer(mTempSoftMax.get())();
        mQkvUpdateInfo.update_kernel_args[1].arg_value = &(*(mKVCacheCLManager->value()))();
    } else {
#endif
        {
            // rearrange key
            uint32_t index = 4;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_rearrange->get().setArg(index++, *mKVCacheCLManager->key());
            ret |= mKernel_rearrange->get().setArg(index++, mKv_seq_len);
            ret |= mKernel_rearrange->get().setArg(index++, mKeyValueMaxlen);
            MNN_CHECK_CL_SUCCESS(ret, "reSetArg rearrange_k");
        }
        {
            // matmul qk
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_qk->get().setArg(index++, mGlobalWorkSizeQk0);
            index++;
            index++;
            ret |= mKernel_qk->get().setArg(index++, *mKVCacheCLManager->key());
            ret |= mKernel_qk->get().setArg(index++, openCLDeferBuffer(mTempQK.get()));
            index++;
            ret |= mKernel_qk->get().setArg(index++, mKv_seq_len);
            ret |= mKernel_qk->get().setArg(index++, mKeyValueMaxlen);
            mGlobalWorkSizeQk[0] = ROUND_UP(mGlobalWorkSizeQk[0], std::max((uint32_t)1, mLocalWorkSizeQk[0]));
            mGlobalWorkSizeQk[1] = ROUND_UP(mGlobalWorkSizeQk[1], std::max((uint32_t)1, mLocalWorkSizeQk[1]));
            MNN_CHECK_CL_SUCCESS(ret, "reSetArg matmul_qk_decode");
        }
        {
            // softmax
            uint32_t index = 3;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_softmax->get().setArg(index++, openCLDeferBuffer(mTempQK.get()));
            ret |= mKernel_softmax->get().setArg(index++, openCLDeferBuffer(mTempSoftMax.get()));
            index++;
            index++;
            ret |= mKernel_softmax->get().setArg(index++, mKv_seq_len);
            MNN_CHECK_CL_SUCCESS(ret, "reSetArg softmax");
        }
        {
            uint32_t index = 4;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_rearrangeV->get().setArg(index++, *mKVCacheCLManager->value());
            ret |= mKernel_rearrangeV->get().setArg(index++, mKv_seq_len);
            ret |= mKernel_rearrangeV->get().setArg(index++, mKeyValueMaxlen);
            
            MNN_CHECK_CL_SUCCESS(ret, "reSetArg rearrange_v");
        }
        // qk * value
        {
            uint32_t index = 2;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_qkv->get().setArg(index++, openCLDeferBuffer(mTempSoftMax.get()));
            ret |= mKernel_qkv->get().setArg(index++, *mKVCacheCLManager->value());
            index++;
            ret |= mKernel_qkv->get().setArg(index++, mKv_seq_len);
            ret |= mKernel_qkv->get().setArg(index++, mKeyValueMaxlen);
            MNN_CHECK_CL_SUCCESS(ret, "reSetArg matmul_qkv_decode");
        }
#ifndef ENABLE_OPENCL_TIME_PROFILER
    }
#endif
    return NO_ERROR;
}

ErrorCode AttentionBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start AttentionBufExecution onExecute !\n");
#endif
    if(mIsDecode){
        mKVCacheCLManager->reallocKVCache();
    }
    UpdateArgs(inputs, outputs);
#ifdef ENABLE_OPENCL_TIME_PROFILER
    if(mLongPrefill) {
        cl::Event event0, event1, event2, event3, event4, event5, event6;
        run3DKernelDefault(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg, mOpenCLBackend->getOpenCLRuntime(), &event0);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_qkv", event0});
        run3DKernelDefault(mKernel_mask, mGlobalWorkSizeMask, mLocalWorkSizeMask, mOpenCLBackend->getOpenCLRuntime(), &event1);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_mask", event1});
        run3DKernelDefault(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk, mOpenCLBackend->getOpenCLRuntime(), &event2);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"matmul_qk_div_mask", event2});
        run3DKernelDefault(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax, mOpenCLBackend->getOpenCLRuntime(), &event3);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"softmax", event3});
        run3DKernelDefault(mKernel_trans, mGlobalWorkSizeTrans, mLocalWorkSizeTrans, mOpenCLBackend->getOpenCLRuntime(), &event4);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"transpose_softmax", event4});
        run3DKernelDefault(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv, mOpenCLBackend->getOpenCLRuntime(), &event5);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"matmul_qkv", event5});
        run3DKernelDefault(mKernel_clip, mGlobalWorkSizeClip, mLocalWorkSizeClip, mOpenCLBackend->getOpenCLRuntime(), &event6);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_output", event6});
    } else{
        if(mIsDecode){
            cl::Event event0, event1, event2, event3, event4;
            run3DKernelDefault(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg, mOpenCLBackend->getOpenCLRuntime(), &event0);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_k", event0});
            runKernel2D(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk, mOpenCLBackend->getOpenCLRuntime(), &event1);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"matmul_qk_div_mask", event1});
            run3DKernelDefault(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax, mOpenCLBackend->getOpenCLRuntime(), &event2);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"softmax", event2});
            run3DKernelDefault(mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV, mOpenCLBackend->getOpenCLRuntime(), &event3);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_v", event3});
            runKernel2D(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv, mOpenCLBackend->getOpenCLRuntime(), &event4);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"matmul_qkv", event4});
        }else{
            cl::Event event0, event1, event2, event3, event4, event5;
            run3DKernelDefault(mKernel_rearrangeQ, mGlobalWorkSizeRearrgQ, mLocalWorkSizeRearrgQ, mOpenCLBackend->getOpenCLRuntime(), &event0);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_q", event0});
            run3DKernelDefault(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg, mOpenCLBackend->getOpenCLRuntime(), &event1);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_k", event1});
            run3DKernelDefault(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk, mOpenCLBackend->getOpenCLRuntime(), &event2);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"matmul_qk_div_mask", event2});
            run3DKernelDefault(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax, mOpenCLBackend->getOpenCLRuntime(), &event3);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"softmax", event3});
            run3DKernelDefault(mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV, mOpenCLBackend->getOpenCLRuntime(), &event4);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_v", event4});
            run3DKernelDefault(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv, mOpenCLBackend->getOpenCLRuntime(), &event5);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"matmul_qkv", event5});
        }
    }
#else
    if(mOpenCLBackend->isUseRecordQueue()){
        mOpenCLBackend->addRecord(mRecording, mOpRecordUpdateInfo);
#ifdef LOG_VERBOSE
        MNN_PRINT("End AttentionBufExecution onExecute... \n");
#endif
        return NO_ERROR;
    }
    
    if(mLongPrefill) {
        run3DKernelDefault(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg, mOpenCLBackend->getOpenCLRuntime());
        run3DKernelDefault(mKernel_mask, mGlobalWorkSizeMask, mLocalWorkSizeMask, mOpenCLBackend->getOpenCLRuntime());
        run3DKernelDefault(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk, mOpenCLBackend->getOpenCLRuntime());
        run3DKernelDefault(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax, mOpenCLBackend->getOpenCLRuntime());
        run3DKernelDefault(mKernel_trans, mGlobalWorkSizeTrans, mLocalWorkSizeTrans, mOpenCLBackend->getOpenCLRuntime());
        run3DKernelDefault(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv, mOpenCLBackend->getOpenCLRuntime());
        run3DKernelDefault(mKernel_clip, mGlobalWorkSizeClip, mLocalWorkSizeClip, mOpenCLBackend->getOpenCLRuntime());
    } else{
        if(mIsDecode){
            run3DKernelDefault(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg, mOpenCLBackend->getOpenCLRuntime());
            runKernel2D(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk, mOpenCLBackend->getOpenCLRuntime());
            run3DKernelDefault(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax, mOpenCLBackend->getOpenCLRuntime());
            run3DKernelDefault(mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV, mOpenCLBackend->getOpenCLRuntime());
            runKernel2D(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv, mOpenCLBackend->getOpenCLRuntime());
        }else{
            run3DKernelDefault(mKernel_rearrangeQ, mGlobalWorkSizeRearrgQ, mLocalWorkSizeRearrgQ, mOpenCLBackend->getOpenCLRuntime());
            run3DKernelDefault(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg, mOpenCLBackend->getOpenCLRuntime());
            run3DKernelDefault(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk, mOpenCLBackend->getOpenCLRuntime());
            run3DKernelDefault(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax, mOpenCLBackend->getOpenCLRuntime());
            run3DKernelDefault(mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV, mOpenCLBackend->getOpenCLRuntime());
            run3DKernelDefault(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv, mOpenCLBackend->getOpenCLRuntime());
        }
    }
#endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end AttentionBufExecution onExecute !\n");
#endif

    return NO_ERROR;
}

AttentionBufExecution::AttentionBufExecution(const MNN::Op *op, Backend* backend, bool kv_cahce) : CommonExecution(backend, op) {
    mKVCacheCLManager.reset(new KVCacheCLManager(backend, kv_cahce));
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("softmax_buf", "softmax_buf", {"-DSOFTMAX_LOCAL_SIZE=512"});
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
}

AttentionBufExecution::AttentionBufExecution(std::shared_ptr<KVCacheCLManager> manager, const MNN::Op *op, Backend *backend) : CommonExecution(backend, op), mKVCacheCLManager(manager) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("softmax_buf", "softmax_buf", {"-DSOFTMAX_LOCAL_SIZE=512"});
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
}

bool AttentionBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    *dst = new AttentionBufExecution(mKVCacheCLManager, op, bn);
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
