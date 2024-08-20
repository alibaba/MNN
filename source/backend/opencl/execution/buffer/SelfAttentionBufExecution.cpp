//
//  FmhaV2Execution.cpp
//  MNN
//
//  Created by MNN on 2024/06/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <iostream>
#include <fstream>
#include "backend/opencl/execution/buffer/SelfAttentionBufExecution.hpp"

namespace MNN {
namespace OpenCL {

SelfAttentionBufImpl::SelfAttentionBufImpl(const MNN::Op *op, Backend *backend){
    auto fmha_v2_param = op->main_as_FmhaV2Param();
    mNumHead = fmha_v2_param->heads();
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("self_attention_buf", "softmax_inside", {"-DSOFTMAX_LOCAL_SIZE=512"});
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
}

int SelfAttentionBufImpl::getLocalSize(int size, int maxGroupSize){
    int local_size = 1;
    while(local_size * 2 <= maxGroupSize && local_size * 2 <= size){
        local_size *= 2;
    }
    return local_size;
}
// [B, seqlen, HeadNum*3*HeadDim] -> [B, seqlen, HeadNum*HeadDim]
ErrorCode SelfAttentionBufImpl::onResize(Backend *backend, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    mOpenCLBackend->startRecord(mRecording);
    
    auto input = inputs[0];// [Batch, seqLen, mNumHead * 3 * mHeadDim]

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto shape = input->shape();
    
    int tile_mn = 32;
    int tile_k = 4; // for gemm alignment
    int batch = shape[0];
    int seq_len = shape[1];
    mHeadDim = shape[2] / mNumHead / 3;
    mScale = 1.0 / sqrt(mHeadDim);
    
    if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()){
        mByte = 2;
    }
    
    // split several pieces for memory save
    if(seq_len > 1024) {
        mQseqSplitNum = (seq_len >= 4096 && seq_len % 64 == 0) ? 8 : ((seq_len < 2048) ? 2 : 4);
    }
    int buffer_size = batch * mNumHead * ROUND_UP(mHeadDim, tile_k) * ROUND_UP(seq_len, tile_mn);
    int buffer_qk_size = batch * mNumHead * ROUND_UP(seq_len, tile_mn) * ROUND_UP(seq_len, tile_mn) / mQseqSplitNum;
    int buffer_v_size = batch * mNumHead * ROUND_UP(mHeadDim, tile_mn) * ROUND_UP(seq_len, tile_mn);
    mTempQ.reset(Tensor::createDevice<float>(std::vector<int>{buffer_size / mQseqSplitNum}));
    mTempK.reset(Tensor::createDevice<float>(std::vector<int>{buffer_size}));
    mTempV.reset(Tensor::createDevice<float>(std::vector<int>{buffer_v_size}));
    mTempQK.reset(Tensor::createDevice<float>(std::vector<int>{buffer_qk_size}));
    mTempTrans.reset(Tensor::createDevice<float>(std::vector<int>{buffer_qk_size}));
    mTempSoftMax.reset(Tensor::createDevice<float>(std::vector<int>{buffer_qk_size}));
    mTempQKV.reset(Tensor::createDevice<float>(std::vector<int>{buffer_v_size / mQseqSplitNum}));
    
//    printf("buffer size x2:%f MB, buffer qk size x3:%f MB, buffer v size x2 :%f MB\n", buffer_size * 2.0 / 1024.0 / 1024.0, buffer_qk_size * 2.0 / 1024.0 / 1024.0, buffer_v_size * 2.0 / 1024.0 / 1024.0);
    mOpenCLBackend->onAcquireBuffer(mTempQ.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempK.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempV.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC);
    
    mOpenCLBackend->onReleaseBuffer(mTempQ.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mTempK.get(), Backend::DYNAMIC);
    
    mOpenCLBackend->onAcquireBuffer(mTempSoftMax.get(), Backend::DYNAMIC);
    
    mOpenCLBackend->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempTrans.get(), Backend::DYNAMIC);

    mOpenCLBackend->onReleaseBuffer(mTempSoftMax.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempQKV.get(), Backend::DYNAMIC);


    mOpenCLBackend->onReleaseBuffer(mTempV.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mTempTrans.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mTempQKV.get(), Backend::DYNAMIC);
    
    mKernel_split.resize(mQseqSplitNum);
    mKernel_qk.resize(mQseqSplitNum);
    mKernel_softmax.resize(mQseqSplitNum);
    mKernel_qkv.resize(mQseqSplitNum);
    mKernel_clip.resize(mQseqSplitNum);
    mKernel_trans.resize(mQseqSplitNum);
    mGlobalWorkSizeSplit.resize(mQseqSplitNum);
    mLocalWorkSizeSplit.resize(mQseqSplitNum);
    mGlobalWorkSizeClip.resize(mQseqSplitNum);
    mLocalWorkSizeClip.resize(mQseqSplitNum);
    mGlobalWorkSizeQk.resize(mQseqSplitNum);
    mLocalWorkSizeQk.resize(mQseqSplitNum);
    mGlobalWorkSizeSoftMax.resize(mQseqSplitNum);
    mLocalWorkSizeSoftMax.resize(mQseqSplitNum);
    mGlobalWorkSizeQkv.resize(mQseqSplitNum);
    mLocalWorkSizeQkv.resize(mQseqSplitNum);
    mGlobalWorkSizeTrans.resize(mQseqSplitNum);
    mLocalWorkSizeTrans.resize(mQseqSplitNum);
    
    for(int seq_idx = 0; seq_idx < mQseqSplitNum; seq_idx++) {
        // Split input to q k v
        {
            // [Batch, seqLen, mNumHead * 3 * mHeadDim] ->
            // Q : [Batch * mNumHead, ROUND_UP(mHeadDim, tile_k), ROUND_UP(seqLen, tile_mn)]
            // K : [Batch * mNumHead, ROUND_UP(mHeadDim, tile_k), ROUND_UP(seqLen, tile_mn)]
            // V : [Batch * mNumHead, ROUND_UP(seqLen, tile_mn), ROUND_UP(mHeadDim, tile_mn)]
            std::set<std::string> buildOption;
            if((mHeadDim % 4) != 0){
                buildOption.emplace("-DHEADDIM_LEAVE");
            }
            if((seq_len % 4) != 0){
                buildOption.emplace("-DSEQLEN_LEAVE");
            }
            
            int seq_len_pack_mn = ROUND_UP(seq_len, tile_mn);
            int head_dim_pack_mn = ROUND_UP(mHeadDim, tile_mn);
            int head_dim_pack_k = ROUND_UP(mHeadDim, tile_k);
            int seq_len_piece = seq_len_pack_mn/mQseqSplitNum;
            
            mKernel_split[seq_idx] = runtime->buildKernel("self_attention_buf", "split_transpose_qkv", buildOption, inputs[0], outputs[0]);
            auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_split[seq_idx]));
            
            mGlobalWorkSizeSplit[seq_idx] = {static_cast<uint32_t>(UP_DIV(seq_len_pack_mn, 4)), static_cast<uint32_t>(UP_DIV(head_dim_pack_mn, 4)), static_cast<uint32_t>(batch*mNumHead)};
            
            if(seq_idx > 0) {
                mGlobalWorkSizeSplit[seq_idx][0] = static_cast<uint32_t>(UP_DIV(seq_len_piece, 4));
            }
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_split[seq_idx]->get().setArg(index++, mGlobalWorkSizeSplit[seq_idx][0]);
            ret |= mKernel_split[seq_idx]->get().setArg(index++, mGlobalWorkSizeSplit[seq_idx][1]);
            ret |= mKernel_split[seq_idx]->get().setArg(index++, mGlobalWorkSizeSplit[seq_idx][2]);
            ret |= mKernel_split[seq_idx]->get().setArg(index++, openCLBuffer(input));
            ret |= mKernel_split[seq_idx]->get().setArg(index++, openCLBuffer(mTempQ.get()));
            ret |= mKernel_split[seq_idx]->get().setArg(index++, openCLBuffer(mTempK.get()));
            ret |= mKernel_split[seq_idx]->get().setArg(index++, openCLBuffer(mTempV.get()));
            ret |= mKernel_split[seq_idx]->get().setArg(index++, seq_len_pack_mn);
            ret |= mKernel_split[seq_idx]->get().setArg(index++, seq_len_piece);
            ret |= mKernel_split[seq_idx]->get().setArg(index++, head_dim_pack_mn);
            ret |= mKernel_split[seq_idx]->get().setArg(index++, head_dim_pack_k);
            ret |= mKernel_split[seq_idx]->get().setArg(index++, seq_len);
            ret |= mKernel_split[seq_idx]->get().setArg(index++, mNumHead);
            ret |= mKernel_split[seq_idx]->get().setArg(index++, mHeadDim);
            ret |= mKernel_split[seq_idx]->get().setArg(index++, seq_idx);
            MNN_CHECK_CL_SUCCESS(ret, "setArg split_transpose_qkv");
            mLocalWorkSizeSplit[seq_idx] = localWS3DDefault(mGlobalWorkSizeSplit[seq_idx], maxWorkGroupSize, runtime, "split_transpose_qkv", mKernel_split[seq_idx]).first;
            mGlobalWorkSizeSplit[seq_idx][0] = ROUND_UP(mGlobalWorkSizeSplit[seq_idx][0], std::max((uint32_t)1, mLocalWorkSizeSplit[seq_idx][0]));
            mGlobalWorkSizeSplit[seq_idx][1] = ROUND_UP(mGlobalWorkSizeSplit[seq_idx][1], std::max((uint32_t)1, mLocalWorkSizeSplit[seq_idx][1]));
            mGlobalWorkSizeSplit[seq_idx][2] = ROUND_UP(mGlobalWorkSizeSplit[seq_idx][2], std::max((uint32_t)1, mLocalWorkSizeSplit[seq_idx][2]));
            mOpenCLBackend->recordKernel3d(mKernel_split[seq_idx], mGlobalWorkSizeSplit[seq_idx], mLocalWorkSizeSplit[seq_idx]);
        }
        
        // query * key -> div
        {
            // Q : [Batch * mNumHead, ROUND_UP(mHeadDim, tile_k), ROUND_UP(seqLen, tile_mn)] -> [B, K, M]
            // K : [Batch * mNumHead, ROUND_UP(mHeadDim, tile_k), ROUND_UP(seqLen, tile_mn)] -> [B, K, N]
            // QV: [Batch * mNumHead, ROUND_UP(seqLen, tile_mn), ROUND_UP(seqLen, tile_mn)]   -> [B, N, M]
            int loop = batch * mNumHead;
            int e_pack = ROUND_UP(seq_len, tile_mn) / mQseqSplitNum;
            int l_pack = ROUND_UP(mHeadDim, tile_k);
            int h_pack = ROUND_UP(seq_len, tile_mn);
            
            std::set<std::string> buildOptions;

            uint32_t layout = 4;
            auto param = getGemmParams({(uint32_t)e_pack, (uint32_t)h_pack, (uint32_t)l_pack, layout, (uint32_t)loop, (uint32_t)0}, {openCLBuffer(mTempQ.get()), openCLBuffer(mTempK.get()), openCLBuffer(mTempQK.get())}, mOpenCLBackend->getOpenCLRuntime());
            
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
            mKernel_qk[seq_idx] = mOpenCLBackend->getOpenCLRuntime()->buildKernel("matmul_params_buf", "XgemmBatched", buildOptions);
            
            int out_per_thread_m = tileM / localM;
            int out_per_thread_n = tileN / localN;
            
            mGlobalWorkSizeQk[seq_idx] = {static_cast<uint32_t>(e_pack/out_per_thread_m), static_cast<uint32_t>(h_pack/out_per_thread_n), static_cast<uint32_t>(loop)};
            mLocalWorkSizeQk[seq_idx] = {static_cast<uint32_t>(localM), static_cast<uint32_t>(localN), 1};
            
            float alpha = mScale;
            float beta = 0.0f;
            int batch_offset_a = e_pack * l_pack;
            int batch_offset_b = h_pack * l_pack;
            int batch_offset_c = e_pack * h_pack;
            int idx            = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_qk[seq_idx]->get().setArg(idx++, static_cast<int>(e_pack));
            ret |= mKernel_qk[seq_idx]->get().setArg(idx++, static_cast<int>(h_pack));
            ret |= mKernel_qk[seq_idx]->get().setArg(idx++, static_cast<int>(l_pack));
            ret |= mKernel_qk[seq_idx]->get().setArg(idx++, alpha);
            ret |= mKernel_qk[seq_idx]->get().setArg(idx++, beta);
            ret |= mKernel_qk[seq_idx]->get().setArg(idx++, openCLBuffer(mTempQ.get()));
            ret |= mKernel_qk[seq_idx]->get().setArg(idx++, batch_offset_a);
            ret |= mKernel_qk[seq_idx]->get().setArg(idx++, openCLBuffer(mTempK.get()));
            ret |= mKernel_qk[seq_idx]->get().setArg(idx++, batch_offset_b);
            ret |= mKernel_qk[seq_idx]->get().setArg(idx++, openCLBuffer(mTempQK.get()));
            ret |= mKernel_qk[seq_idx]->get().setArg(idx++, batch_offset_c);
            MNN_CHECK_CL_SUCCESS(ret, "setArg Self-Attention batchmatmul qk Kernel");
            mOpenCLBackend->recordKernel3d(mKernel_qk[seq_idx], mGlobalWorkSizeQk[seq_idx], mLocalWorkSizeQk[seq_idx]);
            
        }
        
        // softmax
        {
            // QV:     [Batch * mNumHead, ROUND_UP(seqLen, tile_mn), ROUND_UP(seqLen, tile_mn)]
            // Sotmax: [Batch * mNumHead, ROUND_UP(seqLen, tile_mn), ROUND_UP(seqLen, tile_mn)]
            // axis  : 1 (middle dim)
            mSoftmaxShape[0] = batch*mNumHead;
            mSoftmaxShape[1] = ROUND_UP(seq_len, tile_mn)/mQseqSplitNum;
            mSoftmaxShape[2] = ROUND_UP(seq_len, tile_mn);
            
            auto MaxLocalSize = std::min(std::min(runtime->getMaxWorkItemSizes()[0], mMaxWorkGroupSize), static_cast<uint32_t>(256));
            int localSize = getLocalSize(mSoftmaxShape[1], MaxLocalSize);
            if(localSize < 4){
                localSize = 1;
            }
            
            std::set<std::string> buildOption;
            buildOption.emplace("-DSOFTMAX_LOCAL_SIZE=" + std::to_string(localSize));
//            buildOption.emplace("-DOUTPUT_TRANSPOSE");
            
            mKernel_softmax[seq_idx] = runtime->buildKernel("self_attention_buf", "softmax_inside", buildOption, inputs[0], outputs[0]);
            mGlobalWorkSizeSoftMax[seq_idx] =  {static_cast<uint32_t>(localSize), static_cast<uint32_t>(mSoftmaxShape[1]), static_cast<uint32_t>(mSoftmaxShape[0])};
            
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_softmax[seq_idx]->get().setArg(index++, mGlobalWorkSizeSoftMax[seq_idx][0]);
            ret |= mKernel_softmax[seq_idx]->get().setArg(index++, mGlobalWorkSizeSoftMax[seq_idx][1]);
            ret |= mKernel_softmax[seq_idx]->get().setArg(index++, mGlobalWorkSizeSoftMax[seq_idx][2]);
            ret |= mKernel_softmax[seq_idx]->get().setArg(index++, openCLBuffer(mTempQK.get()));
            ret |= mKernel_softmax[seq_idx]->get().setArg(index++, openCLBuffer(mTempSoftMax.get()));
            ret |= mKernel_softmax[seq_idx]->get().setArg(index++, seq_len);
            ret |= mKernel_softmax[seq_idx]->get().setArg(index++, mSoftmaxShape);
            MNN_CHECK_CL_SUCCESS(ret, "setArg Self-Attention softmax");
            
            mLocalWorkSizeSoftMax[seq_idx] = {static_cast<uint32_t>(localSize), 1, 1};
            mOpenCLBackend->recordKernel3d(mKernel_softmax[seq_idx], mGlobalWorkSizeSoftMax[seq_idx], mLocalWorkSizeSoftMax[seq_idx]);
        }
        {
            int loop = batch * mNumHead;
            int transDimW = ROUND_UP(seq_len, tile_mn) / mQseqSplitNum;
            int transDimH = ROUND_UP(seq_len, tile_mn);
            
            std::set<std::string> buildOptions;
            mKernel_trans[seq_idx] = runtime->buildKernel("self_attention_buf", "trans_3d_buf", buildOptions, inputs[0], outputs[0]);
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel_trans[seq_idx]));

            mGlobalWorkSizeTrans[seq_idx] = {(uint32_t)transDimW/8, (uint32_t)transDimH/8, (uint32_t)(loop)};
            
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_trans[seq_idx]->get().setArg(index++, openCLBuffer(mTempSoftMax.get()));
            ret |= mKernel_trans[seq_idx]->get().setArg(index++, openCLBuffer(mTempTrans.get()));
            ret |= mKernel_trans[seq_idx]->get().setArg(index++, loop);
            ret |= mKernel_trans[seq_idx]->get().setArg(index++, transDimW);
            ret |= mKernel_trans[seq_idx]->get().setArg(index++, transDimH);
            MNN_CHECK_CL_SUCCESS(ret, "setArg Self-Attention transpose");
            mLocalWorkSizeTrans[seq_idx] = localWS3DDefault(mGlobalWorkSizeTrans[seq_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "trans_3d_buf", mKernel_trans[seq_idx]).first;
            
            mOpenCLBackend->recordKernel3d(mKernel_trans[seq_idx], mGlobalWorkSizeTrans[seq_idx], mLocalWorkSizeTrans[seq_idx]);
        }
        
        // qk * value
        {
            // Sotmax: [Batch * mNumHead, ROUND_UP(seqLen, tile), ROUND_UP(seqLen, tile)]   -> [B, K, M]
            // V :     [Batch * mNumHead, ROUND_UP(seqLen, tile), ROUND_UP(mHeadDim, tile)] -> [B, K, N]
            // QKV :   [Batch * mNumHead, ROUND_UP(mHeadDim, tile), ROUND_UP(seqLen, tile)] -> [B, N, M]
            
            int loop = batch * mNumHead;
            int e_pack = ROUND_UP(seq_len, tile_mn) / mQseqSplitNum;
            int l_pack = ROUND_UP(seq_len, tile_mn);
            int h_pack = ROUND_UP(mHeadDim, tile_mn);
            
            std::set<std::string> buildOptions;
            /*
             0 -> A:[K, M] B:[K, N] C:[N, M]
             1 -> A:[K, M] B:[N, K] C:[N, M]
             2 -> A:[M, K] B:[K, N] C:[N, M]
             3 -> A:[M, K] B:[N, K] C:[N, M]
             4 -> A:[K, M] B:[K, N] C:[M, N]
             5 -> A:[K, M] B:[N, K] C:[M, N]
             6 -> A:[M, K] B:[K, N] C:[M, N]
             7 -> A:[M, K] B:[N, K] C:[M, N]
             */
            uint32_t layout = 0;
            auto param = getGemmParams({(uint32_t)e_pack, (uint32_t)h_pack, (uint32_t)l_pack, layout, (uint32_t)loop, (uint32_t)0}, {openCLBuffer(mTempTrans.get()), openCLBuffer(mTempV.get()), openCLBuffer(mTempQKV.get())}, mOpenCLBackend->getOpenCLRuntime());

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

            mKernel_qkv[seq_idx] = mOpenCLBackend->getOpenCLRuntime()->buildKernel("matmul_params_buf", "XgemmBatched", buildOptions);
            
            int out_per_thread_m = tileM / localM;
            int out_per_thread_n = tileN / localN;
            
            mGlobalWorkSizeQkv[seq_idx] = {static_cast<uint32_t>(e_pack/out_per_thread_m), static_cast<uint32_t>(h_pack/out_per_thread_n), static_cast<uint32_t>(loop)};
            mLocalWorkSizeQkv[seq_idx] = {static_cast<uint32_t>(localM), static_cast<uint32_t>(localN), 1};
            
            float alpha = 1.0f;
            float beta = 0.0f;
            int batch_offset_a = e_pack * l_pack;
            int batch_offset_b = h_pack * l_pack;
            int batch_offset_c = e_pack * h_pack;
            int idx            = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_qkv[seq_idx]->get().setArg(idx++, static_cast<int>(e_pack));
            ret |= mKernel_qkv[seq_idx]->get().setArg(idx++, static_cast<int>(h_pack));
            ret |= mKernel_qkv[seq_idx]->get().setArg(idx++, static_cast<int>(l_pack));
            ret |= mKernel_qkv[seq_idx]->get().setArg(idx++, alpha);
            ret |= mKernel_qkv[seq_idx]->get().setArg(idx++, beta);
            ret |= mKernel_qkv[seq_idx]->get().setArg(idx++, openCLBuffer(mTempTrans.get()));
            ret |= mKernel_qkv[seq_idx]->get().setArg(idx++, batch_offset_a);
            ret |= mKernel_qkv[seq_idx]->get().setArg(idx++, openCLBuffer(mTempV.get()));
            ret |= mKernel_qkv[seq_idx]->get().setArg(idx++, batch_offset_b);
            ret |= mKernel_qkv[seq_idx]->get().setArg(idx++, openCLBuffer(mTempQKV.get()));
            ret |= mKernel_qkv[seq_idx]->get().setArg(idx++, batch_offset_c);
            MNN_CHECK_CL_SUCCESS(ret, "setArg Self-Attention batchmatmul qkv Kernel");
            mOpenCLBackend->recordKernel3d(mKernel_qkv[seq_idx], mGlobalWorkSizeQkv[seq_idx], mLocalWorkSizeQkv[seq_idx]);
        }
        
        // transpose to output
        {
            // QKV :   [Batch * mNumHead, ROUND_UP(mHeadDim, tile_mn), ROUND_UP(seqLen, tile_mn)] -> [B, N, M]
            // output: [Batch, seqLen, mNumHead * mHeadDim]
            std::set<std::string> buildOption;
            
            mKernel_clip[seq_idx] = runtime->buildKernel("self_attention_buf", "clip_transpose_qkv", buildOption, inputs[0], outputs[0]);
            auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_clip[seq_idx]));
            
            int seq_len_piece = ROUND_UP(seq_len, tile_mn) / mQseqSplitNum;
            
            mGlobalWorkSizeClip[seq_idx] = {static_cast<uint32_t>(UP_DIV(seq_len_piece, 4)), static_cast<uint32_t>(UP_DIV(mHeadDim, 4)), static_cast<uint32_t>(batch*mNumHead)};
            
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_clip[seq_idx]->get().setArg(index++, mGlobalWorkSizeClip[seq_idx][0]);
            ret |= mKernel_clip[seq_idx]->get().setArg(index++, mGlobalWorkSizeClip[seq_idx][1]);
            ret |= mKernel_clip[seq_idx]->get().setArg(index++, mGlobalWorkSizeClip[seq_idx][2]);
            ret |= mKernel_clip[seq_idx]->get().setArg(index++, openCLBuffer(mTempQKV.get()));
            ret |= mKernel_clip[seq_idx]->get().setArg(index++, openCLBuffer(outputs[0]));
            ret |= mKernel_clip[seq_idx]->get().setArg(index++, tile_mn);
            ret |= mKernel_clip[seq_idx]->get().setArg(index++, seq_len);
            ret |= mKernel_clip[seq_idx]->get().setArg(index++, seq_len_piece);
            ret |= mKernel_clip[seq_idx]->get().setArg(index++, mNumHead);
            ret |= mKernel_clip[seq_idx]->get().setArg(index++, mHeadDim);
            ret |= mKernel_clip[seq_idx]->get().setArg(index++, seq_idx);

            mLocalWorkSizeClip[seq_idx] = localWS3DDefault(mGlobalWorkSizeClip[seq_idx], maxWorkGroupSize, runtime, "clip_transpose_qkv", mKernel_clip[seq_idx]).first;
            mGlobalWorkSizeClip[seq_idx][0] = ROUND_UP(mGlobalWorkSizeClip[seq_idx][0], std::max((uint32_t)1, mLocalWorkSizeClip[seq_idx][0]));
            mGlobalWorkSizeClip[seq_idx][1] = ROUND_UP(mGlobalWorkSizeClip[seq_idx][1], std::max((uint32_t)1, mLocalWorkSizeClip[seq_idx][1]));
            mGlobalWorkSizeClip[seq_idx][2] = ROUND_UP(mGlobalWorkSizeClip[seq_idx][2], std::max((uint32_t)1, mLocalWorkSizeClip[seq_idx][2]));

            MNN_CHECK_CL_SUCCESS(ret, "setArg clip_transpose_qkv");
            mOpenCLBackend->recordKernel3d(mKernel_clip[seq_idx], mGlobalWorkSizeClip[seq_idx], mLocalWorkSizeClip[seq_idx]);
        }
    }
    mOpenCLBackend->endRecord(mRecording);
    return NO_ERROR;
}
    
ErrorCode SelfAttentionBufImpl::onExecute(Backend *backend, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start SelfAttentionBufExecution onExecute !\n");
#endif
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
#ifdef ENABLE_OPENCL_TIME_PROFILER
    int batch = inputs[0]->shape()[0];
    int seqLen = inputs[0]->shape()[1];
    int headDim = inputs[0]->shape()[2]/3/mNumHead;
    
    std::string name;
    name += "-b" + std::to_string(batch);
    name += "-s" + std::to_string(seqLen);
    name += "-h" + std::to_string(mNumHead);
    name += "-d" + std::to_string(headDim);

    for(int seq_idx = 0; seq_idx < mQseqSplitNum; seq_idx++) {
        {
            cl::Event event;
            run3DKernelDefault(mKernel_split[seq_idx], mGlobalWorkSizeSplit[seq_idx], mLocalWorkSizeSplit[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime(), &event);
            
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"While-gemm-split" + name, event});
        }
        {
            cl::Event event;
            run3DKernelDefault(mKernel_qk[seq_idx], mGlobalWorkSizeQk[seq_idx], mLocalWorkSizeQk[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime(), &event);
            
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"While-gemm-batchgemm" + name, event});
        }
        {
            cl::Event event;
            run3DKernelDefault(mKernel_softmax[seq_idx], mGlobalWorkSizeSoftMax[seq_idx], mLocalWorkSizeSoftMax[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime(), &event);
            
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"While-gemm-softmax" + name, event});
        }
        {
            cl::Event event;
            run3DKernelDefault(mKernel_trans[seq_idx], mGlobalWorkSizeTrans[seq_idx], mLocalWorkSizeTrans[seq_idx], mOpenCLBackend->getOpenCLRuntime(), &event);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"While-gemm-trans-1" + name, event});
        }
        {
            cl::Event event;
            run3DKernelDefault(mKernel_qkv[seq_idx], mGlobalWorkSizeQkv[seq_idx], mLocalWorkSizeQkv[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime(), &event);
            
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"While-gemm-batchgemm" + name, event});
        }
        {
            cl::Event event;
            run3DKernelDefault(mKernel_clip[seq_idx], mGlobalWorkSizeClip[seq_idx], mLocalWorkSizeClip[seq_idx],
                               mOpenCLBackend->getOpenCLRuntime(), &event);
            
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"While-gemm-clip" + name, event});
        }
    }
#else
    for(int seq_idx = 0; seq_idx < mQseqSplitNum; seq_idx++) {
        run3DKernelDefault(mKernel_split[seq_idx], mGlobalWorkSizeSplit[seq_idx], mLocalWorkSizeSplit[seq_idx], mOpenCLBackend->getOpenCLRuntime());
        run3DKernelDefault(mKernel_qk[seq_idx], mGlobalWorkSizeQk[seq_idx], mLocalWorkSizeQk[seq_idx], mOpenCLBackend->getOpenCLRuntime());
        run3DKernelDefault(mKernel_softmax[seq_idx], mGlobalWorkSizeSoftMax[seq_idx], mLocalWorkSizeSoftMax[seq_idx], mOpenCLBackend->getOpenCLRuntime());
        
        run3DKernelDefault(mKernel_trans[seq_idx], mGlobalWorkSizeTrans[seq_idx], mLocalWorkSizeTrans[seq_idx], mOpenCLBackend->getOpenCLRuntime());
        run3DKernelDefault(mKernel_qkv[seq_idx], mGlobalWorkSizeQkv[seq_idx], mLocalWorkSizeQkv[seq_idx], mOpenCLBackend->getOpenCLRuntime());
        run3DKernelDefault(mKernel_clip[seq_idx], mGlobalWorkSizeClip[seq_idx], mLocalWorkSizeClip[seq_idx], mOpenCLBackend->getOpenCLRuntime());
        
        #ifdef DUMP_INTERNAL_LOG
        {
            std::ofstream outFile("qk.txt");
            std::vector<float> hostPtr_3(16*4096*4096);
            mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueReadBuffer(openCLBuffer(mTempQK.get()), CL_TRUE, 0, 16*4096*4096*4, hostPtr_3.data());
            float max_ = -1000000.0;
            float min_ = 10000000.0;
            float total = 0.0;
            for(int i=1; i<hostPtr_3.size(); i++) {
                float temp = hostPtr_3[i];
                outFile << hostPtr_3[i] << "\n";
                total += temp/(16*4096*4096);
                if(max_ < temp) max_ = temp;
                if(min_ > temp) min_ = temp;
            }
            outFile.close();
            printf("qk max:%f min:%f avg:%f\n", max_, min_, hostPtr_3[0]+total);
        }
        #endif
    }
#endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end SelfAttentionBufExecution onExecute !\n");
#endif

    return NO_ERROR;
}

SelfAttentionBufExecution::SelfAttentionBufExecution(const MNN::Op *op, Backend* backend) : CommonExecution(backend, op) {
    mImpl.reset(new SelfAttentionBufImpl(op, backend));
}

SelfAttentionBufExecution::SelfAttentionBufExecution(std::shared_ptr<SelfAttentionBufImpl> impl, const MNN::Op *op, Backend *backend) : CommonExecution(backend, op), mImpl(impl) {}

ErrorCode SelfAttentionBufExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return mImpl->onResize(backend(), inputs, outputs);
}

ErrorCode SelfAttentionBufExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return mImpl->onExecute(backend(), inputs, outputs);
}

bool SelfAttentionBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    *dst = new SelfAttentionBufExecution(mImpl, op, bn);
    return true;
}

class SelfAttentionBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        return new SelfAttentionBufExecution(op, backend);
    }
};
REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(SelfAttentionBufCreator, OpType_FmhaV2, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
