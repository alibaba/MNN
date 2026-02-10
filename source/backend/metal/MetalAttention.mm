//
//  MetalAttention.mm
//  MNN
//
//  Created by MNN on b'2024/04/29'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalCast.hpp"
#import "MetalAttention.hpp"
#import "MNNMetalContext.h"
#import "MetalAttentionShader.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {

struct Param {
    int query_seq_len;
    int q_seq_piece_len;
    int key_seq_len;
    int head_num;
    int group;
    int head_dim;
    float scale;
    int max_kv_len;
    int batch;
    int kv_align_len;
};
AttentionBufExecution::AttentionBufExecution(Backend *backend, bool kv_cahce)
    : MetalExecution(backend) , mKVCache(kv_cahce) {
    _init();
}
void AttentionBufExecution::_init() {
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mMeta = (KVMeta*)(mtbn->getMetaPtr());

    mParamQKV = [context newDeviceBuffer:sizeof(Param) access:CPUWriteOnly];
    mParamSoftmax = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    mParamCopy = [context newDeviceBuffer:6 * sizeof(int) access:CPUWriteOnly];
    mTempQK.reset(Tensor::createDevice<float>({0, 0}));
    mTempSoftMax.reset(Tensor::createDevice<float>({0, 0}));
    mRunningStats.reset(Tensor::createDevice<uint8_t>({0, 0, 0, 0}));
    mCorrectionScale.reset(Tensor::createDevice<uint8_t>({0, 0, 0}));
    mTempOutput.reset(Tensor::createDevice<uint8_t>({0, 0}));

    MNN::MetalKVCacheManager::KVCacheConfig kvconfig;
    kvconfig.mKVCacheDir = mtbn->getRuntime()->hint().kvcacheDirPath;
    kvconfig.mPrefixCacheDir = mtbn->getRuntime()->hint().prefixcacheDirPath;
    kvconfig.mExpandChunk = 64;
    kvconfig.mKvAlignNum = mKvAlignNum;

    mKVCacheManager.reset(new MetalKVCacheManager(backend(), kvconfig));
    mKvInDisk = !kvconfig.mKVCacheDir.empty();
}

void AttentionBufExecution::compilerShader(const std::vector<Tensor *> &inputs) {
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto rt = (MetalRuntime*)mtbn->runtime();
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    
    auto seq_len = inputs[0]->length(1);
    int group_size = inputs[0]->length(2) / inputs[1]->length(2);
    std::string group_str = std::to_string(group_size);
    
    // Init Kernel
    std::string ftype = "float";
    std::string ftype4 = "float4";
    if (mtbn->useFp16InsteadFp32()) {
        ftype = "half";
        ftype4 = "half4";
    }
    std::vector<std::string> qkKeys = {
        {"matmul_qk_div_mask", ftype, group_str}
    };
    if(mHeadDim % 4 != 0) {
        qkKeys.emplace_back("HEAD_DIM_UNALIGNED_4");
    }
    
    std::vector<std::string> qkvKeys = {
        {"matmul_qkv", ftype, group_str}
    };
    if(mQkvSimdReduce) {
        qkvKeys.emplace_back("SIMD_GROUP_REDUCE");
    }
    std::vector<std::string> qkPrefillKeys = {
        {"matmul_qk_div_mask", ftype, group_str, "FOR_PREFILL"}
    };
    if(mHasMask) {
        if (mIsAddMask) {
            qkPrefillKeys.emplace_back("ADD_MASK");
            if(seq_len > 1) {
                qkKeys.emplace_back("ADD_MASK");
            }
        } else {
            qkPrefillKeys.emplace_back("SET_MASK");
            if(seq_len > 1) {
                qkKeys.emplace_back("SET_MASK");
            }
        }
    }
    if(mQkSimdMatrix) {
        qkPrefillKeys.emplace_back("SIMD_GROUP_MATRIX");
    }
    std::vector<std::string> qkvPrefillKeys = {
        {"matmul_qkv", ftype, group_str, "FOR_PREFILL"}
    };
    if(mQkvSimdMatrix) {
        qkvPrefillKeys.emplace_back("SIMD_GROUP_MATRIX");
    }
    if (mtbn->useFp16InsteadFp32()) {
        qkPrefillKeys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
        qkvPrefillKeys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
    }
    std::vector<std::string> copyPastKeys = {
        {"pastkv_copy", ftype, group_str}
    };
    std::vector<std::string> shaders = {
        "decode_qk",
        "decode_qkv",
        "prefill_qk",
        "prefill_qkv",
        "copy"
    };
    if(mQkTensorMatrix) {
        shaders[2] = "prefill_qk_tensor";
        shaders[3] = "prefill_qkv_tensor";
        qkPrefillKeys.emplace_back("USE_METAL_TENSOR_OPS");
        qkvPrefillKeys.emplace_back("USE_METAL_TENSOR_OPS");
    }
    std::vector<std::vector<std::string>> keys = {
        qkKeys,
        qkvKeys,
        qkPrefillKeys,
        qkvPrefillKeys,
        copyPastKeys
    };
    std::vector<const char*> sources = {
        gMatMulDivMask,
        gMatMulQKV,
        gMatMulDivMask,
        gMatMulQKV,
        gCopyPastKV
    };

    std::vector<id<MTLComputePipelineState>> pipelines(keys.size());
    for (int i=0; i<keys.size(); ++i) {
        auto pipeline = rt->findPipeline(keys[i]);
        if (nil == pipeline) {
            // Rebuild Pipeline
            MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:@(keys[i][1].c_str()) forKey:@"ftype"];
            [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
            [dic setValue:@(keys[i][2].c_str()) forKey:@"GROUP_SIZE"];
            for (int j=3; j<keys[i].size(); ++j) {
                [dic setValue:@"1" forKey:@(keys[i][j].c_str())];
            }
            option.preprocessorMacros = dic;
            
            pipeline = mtbn->makeComputePipelineWithSourceOption(sources[i], shaders[i].c_str(), option);
            rt->insertPipeline(keys[i], pipeline);
        }
        pipelines[i] = pipeline;
    }
    mKernel_qk = pipelines[0];
    mKernel_qkv = pipelines[1];
    mKernelPrefill_qk = pipelines[2];
    mKernelPrefill_qkv = pipelines[3];
    mKernel_copy = pipelines[4];
    MNN_ASSERT(nil != mKernel_qk);
    MNN_ASSERT(nil != mKernel_qkv);
    MNN_ASSERT(nil != mKernelPrefill_qk);
    MNN_ASSERT(nil != mKernelPrefill_qkv);
    MNN_ASSERT(nil != mKernel_copy);

    if(mSftmSimdReduce) {

        MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
        auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
        option.preprocessorMacros = @{
            @"ftype" : @(ftype.c_str()),
            @"ftype4" : @(ftype4.c_str()),
        };
        std::vector<std::string> keys = {"softmax_sg_reduce", ftype};
        keys.emplace_back("softmax_plane_sg");
        auto pipeline = rt->findPipeline(keys);
        if (nil == pipeline) {
            pipeline = mtbn->makeComputePipelineWithSourceOption(gSoftmaxSgReduce, keys.back().c_str(), option);
            rt->insertPipeline(keys, pipeline);
        }
        mKernel_softmax = pipeline;
    } else {
        mKernel_softmax = [context pipelineWithName:@"softmax_plane" fp16:mtbn->useFp16InsteadFp32()];
    }

    if(mUseFlashAttention)
    {
        std::vector<std::vector<std::string>> flashKeys = {
            {"flash_softmax", ftype},
            {"flash_matmul_qkv", ftype},
            {"flash_scale", ftype}
        };
        
        MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
        auto basicDic = [NSMutableDictionary dictionaryWithCapacity:0];
        [basicDic setValue:@(ftype.c_str()) forKey:@"ftype"];
        
        {
            NSMutableDictionary *dic = [basicDic mutableCopy];
            if(mSftmSimdReduce) {
                [dic setValue:@"1" forKey:@"SIMD_GROUP_REDUCE"];
                flashKeys[0].emplace_back("SIMD_GROUP_REDUCE");
            }
            option.preprocessorMacros = dic;
            
            
            auto pipeline = rt->findPipeline(flashKeys[0]);
            if (nil == pipeline) {
                pipeline = mtbn->makeComputePipelineWithSourceOption(gFlashSoftmax, "flash_softmax", option);
                rt->insertPipeline(flashKeys[0], pipeline);
            }
            
            mKernel_flash_softmax = pipeline;//mtbn->makeComputePipelineWithSourceOption(gFlashSoftmax, "flash_softmax", option);
        }
        {
            NSMutableDictionary *dic = [basicDic mutableCopy];
            if(mQkvSimdMatrix) {
                [dic setValue:@"1" forKey:@"SIMD_GROUP_MATRIX"];
                flashKeys[1].emplace_back("SIMD_GROUP_MATRIX");
            }
            if(mQkvSimdReduce) {
                [dic setValue:@"1" forKey:@"SIMD_GROUP_REDUCE"];
                flashKeys[1].emplace_back("SIMD_GROUP_REDUCE");
            }
            if (mtbn->useFp16InsteadFp32()) {
                [dic setValue:@"1" forKey:@"MNN_METAL_FLOAT16_STORAGE"];
                flashKeys[1].emplace_back("MNN_METAL_FLOAT16_STORAGE");
            }
            
            option.preprocessorMacros = dic;
            
            auto pipeline = rt->findPipeline(flashKeys[1]);
            if (nil == pipeline) {
                pipeline = mtbn->makeComputePipelineWithSourceOption(gFlashMatMulQKV, "flash_matmul_qkv", option);
                rt->insertPipeline(flashKeys[1], pipeline);
            }
            
            mKernel_flash_matmul_qkv = pipeline;
            //            mKernel_flash_matmul_qkv = mtbn->makeComputePipelineWithSourceOption(gFlashMatMulQKV, "flash_matmul_qkv", option);
        }
        {
            NSMutableDictionary *dic = [basicDic mutableCopy];
            option.preprocessorMacros = dic;
            
            auto pipeline = rt->findPipeline(flashKeys[2]);
            if (nil == pipeline) {
                pipeline = mtbn->makeComputePipelineWithSourceOption(gFlashScale, "flash_scale", option);
                rt->insertPipeline(flashKeys[2], pipeline);
            }
            
            mKernel_flash_scale = pipeline;
            mKernel_flash_scale = pipeline;
            //            mKernel_flash_scale = mtbn->makeComputePipelineWithSourceOption(gFlashScale, "flash_scale", option);
        }
    }
    if(mUseFlashAttentionFused) {
        MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
        auto basicDic = [NSMutableDictionary dictionaryWithCapacity:0];
        [basicDic setValue:@(ftype.c_str()) forKey:@"ftype"];
        [basicDic setValue:@(ftype4.c_str()) forKey:@"ftype4"];

        std::vector<std::string> keys = {"flash_attention_fused", ftype};
        {
            // Fused Attention (Naive/Simd)
            NSMutableDictionary *dic = [basicDic mutableCopy];
//            if(mSftmSimdReduce) {
//                [dic setValue:@"1" forKey:@"SIMD_GROUP_REDUCE"];
//                keys.emplace_back("SIMD_GROUP_REDUCE");
//            }
            if(mQkvSimdMatrix) {
                [dic setValue:@"1" forKey:@"SIMD_GROUP_MATRIX"];
                keys.emplace_back("SIMD_GROUP_MATRIX");
            }
            if (mtbn->useFp16InsteadFp32()) {
                [dic setValue:@"1" forKey:@"MNN_METAL_FLOAT16_STORAGE"];
                keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
            }
            if(mHasMask) {
                 if(mIsAddMask) {
                     [dic setValue:@"1" forKey:@"ADD_MASK"];
                 } else {
                     [dic setValue:@"1" forKey:@"SET_MASK"];
                 }
            }
            option.preprocessorMacros = dic;
            
            auto pipeline = rt->findPipeline(keys);
            if (nil == pipeline) {
                pipeline = mtbn->makeComputePipelineWithSourceOption(gFlashAttentionFused, "flash_attention_fused", option);
                rt->insertPipeline(keys, pipeline);
            }
            mKernel_flash_fused = pipeline;
        }
    }
}

void AttentionBufExecution::handleKVAllocMemory() {
    if(mKVCache) {
        mKVCacheManager->setPastLength(mMeta != nullptr ? mMeta->previous : 0);

        if (nullptr == mMeta || mMeta->previous == mMeta->remove) {
            mKVCacheManager->onClear();
            mKVCacheManager->onAlloc(mMeta, mSeqLen);
        } else {
            MNN_ASSERT(mMeta->previous == mKVCacheManager->kvLength());
            mKVCacheManager->onRealloc(mMeta);
        }
        
        mKvSeqLen = mKVCacheManager->kvLength() + mSeqLen;
        mKvMaxLen = mKVCacheManager->maxLength();
        
        float useMemorySize = 1.0 * mKvMaxLen / 1024.0 * mSeqLen / 1024.0 * mBatch * mNumHead;
        // elementSize larger than 32M
        mQseqSplitNum = 1;
        
        if(mUseFlashAttentionFused) {
            // no need temp memory
            return;
        }

        int qSeqLenPiece = UP_DIV(mSeqLen, mQseqSplitNum);
        // temp tensor alloc memory
        bool needMalloc = mTempQK->length(0) != mBatch * mNumHead;
        if (mTempQK->length(1) != qSeqLenPiece * mKvMaxLen) {
            needMalloc = true;
        }

        if (needMalloc) {
            mTempQK->setLength(0, mBatch * mNumHead);
            mTempQK->setLength(1, qSeqLenPiece * mKvMaxLen);
            mTempSoftMax->setLength(0, mBatch * mNumHead);
            mTempSoftMax->setLength(1, qSeqLenPiece * mKvMaxLen);
            
            if (mUseFlashAttention) {
                // Flash Attention
                int blockSize = MNN_FLASH_ATTENTION_BLOCK_SIZE;
                mTempQK->setLength(1, mSeqLen * blockSize);
                mTempSoftMax->setLength(1, mSeqLen * blockSize);
                
                mRunningStats->setLength(0, mBatch);
                mRunningStats->setLength(1, mNumHead);
                mRunningStats->setLength(2, mSeqLen);
                mRunningStats->setLength(3, 2 * 4/*sizeof(float)*/);
                
                mCorrectionScale->setLength(0, mBatch);
                mCorrectionScale->setLength(1, mNumHead);
                mCorrectionScale->setLength(2, mSeqLen * 4/*sizeof(float)*/);
                
                mTempOutput->setLength(0, mSeqLen * mBatch);
                mTempOutput->setLength(1, mNumHead * mHeadDim * 4/*sizeof(float)*/);
            }
            
            auto res = backend()->onAcquireBuffer(mTempQK.get(), Backend::STATIC) && backend()->onAcquireBuffer(mTempSoftMax.get(), Backend::STATIC);
            if (mUseFlashAttention) {
                res = res && backend()->onAcquireBuffer(mRunningStats.get(), Backend::STATIC);
                res = res && backend()->onAcquireBuffer(mCorrectionScale.get(), Backend::STATIC);
                res = res && backend()->onAcquireBuffer(mTempOutput.get(), Backend::STATIC);
            }
            if (!res) {
                MNN_ERROR("MNN::Metal: OUT_OF_MEMORY when execute attention metal %d\n", res);
                return;
            }
        }
    }
}
ErrorCode AttentionBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mHasMask = inputs.size() > 3;
    if(mHasMask) {
        mIsAddMask = (inputs[3]->getType() == halide_type_of<float>()); 
    }
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    auto shape = query->shape();
    mBatch = shape[0];
    mSeqLen = shape[1];
    mNumHead = shape[2];
    mHeadDim = shape[3];
    mScale = 1.0 / sqrt(mHeadDim);
    // TODO : define short_seq more accurately
    mShortSeq = mSeqLen < 16;
    
    int attentionOption = static_cast<MetalBackend *>(backend())->getRuntime()->hint().attentionOption;
    // hardware resource limit
    mUseFlashAttentionFused = !mShortSeq && (attentionOption / 8 == 2) && mHeadDim <= 128;
    mUseFlashAttention = !mShortSeq && (attentionOption / 8 >= 1) && !mUseFlashAttentionFused;

    mUseSimpleAttention = !mUseFlashAttentionFused && !mUseFlashAttention;
    // Check Env
    mKvNumHead = key->shape()[2];
    mKvSeqLen = key->shape()[1];
    // Align to mKvAlignNum, for simd/tensor matrix load
    mKvMaxLen = ROUND_UP(mKvSeqLen, mKvAlignNum);
    
    if(mKVCache) {
        mKVCacheManager->onResize(mKvNumHead, mHeadDim);
        return NO_ERROR;
    }
    
    float useMemorySize = 1.0 * mKvMaxLen / 1024.0 * mSeqLen / 1024.0 * mBatch * mNumHead;
    // elementSize larger than 32M
    mQseqSplitNum = 1;
    
    // no kv_cache memory, should create temp q/k memory
    mTempK.reset(Tensor::createDevice<float>({mKvMaxLen * mHeadDim * mBatch * mKvNumHead}));
    mTempV.reset(Tensor::createDevice<float>({mKvMaxLen * mHeadDim * mBatch * mKvNumHead}));
    if (mUseSimpleAttention) {
        mTempQK.reset(Tensor::createDevice<float>({mKvMaxLen * UP_DIV(mSeqLen, mQseqSplitNum) * mBatch * mNumHead}));
        mTempSoftMax.reset(Tensor::createDevice<float>({mKvMaxLen * UP_DIV(mSeqLen, mQseqSplitNum) * mBatch * mNumHead}));
    } else if(mUseFlashAttention){
        int blockSize = MNN_FLASH_ATTENTION_BLOCK_SIZE;
        mTempQK.reset(Tensor::createDevice<float>({blockSize * mSeqLen * mBatch * mNumHead}));
        mTempSoftMax.reset(Tensor::createDevice<float>({blockSize * mSeqLen * mBatch * mNumHead}));
        mRunningStats.reset(Tensor::createDevice<uint8_t>({(int)mBatch * mNumHead * mSeqLen * 2 * 4/*sizeof(float)*/}));
        mCorrectionScale.reset(Tensor::createDevice<uint8_t>({mBatch * mNumHead * mSeqLen * 4/*sizeof(float)*/}));
        mTempOutput.reset(Tensor::createDevice<uint8_t>({mBatch * mNumHead * mSeqLen * mHeadDim * 4/*sizeof(float)*/}));

        
        backend()->onAcquireBuffer(mRunningStats.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mCorrectionScale.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);

    }
    
    backend()->onAcquireBuffer(mTempK.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mTempV.get(), Backend::DYNAMIC);
    if (mUseSimpleAttention) {
        backend()->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mTempSoftMax.get(), Backend::DYNAMIC);
    }
    backend()->onReleaseBuffer(mTempK.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempV.get(), Backend::DYNAMIC);
    if (mUseSimpleAttention) {
        backend()->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mTempSoftMax.get(), Backend::DYNAMIC);
    }
    if (mUseFlashAttention) {
        backend()->onReleaseBuffer(mRunningStats.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mCorrectionScale.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}
void AttentionBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    auto rt = (MetalRuntime*)mtbn->runtime();

    int group_size = mNumHead / mKvNumHead;
    
    // temp memory alloc, handle variable set
    Tensor* tempTensorK;
    Tensor* tempTensorV;
    handleKVAllocMemory();
    
    id<MTLBuffer> tempBufferK;
    id<MTLBuffer> tempBufferV;
    if(mKvInDisk) {
        tempBufferK = mKVCacheManager->getKeyBuffer();
        tempBufferV = mKVCacheManager->getValueBuffer();
    } else if(mKVCache) {
        tempTensorK = mKVCacheManager->getKeyTensor();
        tempTensorV = mKVCacheManager->getValueTensor();
    } else {
        tempTensorK = mTempK.get();
        tempTensorV = mTempV.get();
    }
    
    // whether use simdgroup
    bool supportSimdReduce = rt->supportSimdGroupReduce();
    bool supportSimdMatrix = rt->supportSimdGroupMatrix();
    bool supportTensorMatrix = mtbn->isSupportTensorApi();// rt->supportTensorOps();

    // decode and thread number not too large
    mQkSimdReduce = supportSimdReduce && mShortSeq;
    // loop_k can divide 8, thus avoid branch
    mQkSimdMatrix = supportSimdMatrix && mSeqLen >= 16 && mHeadDim % 8 == 0;
    // 32x32x32 tensor block
    mQkTensorMatrix = supportTensorMatrix && mSeqLen >= 128 && mHeadDim % 32 == 0;

    mSftmSimdReduce = supportSimdReduce;
    mQkvSimdReduce = supportSimdReduce && mShortSeq && mHeadDim * mNumHead < mKvSeqLen * 32;
    mQkvSimdMatrix = supportSimdMatrix && mSeqLen >= 16;
    
    // start to compile attention shaders
    compilerShader(inputs);
    
    // Run Copy and Format-Convert Kernel
    {
        auto copyp = (int*)mParamCopy.contents;
        /*
         Key -> K-Cache :   [mBatch, mKvSeqLen, mKvNumHead, mHeadDim] -> [mKvMaxLen, mBatch, mKvNumHead, mHeadDim]
         Value -> V-Cache : [mBatch, mKvSeqLen, mKvNumHead, mHeadDim] -> [mBatch, mKvNumHead, mHeadDim, mKvMaxLen (fill when decode)]
         */
        copyp[0] = mKvNumHead * mHeadDim;
        // current new kv_len
        copyp[1] = key->shape()[1];
        copyp[2] = mKvMaxLen;
        copyp[3] = mKVCacheManager->kvLength() * copyp[0];
        copyp[4] = mKVCacheManager->kvLength();
        copyp[5] = mBatch;
        int copy_line = key->shape()[1];

        id<MTLComputePipelineState> pipeline = mKernel_copy;
        [encoder setComputePipelineState:pipeline];
        MetalBackend::setTensor(key, encoder, 0);
        MetalBackend::setTensor(value, encoder, 1);
        if(mKvInDisk) {
            MetalBackend::setBuffer(tempBufferK, 0, encoder, 2);
            MetalBackend::setBuffer(tempBufferV, 0, encoder, 3);
        } else {
            MetalBackend::setTensor(tempTensorK, encoder, 2);
            MetalBackend::setTensor(tempTensorV, encoder, 3);
        }
        [encoder setBuffer:mParamCopy offset:0 atIndex:4];
        
        std::pair<MTLSize, MTLSize> gl;
        gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(mKvNumHead * mHeadDim, copy_line, mBatch)];

        [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];

    }
    
    // Update Parameters
    int seqLenPiece = UP_DIV(mSeqLen, mQseqSplitNum);
    {
        auto param = (Param*)mParamQKV.contents;
        param->scale = mScale;
        param->head_dim = mHeadDim;
        param->key_seq_len = mKvSeqLen;
        param->head_num = mNumHead;
        param->group = group_size;
        param->query_seq_len = mSeqLen;
        param->q_seq_piece_len = seqLenPiece;
        param->max_kv_len = mKvMaxLen;
        param->batch = mBatch;
        param->kv_align_len = mKvAlignNum;
    }
    
    if (mUseSimpleAttention) {
        for(int seq_idx = 0; seq_idx < mQseqSplitNum; seq_idx++) {
            // Run QK Kernel
            {
                id<MTLComputePipelineState> pipeline;
                 if (mShortSeq) {
                     pipeline = mKernel_qk;
                 } else {
                     pipeline = mKernelPrefill_qk;
                 }
                //pipeline = mKernel_qk;
                [encoder setComputePipelineState:pipeline];
                // [mBatch, mSeqLen, mNumHead, mHeadDim]
                MetalBackend::setTensor(query, encoder, 0);
                // [mBatch, mNumHead, mSeqLen, mKvSeqLen]
                MetalBackend::setTensor(mTempQK.get(), encoder, 1);
                // [mKvSeqLen, mBatch, mKvNumHead, mHeadDim]
                if(mKvInDisk) {
                    MetalBackend::setBuffer(tempBufferK, 0, encoder, 2);
                } else {
                    MetalBackend::setTensor(tempTensorK, encoder, 2);
                }
                [encoder setBytes:&seq_idx length:sizeof(seq_idx) atIndex:3];
                [encoder setBuffer:mParamQKV offset:0 atIndex:4];
                int kv_start = 0, current_block_len = mKvSeqLen;
                [encoder setBytes:&kv_start length:sizeof(kv_start) atIndex:5];
                [encoder setBytes:&current_block_len length:sizeof(int) atIndex:6];
                if(mHasMask) {
                    MetalBackend::setTensor(inputs[3], encoder, 7);
                }

                int decode_grid_y = mBatch * mNumHead;
                std::pair<MTLSize, MTLSize> gl;
            if(mShortSeq) {
                gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(seqLenPiece, decode_grid_y / group_size, mKvSeqLen)];
            } else if(mQkTensorMatrix) {
                gl = std::make_pair(MTLSizeMake(UP_DIV(seqLenPiece, 32), UP_DIV(mKvSeqLen, 32), decode_grid_y), MTLSizeMake(128, 1, 1));
            } else if(mQkSimdMatrix) {
                gl = std::make_pair(MTLSizeMake(UP_DIV(seqLenPiece, 16), UP_DIV(mKvSeqLen, 16), decode_grid_y), MTLSizeMake(32, 1, 1));
            } else {
                gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(seqLenPiece, decode_grid_y, mKvSeqLen)];
            }
            [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
            
            }
            // Run Softmax Kernel
            {
                // For softmax parameter
                // [mBatch, mNumHead, mSeqLen, mKvSeqLen]
                int inside = 1;
                int outside = mBatch * mNumHead * seqLenPiece;
                int axis = mKvSeqLen;
                int axis_align = ROUND_UP(axis, mKvAlignNum);
                {
                    auto softmax = (int*)mParamSoftmax.contents;
                    // Inside, axis, outside, plane(invalid)
                    softmax[0] = inside;
                    softmax[1] = axis;
                    softmax[2] = outside;
                    softmax[3] = axis_align;
                }
                [encoder setComputePipelineState:mKernel_softmax];
                // [mBatch, mNumHead, mSeqLen, mKvSeqLen]
                MetalBackend::setTensor(mTempQK.get(), encoder, 0);
                // [mBatch, mNumHead, mSeqLen, ROUND_UP(mKvSeqLen, mKvAlignNum)]
                MetalBackend::setTensor(mTempSoftMax.get(), encoder, 1);
                [encoder setBuffer:mParamSoftmax offset:0 atIndex:2];
                
                int thread_group_size = 32;
                std::pair<MTLSize, MTLSize> gl;
                if(mSftmSimdReduce) {
                    gl = std::make_pair(MTLSizeMake(inside, outside, 1), MTLSizeMake(thread_group_size, 1, 1));
                } else {
                    gl = [context computeBestGroupAndLocal: mKernel_softmax threads:MTLSizeMake(inside, outside, 1)];
                }
                
                [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
                
            }
            // Run QKV Kernel
            {
                
                id<MTLComputePipelineState> pipeline;
                if (mShortSeq) {
                    pipeline = mKernel_qkv;
                } else {
                    pipeline = mKernelPrefill_qkv;
                }
                [encoder setComputePipelineState:pipeline];
                // [mBatch, mNumHead, mSeqLen, ROUND_UP(mKvSeqLen, mKvAlignNum)]
                MetalBackend::setTensor(mTempSoftMax.get(), encoder, 0);
                // [mBatch, mSeqLen, mNumHead, mHeadDim]
                MetalBackend::setTensor(outputs[0], encoder, 1);
                // [mBatch, mKvNumHead, mHeadDim, mMaxSeqLen]
                if(mKvInDisk) {
                    MetalBackend::setBuffer(tempBufferV, 0, encoder, 2);
                } else {
                    MetalBackend::setTensor(tempTensorV, encoder, 2);
                }
                [encoder setBytes:&seq_idx length:sizeof(seq_idx) atIndex:3];
                [encoder setBuffer:mParamQKV offset:0 atIndex:4];
                std::pair<MTLSize, MTLSize> gl;
                if(mQkvSimdReduce) {
                    gl = std::make_pair(MTLSizeMake(seqLenPiece, mBatch * mNumHead, mHeadDim), MTLSizeMake(32, 1, 1));
                } else if(mQkTensorMatrix){
                    gl = std::make_pair(MTLSizeMake(UP_DIV(seqLenPiece, 32), UP_DIV(mHeadDim, 32), mBatch * mNumHead), MTLSizeMake(128, 1, 1));
                } else if(mQkvSimdMatrix){
                    gl = std::make_pair(MTLSizeMake(UP_DIV(seqLenPiece, 16), UP_DIV(mHeadDim, 16), mBatch * mNumHead), MTLSizeMake(32, 1, 1));
                } else {
                    gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(seqLenPiece, mBatch * mNumHead, mHeadDim)];
                }
                [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
            }
        }
    } else {
        // Flash Attention
        if (mUseFlashAttentionFused) {
            id<MTLComputePipelineState> pipeline = mKernel_flash_fused;
            [encoder setComputePipelineState:pipeline];
            MetalBackend::setTensor(query, encoder, 0);
            if(mKvInDisk) {
                MetalBackend::setBuffer(tempBufferK, 0, encoder, 1);
                MetalBackend::setBuffer(tempBufferV, 0, encoder, 2);
            } else {
                MetalBackend::setTensor(tempTensorK, encoder, 1);
                MetalBackend::setTensor(tempTensorV, encoder, 2);
            }
            if(mHasMask) {
                 MetalBackend::setTensor(inputs[3], encoder, 3);
            }
            MetalBackend::setTensor(outputs[0], encoder, 4);
            [encoder setBuffer:mParamQKV offset:0 atIndex:5];
            
            // TEMPORARY: Revert to stable configuration for debugging
            // Grid: [q_seqlen/16, batch*headNum, 1], Threadgroup: 32 threads
            if(mQkvSimdMatrix) {
                [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(mSeqLen, 8), mBatch * mNumHead, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                [encoder dispatchThreadgroups:MTLSizeMake(mSeqLen, mBatch * mNumHead, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
        } else {
            int blockSize = MNN_FLASH_ATTENTION_BLOCK_SIZE;
            int kv_blocks = UP_DIV(mKvSeqLen, blockSize);
            
            {
                auto param = (Param*)mParamQKV.contents;
                // Original logic updates Param per piece, but here we run full seq.
                // Adjust param for KV block if needed?
                // The prefill_qk uses param.key_seq_len for loop bound.
                // We need to update this per block or just reuse shader carefully?
                // Reuse prefill_qk: keys logic relies on param.
            }

            int seq_idx = 0; // prefill usually 1 piece

            for (int i = 0; i < kv_blocks; ++i) {
                int kv_start = i * blockSize;
                int current_block_len = std::min(blockSize, mKvSeqLen - kv_start);

                // 1. MatMul QK -> TempQK
                {
                    id<MTLComputePipelineState> pipeline = mKernelPrefill_qk;
                    [encoder setComputePipelineState:pipeline];
                    MetalBackend::setTensor(query, encoder, 0);
                    MetalBackend::setTensor(mTempQK.get(), encoder, 1);
                    if(mKvInDisk) {
                        MetalBackend::setBuffer(tempBufferK, 0, encoder, 2);
                    } else {
                        MetalBackend::setTensor(tempTensorK, encoder, 2);
                    }
                    
                    [encoder setBytes:&seq_idx length:sizeof(seq_idx) atIndex:3];
                    [encoder setBuffer:mParamQKV offset:0 atIndex:4];
                    [encoder setBytes:&kv_start length:sizeof(kv_start) atIndex:5];
                    [encoder setBytes:&current_block_len length:sizeof(int) atIndex:6];
                    if(mHasMask) {
                        MetalBackend::setTensor(inputs[3], encoder, 7);
                    }
                    
                    int decode_grid_y = mBatch * mNumHead;
                    std::pair<MTLSize, MTLSize> gl;
                    
                    // Block len logic mirroring original
                    if(mQkTensorMatrix) {
                        gl = std::make_pair(MTLSizeMake(UP_DIV(mSeqLen, 32), UP_DIV(current_block_len, 32), decode_grid_y), MTLSizeMake(128, 1, 1));
                    } else if(mQkSimdMatrix) {
                        gl = std::make_pair(MTLSizeMake(UP_DIV(mSeqLen, 16), UP_DIV(current_block_len, 16), decode_grid_y), MTLSizeMake(32, 1, 1));
                    } else {
                        gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(mSeqLen, decode_grid_y, current_block_len)];
                    }
                    [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
                }

                // 2. Flash Softmax
                {
                    [encoder setComputePipelineState:mKernel_flash_softmax];
                    MetalBackend::setTensor(mTempQK.get(), encoder, 0);
                    MetalBackend::setTensor(mTempSoftMax.get(), encoder, 1);
                    MetalBackend::setTensor(mRunningStats.get(), encoder, 2);
                    MetalBackend::setTensor(mCorrectionScale.get(), encoder, 3);
                    [encoder setBytes:&current_block_len length:sizeof(int) atIndex:4];
                    [encoder setBuffer:mParamQKV offset:0 atIndex:5];
                    [encoder setBytes:&kv_start length:sizeof(int) atIndex:6];
                    
                    // Grid: [SeqLen, Batch*Head, 1]
                    std::pair<MTLSize, MTLSize> gl;
                    if (mSftmSimdReduce) {
                        gl = std::make_pair(MTLSizeMake(mSeqLen, mBatch * mNumHead, 1), MTLSizeMake(32, 1, 1));
                    } else {
                        gl = [context computeBestGroupAndLocal:mKernel_flash_softmax threads:MTLSizeMake(mSeqLen, mBatch * mNumHead, 1)];
                    }
                    [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
                }
                
                // 3. Flash MatMul QKV
                {
                    [encoder setComputePipelineState:mKernel_flash_matmul_qkv];
                    MetalBackend::setTensor(mTempSoftMax.get(), encoder, 0); // P_block
                    MetalBackend::setTensor(mTempOutput.get(), encoder, 1);         // tempOutput
                    
                    // V_block: needs to be just V tensor
                    if(mKvInDisk) {
                         MetalBackend::setBuffer(tempBufferV, 0, encoder, 2);
                    } else {
                         MetalBackend::setTensor(tempTensorV, encoder, 2);
                    }
                    
                    MetalBackend::setTensor(mCorrectionScale.get(), encoder, 3);
                    [encoder setBytes:&kv_start length:sizeof(int) atIndex:4];
                    [encoder setBytes:&current_block_len length:sizeof(int) atIndex:5];
                    [encoder setBuffer:mParamQKV offset:0 atIndex:6];
                    
                    // Grid: [HeadDim/4, SeqLen, Batch*Head]
                    // We use float4 for HeadDim
                    std::pair<MTLSize, MTLSize> gl;
                    if(mQkvSimdReduce) {
                        gl = std::make_pair(MTLSizeMake(UP_DIV(mHeadDim, 4), mSeqLen, mBatch * mNumHead), MTLSizeMake(32, 1, 1));
                    } else if(mQkvSimdMatrix){
                        gl = std::make_pair(MTLSizeMake(UP_DIV(mSeqLen, 16), UP_DIV(mHeadDim, 16), mBatch * mNumHead), MTLSizeMake(32, 1, 1));
                    } else {
                        gl = [context computeBestGroupAndLocal:mKernel_flash_matmul_qkv threads:MTLSizeMake(UP_DIV(mHeadDim, 4), mSeqLen, mBatch * mNumHead)];
                    }
                    
                    [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
                }
            }
            
            // 4. Flash Scale
            {
                [encoder setComputePipelineState:mKernel_flash_scale];
                MetalBackend::setTensor(mTempOutput.get(), encoder, 0);         // tempOutput
                MetalBackend::setTensor(outputs[0], encoder, 1);
                MetalBackend::setTensor(mRunningStats.get(), encoder, 2);
                [encoder setBuffer:mParamQKV offset:0 atIndex:3];
                
                auto gl = [context computeBestGroupAndLocal:mKernel_flash_scale threads:MTLSizeMake(UP_DIV(mHeadDim, 4), mSeqLen, mBatch * mNumHead)];
                [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
            }
        }
        
    }
    // Update status
    if(mKVCache) {
        mKVCacheManager->setPastLength(mKVCacheManager->kvLength() + mSeqLen);
    }
    return;
}

class AttentionBufCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *> &outputs) const override {
        auto param = op->main_as_AttentionParam();
        return new AttentionBufExecution(backend, param->kv_cache());
    }
};
REGISTER_METAL_OP_TRANSFORMER_CREATOR(AttentionBufCreator, OpType_Attention);

} // namespace MNN
#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif

