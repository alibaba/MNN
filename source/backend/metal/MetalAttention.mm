//
//  MetalAttention.mm
//  MNN
//
//  Created by MNN on b'2024/04/29'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalCast.hpp"
#import "MNNMetalContext.h"
#import "MetalAttentionShader.hpp"
#import "MetalFlashAttnShader.hpp"
#import "MetalSoftmaxShader.hpp"
#import "MetalAttention.hpp"
#import "MetalEnv.hpp"
#include "core/TensorUtils.hpp"

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
    int mask_batch;
    int mask_head_num;
    int mask_q_len;
    int mask_k_len;
    float v_scale;
    float k_scale;
};

struct CopyParam {
    int head_count;
    int kv_seq_len;
    int max_kv_len;
    int dst_k_offset;
    int dst_v_offset;
    int batch;
    int value_c4;
    float v_scale;
    float k_scale;
};

AttentionBufExecution::AttentionBufExecution(Backend* backend, bool kvCache, bool outputC4, float attnScale,
                                             std::shared_ptr<KVQuantParameter> kvQuantParam)
    : MetalExecution(backend),
      mKVCache(kvCache),
      mOutputC4(outputC4),
      mAttnScale(attnScale),
      mKVQuantParameter(kvQuantParam) {
    _init();
}
void AttentionBufExecution::_init() {
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto context = (__bridge MNNMetalContext*)mtbn->context();
    mMeta = (KVMeta*)(mtbn->getMetaPtr());

    mParamQKV = [context newDeviceBuffer:sizeof(Param) access:CPUWriteOnly];
    mParamSoftmax = [context newDeviceBuffer:6 * sizeof(int) access:CPUWriteOnly];
    mParamCopy = [context newDeviceBuffer:sizeof(CopyParam) access:CPUWriteOnly];
    mTempQK.reset(Tensor::createDevice<float>({0, 0}));
    mTempSoftMax.reset(Tensor::createDevice<float>({0, 0}));

    MNN::MetalKVCacheManager::KVCacheConfig kvconfig;
    kvconfig.mKVCacheDir = mtbn->getRuntime()->hint().kvcacheDirPath;
    kvconfig.mPrefixCacheDir = mtbn->getRuntime()->hint().prefixcacheDirPath;
    kvconfig.mExpandChunk = 64;
    kvconfig.mKvAlignNum = mKvAlignNum;

    mKVCacheManager.reset(new MetalKVCacheManager(backend(), kvconfig));
    mKvInDisk = mKVCache && !kvconfig.mKVCacheDir.empty();
    mKVCacheManager->setKVQuantParameter(mKVQuantParameter);
}

void AttentionBufExecution::compilerShader(const std::vector<Tensor*>& inputs) {
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto rt = (MetalRuntime*)mtbn->runtime();
    auto context = (__bridge MNNMetalContext*)mtbn->context();

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
    const bool staticQuantK = mQuantKey && mKVQuantParameter != nullptr && mKVQuantParameter->kScale != 0.0f;
    const bool staticQuantV = mQuantValue && mKVQuantParameter != nullptr && mKVQuantParameter->vScale != 0.0f;
    const bool dynamicQuantK = mQuantKey && !staticQuantK;
    const bool dynamicQuantV = mQuantValue && !staticQuantV;
    std::vector<std::string> qkKeys = {{"matmul_qk_div_mask", ftype, group_str}};

    std::vector<std::string> qkvKeys = {{"matmul_qkv", ftype, group_str}};
    if (mQkvSimdReduce) {
        qkvKeys.emplace_back("SIMD_GROUP_REDUCE");
    }
    std::vector<std::string> qkPrefillKeys = {{"matmul_qk_div_mask", ftype, group_str, "FOR_PREFILL"}};
    if (mHasMask) {
        if (mIsAddMask) {
            qkPrefillKeys.emplace_back("ADD_MASK");
            if (seq_len > 1) {
                qkKeys.emplace_back("ADD_MASK");
            }
        } else {
            qkPrefillKeys.emplace_back("SET_MASK");
            if (seq_len > 1) {
                qkKeys.emplace_back("SET_MASK");
            }
        }
    } else if (mKVCache) {
        qkPrefillKeys.emplace_back("DEFAULT_MASK");
        if (seq_len > 1) {
            qkKeys.emplace_back("DEFAULT_MASK");
        }
    }
    if (mQkSimdMatrix) {
        qkPrefillKeys.emplace_back("SIMD_GROUP_MATRIX");
    }
    if (mQkCausalTri) {
        qkPrefillKeys.emplace_back("CAUSAL_TRI");
    }
    std::vector<std::string> qkvPrefillKeys = {{"matmul_qkv", ftype, group_str, "FOR_PREFILL"}};
    if (mQkvSimdMatrix) {
        qkvPrefillKeys.emplace_back("SIMD_GROUP_MATRIX");
    }
    if (mCausalBound) {
        // activates av_k_upper causal truncation in prefill_qkv (both non-tensor
        // and tensor variants of prefill_qkv observe CAUSAL_BOUND).
        qkvPrefillKeys.emplace_back("CAUSAL_BOUND");
    }
    if (mtbn->useFp16InsteadFp32()) {
        qkPrefillKeys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
        qkvPrefillKeys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
    }
    if (mQuantKey) {
        qkKeys.emplace_back("QUANT_K");
        qkPrefillKeys.emplace_back("QUANT_K");
        if (dynamicQuantK) {
            qkKeys.emplace_back("DYNAMIC_QUANT_K");
            qkPrefillKeys.emplace_back("DYNAMIC_QUANT_K");
        }
    }
    if (mQuantValue) {
        qkvKeys.emplace_back("QUANT_V");
        qkvPrefillKeys.emplace_back("QUANT_V");
        if (dynamicQuantV) {
            qkvKeys.emplace_back("DYNAMIC_QUANT_V");
            qkvPrefillKeys.emplace_back("DYNAMIC_QUANT_V");
        }
    }
    std::vector<std::string> copyPastKeys = {{"pastkv_copy", ftype, group_str}};
    if (mQuantValue) {
        copyPastKeys.emplace_back("KV_QUANT_V");
    }
    if (mQuantKey) {
        copyPastKeys.emplace_back("KV_QUANT_K");
    }
    if (dynamicQuantK || dynamicQuantV) {
        copyPastKeys.emplace_back("DYNAMIC_QUANT");
        if (mCopySimdReduce) {
            copyPastKeys.emplace_back("SIMD_GROUP_REDUCE");
        }
    }
    std::vector<std::string> shaders = {"decode_qk", "decode_qkv", "prefill_qk", "prefill_qkv", "copy"};
    if (mQkTensorMatrix) {
        shaders[2] = "prefill_qk_tensor";
        shaders[3] = "prefill_qkv_tensor";
        qkPrefillKeys.emplace_back("USE_METAL_TENSOR_OPS");
        qkvPrefillKeys.emplace_back("USE_METAL_TENSOR_OPS");
    }
    if (mOutputC4) {
        qkvKeys.emplace_back("ATTENTION_C4");
        qkvPrefillKeys.emplace_back("ATTENTION_C4");
        if (mQkvSimdReduce) {
            qkvKeys.emplace_back("ATTENTION_C4_VEC2");
            shaders[1] = "decode_qkv_c2";
        }
    }
    std::vector<std::vector<std::string>> keys = {qkKeys, qkvKeys, qkPrefillKeys, qkvPrefillKeys, copyPastKeys};
    std::vector<const char*> sources = {gMatMulDivMask, gMatMulQKV, gMatMulDivMask, gMatMulQKV, gCopyPastKV};

    std::vector<id<MTLComputePipelineState>> pipelines(keys.size());
    for (int i = 0; i < keys.size(); ++i) {
        auto pipeline = rt->findPipeline(keys[i]);
        if (nil == pipeline) {
            // Rebuild Pipeline
            MTLCompileOptions* option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:@(keys[i][1].c_str()) forKey:@"ftype"];
            [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
            [dic setValue:@(keys[i][2].c_str()) forKey:@"GROUP_SIZE"];
            for (int j = 3; j < keys[i].size(); ++j) {
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

    MTLCompileOptions* option = [[MTLCompileOptions alloc] init];
    auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
    [dic setValue:@(ftype.c_str()) forKey:@"ftype"];
    [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
    if (mCausalBound) {
        // bounded softmax: reduce/write only the causally-valid row prefix
        [dic setValue:@"1" forKey:@"CAUSAL_BOUND"];
    }
    option.preprocessorMacros = dic;
    {
        std::vector<std::string> keys = {"softmax_sg_reduce", ftype};
        keys.emplace_back(mSftmSimdReduce ? "softmax_plane_sg" : "softmax_plane");
        if (mCausalBound) {
            keys.emplace_back("CAUSAL_BOUND");
        }
        auto pipeline = rt->findPipeline(keys);
        if (nil == pipeline) {
            pipeline = mtbn->makeComputePipelineWithSourceOption(gSoftmaxSgReduce, mSftmSimdReduce ? "softmax_plane_sg" : "softmax_plane", option);
            rt->insertPipeline(keys, pipeline);
        }
        mKernel_softmax = pipeline;
    }
    if (mDecodeQkSoftmax) {
        std::string head_dim_str = std::to_string(mHeadDim);
        std::vector<std::string> keys = {"decode_qk_softmax", ftype, group_str, "HEAD_DIM_" + head_dim_str};
        if (mKvSeqLen <= 128) {
            keys.emplace_back("SHORT_KV_128");
        }
        if (mQuantKey) {
            keys.emplace_back("QUANT_K");
            if (dynamicQuantK) {
                keys.emplace_back("DYNAMIC_QUANT_K");
            }
        }
        auto pipeline = rt->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions* option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:@(ftype.c_str()) forKey:@"ftype"];
            [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
            [dic setValue:@(group_str.c_str()) forKey:@"GROUP_SIZE"];
            [dic setValue:@(head_dim_str.c_str()) forKey:@"HEAD_DIM"];
            for (int j = 4; j < keys.size(); ++j) {
                [dic setValue:@"1" forKey:@(keys[j].c_str())];
            }
            option.preprocessorMacros = dic;
            pipeline = mtbn->makeComputePipelineWithSourceOption(gDecodeQkSoftmax, "decode_qk_softmax", option);
            rt->insertPipeline(keys, pipeline);
        }
        mKernel_qk_softmax = pipeline;
        MNN_ASSERT(nil != mKernel_qk_softmax);
    }
    if (mDecodeSplitKV) {
        std::string head_dim_str = std::to_string(mHeadDim);
        std::vector<std::string> keys = {"decode_splitkv", ftype, group_str, "HEAD_DIM_" + head_dim_str};
        if (mOutputC4) {
            keys.emplace_back("ATTENTION_C4");
        }
        if (mQuantKey) {
            keys.emplace_back("QUANT_K");
            if (dynamicQuantK) {
                keys.emplace_back("DYNAMIC_QUANT_K");
            }
        }
        if (mQuantValue) {
            keys.emplace_back("QUANT_V");
            if (dynamicQuantV) {
                keys.emplace_back("DYNAMIC_QUANT_V");
            }
        }
        std::vector<std::string> keysR = keys;
        keysR[0] = "decode_splitkv_reduce";
        auto pipeline  = rt->findPipeline(keys);
        auto pipelineR = rt->findPipeline(keysR);
        if (nil == pipeline || nil == pipelineR) {
            MTLCompileOptions* option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:@(ftype.c_str()) forKey:@"ftype"];
            [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
            [dic setValue:@(group_str.c_str()) forKey:@"GROUP_SIZE"];
            [dic setValue:@(head_dim_str.c_str()) forKey:@"HEAD_DIM"];
            for (size_t j = 4; j < keys.size(); ++j) {
                [dic setValue:@"1" forKey:@(keys[j].c_str())];
            }
            option.preprocessorMacros = dic;
            pipeline = mtbn->makeComputePipelineWithSourceOption(gDecodeSplitKV, "decode_splitkv", option);
            rt->insertPipeline(keys, pipeline);
            pipelineR = mtbn->makeComputePipelineWithSourceOption(gDecodeSplitKV, "decode_splitkv_reduce", option);
            rt->insertPipeline(keysR, pipelineR);
        }
        mKernel_splitkv = pipeline;
        mKernel_splitkv_reduce = pipelineR;
        MNN_ASSERT(nil != mKernel_splitkv && nil != mKernel_splitkv_reduce);
    }
    if (mFlashAttnPrefill) {
        std::string head_dim_str = std::to_string(mHeadDim);
        std::vector<std::string> keys = {"prefill_flash_attn", ftype, group_str, "HEAD_DIM_" + head_dim_str};
        if (mHasMask) {
            keys.emplace_back("HAS_MASK");
        }
        if (mOutputC4) {
            keys.emplace_back("ATTENTION_C4");
        }
        if (mQuantKey) {
            keys.emplace_back("QUANT_K");
        }
        if (mQuantValue) {
            keys.emplace_back("QUANT_V");
        }
        auto pipeline = rt->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions* option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:@(ftype.c_str()) forKey:@"ftype"];
            [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
            [dic setValue:@(group_str.c_str()) forKey:@"GROUP_SIZE"];
            [dic setValue:@(head_dim_str.c_str()) forKey:@"HEAD_DIM"];
            if (mHasMask) {
                [dic setValue:@"1" forKey:@"HAS_MASK"];
            }
            if (mOutputC4) {
                [dic setValue:@"1" forKey:@"ATTENTION_C4"];
            }
            if (mQuantKey) {
                [dic setValue:@"1" forKey:@"QUANT_K"];
            }
            if (mQuantValue) {
                [dic setValue:@"1" forKey:@"QUANT_V"];
            }
            option.preprocessorMacros = dic;
            pipeline = mtbn->makeComputePipelineWithSourceOption(gPrefillFlashAttn, "prefill_flash_attn", option);
            rt->insertPipeline(keys, pipeline);
        }
        mKernel_flashAttn = pipeline;
        MNN_ASSERT(nil != mKernel_flashAttn);
    }
}

void AttentionBufExecution::handleKVAllocMemory() {
    constexpr auto allocType = Backend::DYNAMIC_IN_EXECUTION;
    if (!mKVCache) {
        mKvSeqLen = mCurrentKvLen;
        mKvMaxLen = ROUND_UP(mKvSeqLen, mKvAlignNum);
        mQseqSplitNum = 1;

        int keySize = mKvMaxLen * mBatch * mKvNumHead * mHeadDim;
        int valueSize = mBatch * mKvNumHead * mHeadDim * mKvMaxLen;
        if (nullptr == mTempK || mTempK->elementSize() != keySize) {
            mTempK.reset(Tensor::createDevice<float>({keySize}));
        }
        if (nullptr == mTempV || mTempV->elementSize() != valueSize) {
            mTempV.reset(Tensor::createDevice<float>({valueSize}));
        }

        int qSeqLenPiece = UP_DIV(mSeqLen, mQseqSplitNum);
        bool needMalloc = mTempQK->length(0) != mBatch * mNumHead;
        if (mTempQK->length(1) != qSeqLenPiece * mKvMaxLen) {
            needMalloc = true;
        }
        if (needMalloc) {
            mTempQK->setLength(0, mBatch * mNumHead);
            mTempQK->setLength(1, qSeqLenPiece * mKvMaxLen);
            mTempSoftMax->setLength(0, mBatch * mNumHead);
            mTempSoftMax->setLength(1, qSeqLenPiece * mKvMaxLen);
        }

        auto res = backend()->onAcquireBuffer(mTempK.get(), allocType) &&
                   backend()->onAcquireBuffer(mTempV.get(), allocType) &&
                   backend()->onAcquireBuffer(mTempQK.get(), allocType) &&
                   backend()->onAcquireBuffer(mTempSoftMax.get(), allocType);
        if (!res) {
            MNN_ERROR("MNN::Metal: OUT_OF_MEMORY when execute attention metal %d\n", res);
            return;
        }
        backend()->onReleaseBuffer(mTempK.get(), allocType);
        backend()->onReleaseBuffer(mTempV.get(), allocType);
        backend()->onReleaseBuffer(mTempQK.get(), allocType);
        backend()->onReleaseBuffer(mTempSoftMax.get(), allocType);
        return;
    }

    if (nullptr == mMeta || mMeta->previous == mMeta->remove) {
        mKVCacheManager->onClear();
        mKVCacheManager->onAlloc(mMeta, mCurrentKvLen);
    } else {
        mKVCacheManager->onRealloc(mMeta);
    }

    mKvSeqLen = mKVCacheManager->kvLength() + mCurrentKvLen;
    mKvMaxLen = mKVCacheManager->maxLength();
    float useMemorySize = 1.0 * mKvMaxLen / 1024.0 * mSeqLen / 1024.0 * mBatch * mNumHead;
    // elementSize larger than 32M
    mQseqSplitNum = 1;

    // Flash-attn prefill path is self-contained: online softmax accumulator lives
    // in threadgroup memory and never materializes the full QK / softmax tensors.
    // Skipping these scratch buffers is the whole point of using flash-attn for
    // long context — mTempQK alone is O(B * H * seq * kv_max) which reaches TB
    // scale at 512K prompts.
    if (mFlashAttnPrefill) {
        return;
    }

    // Split-KV decode: only needs the per-workgroup partial buffer
    // [B*H, nwg_max, head_dim + 2] floats; never touches mTempQK/mTempSoftMax.
    if (mDecodeSplitKV) {
        int splitSize = mBatch * mNumHead * mSplitKVNwgMax * (mHeadDim + 2);
        if (nullptr == mTempSplitKV || mTempSplitKV->elementSize() != splitSize) {
            mTempSplitKV.reset(Tensor::createDevice<float>({splitSize}));
        }
        if (!backend()->onAcquireBuffer(mTempSplitKV.get(), allocType)) {
            MNN_ERROR("MNN::Metal: OUT_OF_MEMORY when execute attention metal splitkv\n");
            return;
        }
        backend()->onReleaseBuffer(mTempSplitKV.get(), allocType);
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
    }

    auto res = backend()->onAcquireBuffer(mTempQK.get(), allocType) &&
               backend()->onAcquireBuffer(mTempSoftMax.get(), allocType);
    if (!res) {
        MNN_ERROR("MNN::Metal: OUT_OF_MEMORY when execute attention metal %d\n", res);
        return;
    }
    backend()->onReleaseBuffer(mTempQK.get(), allocType);
    backend()->onReleaseBuffer(mTempSoftMax.get(), allocType);
}

ErrorCode AttentionBufExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mHasMask = inputs.size() > 3 && inputs[3]->dimensions() >= 2;
    if (mHasMask) {
        mIsAddMask = (inputs[3]->getType() == halide_type_of<float>());
    }
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto context = (__bridge MNNMetalContext*)mtbn->context();
    auto shape = query->shape();
    mBatch = shape[0];
    mSeqLen = shape[1];
    mNumHead = shape[2];
    mHeadDim = shape[3];
    mScale = (mAttnScale == 0.0f) ? (1.0f / sqrt(mHeadDim)) : mAttnScale;
    // TODO : define short_seq more accurately
    mShortSeq = mSeqLen < 16;
    // hardware resource limit
    // Check Env
    mKvNumHead = key->shape()[2];
    mCurrentKvLen = key->shape()[1];
    mKvSeqLen = mCurrentKvLen;
    // Align to mKvAlignNum, for simd/tensor matrix load
    mKvMaxLen = ROUND_UP(mKvSeqLen, mKvAlignNum);
    // Enable static KV quantization only when kv-cache is in memory and mhq_quant provides valid scale
    int attentionOption = static_cast<MetalBackend*>(backend())->getRuntime()->hint().attentionOption;
    bool dynamicQuantK = (attentionOption % 8 >= 1);
    bool dynamicQuantV = (attentionOption % 8 > 1);

    mQuantValue = mKVCache && !mKvInDisk &&
                  ((mKVQuantParameter != nullptr && mKVQuantParameter->vScale != 0.0f) || dynamicQuantV);
    mQuantKey = mKVCache && !mKvInDisk &&
                ((mKVQuantParameter != nullptr && mKVQuantParameter->kScale != 0.0f) || dynamicQuantK);
    if (mKVCache) {
        mKVCacheManager->setKVQuantParameter(mKVQuantParameter);
        mKVCacheManager->setAttenQuantKeyValue(mQuantKey, mQuantValue);
        mKVCacheManager->onResize(mKvNumHead, mHeadDim);
    }
    return NO_ERROR;
}
void AttentionBufExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                     id<MTLComputeCommandEncoder> encoder) {
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto context = (__bridge MNNMetalContext*)mtbn->context();
    auto rt = (MetalRuntime*)mtbn->runtime();

    int group_size = mNumHead / mKvNumHead;

    // Split-KV decode attention (env MNN_METAL_DECODE_SPLITKV=1): decided BEFORE
    // handleKVAllocMemory so the partial buffer is allocated on the first decode
    // step. Fused single-pass QK + online-softmax + AV with KV strided across
    // workgroups + reduce dispatch; no threadgroup-memory KV cap.
    // Restricted to fp16 or int8-quantized KV / seq==1 / causal (trivial or no mask).
    {
        const int sSplitKVThresh = MetalEnv::get().decodeSplitKvThresh;
        bool trivialMask = mHasMask && mIsAddMask && mSeqLen == 1 && inputs[3]->elementSize() == 1;
        // threadgroup floats: sq[GS*HD] + s_vs[4*GS*32] + s_out[4*GS*HD] + s_sm[4*GS*2]
        const int tgBytes = (group_size * mHeadDim + 4 * group_size * 32 +
                             4 * group_size * mHeadDim + 4 * group_size * 2) * (int)sizeof(float);
        const int totalKv = (mKVCache && mKVCacheManager != nullptr ? mKVCacheManager->kvLength() : 0) + mCurrentKvLen;
        mDecodeSplitKV = sSplitKVThresh > 0 && totalKv >= sSplitKVThresh &&
                         mKVCache && mSeqLen == 1 && !mKvInDisk &&
                         (!mHasMask || trivialMask) &&
                         (mHeadDim % 32) == 0 && tgBytes <= 30 * 1024;
    }

    // whether use simdgroup
    bool supportSimdReduce = rt->supportSimdGroupReduce();
    bool supportSimdMatrix = rt->supportSimdGroupMatrix();
    bool supportTensorMatrix = mtbn->isSupportTensorApi(); // rt->supportTensorOps();

    // Fused prefill flash-attention: opt-in.
    // Two ways to enable (either turns FA on):
    //   1. Config-level: attention_mode / 8 >= 1 (i.e. attention_mode in {8, 10, ...})
    //      -- matches the CPU convention documented in docs/transformers/llm.md.
    //      attention_mode encodes both KV quant (% 8) and FA (/ 8), so e.g.
    //      attention_mode=10 gives FA + KV int8 in one config value.
    //   2. Env var MNN_ENABLE_FLASH_ATTN_PREFILL=1 (developer override).
    //      MNN_ENABLE_FLASH_ATTN_PREFILL=0 explicitly disables FA even when the
    //      config asks for it (useful for A/B benchmarking).
    // Eligibility (all must hold):
    //   - simdgroup matrix supported (M2+ / Apple GPU 7+)
    //   - KV in memory (not on disk).  KV quantization is supported via
    //     the QUANT_K/QUANT_V shader path (int8 K/V dequanted per 8x8 tile
    //     into small tg scratch before simdgroup_load).
    //   - head_dim in {64, 128, 256}   (256 for Qwen3.5 memory-bound long context)
    //   - GQA group_size in {1, 2, 4, 8}
    //   - prefill length >= 128 (short seqs already fast via existing paths)
    //
    // head_dim=256 (Qwen3.5): kernel is compute-bound and ~2.8% slower than
    // the three-kernel path in isolation (see prior benchmark note), but at
    // long context the fused path skips the O(seq^2 * B * H) mTempQK /
    // mTempSoftMax scratch allocations, which dominates peak memory.  Trade
    // is acceptable for long-context / constrained-device runs.
    //
    // NOTE: must be decided BEFORE handleKVAllocMemory(), which relies on
    // mFlashAttnPrefill to skip the O(B * H * seq * kv_max) mTempQK /
    // mTempSoftMax scratch allocation.  Deciding it afterwards made the first
    // prefill allocate that scratch with a stale flag (GBs at 4K context on
    // multi-B models), pushing Metal past the app memory limit.
    {
        int attentionOption = static_cast<MetalBackend*>(backend())->getRuntime()->hint().attentionOption;
        bool enableFromConfig = (attentionOption / 8) >= 1;
        const int faEnv = MetalEnv::get().flashAttnPrefill;
        bool envForceOn  = faEnv == 1;
        bool envForceOff = faEnv == -1;
        bool enableFlashAttn = envForceOff ? false : (envForceOn || enableFromConfig);

        // FA shader uses simdgroup_half8x8 for Q/K/V/P — only compiles when
        // ftype=half (fp16 precision).  fp32 precision falls back to the
        // three-kernel pipeline.
        //
        // FA also hard-codes causal masking via `kv_valid_offset = seq_k - seq_q`
        // in the `in_bounds` check, so it's only valid when an explicit mask is
        // present (LLM causal ADD-mask exports).  Non-causal / no-mask attention
        // (e.g. Attention op with kv_cache=false and no mask input) must fall
        // back to the three-kernel pipeline.
        bool eligible = supportSimdMatrix
                        && static_cast<MetalBackend*>(backend())->useFp16InsteadFp32()
                        && mHasMask
                        && !mKvInDisk
                        && (mHeadDim == 64 || mHeadDim == 128 || mHeadDim == 256)
                        && (group_size == 1 || group_size == 2 || group_size == 4 || group_size == 8)
                        && !mShortSeq
                        && mSeqLen >= 128;
        mFlashAttnPrefill = enableFlashAttn && eligible;
        // M4-class demotion (measured on M4 Pro, Qwen3-0.6B/4B): the three-kernel
        // path with CAUSAL_TRI + CAUSAL_BOUND beats the FA kernel and the gap
        // grows with seq (pp512 +2.8%, pp2048 +6.6%, pp3312 +7.8%) because the
        // bounded softmax skips O(seq^2)/2 of QK-write + softmax read/write
        // bandwidth that FA does not. Prefer three-kernel on non-tensor-API
        // M4/A-series devices whenever causal-tri can engage and kv is within
        // the measured range; env MNN_ENABLE_FLASH_ATTN_PREFILL=1 still forces FA
        // (and long context keeps FA for its scratch-memory elimination).
        if (mFlashAttnPrefill && !envForceOn) {
            bool boundUsable = !MetalEnv::get().qkCausalTriOff && (mHasMask || mKVCache) && !mKvInDisk &&
                               mKvSeqLen >= mSeqLen;
            // CAUSAL_TRI (QK trapezoid dispatch) is wired on both the simdgroup-
            // matrix path (16x16 tile, added in the original causal-tri commit)
            // and the tensor-API path (32x32 tile, added by the "extend
            // CAUSAL_TRI to tensor" follow-up).
            bool causalTriUsable  = boundUsable && (mQkSimdMatrix || mQkTensorMatrix);
            // CAUSAL_BOUND (softmax row-prefix + prefill_qkv AV early-exit) is
            // path-agnostic — works on both simd-matrix and tensor QK paths.
            bool causalBoundUsable = boundUsable;
            // preferInShaderPrefillDequant is true on M4-class and above (M1/M2/M3
            // are excluded by device name). On tensor-API devices (M5+) demote to
            // three-kernel path so CAUSAL_BOUND can save O(seq^2/2) softmax read/
            // write + AV K-read bandwidth that FA does not skip.
            bool m4Class = rt->preferInShaderPrefillDequant();
            if ((causalTriUsable || causalBoundUsable) && m4Class && mKvSeqLen <= 8192) {
                mFlashAttnPrefill = false;
            }
        }
        static bool _fa_log_once = false;
        if (mFlashAttnPrefill && !_fa_log_once) {
            _fa_log_once = true;
            MNN_PRINT("[MetalAttention] flash-attn-prefill kernel active (seq=%d, head_dim=%d, group=%d, mask=%d, outC4=%d, quant_k=%d, quant_v=%d).\n",
                      mSeqLen, mHeadDim, group_size, (int)mHasMask, (int)mOutputC4,
                      (int)mQuantKey, (int)mQuantValue);
        }
    }

    // temp memory alloc, handle variable set
    Tensor* tempTensorK;
    Tensor* tempTensorV;
    handleKVAllocMemory();
    id<MTLBuffer> tempBufferK;
    id<MTLBuffer> tempBufferV;
    if (mKvInDisk) {
        tempBufferK = mKVCacheManager->getKeyBuffer();
        tempBufferV = mKVCacheManager->getValueBuffer();
    } else if (mKVCache) {
        tempTensorK = mKVCacheManager->getKeyTensor();
        tempTensorV = mKVCacheManager->getValueTensor();
    } else {
        tempTensorK = mTempK.get();
        tempTensorV = mTempV.get();
    }

    // decode and thread number not too large
    mQkSimdReduce = supportSimdReduce && mShortSeq;
    // loop_k can divide 8, thus avoid branch
    mQkSimdMatrix = supportSimdMatrix && mSeqLen >= 16 && mHeadDim % 8 == 0;
    // 32x32x32 tensor block — minimum seqLen=32 matches tile size
    mQkTensorMatrix = supportTensorMatrix && mSeqLen >= 32 && mHeadDim % 32 == 0;

    mSftmSimdReduce = supportSimdReduce;
    mQkvSimdReduce = supportSimdReduce && mShortSeq && mHeadDim * mNumHead < mKvSeqLen * 32;
    mQkvSimdMatrix = supportSimdMatrix && mSeqLen >= 16;
    mCopySimdReduce = mKVCache && supportSimdReduce && mKVCacheManager->useDynamicScaleBuffer();

    // Causal triangular dispatch for prefill_qk (see MetalAttention.hpp).
    // Requires one of the causal mask macros to be compiled (mHasMask || mKVCache),
    // the simdgroup-matrix tile path, in-memory KV, and kv >= q so the diagonal
    // offset D is non-negative for every seq piece.
    {
        const bool sQkTriOff = MetalEnv::get().qkCausalTriOff;
        mQkCausalTri = !sQkTriOff && !mShortSeq && (mQkSimdMatrix || mQkTensorMatrix) &&
                       !mFlashAttnPrefill && (mHasMask || mKVCache) && !mKvInDisk &&
                       mKvSeqLen >= mSeqLen;
        // CAUSAL_BOUND is path-agnostic: activates on both simd-matrix (M4 and
        // below) and tensor-API (M5+) three-kernel prefill paths, so long as we
        // are not on the FA path and the causal-mask semantics hold.
        mCausalBound = !sQkTriOff && !mShortSeq && !mFlashAttnPrefill &&
                       (mHasMask || mKVCache) && !mKvInDisk &&
                       mKvSeqLen >= mSeqLen;
    }

    bool trivialFloatMask = mHasMask && mIsAddMask && mSeqLen == 1 && inputs[3]->elementSize() == 1;
    // Max KV length for fused decode QK+softmax kernel depends on group_size
    // to stay within 32KB threadgroup memory limit:
    //   group_size<=2: 2048, group_size<=4: 1024, group_size<=8: 512
    int maxKvForFusion = 0;
    if (group_size >= 2 && group_size <= 2) maxKvForFusion = 2048;
    else if (group_size <= 4) maxKvForFusion = 1024;
    else if (group_size <= 8) maxKvForFusion = 512;
    mDecodeQkSoftmax = mKVCache && mShortSeq && mSeqLen <= 8 &&
                       (!mHasMask || trivialFloatMask) && !mKvInDisk &&
                       group_size >= 2 && mHeadDim % 8 == 0 && mKvSeqLen <= maxKvForFusion;

    // Split-KV decode (decided at the top of onEncode, before the KV alloc)
    // supersedes the fused qk_softmax path.
    if (mDecodeSplitKV) {
        mDecodeQkSoftmax = false;
    }

    // start to compile attention shaders
    compilerShader(inputs);

#if MNN_METAL_OP_PROFILE
    // Split Attention into per-subpass command buffers so profile shows QK / Softmax / AV / Copy separately.
    static_cast<MetalBackend*>(backend())->setProfileSubtag("copy");
#endif
    // Run Copy and Format-Convert Kernel
    {
        auto copyp = (CopyParam*)mParamCopy.contents;
        /*
         Key -> K-Cache :   [mBatch, mKvSeqLen, mKvNumHead, mHeadDim] -> [mKvMaxLen, mBatch, mKvNumHead, mHeadDim]
         Value -> V-Cache : [mBatch, mKvSeqLen, mKvNumHead, mHeadDim] -> [mBatch, mKvNumHead, mHeadDim, mKvMaxLen (fill
         when decode)]
         */
        copyp->head_count = mKvNumHead * mHeadDim;
        // current new kv_len
        copyp->kv_seq_len = key->shape()[1];
        copyp->max_kv_len = mKvMaxLen;
        int pastLength = mKVCache ? mKVCacheManager->kvLength() : 0;
        copyp->dst_k_offset = pastLength * copyp->head_count;
        copyp->dst_v_offset = pastLength;
        copyp->batch = mBatch;
        copyp->value_c4 =
            TensorUtils::getDescribe(value)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 ? 1 : 0;
        if (mQuantValue && mKVQuantParameter != nullptr) {
            copyp->v_scale = mKVQuantParameter->vScale;
        } else {
            copyp->v_scale = 0.0f;
        }
        if (mQuantKey && mKVQuantParameter != nullptr) {
            copyp->k_scale = mKVQuantParameter->kScale;
        } else {
            copyp->k_scale = 0.0f;
        }
        int copy_line = key->shape()[1];

        id<MTLComputePipelineState> pipeline = mKernel_copy;
        [encoder setComputePipelineState:pipeline];
        MetalBackend::setTensor(key, encoder, 0);
        MetalBackend::setTensor(value, encoder, 1);
        if (mKvInDisk) {
            MetalBackend::setBuffer(tempBufferK, 0, encoder, 2);
            MetalBackend::setBuffer(tempBufferV, 0, encoder, 3);
        } else {
            MetalBackend::setTensor(tempTensorK, encoder, 2);
            MetalBackend::setTensor(tempTensorV, encoder, 3);
        }
        [encoder setBuffer:mParamCopy offset:0 atIndex:4];
        if (mKVCache && mKVCacheManager->getKScaleBuffer() != nil) {
            [encoder setBuffer:mKVCacheManager->getKScaleBuffer() offset:0 atIndex:8];
            [encoder setBuffer:mKVCacheManager->getVScaleBuffer() offset:0 atIndex:9];
        }

        std::pair<MTLSize, MTLSize> gl;
        if (mKVCache && mKVCacheManager->getKScaleBuffer() != nil) {
            int localSize = mCopySimdReduce ? 32 : 128;
            gl = std::make_pair(MTLSizeMake(1, copy_line, mBatch), MTLSizeMake(localSize, 1, 1));
        } else if (mDecodeQkSoftmax) {
            gl = std::make_pair(MTLSizeMake(UP_DIV(mKvNumHead * mHeadDim, 128), copy_line, mBatch), MTLSizeMake(128, 1, 1));
        } else {
            gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(mKvNumHead * mHeadDim, copy_line, mBatch)];
        }

        [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
    }
#if MNN_METAL_OP_PROFILE
    {
        auto* mtbn = static_cast<MetalBackend*>(backend());
        encoder = mtbn->profileNextSubpass(mShortSeq ? "qk_short" : (mDecodeQkSoftmax ? "qk_softmax_fused" : "qk"));
    }
#endif

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
        param->mask_batch = mHasMask ? inputs[3]->length(0) : 1;
        param->mask_head_num = (mHasMask && inputs[3]->dimensions() > 3) ? inputs[3]->length(1) : 1;
        param->mask_q_len = (mHasMask && inputs[3]->dimensions() > 3) ? inputs[3]->length(2) : 1;
        param->mask_k_len = (mHasMask && inputs[3]->dimensions() > 0) ? inputs[3]->length(inputs[3]->dimensions() - 1) : 1;
        if (mQuantValue && mKVQuantParameter != nullptr) {
            param->v_scale = mKVQuantParameter->vScale;
        } else {
            param->v_scale = 0.0f;
        }
        if (mQuantKey && mKVQuantParameter != nullptr) {
            param->k_scale = mKVQuantParameter->kScale;
        } else {
            param->k_scale = 0.0f;
        }
    }

    for (int seq_idx = 0; seq_idx < mQseqSplitNum; seq_idx++) {
        if (mFlashAttnPrefill) {
            // Fused prefill flash-attention: QK + online softmax + PV in a single dispatch.
            // Writes directly to outputs[0]; mTempQK / mTempSoftMax are never touched.
            [encoder setComputePipelineState:mKernel_flashAttn];
            MetalBackend::setTensor(query, encoder, 0);
            MetalBackend::setTensor(outputs[0], encoder, 1);
            MetalBackend::setTensor(tempTensorK, encoder, 2);
            MetalBackend::setTensor(tempTensorV, encoder, 3);
            [encoder setBuffer:mParamQKV offset:0 atIndex:4];
            [encoder setBytes:&seq_idx length:sizeof(seq_idx) atIndex:5];
            int fa_kv_start = 0;
            int fa_kv_len   = mKvSeqLen;
            [encoder setBytes:&fa_kv_start length:sizeof(int) atIndex:6];
            [encoder setBytes:&fa_kv_len   length:sizeof(int) atIndex:7];
            if (mHasMask) {
                MetalBackend::setTensor(inputs[3], encoder, 8);
            }
            if (mQuantKey && mKVCacheManager->getKScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getKScaleBuffer() offset:0 atIndex:9];
            }
            if (mQuantValue && mKVCacheManager->getVScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getVScaleBuffer() offset:0 atIndex:10];
            }
            // Grid = (ceil(seqLenPiece/16), B*H, 1); threadgroup = (32, NSG=4, 1) = 128 threads.
            // Q_TILE=16 halves K read redundancy per pp2048 layer vs Q_TILE=8.
            auto gl = std::make_pair(
                MTLSizeMake(UP_DIV(seqLenPiece, 16), mBatch * mNumHead, 1),
                MTLSizeMake(32, 4, 1));
            [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
#if MNN_METAL_OP_PROFILE
            {
                auto* mtbn2 = static_cast<MetalBackend*>(backend());
                encoder = mtbn2->profileNextSubpass("flash_attn");
            }
#endif
            continue;   // skip the standard QK / softmax / PV path below
        }
        if (mDecodeSplitKV) {
            // Adaptive workgroup count: each workgroup covers 4 simdgroups x 32 kv
            // per stride; target >= 2 strides per workgroup before capping at nwg_max.
            int nwg = ALIMIN(mSplitKVNwgMax, ALIMAX(1, UP_DIV(mKvSeqLen, 256)));
            [encoder setComputePipelineState:mKernel_splitkv];
            MetalBackend::setTensor(query, encoder, 0);
            MetalBackend::setTensor(mTempSplitKV.get(), encoder, 1);
            MetalBackend::setTensor(tempTensorK, encoder, 2);
            MetalBackend::setTensor(tempTensorV, encoder, 3);
            [encoder setBuffer:mParamQKV offset:0 atIndex:4];
            [encoder setBytes:&nwg length:sizeof(nwg) atIndex:5];
            if (mQuantKey && mKVCacheManager->getKScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getKScaleBuffer() offset:0 atIndex:8];
            }
            if (mQuantValue && mKVCacheManager->getVScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getVScaleBuffer() offset:0 atIndex:9];
            }
            [encoder dispatchThreadgroups:MTLSizeMake(nwg, mBatch * mKvNumHead, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
#if MNN_METAL_OP_PROFILE
            {
                auto* mtbn2 = static_cast<MetalBackend*>(backend());
                encoder = mtbn2->profileNextSubpass("splitkv");
            }
#endif
            [encoder setComputePipelineState:mKernel_splitkv_reduce];
            MetalBackend::setTensor(mTempSplitKV.get(), encoder, 0);
            MetalBackend::setTensor(outputs[0], encoder, 1);
            [encoder setBuffer:mParamQKV offset:0 atIndex:4];
            [encoder setBytes:&nwg length:sizeof(nwg) atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake(mBatch * mNumHead, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
#if MNN_METAL_OP_PROFILE
            {
                auto* mtbn2 = static_cast<MetalBackend*>(backend());
                encoder = mtbn2->profileNextSubpass("splitkv_reduce");
            }
#endif
            continue;   // skip the standard QK / softmax / QKV path below
        }
        if (mDecodeQkSoftmax) {
            [encoder setComputePipelineState:mKernel_qk_softmax];
            MetalBackend::setTensor(query, encoder, 0);
            MetalBackend::setTensor(mTempSoftMax.get(), encoder, 1);
            MetalBackend::setTensor(tempTensorK, encoder, 2);
            [encoder setBytes:&seq_idx length:sizeof(seq_idx) atIndex:3];
            [encoder setBuffer:mParamQKV offset:0 atIndex:4];
            if (mKVCache && mQuantKey && mKVCacheManager->getKScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getKScaleBuffer() offset:0 atIndex:8];
            }
            int qkGroups = mBatch * (mNumHead / group_size) * seqLenPiece;
            int maxLocalSize = ALIMAX(32, ((int)mKernel_qk_softmax.maxTotalThreadsPerThreadgroup / 32) * 32);
            int localSize = qkGroups <= 8 ? ALIMIN(maxLocalSize, ALIMAX(128, ROUND_UP(mKvSeqLen, 32))) :
                            ALIMIN(maxLocalSize, ALIMAX(64, ROUND_UP(UP_DIV(mKvSeqLen, 6), 32)));
            auto gl = std::make_pair(MTLSizeMake(mBatch * (mNumHead / group_size), seqLenPiece, 1), MTLSizeMake(localSize, 1, 1));
            [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
        } else {
            // Run QK Kernel
            id<MTLComputePipelineState> pipeline;
            if (mShortSeq) {
                pipeline = mKernel_qk;
            } else {
                pipeline = mKernelPrefill_qk;
            }
            // Split by tile size so the trapezoid tile-count formula matches
            // the CAUSAL_TRI remap in the shader (16-tile for prefill_qk,
            // 32-tile for prefill_qk_tensor).
            const bool useSimdCausalTri   = mQkCausalTri && mQkSimdMatrix && !mQkTensorMatrix;
            const bool useTensorCausalTri = mQkCausalTri && mQkTensorMatrix;
            // pipeline = mKernel_qk;
            [encoder setComputePipelineState:pipeline];
            // [mBatch, mSeqLen, mNumHead, mHeadDim]
            MetalBackend::setTensor(query, encoder, 0);
            // [mBatch, mNumHead, mSeqLen, mKvSeqLen]
            MetalBackend::setTensor(mTempQK.get(), encoder, 1);
            // [mKvSeqLen, mBatch, mKvNumHead, mHeadDim]
            if (mKvInDisk) {
                MetalBackend::setBuffer(tempBufferK, 0, encoder, 2);
            } else {
                MetalBackend::setTensor(tempTensorK, encoder, 2);
            }
            [encoder setBytes:&seq_idx length:sizeof(seq_idx) atIndex:3];
            [encoder setBuffer:mParamQKV offset:0 atIndex:4];
            if (mKVCache && mKVCacheManager->getKScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getKScaleBuffer() offset:0 atIndex:8];
                [encoder setBuffer:mKVCacheManager->getVScaleBuffer() offset:0 atIndex:9];
            }
            int kv_start = 0, current_block_len = mKvSeqLen;
            [encoder setBytes:&kv_start length:sizeof(kv_start) atIndex:5];
            [encoder setBytes:&current_block_len length:sizeof(int) atIndex:6];
            if (mHasMask) {
                MetalBackend::setTensor(inputs[3], encoder, 7);
            }

            int decode_grid_y = mBatch * mNumHead;
            std::pair<MTLSize, MTLSize> gl;
            if (mShortSeq) {
                gl = [context computeBestGroupAndLocal:pipeline
                                               threads:MTLSizeMake(seqLenPiece, decode_grid_y / group_size, mKvSeqLen)];
            } else if (mQkTensorMatrix) {
                if (useTensorCausalTri) {
                    // Trapezoid tile count for 32x32 tiles — mirrors the
                    // CAUSAL_TRI remap in prefill_qk_tensor. Same closed-form
                    // as the 16-tile variant with 32 substituted for 16.
                    int qt = UP_DIV(seqLenPiece, 32);
                    int kt = UP_DIV(mKvSeqLen, 32);
                    int D = (mKvSeqLen - mSeqLen) + seq_idx * seqLenPiece; // kv_start == 0
                    int base = (D + 31) / 32 + 1;
                    int r = kt - base + 1;
                    r = r < 0 ? 0 : (r > qt ? qt : r);
                    NSUInteger total = (NSUInteger)((long)r * base + (long)r * (r - 1) / 2 + (long)(qt - r) * kt);
                    gl = std::make_pair(MTLSizeMake(total, 1, decode_grid_y), MTLSizeMake(128, 1, 1));
                } else {
                    gl = std::make_pair(MTLSizeMake(UP_DIV(seqLenPiece, 32), UP_DIV(mKvSeqLen, 32), decode_grid_y),
                                        MTLSizeMake(128, 1, 1));
                }
            } else if (mQkSimdMatrix) {
                if (useSimdCausalTri) {
                    // Trapezoid tile count — must mirror the CAUSAL_TRI remap in
                    // prefill_qk: row-tile lq covers v(lq) = min(kt, lq + base)
                    // k-tiles; rows 0..r-1 triangle, rows r..qt-1 full kt.
                    int qt = UP_DIV(seqLenPiece, 16);
                    int kt = UP_DIV(mKvSeqLen, 16);
                    int D = (mKvSeqLen - mSeqLen) + seq_idx * seqLenPiece; // kv_start == 0
                    int base = (D + 15) / 16 + 1;
                    int r = kt - base + 1;
                    r = r < 0 ? 0 : (r > qt ? qt : r);
                    NSUInteger total = (NSUInteger)((long)r * base + (long)r * (r - 1) / 2 + (long)(qt - r) * kt);
                    gl = std::make_pair(MTLSizeMake(total, 1, decode_grid_y), MTLSizeMake(32, 1, 1));
                } else {
                    gl = std::make_pair(MTLSizeMake(UP_DIV(seqLenPiece, 16), UP_DIV(mKvSeqLen, 16), decode_grid_y),
                                        MTLSizeMake(32, 1, 1));
                }
            } else {
                gl = [context computeBestGroupAndLocal:pipeline
                                               threads:MTLSizeMake(seqLenPiece, decode_grid_y, mKvSeqLen)];
            }
            [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
#if MNN_METAL_OP_PROFILE
            {
                auto* mtbn = static_cast<MetalBackend*>(backend());
                encoder = mtbn->profileNextSubpass("softmax");
            }
#endif
            // Run Softmax Kernel
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
                // CAUSAL_BOUND fields (ignored by non-causal softmax variants)
                softmax[4] = seqLenPiece;
                softmax[5] = (mKvSeqLen - mSeqLen) + seq_idx * seqLenPiece + 1;
            }
            [encoder setComputePipelineState:mKernel_softmax];
            // [mBatch, mNumHead, mSeqLen, mKvSeqLen]
            MetalBackend::setTensor(mTempQK.get(), encoder, 0);
            // [mBatch, mNumHead, mSeqLen, ROUND_UP(mKvSeqLen, mKvAlignNum)]
            MetalBackend::setTensor(mTempSoftMax.get(), encoder, 1);
            [encoder setBuffer:mParamSoftmax offset:0 atIndex:2];

            int thread_group_size = 32;
            std::pair<MTLSize, MTLSize> softmaxGl;
            if (mSftmSimdReduce) {
                softmaxGl = std::make_pair(MTLSizeMake(inside, outside, 1), MTLSizeMake(thread_group_size, 1, 1));
            } else {
                softmaxGl = [context computeBestGroupAndLocal:mKernel_softmax threads:MTLSizeMake(inside, outside, 1)];
            }

            [encoder dispatchThreadgroups:softmaxGl.first threadsPerThreadgroup:softmaxGl.second];
        }
#if MNN_METAL_OP_PROFILE
        {
            auto* mtbn = static_cast<MetalBackend*>(backend());
            encoder = mtbn->profileNextSubpass("av");
        }
#endif
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
            if (mKvInDisk) {
                MetalBackend::setBuffer(tempBufferV, 0, encoder, 2);
            } else {
                MetalBackend::setTensor(tempTensorV, encoder, 2);
            }
            [encoder setBytes:&seq_idx length:sizeof(seq_idx) atIndex:3];
            [encoder setBuffer:mParamQKV offset:0 atIndex:4];
            if (mKVCache && mKVCacheManager->getKScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getKScaleBuffer() offset:0 atIndex:8];
                [encoder setBuffer:mKVCacheManager->getVScaleBuffer() offset:0 atIndex:9];
            }
            std::pair<MTLSize, MTLSize> gl;
            if (mQkvSimdReduce) {
                int grid_z = mOutputC4 ? UP_DIV(mHeadDim, 2) : mHeadDim;
                gl = std::make_pair(MTLSizeMake(seqLenPiece, mBatch * mNumHead, grid_z), MTLSizeMake(32, 1, 1));
            } else if (mQkTensorMatrix) {
                gl = std::make_pair(MTLSizeMake(UP_DIV(seqLenPiece, 32), UP_DIV(mHeadDim, 32), mBatch * mNumHead),
                                    MTLSizeMake(128, 1, 1));
            } else if (mQkvSimdMatrix) {
                gl = std::make_pair(MTLSizeMake(UP_DIV(seqLenPiece, 16), UP_DIV(mHeadDim, 16), mBatch * mNumHead),
                                    MTLSizeMake(32, 1, 1));
            } else {
                gl = [context computeBestGroupAndLocal:pipeline
                                               threads:MTLSizeMake(seqLenPiece, mBatch * mNumHead, mHeadDim)];
            }
            [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
        }
    }

    // Update status
    if (mKVCache) {
        mKVCacheManager->setPastLength(mKVCacheManager->kvLength() + mCurrentKvLen);
    }
    return;
}

class AttentionBufCreator : public MetalBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend,
                                const std::vector<Tensor*>& outputs) const override {
        auto param = op->main_as_AttentionParam();
        std::shared_ptr<KVQuantParameter> quantParam;
        if (nullptr != param->mhq_quant() && param->mhq_quant()->size() > 0) {
            MNN_ASSERT(param->mhq_quant()->size() == 4);
            std::vector<float> mhqscale(param->mhq_quant()->size());
            for (int i = 0; i < mhqscale.size(); ++i) {
                mhqscale[i] = param->mhq_quant()->GetAs<TensorQuantInfo>(i)->scale();
            }
            quantParam.reset(new KVQuantParameter);
            quantParam->qScale = mhqscale[0];
            quantParam->kScale = mhqscale[1];
            quantParam->qkScale = mhqscale[2];
            quantParam->vScale = mhqscale[3];
        }
        return new AttentionBufExecution(backend, param->kv_cache(), param->output_c4(), param->attnScale(), quantParam);
    }
};
REGISTER_METAL_OP_TRANSFORMER_CREATOR(AttentionBufCreator, OpType_Attention);

} // namespace MNN
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif