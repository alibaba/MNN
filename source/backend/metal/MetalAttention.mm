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
#include "core/AttentionCommon.hpp"
#include "core/TensorUtils.hpp"
#include <algorithm>
#include <cmath>

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
namespace MNN {

static constexpr int kSdpaKvBlock = 512;
static constexpr int kSdpaMinKv = 2;
static constexpr int kSdpaTwoPassMinKv = 2048;

AttentionBufExecution::AttentionBufExecution(Backend* backend, bool kvCache, bool outputC4,
                                             std::shared_ptr<KVQuantParameter> kvQuantParam)
    : MetalExecution(backend),
      mKVCache(kvCache),
      mOutputC4(outputC4),
      mKVQuantParameter(kvQuantParam) {
    _init();
}
void AttentionBufExecution::_init() {
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto context = (__bridge MNNMetalContext*)mtbn->context();
    mMeta = (KVMeta*)(mtbn->getMetaPtr());

    mTempQK.reset(Tensor::createDevice<float>({0, 0}));
    mTempSoftMax.reset(Tensor::createDevice<float>({0, 0}));

    MNN::MetalKVCacheManager::KVCacheConfig kvconfig;
    kvconfig.mKVCacheDir = mtbn->getRuntime()->hint().kvcacheDirPath;
    kvconfig.mPrefixCacheDir = mtbn->getRuntime()->hint().prefixcacheDirPath;
    kvconfig.mExpandChunk = 64;
    kvconfig.mKvAlignNum = mKvAlignNum;

    mKVCacheManager.reset(new MetalKVCacheManager(backend(), kvconfig));
    mKVCacheManager->addReplayUser(this);
    mKvDirOnDisk = mKVCache && !kvconfig.mKVCacheDir.empty();
    mKvInDisk = mKvDirOnDisk;
    mKVCacheManager->setKVQuantParameter(mKVQuantParameter);
}

AttentionBufExecution::~AttentionBufExecution() {
    if (mKVCacheManager) {
        mKVCacheManager->removeReplayUser(this);
    }
}

void AttentionBufExecution::invalidateReplay() {
    MetalExecution::invalidateReplay();
    mLastKTensor = nullptr;
    mLastVTensor = nullptr;
    mLastSdpaPartials = nullptr;
    mLastKScaleBuffer = nil;
    mLastVScaleBuffer = nil;
}

ErrorCode AttentionBufExecution::compilePipelines(const std::vector<Tensor*>& inputs) {
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto rt = (MetalRuntime*)mtbn->runtime();
    auto context = (__bridge MNNMetalContext*)mtbn->context();

    auto seq_len = inputs[0]->length(1);
    int group_size = inputs[0]->length(2) / inputs[1]->length(2);
    std::string group_str = std::to_string(group_size);

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
    if (mPlan.qkvSimdReduce) {
        qkvKeys.emplace_back("SIMD_GROUP_REDUCE");
    }
    std::vector<std::string> qkPrefillKeys = {{"matmul_qk_div_mask", ftype, group_str, "FOR_PREFILL"}};
    if (mHasTensorMask) {
        // Decode reads the mask too: a sliding window hides old keys even for a single query.
        if (mIsAddMask) {
            qkPrefillKeys.emplace_back("ADD_MASK");
            qkKeys.emplace_back("ADD_MASK");
        } else {
            qkPrefillKeys.emplace_back("SET_MASK");
            qkKeys.emplace_back("SET_MASK");
        }
    } else if (mKVCache && !mFacts.bidirectionalMask) {
        qkPrefillKeys.emplace_back("DEFAULT_MASK");
        if (seq_len > 1) {
            qkKeys.emplace_back("DEFAULT_MASK");
        }
    }
    if (mPlan.qkSimdMatrix) {
        qkPrefillKeys.emplace_back("SIMD_GROUP_MATRIX");
    }
    if (mPlan.causalTri) {
        qkPrefillKeys.emplace_back("CAUSAL_TRI");
    }
    std::vector<std::string> qkvPrefillKeys = {{"matmul_qkv", ftype, group_str, "FOR_PREFILL"}};
    if (mPlan.qkvSimdMatrix) {
        qkvPrefillKeys.emplace_back("SIMD_GROUP_MATRIX");
    }
    if (mPlan.causalBound) {
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
        if (mPlan.copySimdReduce) {
            copyPastKeys.emplace_back("SIMD_GROUP_REDUCE");
        }
    }
    std::vector<std::string> shaders = {"decode_qk", "decode_qkv", "prefill_qk", "prefill_qkv", "copy"};
    const bool tensorPrefillShader = mPlan.qkTensorMatrix;
    if (tensorPrefillShader) {
        shaders[2] = "prefill_qk_tensor";
        shaders[3] = "prefill_qkv_tensor";
        qkPrefillKeys.emplace_back("USE_METAL_TENSOR_OPS");
        qkvPrefillKeys.emplace_back("USE_METAL_TENSOR_OPS");
    }
    if (mOutputC4) {
        qkvKeys.emplace_back("ATTENTION_C4");
        qkvPrefillKeys.emplace_back("ATTENTION_C4");
        if (mPlan.qkvSimdReduce) {
            qkvKeys.emplace_back("ATTENTION_C4_VEC2");
            shaders[1] = "decode_qkv_c2";
        }
    }
    std::vector<std::vector<std::string>> keys = {qkKeys, qkvKeys, qkPrefillKeys, qkvPrefillKeys, copyPastKeys};
    std::vector<const char*> sources = {gMatMulDivMask, gMatMulQKV, gMatMulDivMask, gMatMulQKV, gCopyPastKV};

    std::vector<id<MTLComputePipelineState>> pipelines(keys.size());
    for (int i = 0; i < keys.size(); ++i) {
        auto pipeline = rt->findPipeline(keys[i]);
        if (nil == pipeline && !rt->pipelineCompileFailed(keys[i])) {
            MTLCompileOptions* option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:@(keys[i][1].c_str()) forKey:@"ftype"];
            [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
            [dic setValue:@(keys[i][2].c_str()) forKey:@"GROUP_SIZE"];
            for (int j = 3; j < keys[i].size(); ++j) {
                [dic setValue:@"1" forKey:@(keys[i][j].c_str())];
            }
            option.preprocessorMacros = dic;

            pipeline = mtbn->makeComputePipelineWithSourceOption(sources[i], shaders[i].c_str(), option,
                                                                 tensorPrefillShader && (i == 2 || i == 3));
            // Caching nil marks the compile failure so resize stops retrying.
            rt->insertPipeline(keys[i], pipeline);
        }
        const bool tensorPrefill = tensorPrefillShader && (i == 2 || i == 3);
        if (nil == pipeline || (tensorPrefill &&
            (pipeline.maxTotalThreadsPerThreadgroup < 128 ||
             pipeline.staticThreadgroupMemoryLength > context.device.maxThreadgroupMemoryLength))) {
            if (tensorPrefill) {
                MNN_PRINT("Metal Attention Tensor QK/PV unavailable; retrying non-Tensor path.\n");
                mVeto |= AttnVetoTensorApi | AttnVetoFaTc;
                _resolveRoute(inputs, mFacts.kvLen);
                return compilePipelines(inputs);
            }
            MNN_ERROR("Metal Attention required pipeline unavailable: %s.\n", shaders[i].c_str());
            return NOT_SUPPORT;
        }
        pipelines[i] = pipeline;
    }
    mKernel_qk = pipelines[0];
    mKernel_qkv = pipelines[1];
    mKernelPrefill_qk = pipelines[2];
    mKernelPrefill_qkv = pipelines[3];
    mKernel_copy = pipelines[4];

    MTLCompileOptions* option = [[MTLCompileOptions alloc] init];
    auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
    [dic setValue:@(ftype.c_str()) forKey:@"ftype"];
    [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
    if (mPlan.causalBound) {
        // bounded softmax: reduce/write only the causally-valid row prefix
        [dic setValue:@"1" forKey:@"CAUSAL_BOUND"];
    }
    if (mHasSinks) {
        [dic setValue:@"1" forKey:@"ATTENTION_SINKS"];
    }
    option.preprocessorMacros = dic;
    {
        std::vector<std::string> keys = {"softmax_sg_reduce", ftype};
        keys.emplace_back(mPlan.softmaxSimdReduce ? "softmax_plane_sg" : "softmax_plane");
        if (mPlan.causalBound) {
            keys.emplace_back("CAUSAL_BOUND");
        }
        if (mHasSinks) {
            keys.emplace_back("ATTENTION_SINKS");
        }
        auto pipeline = rt->findPipeline(keys);
        if (nil == pipeline) {
            pipeline = mtbn->makeComputePipelineWithSourceOption(gSoftmaxSgReduce, mPlan.softmaxSimdReduce ? "softmax_plane_sg" : "softmax_plane", option);
            rt->insertPipeline(keys, pipeline);
        }
        mKernel_softmax = pipeline;
        if (nil == pipeline) return NOT_SUPPORT;
    }
    const bool fusedQkSoftmax = mPlan.route == AttnRoute::DecodeFusedQkSoftmax;
    if (fusedQkSoftmax) {
        std::string head_dim_str = std::to_string(mHeadDim);
        std::vector<std::string> keys = {"decode_qk_softmax", ftype, group_str, "HEAD_DIM_" + head_dim_str};
        // qkQsplit flips per token, so each variant needs its own pipeline.
        if (mPlan.qkQsplit) {
            keys.emplace_back("QK_QSPLIT");
        }
        if (mFacts.kvLen <= 128) {
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
        if (nil == mKernel_qk_softmax) return NOT_SUPPORT;
    }
    if (mPlan.route == AttnRoute::DecodeSdpa) {
        std::string head_dim_str = std::to_string(mHeadDim);
        auto buildSdpa = [&](int nsg, int kvb) -> id<MTLComputePipelineState> {
            std::vector<std::string> keys = {"decode_splitkv_sdpa", ftype, group_str,
                                             "HEAD_DIM_" + head_dim_str, "NSG_" + std::to_string(nsg)};
            if (mPlan.sdpaQHeadsPerTg > 1) {
                keys.emplace_back("QHTG_" + std::to_string(mPlan.sdpaQHeadsPerTg));
            }
            if (kvb > 1) {
                keys.emplace_back("KVB_" + std::to_string(kvb));
            }
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
            auto pipeline = rt->findPipeline(keys);
            if (nil == pipeline && !rt->pipelineCompileFailed(keys)) {
                MTLCompileOptions* option = [[MTLCompileOptions alloc] init];
                auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
                [dic setValue:@(ftype.c_str()) forKey:@"ftype"];
                [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
                [dic setValue:@(group_str.c_str()) forKey:@"GROUP_SIZE"];
                [dic setValue:@(head_dim_str.c_str()) forKey:@"HEAD_DIM"];
                [dic setValue:@(std::to_string(nsg).c_str()) forKey:@"SPLITKV_NSG"];
                // keys[0..4] plus the optional QHTG/KVB entries are structural
                // (they map to valued macros, not flags); the rest are plain
                // =1 flags.
                size_t flagStart = 5;
                if (mPlan.sdpaQHeadsPerTg > 1) {
                    [dic setValue:@(std::to_string(mPlan.sdpaQHeadsPerTg).c_str()) forKey:@"SDPA_QHEADS_PER_TG"];
                    flagStart = 6;
                }
                if (kvb > 1) {
                    [dic setValue:@(std::to_string(kvb).c_str()) forKey:@"SDPA_KV_TOKENS_PER_ITER"];
                    flagStart += 1;
                }
                for (size_t j = flagStart; j < keys.size(); ++j) {
                    [dic setValue:@"1" forKey:@(keys[j].c_str())];
                }
                option.preprocessorMacros = dic;
                pipeline = mtbn->makeComputePipelineWithSourceOption(gDecodeSplitKV, "decode_splitkv", option, true);
                rt->insertPipeline(keys, pipeline);
            }
            return pipeline;
        };
        // Degrade NSG until the compiled pipeline can actually host 32*NSG threads.
        int nsg = mPlan.sdpaNsg;
        id<MTLComputePipelineState> pipeline = nil;
        while (nsg >= 4) {
            pipeline = buildSdpa(nsg, mPlan.sdpaKvTokensPerIter);
            if (nil != pipeline && (int)pipeline.maxTotalThreadsPerThreadgroup >= 32 * nsg &&
                pipeline.staticThreadgroupMemoryLength <= context.device.maxThreadgroupMemoryLength) {
                break;
            }
            if (nil != pipeline) {
                MNN_PRINT("MNN::Metal SDPA: NSG %d exceeds pipeline thread cap %d, degrading\n",
                          nsg, (int)pipeline.maxTotalThreadsPerThreadgroup);
            }
            pipeline = nil;
            nsg /= 2;
        }
        if (nil == pipeline) {
            MNN_ERROR("MNN::Metal SDPA: no viable NSG, falling back to legacy decode path\n");
            mVeto |= AttnVetoSdpa;
            _resolveRoute(inputs, mFacts.kvLen);
            return compilePipelines(inputs);
        } else {
            mPlan.sdpaNsg = nsg;
            mKernel_sdpa = pipeline;
        }
        if (mFacts.kvLen >= kSdpaTwoPassMinKv) {
            std::vector<std::string> pkeys = {"decode_splitkv_partial", ftype, group_str,
                                              "HEAD_DIM_" + head_dim_str, "NSG_" + std::to_string(mPlan.sdpaNsg)};
            if (mPlan.sdpaQHeadsPerTg > 1) pkeys.emplace_back("QHTG_" + std::to_string(mPlan.sdpaQHeadsPerTg));
            if (mPlan.sdpaKvTokensPerIter > 1) pkeys.emplace_back("KVB_" + std::to_string(mPlan.sdpaKvTokensPerIter));
            if (mQuantKey) { pkeys.emplace_back("QUANT_K"); if (dynamicQuantK) pkeys.emplace_back("DYNAMIC_QUANT_K"); }
            if (mQuantValue) { pkeys.emplace_back("QUANT_V"); if (dynamicQuantV) pkeys.emplace_back("DYNAMIC_QUANT_V"); }
            auto ppipeline = rt->findPipeline(pkeys);
            if (nil == ppipeline && !rt->pipelineCompileFailed(pkeys)) {
                MTLCompileOptions* option = [[MTLCompileOptions alloc] init];
                auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
                [dic setValue:@(ftype.c_str()) forKey:@"ftype"];
                [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
                [dic setValue:@(group_str.c_str()) forKey:@"GROUP_SIZE"];
                [dic setValue:@(head_dim_str.c_str()) forKey:@"HEAD_DIM"];
                [dic setValue:@(std::to_string(mPlan.sdpaNsg).c_str()) forKey:@"SPLITKV_NSG"];
                [dic setValue:@(std::to_string(mPlan.sdpaQHeadsPerTg).c_str()) forKey:@"SDPA_QHEADS_PER_TG"];
                [dic setValue:@(std::to_string(mPlan.sdpaKvTokensPerIter).c_str()) forKey:@"SDPA_KV_TOKENS_PER_ITER"];
                [dic setValue:@(std::to_string(kSdpaKvBlock).c_str()) forKey:@"SPLITKV_BLK"];
                [dic setValue:@"1" forKey:@"SPLITKV_PARTIAL"];
                if (mQuantKey) { [dic setValue:@"1" forKey:@"QUANT_K"]; if (dynamicQuantK) [dic setValue:@"1" forKey:@"DYNAMIC_QUANT_K"]; }
                if (mQuantValue) { [dic setValue:@"1" forKey:@"QUANT_V"]; if (dynamicQuantV) [dic setValue:@"1" forKey:@"DYNAMIC_QUANT_V"]; }
                option.preprocessorMacros = dic;
                ppipeline = mtbn->makeComputePipelineWithSourceOption(gDecodeSplitKV, "decode_splitkv", option, true);
                rt->insertPipeline(pkeys, ppipeline);
            }
            std::vector<std::string> rkeys = {"decode_splitkv_reduce", ftype, group_str, "HEAD_DIM_" + head_dim_str};
            if (mOutputC4) rkeys.emplace_back("ATTENTION_C4");
            auto rpipeline = rt->findPipeline(rkeys);
            if (nil == rpipeline && !rt->pipelineCompileFailed(rkeys)) {
                MTLCompileOptions* option = [[MTLCompileOptions alloc] init];
                auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
                [dic setValue:@(ftype.c_str()) forKey:@"ftype"];
                [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
                [dic setValue:@(group_str.c_str()) forKey:@"GROUP_SIZE"];
                [dic setValue:@(head_dim_str.c_str()) forKey:@"HEAD_DIM"];
                if (mOutputC4) [dic setValue:@"1" forKey:@"ATTENTION_C4"];
                option.preprocessorMacros = dic;
                rpipeline = mtbn->makeComputePipelineWithSourceOption(gDecodeSplitKV, "decode_splitkv_reduce", option, true);
                rt->insertPipeline(rkeys, rpipeline);
            }
            const bool partialOk = nil != ppipeline &&
                (int)ppipeline.maxTotalThreadsPerThreadgroup >= 32 * mPlan.sdpaNsg &&
                ppipeline.staticThreadgroupMemoryLength <= context.device.maxThreadgroupMemoryLength;
            if (partialOk && nil != rpipeline) {
                mPlan.sdpaTwoPass = true;
                mKernelSdpaPartial = ppipeline;
                mKernelSdpaReduce = rpipeline;
            }
        }
    }
    if (mPlan.route == AttnRoute::PrefillFaTc) {
        std::string head_dim_str = std::to_string(mHeadDim);
        std::vector<std::string> keys = {"prefill_flash_attn_tc", ftype, "HEAD_DIM_" + head_dim_str};
        const bool preGen17CoopLayout = rt->gpuInfo().gen >= 13 && rt->gpuInfo().gen < 17;
        if (preGen17CoopLayout) {
            keys.emplace_back("MNN_METAL_COOP_PRE_G17");
        }
        if (mHasTensorMask) {
            keys.emplace_back("HAS_MASK");
        }
        if (mOutputC4) {
            keys.emplace_back("ATTENTION_C4");
        }
        keys.emplace_back("FATC_Q_REG");
        keys.emplace_back("FATC_QK_K32");
        keys.emplace_back("FATC_O_CT");
        // K/V read in place as matmul2d tensor handles.
        keys.emplace_back("FATC_KV_DEV_TENSOR");
        keys.emplace_back("FATC_QREV");
        // Persistent QK left-input CTs replace the q_reg array.
        keys.emplace_back("FATC_Q_CT");
        keys.emplace_back("FATC_PV_K32");
        // Must match the dispatch threadgroup width (32 * faTcQueryNsg * faTcHeadDimNsg).
        const int faTcQueryNsg = mPlan.faTcQueryNsg;
        const int faTcHeadDimNsg = mPlan.faTcHeadDimNsg;
        std::string wm_str = std::to_string(faTcQueryNsg);
        std::string wn_str = std::to_string(faTcHeadDimNsg);
        if (faTcQueryNsg != 4) {
            keys.emplace_back("FATC_QUERY_NSG_" + wm_str);
        }
        if (faTcHeadDimNsg > 1) {
            keys.emplace_back("FATC_HEADDIM_NSG_" + wn_str);
        }
        auto pipeline = rt->findPipeline(keys);
        if (nil == pipeline && !rt->pipelineCompileFailed(keys)) {
            MTLCompileOptions* option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            if (preGen17CoopLayout) {
                [dic setValue:@"1" forKey:@"MNN_METAL_COOP_PRE_G17"];
            }
            [dic setValue:@(ftype.c_str()) forKey:@"ftype"];
            [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
            [dic setValue:@(head_dim_str.c_str()) forKey:@"HEAD_DIM"];
            if (mHasTensorMask) {
                [dic setValue:@"1" forKey:@"HAS_MASK"];
            }
            if (mOutputC4) {
                [dic setValue:@"1" forKey:@"ATTENTION_C4"];
            }
            [dic setValue:@"1" forKey:@"FATC_Q_REG"];
            [dic setValue:@"1" forKey:@"FATC_QK_K32"];
            [dic setValue:@"1" forKey:@"FATC_O_CT"];
            [dic setValue:@"1" forKey:@"FATC_KV_DEV_TENSOR"];
            [dic setValue:@"1" forKey:@"FATC_QREV"];
            [dic setValue:@"1" forKey:@"FATC_Q_CT"];
            [dic setValue:@"1" forKey:@"FATC_PV_K32"];
            if (faTcQueryNsg != 4) {
                [dic setValue:@(wm_str.c_str()) forKey:@"FATC_QUERY_NSG"];
            }
            if (faTcHeadDimNsg > 1) {
                [dic setValue:@(wn_str.c_str()) forKey:@"FATC_HEADDIM_NSG"];
            }
            option.preprocessorMacros = dic;
            pipeline = mtbn->makeComputePipelineWithSourceOption(
                gPrefillFlashAttnTc, "prefill_flash_attn_tc", option, true);
            rt->insertPipeline(keys, pipeline);
        }
        if (nil == pipeline || (int)pipeline.maxTotalThreadsPerThreadgroup < 32 * faTcQueryNsg * faTcHeadDimNsg ||
            pipeline.staticThreadgroupMemoryLength > context.device.maxThreadgroupMemoryLength) {
            MNN_ERROR("MNN::Metal FA-TC: pipeline unavailable (cap %d), falling back to three-stage prefill\n",
                      pipeline ? (int)pipeline.maxTotalThreadsPerThreadgroup : -1);
            // The staged fallback differs in causal optimizations and scratch requirements.
            mVeto |= AttnVetoFaTc;
            _resolveRoute(inputs, mFacts.kvLen);
            return compilePipelines(inputs);
        } else {
            mKernel_faTc = pipeline;
        }
    }
    if (mPlan.route == AttnRoute::PrefillFaSg) {
        std::string head_dim_str = std::to_string(mHeadDim);
        // The batched V loop is unrolled over head_dim fragments, so a batch width
        // that does not divide their count would index past mO.
        const int fasgVb = MetalEnv::get().fasgLoadBatch;
        const int fasgLoadBatch = (fasgVb > 0 && (mHeadDim / 8) % fasgVb == 0) ? fasgVb : 0;
        // The kernel keeps the HEAD_DIM/8 Q fragments in registers instead of
        // re-issuing them from smem on every kv tile.
        std::vector<std::string> keys = {"prefill_flash_attn_sg", ftype, "HEAD_DIM_" + head_dim_str};
        if (mHasTensorMask) {
            keys.emplace_back("HAS_MASK");
        }
        if (mOutputC4) {
            keys.emplace_back("ATTENTION_C4");
        }
        if (fasgLoadBatch > 0) {
            keys.emplace_back("FASG_LOADBATCH_" + std::to_string(fasgLoadBatch));
        }
        auto pipeline = rt->findPipeline(keys);
        if (nil == pipeline && !rt->pipelineCompileFailed(keys)) {
            MTLCompileOptions* option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:@(ftype.c_str()) forKey:@"ftype"];
            [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
            [dic setValue:@(head_dim_str.c_str()) forKey:@"HEAD_DIM"];
            if (mHasTensorMask) {
                [dic setValue:@"1" forKey:@"HAS_MASK"];
            }
            if (mOutputC4) {
                [dic setValue:@"1" forKey:@"ATTENTION_C4"];
            }
            if (fasgLoadBatch > 0) {
                [dic setValue:@"1" forKey:@"FASG_LOADBATCH"];
                [dic setValue:@(std::to_string(fasgLoadBatch).c_str()) forKey:@"FASG_VB"];
            }
            option.preprocessorMacros = dic;
            pipeline = mtbn->makeComputePipelineWithSourceOption(
                gPrefillFlashAttnSg, "prefill_flash_attn_sg", option, true);
            rt->insertPipeline(keys, pipeline);
        }
        if (nil == pipeline || (int)pipeline.maxTotalThreadsPerThreadgroup < 32 * mPlan.faSgNsg ||
            pipeline.staticThreadgroupMemoryLength > context.device.maxThreadgroupMemoryLength) {
            MNN_ERROR("MNN::Metal FA-SG: pipeline unavailable (cap %d), falling back to three-stage prefill\n",
                      pipeline ? (int)pipeline.maxTotalThreadsPerThreadgroup : -1);
            mVeto |= AttnVetoFaSg;
            _resolveRoute(inputs, mFacts.kvLen);
            return compilePipelines(inputs);
        }
        mKernel_faSg = pipeline;
    }
    if (fusedQkSoftmax && nil == mKernel_qk_softmax) return NOT_SUPPORT;
    return NO_ERROR;
}

int AttentionBufExecution::_resolveQseqSplit(const AttnFacts& f, AttnRoute route) const {
    if (route != AttnRoute::PrefillStaged || f.seqLen <= 32) {
        return 1;
    }
    // Both scratch tensors are live per q row: 2 * B * H * aligned_kv elements.
    const int elemBytes = f.fp16 ? 2 : 4;
    const int scratchKvLen = ROUND_UP(f.kvLen, mKvAlignNum);
    const int64_t bytesPerQRow = (int64_t)2 * f.batch * f.numHead * scratchKvLen * elemBytes;
    int split = 1;
    if (bytesPerQRow > 0) {
        const int64_t budget = (int64_t)MetalEnv::get().attnQSplitMB << 20;
        const int64_t maxRows = budget / bytesPerQRow;
        if (maxRows < f.seqLen) {
            split = UP_DIV(f.seqLen, (int)ALIMAX((int64_t)1, maxRows));
        }
    }
    // Round up to a power of two: the replay signature stores log2(split), and
    // equal-sized pieces keep the CAUSAL_TRI trapezoid counts uniform.
    int pow2 = 1;
    while (pow2 < split) {
        pow2 <<= 1;
    }
    // Each piece keeps a full 32-row QK tile; log2 must fit the 4-bit signature field.
    const int maxSplit = ALIMIN(f.seqLen / 32, 1 << 15);
    return ALIMAX(1, ALIMIN(pow2, maxSplit));
}

int AttentionBufExecution::_pastKvLength() const {
    // The owning layer encodes first and has already committed this step's rows.
    return mIsKVShared ? mKVCacheManager->kvLength() - mCurrentKvLen : mKVCacheManager->kvLength();
}

AttentionBufExecution::AttnFacts AttentionBufExecution::_collectFacts(const std::vector<Tensor*>& inputs,
                                                                     int totalKv) const {
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto rt = (MetalRuntime*)mtbn->runtime();
    const MetalGpuInfo& gpu = rt->gpuInfo();
    AttnFacts f;
    f.seqLen = mSeqLen;
    f.kvLen = totalKv;
    f.kvMaxLen = mKvMaxLen;
    f.decode = mSeqLen < 16;
    f.simdReduce = rt->supportSimdGroupReduce();
    f.simdMatrix = rt->supportSimdGroupMatrix();
    f.tensorApi = mtbn->isSupportTensorApi();
    f.fp16 = mtbn->useFp16InsteadFp32();
    f.gpuGen = gpu.gen;
    f.highBandwidth = gpu.tier >= MetalGpuInfo::TierPro;
    f.kvCache = mKVCache;
    f.kvInDisk = mKvInDisk;
    f.quantKey = mQuantKey;
    f.quantValue = mQuantValue;
    f.dynamicScaleBuffer = mKVCacheManager->useDynamicScaleBuffer();
    f.sinks = mHasSinks;
    // Shape only: the h2d blit is still queued, so mask bytes may be stale at encode time.
    const bool hasMask = inputs.size() > 3;
    const int maskDims = hasMask ? inputs[3]->dimensions() : 0;
    f.bidirectionalMask = IsAttentionMaskBidirectionalSentinel(
        hasMask, maskDims, hasMask ? inputs[3]->elementSize() : 0);
    // Same predicate as CPUAttention: the sentinel only candidates causality, kv_cache licenses it.
    f.causal = IsAttentionMaskCausal(hasMask, maskDims, mKVCache);
    f.trivialDecodeMask = mHasTensorMask && mIsAddMask && mSeqLen == 1 &&
                          hasMask && inputs[3]->elementSize() == 1;
    f.batch = mBatch;
    f.numHead = mNumHead;
    f.kvNumHead = mKvNumHead;
    f.headDim = mHeadDim;
    f.groupSize = mKvNumHead > 0 ? mNumHead / mKvNumHead : 1;
    // GQA kernels recover the kv head as h / param.group, so the split must be even.
    f.groupDivides = mKvNumHead > 0 && f.groupSize * mKvNumHead == mNumHead;
    return f;
}

bool AttentionBufExecution::_resolveSdpa(const AttnFacts& f, uint8_t veto, AttnPlan& p) {
    const int kvMul = (f.kvLen >= kSdpaTwoPassMinKv) ? 4 : 1;
    const int maxQh = (f.headDim >= 256) ? 2 : f.groupSize;
    p.sdpaQHeadsPerTg = 1;
    for (int cand = 2; cand <= ALIMIN(f.groupSize, maxQh); cand *= 2) {
        if (f.groupSize % cand != 0 || f.numHead % cand != 0) {
            continue;
        }
        const int threadgroups = kvMul * (f.batch * f.numHead / cand);
        if (threadgroups < 8) {
            break;
        }
        if (threadgroups < 16 && f.groupSize / cand < 2) {
            break;
        }
        p.sdpaQHeadsPerTg = cand;
    }
    if (f.groupSize % p.sdpaQHeadsPerTg != 0 || f.numHead % p.sdpaQHeadsPerTg != 0) {
        p.sdpaQHeadsPerTg = 1;
    }
    // nsg widens each threadgroup to hold tgCount*nsg constant, so qh must be resolved first.
    const int tgCount = ALIMAX(kvMul * (f.batch * f.numHead / p.sdpaQHeadsPerTg), 1);
    if (f.gpuGen >= 17) {
        // The tgCount*nsg rule does not hold on this generation.
        p.sdpaNsg = (f.kvLen >= kSdpaTwoPassMinKv && p.sdpaQHeadsPerTg * f.headDim >= 512) ? 16 : 32;
    } else {
        const int product = f.highBandwidth ? 512 : 256;
        // NSG must divide SIMD_GROUP_WIDTH: the decode_splitkv reduce visits
        // lanes lp = sgitg + rep * NSG.
        const int nsgWant = ALIMIN(ALIMAX(product / tgCount, 4), 32);
        p.sdpaNsg = 4;
        while (p.sdpaNsg * 2 <= nsgWant) {
            p.sdpaNsg *= 2;
        }
        if (f.kvLen >= kSdpaTwoPassMinKv && p.sdpaQHeadsPerTg * f.headDim >= 512) {
            // Keep two-pass split-KV at <=16 SGs/TG when per-thread O/Q state is wide
            // so each SG amortizes the cross-SG epilogue over >=32 tokens without spilling.
            p.sdpaNsg = ALIMIN(p.sdpaNsg, 16);
        }
        const bool shortKvNeedsNarrow =
            (!f.highBandwidth && f.kvLen < 512 && tgCount < 16) ||
            (f.highBandwidth && f.kvLen < 256 && tgCount < 16 &&
             p.sdpaQHeadsPerTg * f.headDim > 256);
        if (shortKvNeedsNarrow) {
            p.sdpaNsg = ALIMIN(p.sdpaNsg, 16);
        }
    }
    // threadgroup floats: s_out[NSG*32] + s_sm[NSG*qh*2]
    const int tgBytes = (p.sdpaNsg * 32 + p.sdpaNsg * p.sdpaQHeadsPerTg * 2) * (int)sizeof(float);
    const bool chosen = !(veto & AttnVetoSdpa) && !f.sinks && f.kvCache && !f.kvInDisk &&
                        f.seqLen == 1 && f.kvLen >= kSdpaMinKv &&
                        (f.causal || f.trivialDecodeMask) && (f.headDim % 32) == 0 &&
                        tgBytes <= 30 * 1024;
    // KVB overlaps adjacent kv tokens' dependency chains; it only pays when TGs cannot hide them.
    const int kvbEnv = MetalEnv::get().decodeSdpaKvb;
    const bool wideKvbAuto = (f.gpuGen >= 18) || (f.gpuGen >= 16 && f.highBandwidth);
    const int kvbMinKv = wideKvbAuto ? 512 : 2048;
    const int kvbMaxTg = wideKvbAuto ? 17 : 16;
    int kvb = (kvbEnv > 0) ? kvbEnv : ((chosen && f.kvLen >= kvbMinKv && tgCount < kvbMaxTg) ? 2 : 1);
    // KVB scales the per-lane K/V staging arrays (2*KVB*DPT floats): register pressure clamp.
    if (kvb > 2 && f.headDim > 128) {
        kvb = 2;
    }
    p.sdpaKvTokensPerIter = kvb;
    p.sdpaKvBlocks = UP_DIV(f.kvLen, kSdpaKvBlock);
    return chosen;
}

// Set by threadgroup memory.
static int FusedQkSoftmaxMaxKv(int groupSize) {
    if (groupSize < 2) return 0;
    if (groupSize == 2) return 2048;
    if (groupSize <= 4) return 1024;
    if (groupSize <= 8) return 512;
    return 0;
}

AttentionBufExecution::AttnPlan AttentionBufExecution::_resolvePlan(const AttnFacts& f) const {
    AttnPlan p;
    const bool tensorApi = f.tensorApi && !(mVeto & AttnVetoTensorApi);

    // Inert on fused routes but still part of the signature.
    p.qkSimdReduce = f.simdReduce && f.decode;
    p.qkSimdMatrix = f.simdMatrix && f.seqLen >= 16 && f.headDim % 8 == 0;
    p.qkTensorMatrix = tensorApi && f.seqLen >= 32 && f.headDim % 32 == 0;
    p.qkvSimdReduce = f.simdReduce && f.decode && f.headDim * f.numHead < f.kvLen * 32;
    p.qkvSimdMatrix = f.simdMatrix && f.seqLen >= 16;
    p.softmaxSimdReduce = f.simdReduce;
    p.copySimdReduce = f.kvCache && f.simdReduce && f.dynamicScaleBuffer;

    // Scores stay on-chip; kv >= q keeps the diagonal offset non-negative.
    const bool fusedPrefillOk = !f.sinks && f.fp16 && f.causal && !f.kvInDisk &&
                                !f.quantKey && !f.quantValue && f.kvLen >= f.seqLen;
    const bool faTcHeadDim = f.headDim == 64 || f.headDim == 128 || f.headDim == 256;
    // Older generations cannot hold persistent Q/O cooperative tensors alongside the tile.
    const bool preferStagedTensor = f.headDim == 256 && f.gpuGen <= 16;
    const bool faTcOk = tensorApi && fusedPrefillOk && faTcHeadDim && !preferStagedTensor &&
                        f.seqLen >= 64 && !(mVeto & AttnVetoFaTc);
    const bool faSgHeadDim = f.headDim == 64 || f.headDim == 128;
    const int faSgSeqFloor = f.highBandwidth ? 512 : 1024;
    const bool faSgOk = !tensorApi && fusedPrefillOk && faSgHeadDim && f.seqLen >= faSgSeqFloor &&
                        f.simdMatrix && f.groupDivides && !(mVeto & AttnVetoFaSg);

    if (f.decode) {
        if (_resolveSdpa(f, mVeto, p)) {
            p.route = AttnRoute::DecodeSdpa;
        } else if (!f.sinks && f.kvCache && !f.kvInDisk && f.seqLen <= 8 &&
                   (f.causal || f.trivialDecodeMask) && f.headDim % 8 == 0 &&
                   f.kvLen <= FusedQkSoftmaxMaxKv(f.groupSize)) {
            p.route = AttnRoute::DecodeFusedQkSoftmax;
            p.qkQsplit = f.groupSize == 2 && f.gpuGen < 17 && f.kvLen >= 512;
        } else {
            p.route = AttnRoute::DecodeStaged;
        }
    } else if (faTcOk) {
        p.route = AttnRoute::PrefillFaTc;
        // Each SIMD-group computes one head-dimension slice; reduce partial QK before softmax.
        const int wn = (f.headDim == 256) ? 2 : 1;
        p.faTcHeadDimNsg = (f.headDim % (32 * wn) == 0) ? wn : 1;
        int wm = 4;
        if (p.faTcHeadDimNsg > 1 && f.seqLen <= 64) {
            wm = 1;
        } else if (p.faTcHeadDimNsg > 1 && f.seqLen <= 128) {
            wm = 2;
        } else {
            const int tg64 = UP_DIV(f.seqLen, 64) * f.batch * f.numHead;
            if (tg64 < 16) {
                wm = (tg64 * 2 < 16) ? 1 : 2;
            }
        }
        p.faTcQueryNsg = wm;
    } else if (faSgOk) {
        p.route = AttnRoute::PrefillFaSg;
    } else {
        p.route = AttnRoute::PrefillStaged;
        // Fused kernels early-exit per tile themselves.
        const bool causalUsable = f.causal && !f.kvInDisk && f.kvLen >= f.seqLen;
        p.causalTri = causalUsable && (p.qkSimdMatrix || p.qkTensorMatrix);
        p.causalBound = causalUsable;
    }
    p.seqLenSplitCount = _resolveQseqSplit(f, p.route);
    return p;
}

void AttentionBufExecution::_resolveRoute(const std::vector<Tensor*>& inputs, int totalKv) {
    mFacts = _collectFacts(inputs, totalKv);
    mPlan = _resolvePlan(mFacts);
}

uint64_t AttentionBufExecution::pipelineSignature() const {
    uint64_t signature = _pathSignature(false);
    if (mPlan.route == AttnRoute::DecodeSdpa) {
        signature |= uint64_t(mPlan.sdpaNsg) << 32;
        signature |= uint64_t(mPlan.sdpaKvTokensPerIter) << 40;
    }
    // A vetoed SDPA falls back to routes that flip at more kv boundaries.
    if (mPlan.isDecode()) {
        const int boundaries[] = {2, 129, 256, 512, 513, 1025, 2048, 2049};
        unsigned interval = 0;
        for (int boundary : boundaries) interval += mFacts.kvLen >= boundary;
        signature |= uint64_t(interval) << 48;
    }
    return signature;
}

ErrorCode AttentionBufExecution::preparePipelines(const std::vector<Tensor*>& inputs) {
    const int maskDims = inputs.size() > 3 ? inputs[3]->dimensions() : -1;
    const int maskElements = maskDims >= 0 ? inputs[3]->elementSize() : 0;
    auto mtbn = static_cast<MetalBackend*>(backend());
    const std::vector<int> shape = {mBatch, mSeqLen, mNumHead, mHeadDim, mKvNumHead, mCurrentKvLen,
                                   int(mHasTensorMask), int(mIsAddMask), maskDims, maskElements,
                                   int(mQuantKey), int(mQuantValue), int(inputs.size() > 4),
                                   mtbn->getRuntime()->hint().attentionOption, int(mKvInDisk)};
    if (!mPipelinePlans.empty() && mPreparedShape == shape) {
        return NO_ERROR;
    }
    mPreparedShape.clear();
    mPipelinePlans.clear();
    std::vector<int> kvLengths = {mCurrentKvLen};
    if (mKVCache) {
        // kv boundaries where routing or shader specialization flips; lengths only, no cache data.
        const int boundaries[] = {kSdpaMinKv, 129, 256, 512, 513, 1025,
                                  kSdpaTwoPassMinKv, kSdpaTwoPassMinKv + 1, mSeqLen,
                                  int(int64_t(mHeadDim) * mNumHead / 32 + 1)};
        for (int boundary : boundaries) {
            for (int delta = -1; delta <= 0; ++delta) {
                const int kvLen = boundary + delta;
                if (kvLen >= mCurrentKvLen) kvLengths.push_back(kvLen);
            }
        }
    }
    std::sort(kvLengths.begin(), kvLengths.end());
    kvLengths.erase(std::unique(kvLengths.begin(), kvLengths.end()), kvLengths.end());
    ErrorCode code = NO_ERROR;
    for (int kvLen : kvLengths) {
        mVeto = AttnVetoNone;
        _resolveRoute(inputs, kvLen);
        // Key with the un-vetoed resolution: execution looks up that same key, then applies plan.veto.
        const uint64_t signature = pipelineSignature();
        if (mPipelinePlans.find(signature) != mPipelinePlans.end()) continue;
        mKernel_qk = mKernel_qkv = mKernel_copy = mKernel_softmax = nil;
        mKernelPrefill_qk = mKernelPrefill_qkv = mKernel_qk_softmax = nil;
        mKernel_sdpa = mKernel_faTc = mKernel_faSg = nil;
        mKernelSdpaPartial = mKernelSdpaReduce = nil;
        code = compilePipelines(inputs);
        if (code != NO_ERROR) break;
        PipelinePlan plan;
        plan.qk = mKernel_qk;
        plan.qkv = mKernel_qkv;
        plan.copy = mKernel_copy;
        plan.softmax = mKernel_softmax;
        plan.prefillQk = mKernelPrefill_qk;
        plan.prefillQkv = mKernelPrefill_qkv;
        plan.qkSoftmax = mKernel_qk_softmax;
        plan.sdpa = mKernel_sdpa;
        plan.flashTc = mKernel_faTc;
        plan.flashSg = mKernel_faSg;
        plan.sdpaPartial = mKernelSdpaPartial;
        plan.sdpaReduce = mKernelSdpaReduce;
        plan.veto = mVeto;
        plan.sdpaNsg = mPlan.sdpaNsg;
        plan.sdpaTwoPass = mPlan.sdpaTwoPass;
        mPipelinePlans.emplace(signature, plan);
    }
    if (code == NO_ERROR) {
        mPreparedShape = shape;
    } else {
        mPipelinePlans.clear();
    }
    return code;
}

ErrorCode AttentionBufExecution::updateExecutionState(const std::vector<Tensor*>& inputs) {
    auto code = updateKVCache();
    if (code != NO_ERROR) return code;
    mVeto = AttnVetoNone;
    _resolveRoute(inputs, mKvSeqLen);
    auto iter = mPipelinePlans.find(pipelineSignature());
    if (iter == mPipelinePlans.end()) {
        // A prefix read/write flips storage mode without a shape change, so resize is skipped.
        code = preparePipelines(inputs);
        if (code != NO_ERROR) return code;
        mVeto = AttnVetoNone;
        _resolveRoute(inputs, mKvSeqLen);
        iter = mPipelinePlans.find(pipelineSignature());
    }
    if (iter == mPipelinePlans.end()) {
        MNN_ERROR("Metal Attention pipeline was not prepared by resize; execution stopped.\n");
        return NOT_SUPPORT;
    }
    const auto& plan = iter->second;
    // The route must match the plan's compiled artifacts.
    mVeto = plan.veto;
    if (plan.veto != AttnVetoNone) {
        _resolveRoute(inputs, mKvSeqLen);
    }
    // Compile outcomes, not routing inputs: both stay out of the signature.
    mPlan.sdpaNsg = plan.sdpaNsg;
    mPlan.sdpaTwoPass = plan.sdpaTwoPass;
    mKernel_qk = plan.qk;
    mKernel_qkv = plan.qkv;
    mKernel_copy = plan.copy;
    mKernel_softmax = plan.softmax;
    mKernelPrefill_qk = plan.prefillQk;
    mKernelPrefill_qkv = plan.prefillQkv;
    mKernel_qk_softmax = plan.qkSoftmax;
    mKernel_sdpa = plan.sdpa;
    mKernelSdpaPartial = plan.sdpaPartial;
    mKernelSdpaReduce = plan.sdpaReduce;
    mKernel_faTc = plan.flashTc;
    mKernel_faSg = plan.flashSg;
    return handleKVAllocMemory();
}

// Mirrors the manager's alloc dispatch: fresh allocs may go to disk, realloc keeps the current storage.
// A prefix write/read leaves the cache on disk after file_flag returns to NoChange, so consult the manager.
bool AttentionBufExecution::_predictKvInDisk() const {
    if (!mKVCache) {
        return false;
    }
    if (mKvDirOnDisk) {
        return true;
    }
    if (mIsKVShared) {
        return mKVCacheManager->inDisk();
    }
    if (nullptr == mMeta || mMeta->previous == mMeta->remove) {
        return mMeta != nullptr && mMeta->file_name.size() > 0 &&
               (mMeta->file_flag == KVMeta::PendingWrite || mMeta->file_flag == KVMeta::PendingRead);
    }
    return mKVCacheManager->inDisk();
}

ErrorCode AttentionBufExecution::updateKVCache() {
    if (!mKVCache) {
        mKvSeqLen = mCurrentKvLen;
        mKvMaxLen = ROUND_UP(mKvSeqLen, mKvAlignNum);
        return NO_ERROR;
    }
    if (!mIsKVShared) {
        if (nullptr == mMeta || mMeta->previous == mMeta->remove) {
            // Settle mode and K/V widths before alloc so cache bytes, pipelines and bindings agree.
            _resolveKvFormat();
            mKVCacheManager->onClear();
            auto code = mKVCacheManager->onAlloc(mMeta, mCurrentKvLen);
            if (NO_ERROR != code) {
                static_cast<MetalBackend*>(backend())->runtime()->markResourceAllocFailed();
                return code;
            }
        } else {
            auto code = mKVCacheManager->onRealloc(mMeta);
            if (NO_ERROR != code) {
                static_cast<MetalBackend*>(backend())->runtime()->markResourceAllocFailed();
                return code;
            }
        }
    }
    mKvSeqLen = _pastKvLength() + mCurrentKvLen;
    if (!mIsKVShared) {
        // Padded block decode writes mCurrentKvLen rows while meta->add commits fewer.
        auto code = mKVCacheManager->ensureCapacity(mKvSeqLen);
        if (NO_ERROR != code) {
            static_cast<MetalBackend*>(backend())->runtime()->markResourceAllocFailed();
            return code;
        }
    }
    mKvMaxLen = mKVCacheManager->maxLength();
    // Manager state is ground truth after alloc, incl. prefix-only disk caches.
    mKvInDisk = mKVCacheManager->inDisk();
    if (mIsKVShared) {
        // The owner picked the layout; a shared layer only reads it.
        mQuantKey = mKVCacheManager->quantKey();
        mQuantValue = mKVCacheManager->quantValue();
    }
    if (mKvMaxLen < mKvSeqLen ||
        (mKvInDisk ? (nil == mKVCacheManager->getKeyBuffer() || nil == mKVCacheManager->getValueBuffer())
                   : (nullptr == mKVCacheManager->getKeyTensor() || nullptr == mKVCacheManager->getValueTensor()))) {
        MNN_ERROR("Metal Attention KV cache preparation failed.\n");
        return OUT_OF_MEMORY;
    }
    return NO_ERROR;
}

ErrorCode AttentionBufExecution::handleKVAllocMemory() {
    constexpr auto allocType = Backend::DYNAMIC_IN_EXECUTION;
    // The scratch only holds the current KV window, not the whole cache capacity.
    const int scratchKvLen = ROUND_UP(mKvSeqLen, mKvAlignNum);
    if (!mKVCache) {
        mKvSeqLen = mCurrentKvLen;
        mKvMaxLen = ROUND_UP(mKvSeqLen, mKvAlignNum);

        int keySize = mKvMaxLen * mBatch * mKvNumHead * mHeadDim;
        int valueSize = mBatch * mKvNumHead * mHeadDim * mKvMaxLen;
        // Only safe because this !mKVCache block runs while onReplayUpdate
        // already bails on !mKVCache (see MetalReplay.hpp). Re-creating mTempK/V
        // on an mKVCache path would dangle recorded Tensor* bindings.
        if (nullptr == mTempK || mTempK->elementSize() != keySize) {
            mTempK.reset(Tensor::createDevice<float>({keySize}));
        }
        if (nullptr == mTempV || mTempV->elementSize() != valueSize) {
            mTempV.reset(Tensor::createDevice<float>({valueSize}));
        }

        int qSeqLenPiece = UP_DIV(mSeqLen, mPlan.seqLenSplitCount);
        bool needMalloc = mTempQK->length(0) != mBatch * mNumHead;
        if (mTempQK->length(1) != qSeqLenPiece * scratchKvLen) {
            needMalloc = true;
        }
        if (needMalloc) {
            mTempQK->setLength(0, mBatch * mNumHead);
            mTempQK->setLength(1, qSeqLenPiece * scratchKvLen);
            mTempSoftMax->setLength(0, mBatch * mNumHead);
            mTempSoftMax->setLength(1, qSeqLenPiece * scratchKvLen);
        }

        const bool gotK = backend()->onAcquireBuffer(mTempK.get(), allocType);
        const bool gotV = gotK && backend()->onAcquireBuffer(mTempV.get(), allocType);
        const bool gotQK = gotV && backend()->onAcquireBuffer(mTempQK.get(), allocType);
        const bool gotSoftMax = gotQK && backend()->onAcquireBuffer(mTempSoftMax.get(), allocType);
        // Return exactly what this attempt took; a buffer that was never
        // acquired must not be handed back to the pool.
        if (gotSoftMax) {
            backend()->onReleaseBuffer(mTempSoftMax.get(), allocType);
        }
        if (gotQK) {
            backend()->onReleaseBuffer(mTempQK.get(), allocType);
        }
        if (gotV) {
            backend()->onReleaseBuffer(mTempV.get(), allocType);
        }
        if (gotK) {
            backend()->onReleaseBuffer(mTempK.get(), allocType);
        }
        if (!gotSoftMax) {
            MNN_ERROR("MNN::Metal: OUT_OF_MEMORY when execute attention metal\n");
            static_cast<MetalBackend*>(backend())->runtime()->markResourceAllocFailed();
            return OUT_OF_MEMORY;
        }
        return NO_ERROR;
    }
    // Fused prefill keeps the accumulator in threadgroup memory, so no O(B*H*seq*kv) scratch.
    if (mPlan.isFusedPrefill()) {
        return NO_ERROR;
    }

    // SDPA writes the output itself; only the two-pass variant needs its partials.
    if (mPlan.route == AttnRoute::DecodeSdpa) {
        if (mPlan.sdpaTwoPass) {
            // Byte-allocated: a float tensor would be halved by the fp16 backend, the shader writes 4B.
            const int bytes = mPlan.sdpaKvBlocks * mBatch * mNumHead * (mHeadDim + 2) * (int)sizeof(float);
            if (nullptr == mSdpaPartials || mSdpaPartials->elementSize() < bytes) {
                mSdpaPartials.reset(Tensor::createDevice<uint8_t>({bytes}));
            }
            if (!backend()->onAcquireBuffer(mSdpaPartials.get(), allocType)) {
                MNN_ERROR("MNN::Metal: OUT_OF_MEMORY for split-KV partials\n");
                static_cast<MetalBackend*>(backend())->runtime()->markResourceAllocFailed();
                return OUT_OF_MEMORY;
            }
            backend()->onReleaseBuffer(mSdpaPartials.get(), allocType);
        }
        return NO_ERROR;
    }

    int qSeqLenPiece = UP_DIV(mSeqLen, mPlan.seqLenSplitCount);
    // decode_qk_softmax writes probabilities straight into mTempSoftMax.
    const bool needQkScratch = mPlan.route != AttnRoute::DecodeFusedQkSoftmax;
    bool needMalloc = mTempSoftMax->length(0) != mBatch * mNumHead ||
                      mTempSoftMax->length(1) != qSeqLenPiece * scratchKvLen;
    if (needMalloc) {
        mTempQK->setLength(0, mBatch * mNumHead);
        mTempQK->setLength(1, qSeqLenPiece * scratchKvLen);
        mTempSoftMax->setLength(0, mBatch * mNumHead);
        mTempSoftMax->setLength(1, qSeqLenPiece * scratchKvLen);
    }

    const bool gotQK = !needQkScratch || backend()->onAcquireBuffer(mTempQK.get(), allocType);
    const bool gotSoftMax = gotQK && backend()->onAcquireBuffer(mTempSoftMax.get(), allocType);
    if (gotSoftMax) {
        backend()->onReleaseBuffer(mTempSoftMax.get(), allocType);
    }
    if (needQkScratch && gotQK) {
        backend()->onReleaseBuffer(mTempQK.get(), allocType);
    }
    if (!gotSoftMax) {
        MNN_ERROR("MNN::Metal: OUT_OF_MEMORY when execute attention metal\n");
        static_cast<MetalBackend*>(backend())->runtime()->markResourceAllocFailed();
        return OUT_OF_MEMORY;
    }
    return NO_ERROR;
}

ErrorCode AttentionBufExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mVeto = AttnVetoNone;
    // dims>=2 is a per-element tensor mask; lower ranks are sentinels resolved by _collectFacts.
    // core/AttentionCommon.hpp -- predicate must stay identical to CPUAttention.
    mHasTensorMask = inputs.size() > 3 && inputs[3]->dimensions() >= 2;
    if (mHasTensorMask) {
        mIsAddMask = (inputs[3]->getType() == halide_type_of<float>());
    }
    mHasSinks = inputs.size() > 4;
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
    mScale = (nullptr != mMeta && mMeta->attn_scale > 0.0f) ? mMeta->attn_scale : (1.0f / sqrt(mHeadDim));
    mKvNumHead = key->shape()[2];
    mCurrentKvLen = key->shape()[1];
    mKvSeqLen = mCurrentKvLen;
    // Align to mKvAlignNum, for simd/tensor matrix load
    mKvMaxLen = ROUND_UP(mKvSeqLen, mKvAlignNum);
    _resolveKvFormat();
    return preparePipelines(inputs);
}

// Disk KV has no quantized layout; expansion keeps the committed one, so only resize/fresh alloc resolve.
void AttentionBufExecution::_resolveKvFormat() {
    mKvInDisk = _predictKvInDisk();
    int attentionOption = static_cast<MetalBackend*>(backend())->getRuntime()->hint().attentionOption;
    bool dynamicQuantK = (attentionOption % 8 >= 1);
    bool dynamicQuantV = (attentionOption % 8 > 1);
    mQuantValue = mKVCache && !mKvInDisk &&
                  ((mKVQuantParameter != nullptr && mKVQuantParameter->vScale != 0.0f) || dynamicQuantV);
    mQuantKey = mKVCache && !mKvInDisk &&
                ((mKVQuantParameter != nullptr && mKVQuantParameter->kScale != 0.0f) || dynamicQuantK);
    if (mKVCache && !mIsKVShared) {
        mKVCacheManager->setBatch(mBatch);
        mKVCacheManager->setKVQuantParameter(mKVQuantParameter);
        mKVCacheManager->setAttenQuantKeyValue(mQuantKey, mQuantValue);
        mKVCacheManager->onResize(mKvNumHead, mHeadDim);
    }
}
uint64_t AttentionBufExecution::_pathSignature(bool forReplay) const {
    // Any field that changes kernels or dispatch must appear here, or replay runs an old kernel.
    // Non-replay bits must stay below bit 32 so pipelineSignature() can pack SDPA metadata above.
    uint64_t sig = 0;
    int bit = 0;
    auto put = [&](uint64_t value, int width) {
        sig |= (value & ((1ull << width) - 1)) << bit;
        bit += width;
    };
    put((uint64_t)mPlan.route, 3);
    put(mPlan.qkSimdReduce, 1);
    put(mPlan.qkSimdMatrix, 1);
    put(mPlan.qkTensorMatrix, 1);
    put(mPlan.qkvSimdReduce, 1);
    put(mPlan.qkvSimdMatrix, 1);
    put(mPlan.softmaxSimdReduce, 1);
    put(mPlan.copySimdReduce, 1);
    put(mPlan.causalTri, 1);
    put(mPlan.causalBound, 1);
    put(mHasTensorMask, 1);
    put(mIsAddMask, 1);
    // The bidirectional sentinel drops the DEFAULT_MASK shader key.
    put(mFacts.bidirectionalMask, 1);
    put(mQuantKey, 1);
    put(mQuantValue, 1);
    put(mKVCache, 1);
    put(mKvInDisk, 1);
    put(mOutputC4, 1);
    // SHORT_KV_128 is a separately compiled variant; replay must not outlive it.
    put(mPlan.route == AttnRoute::DecodeFusedQkSoftmax && mFacts.kvLen <= 128, 1);
    put(mPlan.qkQsplit, 1);
    put(mPlan.faTcHeadDimNsg > 1, 1); // FATC_HEADDIM_NSG changes the kernel and threadgroup width.
    int wmLog2 = 0;
    while ((1 << wmLog2) < mPlan.faTcQueryNsg && wmLog2 < 3) {
        wmLog2++;
    }
    put((uint64_t)wmLog2, 2); // FATC_QUERY_NSG (1/2/4) changes BQ and threadgroup width.
    // SDPA_KV_TOKENS_PER_ITER and SDPA_QHEADS_PER_TG each change both the kernel and the dispatch grid.
    put(mPlan.sdpaKvTokensPerIter > 1, 1);
    int qhLog2 = 0;
    while ((1 << qhLog2) < mPlan.sdpaQHeadsPerTg && qhLog2 < 7) {
        qhLog2++;
    }
    put((uint64_t)qhLog2, 3);
    if (forReplay) {
        // Changes event layout and is a compile outcome: replay key only.
        put(mPlan.sdpaTwoPass, 1);
        // log2 in 4 bits: scratch bounding needs split counts past 15.
        int qsplitLog2 = 0;
        while ((1 << qsplitLog2) < mPlan.seqLenSplitCount) {
            qsplitLog2++;
        }
        put((uint64_t)qsplitLog2, 4);
    }
    return sig;
}

// KV-copy uniforms go through setBytes (encoder-owned storage) rather than a
// persistent MTLBuffer: a shared buffer rewritten at encode time can race with
// the previous forward's copy. QK/QKV/softmax uniforms share the same hazard.
AttentionCopyParam AttentionBufExecution::_makeCopyParam(const Tensor* key, const Tensor* value) const {
    AttentionCopyParam p;
    /*
     Key -> K-Cache :   [mBatch, mKvSeqLen, mKvNumHead, mHeadDim] -> [mKvMaxLen, mBatch, mKvNumHead, mHeadDim]
     Value -> V-Cache : [mBatch, mKvSeqLen, mKvNumHead, mHeadDim] -> [mKvMaxLen, mBatch, mKvNumHead, mHeadDim]
     */
    p.head_count = mKvNumHead * mHeadDim;
    p.kv_seq_len = key->shape()[1];
    p.max_kv_len = mKvMaxLen;
    int pastLength = mKVCache ? _pastKvLength() : 0;
    p.past_kv_len = pastLength;
    p.reserved = 0;
    p.batch = mBatch;
    p.value_c4 =
        TensorUtils::getDescribe(value)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 ? 1 : 0;
    if (mQuantValue && mKVQuantParameter != nullptr) {
        p.v_scale = mKVQuantParameter->vScale;
    } else {
        p.v_scale = 0.0f;
    }
    if (mQuantKey && mKVQuantParameter != nullptr) {
        p.k_scale = mKVQuantParameter->kScale;
    } else {
        p.k_scale = 0.0f;
    }
    return p;
}

AttentionQKVParam AttentionBufExecution::_makeQKVParam(const std::vector<Tensor*>& inputs, int seqLenPiece) const {
    int group_size = mNumHead / mKvNumHead;
    AttentionQKVParam p;
    p.scale = mScale;
    p.head_dim = mHeadDim;
    p.key_seq_len = mKvSeqLen;
    p.head_num = mNumHead;
    p.group = group_size;
    p.query_seq_len = mSeqLen;
    p.q_seq_piece_len = seqLenPiece;
    p.max_kv_len = mKvMaxLen;
    p.batch = mBatch;
    p.kv_align_len = mKvAlignNum;
    p.mask_batch = mHasTensorMask ? inputs[3]->length(0) : 1;
    p.mask_head_num = (mHasTensorMask && inputs[3]->dimensions() > 3) ? inputs[3]->length(1) : 1;
    // q/k are the mask's two trailing dims at any rank: [b,h,q,k] / [b,q,k] / [q,k].
    p.mask_q_len = (mHasTensorMask && inputs[3]->dimensions() > 2) ? inputs[3]->length(inputs[3]->dimensions() - 2) : 1;
    p.mask_k_len = (mHasTensorMask && inputs[3]->dimensions() > 0) ? inputs[3]->length(inputs[3]->dimensions() - 1) : 1;
    if (mQuantValue && mKVQuantParameter != nullptr) {
        p.v_scale = mKVQuantParameter->vScale;
    } else {
        p.v_scale = 0.0f;
    }
    if (mQuantKey && mKVQuantParameter != nullptr) {
        p.k_scale = mKVQuantParameter->kScale;
    } else {
        p.k_scale = 0.0f;
    }
    return p;
}
AttentionSoftmaxParam AttentionBufExecution::_makeSoftmaxParam(int seqLenPiece) const {
    // [mBatch, mNumHead, mSeqLen, mKvSeqLen]
    int inside = 1;
    int outside = mBatch * mNumHead * seqLenPiece;
    int axis = mKvSeqLen;
    int axis_align = ROUND_UP(axis, mKvAlignNum);
    AttentionSoftmaxParam sp;
    // Inside, axis, outside, plane(invalid)
    sp.v[0] = inside;
    sp.v[1] = axis;
    sp.v[2] = outside;
    sp.v[3] = axis_align;
    // CAUSAL_BOUND fields (ignored by non-causal softmax variants). causal_base is
    // written piece-independent and the piece offset is bound per dispatch as seq_idx.
    sp.v[4] = seqLenPiece;
    sp.v[5] = (mKvSeqLen - mSeqLen) + 1;
    sp.v[6] = mNumHead;
    return sp;
}
ErrorCode AttentionBufExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    @autoreleasepool {
        auto code = updateExecutionState(inputs);
        if (code != NO_ERROR) return code;
        return MetalExecution::onExecute(inputs, outputs);
    }
}

void AttentionBufExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                     id<MTLComputeCommandEncoder> encoder) {
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto context = (__bridge MNNMetalContext*)mtbn->context();

    int group_size = mNumHead / mKvNumHead;

    Tensor* tempTensorK;
    Tensor* tempTensorV;
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

#if MNN_METAL_OP_PROFILE
    static_cast<MetalBackend*>(backend())->setProfileSubtag("copy");
#endif
    if (!mIsKVShared) {
        const AttentionCopyParam copyParam = _makeCopyParam(key, value);
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
        [encoder setBytes:&copyParam length:sizeof(AttentionCopyParam) atIndex:4];
        if (mKVCache && mKVCacheManager->getKScaleBuffer() != nil) {
            [encoder setBuffer:mKVCacheManager->getKScaleBuffer() offset:0 atIndex:8];
            [encoder setBuffer:mKVCacheManager->getVScaleBuffer() offset:0 atIndex:9];
        }

        std::pair<MTLSize, MTLSize> gl;
        if (mKVCache && mKVCacheManager->getKScaleBuffer() != nil) {
            int localSize = mPlan.copySimdReduce ? 32 : 128;
            gl = std::make_pair(MTLSizeMake(1, copy_line, mBatch), MTLSizeMake(localSize, 1, 1));
        } else if (mPlan.route == AttnRoute::DecodeFusedQkSoftmax) {
            gl = std::make_pair(MTLSizeMake(UP_DIV(mKvNumHead * mHeadDim, 128), copy_line, mBatch), MTLSizeMake(128, 1, 1));
        } else {
            gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(mKvNumHead * mHeadDim, copy_line, mBatch)];
        }

        [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
    }
#if MNN_METAL_OP_PROFILE
    {
        auto* mtbn = static_cast<MetalBackend*>(backend());
        const char* subtag = "qk";
        switch (mPlan.route) {
            case AttnRoute::DecodeSdpa:           subtag = "sdpa_fused"; break;
            case AttnRoute::DecodeFusedQkSoftmax: subtag = "qk_softmax_fused"; break;
            case AttnRoute::DecodeStaged:         subtag = "qk_short"; break;
            default: break;
        }
        encoder = mtbn->profileNextSubpass(subtag);
    }
#endif

    int seqLenPiece = UP_DIV(mSeqLen, mPlan.seqLenSplitCount);
    const AttentionQKVParam qkvParam = _makeQKVParam(inputs, seqLenPiece);

    for (int seq_idx = 0; seq_idx < mPlan.seqLenSplitCount; seq_idx++) {
        if (mPlan.route == AttnRoute::PrefillFaTc) {
            // Fused prefill on the tensor API: one dispatch, S and O in registers.
            [encoder setComputePipelineState:mKernel_faTc];
            MetalBackend::setTensor(query, encoder, 0);
            MetalBackend::setTensor(outputs[0], encoder, 1);
            MetalBackend::setTensor(tempTensorK, encoder, 2);
            MetalBackend::setTensor(tempTensorV, encoder, 3);
            [encoder setBytes:&qkvParam length:sizeof(AttentionQKVParam) atIndex:4];
            [encoder setBytes:&seq_idx length:sizeof(seq_idx) atIndex:5];
            if (mHasTensorMask) {
                MetalBackend::setTensor(inputs[3], encoder, 8);
            }
            const int faTcBQ = mPlan.faTcQueryNsg * 16;
            [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(seqLenPiece, faTcBQ), mBatch * mNumHead, 1)
                    threadsPerThreadgroup:MTLSizeMake(32 * mPlan.faTcQueryNsg * mPlan.faTcHeadDimNsg, 1, 1)];
#if MNN_METAL_OP_PROFILE
            {
                auto* mtbn2 = static_cast<MetalBackend*>(backend());
                encoder = mtbn2->profileNextSubpass("fa_tc");
            }
#endif
            continue;
        }
        if (mPlan.route == AttnRoute::PrefillFaSg) {
            [encoder setComputePipelineState:mKernel_faSg];
            MetalBackend::setTensor(query, encoder, 0);
            MetalBackend::setTensor(outputs[0], encoder, 1);
            MetalBackend::setTensor(tempTensorK, encoder, 2);
            MetalBackend::setTensor(tempTensorV, encoder, 3);
            [encoder setBytes:&qkvParam length:sizeof(AttentionQKVParam) atIndex:4];
            [encoder setBytes:&seq_idx length:sizeof(seq_idx) atIndex:5];
            int fa_kv_start = 0;
            int fa_kv_len   = mKvSeqLen;
            [encoder setBytes:&fa_kv_start length:sizeof(int) atIndex:6];
            [encoder setBytes:&fa_kv_len   length:sizeof(int) atIndex:7];
            if (mHasTensorMask) {
                MetalBackend::setTensor(inputs[3], encoder, 8);
            }
            [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(seqLenPiece, mPlan.faSgSeqLenPerTg), mBatch * mNumHead, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, mPlan.faSgNsg, 1)];
#if MNN_METAL_OP_PROFILE
            {
                auto* mtbn2 = static_cast<MetalBackend*>(backend());
                encoder = mtbn2->profileNextSubpass("flash_attn_sg");
            }
#endif
            continue;
        }
        if (mPlan.route == AttnRoute::DecodeSdpa && mPlan.sdpaTwoPass) {
            [encoder setComputePipelineState:mKernelSdpaPartial];
            MetalBackend::setTensor(query, encoder, 0);
            MetalBackend::setTensor(outputs[0], encoder, 1);
            MetalBackend::setTensor(tempTensorK, encoder, 2);
            MetalBackend::setTensor(tempTensorV, encoder, 3);
            [encoder setBytes:&qkvParam length:sizeof(AttentionQKVParam) atIndex:4];
            MetalBackend::setTensor(mSdpaPartials.get(), encoder, 10);
            if (mQuantKey && mKVCacheManager->getKScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getKScaleBuffer() offset:0 atIndex:8];
            }
            if (mQuantValue && mKVCacheManager->getVScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getVScaleBuffer() offset:0 atIndex:9];
            }
            const int gridGroups = mBatch * (mNumHead / mPlan.sdpaQHeadsPerTg);
            [encoder dispatchThreadgroups:MTLSizeMake(mPlan.sdpaKvBlocks, gridGroups, 1)
                    threadsPerThreadgroup:MTLSizeMake(32 * mPlan.sdpaNsg, 1, 1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [encoder setComputePipelineState:mKernelSdpaReduce];
            MetalBackend::setTensor(outputs[0], encoder, 1);
            MetalBackend::setTensor(mSdpaPartials.get(), encoder, 10);
            [encoder setBytes:&qkvParam length:sizeof(AttentionQKVParam) atIndex:4];
            [encoder setBytes:&mPlan.sdpaKvBlocks length:sizeof(mPlan.sdpaKvBlocks) atIndex:11];
            [encoder dispatchThreadgroups:MTLSizeMake(mBatch * mNumHead, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            continue;
        }
        if (mPlan.route == AttnRoute::DecodeSdpa) {
            // Fused decode SDPA: one threadgroup per q-head group sweeps the
            // whole kv range and writes the final output itself -- one kernel,
            // no reduce dispatch, no partials.
            [encoder setComputePipelineState:mKernel_sdpa];
            MetalBackend::setTensor(query, encoder, 0);
            MetalBackend::setTensor(outputs[0], encoder, 1);
            MetalBackend::setTensor(tempTensorK, encoder, 2);
            MetalBackend::setTensor(tempTensorV, encoder, 3);
            [encoder setBytes:&qkvParam length:sizeof(AttentionQKVParam) atIndex:4];
            if (mQuantKey && mKVCacheManager->getKScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getKScaleBuffer() offset:0 atIndex:8];
            }
            if (mQuantValue && mKVCacheManager->getVScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getVScaleBuffer() offset:0 atIndex:9];
            }
            const int gridGroups = mBatch * (mNumHead / mPlan.sdpaQHeadsPerTg);
            [encoder dispatchThreadgroups:MTLSizeMake(1, gridGroups, 1)
                    threadsPerThreadgroup:MTLSizeMake(32 * mPlan.sdpaNsg, 1, 1)];
            // No trailing profileNextSubpass: "sdpa_fused" subtag was set after
            // the copy flush; a trailing call would add an empty encoder.
            continue;
        }
        if (mPlan.route == AttnRoute::DecodeFusedQkSoftmax) {
            [encoder setComputePipelineState:mKernel_qk_softmax];
            MetalBackend::setTensor(query, encoder, 0);
            MetalBackend::setTensor(mTempSoftMax.get(), encoder, 1);
            MetalBackend::setTensor(tempTensorK, encoder, 2);
            [encoder setBytes:&seq_idx length:sizeof(seq_idx) atIndex:3];
            [encoder setBytes:&qkvParam length:sizeof(AttentionQKVParam) atIndex:4];
            if (mKVCache && mQuantKey && mKVCacheManager->getKScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getKScaleBuffer() offset:0 atIndex:8];
            }
            int gridZ = mPlan.qkQsplit ? group_size : 1;
            int qkGroups = mBatch * (mNumHead / group_size) * seqLenPiece;
            int maxLocalSize = ALIMAX(32, ((int)mKernel_qk_softmax.maxTotalThreadsPerThreadgroup / 32) * 32);
            int localSize = qkGroups <= 8 ? ALIMIN(maxLocalSize, ALIMAX(128, ROUND_UP(mKvSeqLen, 32))) :
                            ALIMIN(maxLocalSize, ALIMAX(64, ROUND_UP(UP_DIV(mKvSeqLen, 6), 32)));
            if (mPlan.qkQsplit) {
                localSize = ALIMIN(maxLocalSize, ALIMAX(128, ROUND_UP(UP_DIV(mKvSeqLen, 2), 32)));
            }
            auto gl = std::make_pair(MTLSizeMake(mBatch * (mNumHead / group_size), seqLenPiece, gridZ), MTLSizeMake(localSize, 1, 1));
            [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
        } else {
            id<MTLComputePipelineState> pipeline;
            if (mPlan.isDecode()) {
                pipeline = mKernel_qk;
            } else {
                pipeline = mKernelPrefill_qk;
            }
            // Split by tile size so the trapezoid tile-count formula matches
            // the CAUSAL_TRI remap in the shader (16-tile for prefill_qk,
            // 32-tile for prefill_qk_tensor).
            const bool useSimdCausalTri   = mPlan.causalTri && mPlan.qkSimdMatrix && !mPlan.qkTensorMatrix;
            const bool useTensorCausalTri = mPlan.causalTri && mPlan.qkTensorMatrix;
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
            [encoder setBytes:&qkvParam length:sizeof(AttentionQKVParam) atIndex:4];
            if (mKVCache && mKVCacheManager->getKScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getKScaleBuffer() offset:0 atIndex:8];
                [encoder setBuffer:mKVCacheManager->getVScaleBuffer() offset:0 atIndex:9];
            }
            int kv_start = 0, current_block_len = mKvSeqLen;
            [encoder setBytes:&kv_start length:sizeof(kv_start) atIndex:5];
            [encoder setBytes:&current_block_len length:sizeof(int) atIndex:6];
            if (mHasTensorMask) {
                MetalBackend::setTensor(inputs[3], encoder, 7);
            }

            int decode_grid_y = mBatch * mNumHead;
            std::pair<MTLSize, MTLSize> gl;
            if (mPlan.isDecode()) {
                gl = [context computeBestGroupAndLocal:pipeline
                                               threads:MTLSizeMake(seqLenPiece, decode_grid_y / group_size, mKvSeqLen)];
            } else if (mPlan.qkTensorMatrix) {
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
            } else if (mPlan.qkSimdMatrix) {
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
            int inside = 1;
            int outside = mBatch * mNumHead * seqLenPiece;
            const AttentionSoftmaxParam softmaxParam = _makeSoftmaxParam(seqLenPiece);
            [encoder setComputePipelineState:mKernel_softmax];
            // [mBatch, mNumHead, mSeqLen, mKvSeqLen]
            MetalBackend::setTensor(mTempQK.get(), encoder, 0);
            // [mBatch, mNumHead, mSeqLen, ROUND_UP(mKvSeqLen, mKvAlignNum)]
            MetalBackend::setTensor(mTempSoftMax.get(), encoder, 1);
            [encoder setBytes:&softmaxParam length:sizeof(AttentionSoftmaxParam) atIndex:2];
            if (mHasSinks) {
                MetalBackend::setTensor(inputs[4], encoder, 4);
            }
            if (mPlan.causalBound) {
                // Both setBytes: each dispatch keeps its own seq_idx regardless of
                // what later pieces of the same command buffer encode.
                [encoder setBytes:&seq_idx length:sizeof(seq_idx) atIndex:3];
            }

            int thread_group_size = 32;
            std::pair<MTLSize, MTLSize> softmaxGl;
            if (mPlan.softmaxSimdReduce) {
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
        {
            id<MTLComputePipelineState> pipeline;
            if (mPlan.isDecode()) {
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
            [encoder setBytes:&qkvParam length:sizeof(AttentionQKVParam) atIndex:4];
            if (mKVCache && mKVCacheManager->getKScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getKScaleBuffer() offset:0 atIndex:8];
                [encoder setBuffer:mKVCacheManager->getVScaleBuffer() offset:0 atIndex:9];
            }
            std::pair<MTLSize, MTLSize> gl;
            if (mPlan.qkvSimdReduce) {
                int grid_z = mOutputC4 ? UP_DIV(mHeadDim, 2) : mHeadDim;
                gl = std::make_pair(MTLSizeMake(seqLenPiece, mBatch * mNumHead, grid_z), MTLSizeMake(32, 1, 1));
            } else if (mPlan.qkTensorMatrix) {
                gl = std::make_pair(MTLSizeMake(UP_DIV(seqLenPiece, 32), UP_DIV(mHeadDim, 32), mBatch * mNumHead),
                                    MTLSizeMake(128, 1, 1));
            } else if (mPlan.qkvSimdMatrix) {
                gl = std::make_pair(MTLSizeMake(UP_DIV(seqLenPiece, 16), UP_DIV(mHeadDim, 16), mBatch * mNumHead),
                                    MTLSizeMake(32, 1, 1));
            } else {
                gl = [context computeBestGroupAndLocal:pipeline
                                               threads:MTLSizeMake(seqLenPiece, mBatch * mNumHead, mHeadDim)];
            }
            [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
        }
    }

    if (mKVCache && !mIsKVShared) {
        mKVCacheManager->setPastLength(mKVCacheManager->kvLength() + mCurrentKvLen);
    }
    // Replay bookkeeping: capture the structural fingerprint + raw scale-buffer
    // identities so onReplayUpdate can detect path switches / KV reallocations.
    mLastEncodeSig = _pathSignature(true);
    mLastKScaleBuffer = (mKVCache && mKVCacheManager) ? mKVCacheManager->getKScaleBuffer() : nil;
    mLastVScaleBuffer = (mKVCache && mKVCacheManager) ? mKVCacheManager->getVScaleBuffer() : nil;
    mLastSdpaPartials = mPlan.sdpaTwoPass ? mSdpaPartials.get() : nullptr;
    if (mKVCache && !mKvInDisk && mKVCacheManager) {
        mLastKTensor = mKVCacheManager->getKeyTensor();
        mLastVTensor = mKVCacheManager->getValueTensor();
    } else {
        mLastKTensor = nullptr;
        mLastVTensor = nullptr;
    }
    return;
}

bool AttentionBufExecution::onReplayUpdate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Replay only for in-memory KV-cache decode with a single seq piece; anything
    // else (prefill, disk KV, split, shared layers) bails and re-encodes.
    // Checks here use last-encode state; the signature below is authoritative.
    if (!mKVCache || mIsKVShared || mKvInDisk || mPlan.isFusedPrefill() || mPlan.seqLenSplitCount != 1) {
        return false;
    }
    // Reserve compaction is not idempotent across replay fallback.
    if (mMeta && mMeta->n_reserve != 0) {
        return false;
    }
    // onExecute already advanced KV, even if validation below falls back to a normal encode.
    if (!mPlan.isDecode() || _pathSignature(true) != mLastEncodeSig) {
        return false;
    }
    // Raw scale-buffer bindings are not tensor-validated by the replay layer;
    // a KV-cache realloc swaps them, so compare identities explicitly.
    auto kScale = mKVCacheManager ? mKVCacheManager->getKScaleBuffer() : nil;
    auto vScale = mKVCacheManager ? mKVCacheManager->getVScaleBuffer() : nil;
    if (kScale != mLastKScaleBuffer || vScale != mLastVScaleBuffer) {
        return false;
    }
    // KV expansion resets mPastKey/V and dangles recorded Tensor* bindings.
    // Detect by tensor identity (buffer identity isn't reliable); pointers
    // are compared, never dereferenced.
    if (mKVCacheManager->getKeyTensor() != mLastKTensor ||
        mKVCacheManager->getValueTensor() != mLastVTensor) {
        return false;
    }
    // Check identity before validation dereferences the possibly destroyed scratch Tensor.
    const bool sdpaTwoPass = mPlan.route == AttnRoute::DecodeSdpa && mPlan.sdpaTwoPass;
    if (sdpaTwoPass && mSdpaPartials.get() != mLastSdpaPartials) {
        return false;
    }
    // Structural layout must match the recording before patching by position.
    const int expectEvents = sdpaTwoPass ? 4 :
                             (mPlan.route == AttnRoute::DecodeSdpa ? 2 :
                              (mPlan.route == AttnRoute::DecodeFusedQkSoftmax ? 3 : 4));
    if ((int)mReplayEvents.size() != expectEvents) {
        return false;
    }
    const auto selectedPipeline = sdpaTwoPass ? mKernelSdpaPartial :
                                  (mPlan.route == AttnRoute::DecodeSdpa ? mKernel_sdpa :
                                   (mPlan.route == AttnRoute::DecodeFusedQkSoftmax ? mKernel_qk_softmax : mKernel_qk));
    if (mReplayEvents[0].pipeline != mKernel_copy || mReplayEvents[1].pipeline != selectedPipeline) {
        return false;
    }
    if (sdpaTwoPass && (mReplayEvents[2].barrierScope != MTLBarrierScopeBuffers ||
                        mReplayEvents[3].pipeline != mKernelSdpaReduce)) {
        return false;
    }
    for (int i = 0; i < expectEvents; ++i) {
        const auto expectedType = sdpaTwoPass && i == 2 ? MetalReplayEvent::BarrierScope : MetalReplayEvent::Dispatch;
        if (mReplayEvents[i].type != expectedType) {
            return false;
        }
    }
    // Validate tensor bindings up front: kvLength is advanced below, and a
    // late metalReplayEmit failure would fall back to onEncode and advance it
    // a second time.
    if (!metalReplayValidate(mReplayEvents)) {
        return false;
    }

    auto mtbn = static_cast<MetalBackend*>(backend());
    auto context = (__bridge MNNMetalContext*)mtbn->context();
    int group_size = mNumHead / mKvNumHead;
    int seqLenPiece = UP_DIV(mSeqLen, mPlan.seqLenSplitCount);

    // Patch per-token state inside recorded events. The three uniform structs
    // travel as setBytes args (see _makeCopyParam) and are keyed by (index, size);
    // sizes are disjoint so matches are unambiguous.
    {
        const AttentionCopyParam copyParam = _makeCopyParam(inputs[1], inputs[2]);
        const AttentionQKVParam qkvParam = _makeQKVParam(inputs, seqLenPiece);
        const AttentionSoftmaxParam softmaxParam = _makeSoftmaxParam(seqLenPiece);
        for (auto& e : mReplayEvents) {
            for (auto& by : e.bytesArgs) {
                if (by.first == 4 && by.second.size() == sizeof(AttentionCopyParam)) {
                    memcpy(by.second.data(), &copyParam, sizeof(AttentionCopyParam));
                } else if (by.first == 4 && by.second.size() == sizeof(AttentionQKVParam)) {
                    memcpy(by.second.data(), &qkvParam, sizeof(AttentionQKVParam));
                } else if (by.first == 2 && by.second.size() == sizeof(AttentionSoftmaxParam)) {
                    memcpy(by.second.data(), &softmaxParam, sizeof(AttentionSoftmaxParam));
                }
            }
        }
    }

    auto patchIntBytes = [](MetalReplayEvent& e, int index, int value) {
        for (auto& by : e.bytesArgs) {
            if (by.first == index && by.second.size() == sizeof(int)) {
                memcpy(by.second.data(), &value, sizeof(int));
            }
        }
    };
    // events[0] is the KV copy: its grid is kv-length independent.
    if (sdpaTwoPass) {
        // events[1] = partial, events[2] = buffer barrier, events[3] = reduce.
        auto& partial = mReplayEvents[1];
        partial.grid = MTLSizeMake(mPlan.sdpaKvBlocks, mBatch * (mNumHead / mPlan.sdpaQHeadsPerTg), 1);
        patchIntBytes(mReplayEvents[3], 11, mPlan.sdpaKvBlocks);
    } else if (mPlan.route == AttnRoute::DecodeSdpa) {
        // events[1] = fused sdpa: grid (1, B*head groups) is kv-length
        // independent; kv reaches the kernel via the param rewrite.
    } else if (mPlan.route == AttnRoute::DecodeFusedQkSoftmax) {
        // events[1] = fused qk_softmax (tg width tracks kv); events[2] = qkv
        // (grid is kv-independent).
        auto& fused = mReplayEvents[1];
        int qkGroups = mBatch * (mNumHead / group_size) * seqLenPiece;
        int maxLocalSize = ALIMAX(32, ((int)fused.pipeline.maxTotalThreadsPerThreadgroup / 32) * 32);
        int localSize = qkGroups <= 8 ? ALIMIN(maxLocalSize, ALIMAX(128, ROUND_UP(mKvSeqLen, 32))) :
                        ALIMIN(maxLocalSize, ALIMAX(64, ROUND_UP(UP_DIV(mKvSeqLen, 6), 32)));
        if (mPlan.qkQsplit) {
            localSize = ALIMIN(maxLocalSize, ALIMAX(128, ROUND_UP(UP_DIV(mKvSeqLen, 2), 32)));
        }
        fused.threads = MTLSizeMake(localSize, 1, 1);
    } else {
        // events[1] = qk (short-seq): grid depth + block length track kv.
        // events[2] = softmax, events[3] = qkv: grids kv-length independent
        // (kv length reaches them through the rewritten param buffers).
        auto& qk = mReplayEvents[1];
        int decode_grid_y = mBatch * mNumHead;
        auto gl = [context computeBestGroupAndLocal:qk.pipeline
                                            threads:MTLSizeMake(seqLenPiece, decode_grid_y / group_size, mKvSeqLen)];
        qk.grid = gl.first;
        qk.threads = gl.second;
        patchIntBytes(qk, 6, mKvSeqLen);
    }

    mKVCacheManager->setPastLength(mKVCacheManager->kvLength() + mCurrentKvLen);
    return true;
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
        return new AttentionBufExecution(backend, param->kv_cache(), param->output_c4(), quantParam);
    }
};
REGISTER_METAL_OP_TRANSFORMER_CREATOR(AttentionBufCreator, OpType_Attention);

} // namespace MNN
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif
