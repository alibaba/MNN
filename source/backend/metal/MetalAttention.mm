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
#include "core/AttentionMaskConvention.hpp"
#include "core/TensorUtils.hpp"

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
namespace MNN {

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
    if (mHasTensorMask) {
        // Decode reads the mask too: a sliding window hides old keys even for a single query.
        if (mIsAddMask) {
            qkPrefillKeys.emplace_back("ADD_MASK");
            qkKeys.emplace_back("ADD_MASK");
        } else {
            qkPrefillKeys.emplace_back("SET_MASK");
            qkKeys.emplace_back("SET_MASK");
        }
    } else if (mKVCache && !mBidirectionalMask) {
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
        // mQkQsplit decided in _computePathFlags (kv/device gated, per token).
        if (mQkQsplit) {
            keys.emplace_back("QK_QSPLIT");
        }
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
    if (mSdpaSinglePass) {
        std::string head_dim_str = std::to_string(mHeadDim);
        auto buildSdpa = [&](int nsg, int kvb) -> id<MTLComputePipelineState> {
            std::vector<std::string> keys = {"decode_splitkv_sdpa", ftype, group_str,
                                             "HEAD_DIM_" + head_dim_str, "NSG_" + std::to_string(nsg)};
            if (mSdpaQhPerTg > 1) {
                keys.emplace_back("QHTG_" + std::to_string(mSdpaQhPerTg));
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
            if (nil == pipeline) {
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
                if (mSdpaQhPerTg > 1) {
                    [dic setValue:@(std::to_string(mSdpaQhPerTg).c_str()) forKey:@"SDPA_QH_PER_TG"];
                    flagStart = 6;
                }
                if (kvb > 1) {
                    [dic setValue:@(std::to_string(kvb).c_str()) forKey:@"SDPA_KVB"];
                    flagStart += 1;
                }
                for (size_t j = flagStart; j < keys.size(); ++j) {
                    [dic setValue:@"1" forKey:@(keys[j].c_str())];
                }
                option.preprocessorMacros = dic;
                pipeline = mtbn->makeComputePipelineWithSourceOption(gDecodeSplitKV, "decode_splitkv", option);
                if (nil != pipeline) {
                    rt->insertPipeline(keys, pipeline);
                }
            }
            return pipeline;
        };
        // Degrade NSG until the compiled pipeline can actually host 32*NSG threads.
        int nsg = mSdpaNsg;
        id<MTLComputePipelineState> pipeline = nil;
        while (nsg >= 4) {
            pipeline = buildSdpa(nsg, mSdpaKvb);
            if (nil != pipeline && (int)pipeline.maxTotalThreadsPerThreadgroup >= 32 * nsg) {
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
            mSdpaSinglePass = false;
            // the single-pass early return skipped mTempQK/mTempSoftMax; re-run
            handleKVAllocMemory();
        } else {
            mSdpaNsg = nsg;
            mKernel_sdpa = pipeline;
            if (mSdpaKvbPrebuild) {
                // Pre-build the other auto-mode KVB variant into the pipeline
                // cache: the auto gate flips KVB 1<->2 as kv crosses its
                // threshold, and without this the flip compiles synchronously
                // inside a mid-decode encode. Only layer 0 pays; the rest hit
                // the cache.
                buildSdpa(nsg, mSdpaKvb == 2 ? 1 : 2);
            }
        }
    }
    if (mFlashAttnPrefill) {
        std::string head_dim_str = std::to_string(mHeadDim);
        std::vector<std::string> keys = {"prefill_flash_attn", ftype, group_str, "HEAD_DIM_" + head_dim_str};
        if (mHasTensorMask) {
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
            if (mHasTensorMask) {
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
    if (mFaTcPrefill) {
        std::string head_dim_str = std::to_string(mHeadDim);
        std::vector<std::string> keys = {"prefill_flash_attn_tc", ftype, "HEAD_DIM_" + head_dim_str};
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
        // Head_dim d-split: a slice must stay a whole multiple of the 32-wide
        // O tile. At head_dim 256 the persistent O accumulator alone occupies
        // 128 float registers per lane, so split it 2-ways.
        int faTcDSplit = (mHeadDim == 256) ? 2 : 1;
        if ((mHeadDim % (32 * faTcDSplit)) != 0) {
            faTcDSplit = 1;
        }
        mFaTcDSplit = faTcDSplit;
        std::string dsplit_str = std::to_string(faTcDSplit);
        if (faTcDSplit > 1) {
            keys.emplace_back("FATC_DSPLIT_" + dsplit_str);
        }
        auto pipeline = rt->findPipeline(keys);
        if (nil == pipeline) {
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
            [dic setValue:@"1" forKey:@"FATC_Q_REG"];
            [dic setValue:@"1" forKey:@"FATC_QK_K32"];
            [dic setValue:@"1" forKey:@"FATC_O_CT"];
            [dic setValue:@"1" forKey:@"FATC_KV_DEV_TENSOR"];
            [dic setValue:@"1" forKey:@"FATC_QREV"];
            [dic setValue:@"1" forKey:@"FATC_Q_CT"];
            [dic setValue:@"1" forKey:@"FATC_PV_K32"];
            if (faTcDSplit > 1) {
                [dic setValue:@(dsplit_str.c_str()) forKey:@"FATC_DSPLIT"];
            }
            option.preprocessorMacros = dic;
            pipeline = mtbn->makeComputePipelineWithSourceOption(gPrefillFlashAttnTc, "prefill_flash_attn_tc", option);
            if (nil != pipeline) {
                rt->insertPipeline(keys, pipeline);
            }
        }
        if (nil == pipeline || (int)pipeline.maxTotalThreadsPerThreadgroup < 128) {
            MNN_ERROR("MNN::Metal FA-TC: pipeline unavailable (cap %d), falling back to three-stage prefill\n",
                      pipeline ? (int)pipeline.maxTotalThreadsPerThreadgroup : -1);
            // Sticky, then recompute: the FA-TC early return skipped the scratch
            // allocation and suppressed causal-tri / causal-bound.
            mFaTcUnavailable = true;
            mFaTcPrefill = false;
            _computePathFlags(inputs);
            compilerShader(inputs);
            return;
        } else {
            mKernel_faTc = pipeline;
            static bool _fatc_log_once = false;
            if (!_fatc_log_once) {
                _fatc_log_once = true;
                MNN_PRINT("[MetalAttention] prefill-flash-attn-tc kernel active (seq=%d, head_dim=%d, qk_k=%d, pv_k=%d, dsplit=%d).\n",
                          mSeqLen, mHeadDim, 32, 32,
                          faTcDSplit);
            }
        }
    }
    if (mFaSgPrefill) {
        std::string head_dim_str = std::to_string(mHeadDim);
        // The batched V loop is unrolled over head_dim fragments, so a batch width
        // that does not divide their count would index past mO.
        const int fasgVb = MetalEnv::get().fasgLoadBatch;
        const int fasgLoadBatch = (fasgVb > 0 && (mHeadDim / 8) % fasgVb == 0) ? fasgVb : 0;
        // 8 simdgroups (256 threads) put 64 q rows in the tile, halving KV
        // re-reads at unchanged register pressure; Qs stays under the tg budget.
        mFaSgNsg = 8;
        mFaSgBq  = mFaSgNsg * 8;
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
        if (nil == pipeline) {
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
            pipeline = mtbn->makeComputePipelineWithSourceOption(gPrefillFlashAttnSg, "prefill_flash_attn_sg", option);
            if (nil != pipeline) {
                rt->insertPipeline(keys, pipeline);
            }
        }
        if (nil == pipeline || (int)pipeline.maxTotalThreadsPerThreadgroup < 32 * mFaSgNsg) {
            MNN_ERROR("MNN::Metal FA-SG: pipeline unavailable (cap %d), falling back to three-stage prefill\n",
                      pipeline ? (int)pipeline.maxTotalThreadsPerThreadgroup : -1);
            mFaSgUnavailable = true;
            mFaSgPrefill = false;
            _computePathFlags(inputs);
            compilerShader(inputs);
            return;
        }
        mKernel_faSg = pipeline;
    }
}

int AttentionBufExecution::_resolveQseqSplit() const {
    // Only the three-stage path allocates the scratch this bounds.
    if (mFlashAttnPrefill || mFaTcPrefill || mFaSgPrefill || mSdpaSinglePass || mSeqLen <= 32) {
        return 1;
    }
    // A piece of q rows costs both scratch tensors at once, so one q row is
    // 2 * B * H * kv_max elements wide.
    const int elemBytes = static_cast<MetalBackend*>(backend())->useFp16InsteadFp32() ? 2 : 4;
    const int64_t bytesPerQRow = (int64_t)2 * mBatch * mNumHead * mKvMaxLen * elemBytes;
    const auto& env = MetalEnv::get();
    int split = 1;
    if (bytesPerQRow > 0) {
        const int64_t budget = (int64_t)env.attnQSplitMB << 20;
        const int64_t maxRows = budget / bytesPerQRow;
        if (maxRows < mSeqLen) {
            split = UP_DIV(mSeqLen, (int)ALIMAX((int64_t)1, maxRows));
        }
    }
    // Round up to a power of two: the replay signature stores log2(split), and
    // equal-sized pieces keep the CAUSAL_TRI trapezoid counts uniform.
    int pow2 = 1;
    while (pow2 < split) {
        pow2 <<= 1;
    }
    // Each piece must keep at least one full 32-row QK tile, and log2 must fit
    // the 4-bit signature field.
    const int maxSplit = ALIMIN(mSeqLen / 32, 1 << 15);
    return ALIMAX(1, ALIMIN(pow2, maxSplit));
}

int AttentionBufExecution::_pastKvLength() const {
    // The owning layer encodes first and has already committed this step's rows.
    return mIsKVShared ? mKVCacheManager->kvLength() - mCurrentKvLen : mKVCacheManager->kvLength();
}

void AttentionBufExecution::handleKVAllocMemory() {
    constexpr auto allocType = Backend::DYNAMIC_IN_EXECUTION;
    if (!mKVCache) {
        mKvSeqLen = mCurrentKvLen;
        mKvMaxLen = ROUND_UP(mKvSeqLen, mKvAlignNum);
        mQseqSplitNum = _resolveQseqSplit();

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

    if (!mIsKVShared) {
        if (nullptr == mMeta || mMeta->previous == mMeta->remove) {
            mKVCacheManager->onClear();
            mKVCacheManager->onAlloc(mMeta, mCurrentKvLen);
        } else {
            mKVCacheManager->onRealloc(mMeta);
        }
    }

    mKvSeqLen = _pastKvLength() + mCurrentKvLen;
    mKvMaxLen = mKVCacheManager->maxLength();
    mQseqSplitNum = _resolveQseqSplit();

    // FA prefill keeps the online-softmax accumulator in tg memory and never
    // materializes mTempQK/mTempSoftMax, which are O(B*H*seq*kv_max) and dominate
    // long-context memory.
    if (mFlashAttnPrefill || mFaTcPrefill || mFaSgPrefill) {
        return;
    }

    // Single-pass fused SDPA writes the final output from the kernel itself;
    // no partial buffer, no mTempQK/mTempSoftMax.
    if (mSdpaSinglePass) {
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
    // dims>=2 is a per-element tensor mask; lower-rank inputs are sentinels
    // resolved into mCausalLayout / mBidirectionalMask. See
    // core/AttentionMaskConvention.hpp -- predicate must stay identical to CPUAttention.
    mHasTensorMask = inputs.size() > 3 && inputs[3]->dimensions() >= 2;
    if (mHasTensorMask) {
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
    mScale = (nullptr != mMeta && mMeta->attn_scale > 0.0f) ? mMeta->attn_scale : (1.0f / sqrt(mHeadDim));
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
    if (mKVCache && !mIsKVShared) {
        mKVCacheManager->setKVQuantParameter(mKVQuantParameter);
        mKVCacheManager->setAttenQuantKeyValue(mQuantKey, mQuantValue);
        mKVCacheManager->onResize(mKvNumHead, mHeadDim);
    }
    return NO_ERROR;
}
void AttentionBufExecution::_computePathFlags(const std::vector<Tensor*>& inputs) {
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto rt = (MetalRuntime*)mtbn->runtime();

    // Same predicate as Conv1x1GpuTier::proTier, so both ops split chips the same way.
    const MetalGpuInfo& gpu = rt->gpuInfo();
    const bool highBandwidthGpu = gpu.tier >= MetalGpuInfo::TierPro;

    int group_size = mNumHead / mKvNumHead;
    // GQA kernels recover the kv head as h / param.group, which only holds when the q
    // heads split evenly over the kv heads. GROUP_SIZE is just a pipeline cache key.
    const bool groupDivides = mKvNumHead > 0 && group_size * mKvNumHead == mNumHead;

    // Shape-only classification (see core/AttentionMaskConvention.hpp). mCausalLayout
    // implies mKVCache. Do not inspect mask buffer contents: the h2d blit is queued,
    // so stale pool data may still sit there at encode time.
    const bool hasMaskInput = inputs.size() > 3;
    const int maskDims = hasMaskInput ? inputs[3]->dimensions() : 0;
    mBidirectionalMask = IsAttentionMaskBidirectionalSentinel(
        hasMaskInput, maskDims, hasMaskInput ? inputs[3]->elementSize() : 0);
    // The same predicate CPUAttention uses: the sentinel only candidates the op for
    // causality, kv_cache licenses it. Equivalent to compilerShader's DEFAULT_MASK gate
    // for every supported shape, so perf flag and correctness key cannot disagree.
    mCausalLayout = IsAttentionMaskCausal(hasMaskInput, maskDims, mKVCache);

    // Single-pass SDPA auto threshold. decode_splitkv is the route down to very
    // small kv: the fallbacks (fused qk_softmax and the three-stage decode_qk path)
    // pay per-token strided reads on the row-major V cache. Only kv==1 keeps them.
    {
        const int sDecodeFusedThresh = 2;
        bool trivialMask = mHasTensorMask && mIsAddMask && mSeqLen == 1 && inputs[3]->elementSize() == 1;
        const int totalKv = (mKVCache && mKVCacheManager != nullptr ? _pastKvLength() : 0) + mCurrentKvLen;

        // Single-pass fused decode SDPA: decode_splitkv runs one threadgroup per q
        // head and writes the output itself, no reduce dispatch. Auto-on; only kv
        // below the threshold stays on the fallback paths.
        mSdpaSinglePass = false;
        const int sdpaThresh = sDecodeFusedThresh;
        // q heads per TG. Must divide group_size so each TG reads a single kv
        // head. Resolved before nsg since it sets grid.y which nsg depends on.
        const int groupSize = (mKvNumHead > 0) ? (mNumHead / mKvNumHead) : 1;
        {
            // Share each KV row across qh heads: cuts KV read requests by qh at
            // the cost of qh fewer TGs. <8 TGs is parallelism-starved; 8..15 only
            // pays while a residual group_size/qh >= 2 keeps requests the bottleneck.
            mSdpaQhPerTg = 1;
            for (int cand = 2; cand <= groupSize; cand *= 2) {
                if (groupSize % cand != 0 || mNumHead % cand != 0) {
                    continue;
                }
                const int threadgroups = mBatch * mNumHead / cand;
                if (threadgroups < 8) {
                    break;
                }
                if (threadgroups < 16 && groupSize / cand < 2) {
                    break;
                }
                mSdpaQhPerTg = cand;
            }
        }
        if (groupSize % mSdpaQhPerTg != 0 || mNumHead % mSdpaQhPerTg != 0) {
            mSdpaQhPerTg = 1;
        }
        // grid.y = batch*head_num/qh is total parallelism; nsg widens each TG.
        // Route keeps threadgroups*nsg roughly constant.
        const int tgCount = ALIMAX(mBatch * mNumHead / mSdpaQhPerTg, 1);
        {
            if (mtbn->isSupportTensorApi()) {
                // Tensor-API tier: fixed nsg32. The threadgroups*nsg rule of the
                // non-tensor branch below does not carry over to it, so this stays
                // a constant rather than an extrapolation.
                mSdpaNsg = 32;
            } else {
                const int product = highBandwidthGpu ? 512 : 256;
                // NSG must divide SIMD_GROUP_WIDTH: the decode_splitkv reduce
                // visits component lanes lp = sgitg + rep * NSG, rep < 32 / NSG.
                const int nsgWant = ALIMIN(ALIMAX(product / tgCount, 4), 32);
                mSdpaNsg = 4;
                while (mSdpaNsg * 2 <= nsgWant) {
                    mSdpaNsg *= 2;
                }
                const bool shortKvNeedsNarrow =
                    (!highBandwidthGpu && totalKv < 512 && tgCount < 16) ||
                    (highBandwidthGpu && totalKv < 256 && tgCount < 16 &&
                     mSdpaQhPerTg * mHeadDim > 256);
                if (shortKvNeedsNarrow) {
                    mSdpaNsg = ALIMIN(mSdpaNsg, 16);
                }
            }
        }
        // threadgroup floats: s_out[NSG*32] + s_sm[NSG*qh*2]
        const int tgBytesSdpa = (mSdpaNsg * 32 + mSdpaNsg * mSdpaQhPerTg * 2) * (int)sizeof(float);
        mSdpaSinglePass = sdpaThresh > 0 && totalKv >= sdpaThresh &&
                          mKVCache && mSeqLen == 1 && !mKvInDisk &&
                          (mCausalLayout || trivialMask) &&
                          (mHeadDim % 32) == 0 && tgBytesSdpa <= 30 * 1024;
        // KVB overlaps kv tokens per iteration to break the per-token dep chain;
        // only pays when TG count can't hide it. Newer/higher-bandwidth dies
        // bind at 16 TGs and shorter KV, so widen the gate there.
        const int kvbEnv = MetalEnv::get().decodeSdpaKvb;
        const bool wideKvbAuto = (gpu.gen >= 18) ||
                                 (gpu.gen >= 16 && highBandwidthGpu);
        const int  kvbMinKv    = wideKvbAuto ? 512  : 2048;
        const int  kvbMaxTg    = wideKvbAuto ? 17   : 16; // strict <
        int kvb = (kvbEnv > 0) ? kvbEnv
                               : ((mSdpaSinglePass && totalKv >= kvbMinKv && tgCount < kvbMaxTg) ? 2 : 1);
        // KVB multiplies the per-lane K/V staging arrays (2 * KVB * DPT floats,
        // DPT = HEAD_DIM/32): KVB=4 at head_dim 256 is 64 float regs, which
        // collapses occupancy. Cap by head_dim, not just the env whitelist.
        if (kvb > 2 && mHeadDim > 128) {
            kvb = 2;
        }
        mSdpaKvb = kvb;
        // Auto mode flips KVB when kv crosses kvbMinKv mid-decode; ask
        // compilerShader to pre-build both variants up front so the flip does
        // not compile synchronously inside an encode.
        mSdpaKvbPrebuild = kvbEnv == 0 && mSdpaSinglePass && tgCount < kvbMaxTg;
    }

    // whether use simdgroup
    bool supportSimdReduce = rt->supportSimdGroupReduce();
    bool supportSimdMatrix = rt->supportSimdGroupMatrix();
    bool supportTensorMatrix = mtbn->isSupportTensorApi();

    // Fused prefill flash-attention (opt-in via attention_mode/8 >= 1, see
    // docs/transformers/llm.md). Decided BEFORE handleKVAllocMemory() so the
    // O(B*H*seq*kv_max) mTempQK/mTempSoftMax scratch can be skipped.
    {
        int attentionOption = static_cast<MetalBackend*>(backend())->getRuntime()->hint().attentionOption;
        bool enableFromConfig = (attentionOption / 8) >= 1;
        bool enableFlashAttn = enableFromConfig;

        // FA shader needs fp16 and hard-codes causal masking; fp32 / non-causal
        // fall back to the three-kernel pipeline.
        bool eligible = supportSimdMatrix
                        && static_cast<MetalBackend*>(backend())->useFp16InsteadFp32()
                        && mCausalLayout
                        && !mKvInDisk
                        && (mHeadDim == 64 || mHeadDim == 128 || mHeadDim == 256)
                        && groupDivides
                        && !mShortSeq
                        && mSeqLen >= 128;
        mFlashAttnPrefill = enableFlashAttn && eligible;
        // Prefer three-kernel when causal-tri/bound can engage: bounded softmax
        // skips O(seq^2)/2 of QK + softmax traffic that FA still pays.
        if (mFlashAttnPrefill) {
            bool boundUsable = mCausalLayout && !mKvInDisk && mKvSeqLen >= mSeqLen;
            bool causalTriUsable  = boundUsable && (mQkSimdMatrix || mQkTensorMatrix);
            bool causalBoundUsable = boundUsable;
            if (causalTriUsable || causalBoundUsable) {
                mFlashAttnPrefill = false;
            }
        }
    }

    // Tensor-API FA prefill (prefill_flash_attn_tc). Keeps S/O in registers so
    // the score matrix is never materialized. Decided before handleKVAllocMemory
    // so the scratch can be skipped. fp16 + fp16 KV + causal only.
    {
        mFaTcPrefill = false;
        const bool faCommon = mtbn->useFp16InsteadFp32() && mCausalLayout && !mKvInDisk &&
                              !mQuantKey && !mQuantValue && mKvSeqLen >= mSeqLen &&
                              (mHeadDim == 64 || mHeadDim == 128 || mHeadDim == 256);
        if (!mFaTcUnavailable) {
            // matmul2d input cooperative tensors are single-simdgroup only.
            mFaTcPrefill = mtbn->isSupportTensorCoopInput() && faCommon && mSeqLen >= 64;
        }
        if (mFaTcPrefill) {
            mFlashAttnPrefill = false;
        }
    }

    // Simdgroup-matrix fused prefill (prefill_flash_attn_sg). Scores stay in simdgroup
    // fragments; mTempQK/mTempSoftMax are not allocated. Must be decided
    // BEFORE handleKVAllocMemory().
    {
        mFaSgPrefill = false;
        const bool sgCommon = mtbn->useFp16InsteadFp32() && mCausalLayout && !mKvInDisk &&
                              !mQuantKey && !mQuantValue && mKvSeqLen >= mSeqLen &&
                              (mHeadDim == 64 || mHeadDim == 128) && !mShortSeq &&
                              mSeqLen >= 64 && supportSimdMatrix &&
                              groupDivides;
        const int sgSeqFloor = highBandwidthGpu ? 512 : 1024;
        const bool autoM4 = !mtbn->isSupportTensorCoopInput() && mSeqLen >= sgSeqFloor;
        if (autoM4 && sgCommon && !mFaTcPrefill && !mFaSgUnavailable) {
            mFaSgPrefill = true;
            mFlashAttnPrefill = false;
        }
        static bool _fasg_log_once = false;
        if (mFaSgPrefill && !_fasg_log_once) {
            _fasg_log_once = true;
            MNN_PRINT("[MetalAttention] prefill-flash-attn-sg kernel active (seq=%d, head_dim=%d, group=%d, outC4=%d).\n",
                      mSeqLen, mHeadDim, group_size, (int)mOutputC4);
        }
    }

    handleKVAllocMemory();

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
    // Gated on mCausalLayout (standard causal mask), the simdgroup-matrix tile
    // path, in-memory KV, and kv >= q so the diagonal offset D is non-negative.
    {
        mQkCausalTri = mCausalLayout && !mShortSeq && (mQkSimdMatrix || mQkTensorMatrix) &&
                       !mFlashAttnPrefill && !mFaTcPrefill && !mFaSgPrefill && !mKvInDisk &&
                       mKvSeqLen >= mSeqLen;
        // CAUSAL_BOUND is path-agnostic: activates on both the simd-matrix and the
        // tensor-API three-kernel prefill paths, so long as we are not on the FA
        // path and the causal-mask semantics hold.
        mCausalBound = mCausalLayout && !mShortSeq && !mFlashAttnPrefill && !mFaTcPrefill &&
                       !mFaSgPrefill && !mKvInDisk && mKvSeqLen >= mSeqLen;
    }

    bool trivialFloatMask = mHasTensorMask && mIsAddMask && mSeqLen == 1 && inputs[3]->elementSize() == 1;
    // Max KV length for fused decode QK+softmax kernel depends on group_size
    // to stay within 32KB threadgroup memory limit:
    //   group_size<=2: 2048, group_size<=4: 1024, group_size<=8: 512
    int maxKvForFusion = 0;
    if (group_size >= 2 && group_size <= 2) maxKvForFusion = 2048;
    else if (group_size <= 4) maxKvForFusion = 1024;
    else if (group_size <= 8) maxKvForFusion = 512;
    mDecodeQkSoftmax = mKVCache && mShortSeq && mSeqLen <= 8 &&
                       (mCausalLayout || trivialFloatMask) && !mKvInDisk &&
                       group_size >= 2 && mHeadDim % 8 == 0 && mKvSeqLen <= maxKvForFusion;

    // Single-pass fused SDPA (decided at the top of onEncode, before the KV
    // alloc) supersedes the fused qk_softmax path.
    if (mSdpaSinglePass) {
        mDecodeQkSoftmax = false;
    }

    // Q-head-split variant of the fused QK+softmax kernel (group_size==2
    // only). Enabled on non-tensor-API devices at kv >= 512; tensor-API
    // devices stay excluded (the override MNN_METAL_QK_QSPLIT was removed).
    mQkQsplit = mDecodeQkSoftmax && group_size == 2 &&
                !mtbn->isSupportTensorApi() && mKvSeqLen >= 512;
}

uint32_t AttentionBufExecution::_pathSignature() const {
    uint32_t sig = 0;
    sig |= (uint32_t)mShortSeq;
    // SDPA_KVB flips at its kv auto gate, which recompiles the kernel.
    sig |= (uint32_t)(mSdpaKvb > 1) << 1;
    sig |= (uint32_t)mDecodeQkSoftmax << 2;
    sig |= (uint32_t)mQkvSimdReduce << 3;
    sig |= (uint32_t)mQkvSimdMatrix << 4;
    sig |= (uint32_t)mQkSimdMatrix << 5;
    sig |= (uint32_t)mQkTensorMatrix << 6;
    sig |= (uint32_t)mQkSimdReduce << 7;
    sig |= (uint32_t)mSftmSimdReduce << 8;
    sig |= (uint32_t)mCopySimdReduce << 9;
    sig |= (uint32_t)mHasTensorMask << 10;
    sig |= (uint32_t)mIsAddMask << 11;
    sig |= (uint32_t)mFlashAttnPrefill << 12;
    sig |= (uint32_t)mQuantKey << 13;
    sig |= (uint32_t)mQuantValue << 14;
    sig |= (uint32_t)mKVCache << 15;
    sig |= (uint32_t)mKvInDisk << 16;
    sig |= (uint32_t)mOutputC4 << 17;
    sig |= (uint32_t)mQkCausalTri << 18;
    sig |= (uint32_t)mCausalBound << 19;
    // N is always a power of two, so the 4-bit field holds log2(N) rather than N
    // itself: bounding the three-stage scratch needs N well past the 15 a raw
    // count would allow.
    {
        int qsplitLog2 = 0;
        while ((1 << qsplitLog2) < mQseqSplitNum) {
            qsplitLog2++;
        }
        sig |= (uint32_t)(qsplitLog2 & 0xF) << 20;
    }
    // The fused decode_qk_softmax pipeline is compiled with a SHORT_KV_128
    // macro while kv <= 128 — replay must not outlive that variant.
    sig |= (uint32_t)((mDecodeQkSoftmax && mKvSeqLen <= 128) ? 1 : 0) << 24;
    // Q-head-split fused kernel variant flips at the kv>=512 auto gate.
    sig |= (uint32_t)mQkQsplit << 25;
    // Single-pass fused SDPA path.
    sig |= (uint32_t)mSdpaSinglePass << 26;
    sig |= (uint32_t)mFaSgPrefill << 27;
    // Bidirectional sentinel suppresses the DEFAULT_MASK shader key, so it changes the
    // compiled kernel and must not share a replay slot with the causal form.
    sig |= (uint32_t)mBidirectionalMask << 28;
    sig |= (uint32_t)mFaTcPrefill << 29;
    // SDPA_QH_PER_TG changes both the compiled kernel and the dispatch grid.
    sig |= (uint32_t)((mSdpaQhPerTg == 8 ? 3 : (mSdpaQhPerTg == 4 ? 2 : (mSdpaQhPerTg == 2 ? 1 : 0))) & 0x3) << 30;
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
    // current new kv_len
    p.kv_seq_len = key->shape()[1];
    p.max_kv_len = mKvMaxLen;
    int pastLength = mKVCache ? _pastKvLength() : 0;
    p.dst_k_offset = pastLength * p.head_count;
    p.dst_v_offset = pastLength * p.head_count;
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
    return sp;
}
void AttentionBufExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                     id<MTLComputeCommandEncoder> encoder) {
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto context = (__bridge MNNMetalContext*)mtbn->context();

    _computePathFlags(inputs);
    int group_size = mNumHead / mKvNumHead;

    // temp memory alloc, handle variable set
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

    // start to compile attention shaders
    compilerShader(inputs);

#if MNN_METAL_OP_PROFILE
    // Split Attention into per-subpass command buffers so profile shows QK / Softmax / AV / Copy separately.
    static_cast<MetalBackend*>(backend())->setProfileSubtag("copy");
#endif
    // Run Copy and Format-Convert Kernel
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
        encoder = mtbn->profileNextSubpass(mSdpaSinglePass ? "sdpa_fused" : (mShortSeq ? "qk_short" : (mDecodeQkSoftmax ? "qk_softmax_fused" : "qk")));
    }
#endif

    // Update Parameters
    int seqLenPiece = UP_DIV(mSeqLen, mQseqSplitNum);
    const AttentionQKVParam qkvParam = _makeQKVParam(inputs, seqLenPiece);

    for (int seq_idx = 0; seq_idx < mQseqSplitNum; seq_idx++) {
        if (mFaTcPrefill) {
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
            [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(seqLenPiece, 64), mBatch * mNumHead, mFaTcDSplit)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
#if MNN_METAL_OP_PROFILE
            {
                auto* mtbn2 = static_cast<MetalBackend*>(backend());
                encoder = mtbn2->profileNextSubpass("fa_tc");
            }
#endif
            continue;   // skip the standard QK / softmax / PV path below
        }
        if (mFaSgPrefill) {
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
            [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(seqLenPiece, mFaSgBq), mBatch * mNumHead, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, mFaSgNsg, 1)];
#if MNN_METAL_OP_PROFILE
            {
                auto* mtbn2 = static_cast<MetalBackend*>(backend());
                encoder = mtbn2->profileNextSubpass("flash_attn_sg");
            }
#endif
            continue;
        }
        if (mFlashAttnPrefill) {
            // Fused prefill flash-attention: QK + online softmax + PV in a single dispatch.
            // Writes directly to outputs[0]; mTempQK / mTempSoftMax are never touched.
            [encoder setComputePipelineState:mKernel_flashAttn];
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
            if (mQuantKey && mKVCacheManager->getKScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getKScaleBuffer() offset:0 atIndex:9];
            }
            if (mQuantValue && mKVCacheManager->getVScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getVScaleBuffer() offset:0 atIndex:10];
            }
            // Grid = (ceil(seqLenPiece/16), B*H, 1); threadgroup = (32, NSG=4, 1) = 128 threads.
            // Q_TILE=16 halves K read redundancy vs Q_TILE=8.
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
        if (mSdpaSinglePass) {
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
            const int gridGroups = mBatch * (mNumHead / mSdpaQhPerTg);
            [encoder dispatchThreadgroups:MTLSizeMake(1, gridGroups, 1)
                    threadsPerThreadgroup:MTLSizeMake(32 * mSdpaNsg, 1, 1)];
            // No trailing profileNextSubpass: "sdpa_fused" subtag was set after
            // the copy flush; a trailing call would add an empty encoder.
            continue;   // skip the standard QK / softmax / QKV path below
        }
        if (mDecodeQkSoftmax) {
            [encoder setComputePipelineState:mKernel_qk_softmax];
            MetalBackend::setTensor(query, encoder, 0);
            MetalBackend::setTensor(mTempSoftMax.get(), encoder, 1);
            MetalBackend::setTensor(tempTensorK, encoder, 2);
            [encoder setBytes:&seq_idx length:sizeof(seq_idx) atIndex:3];
            [encoder setBytes:&qkvParam length:sizeof(AttentionQKVParam) atIndex:4];
            if (mKVCache && mQuantKey && mKVCacheManager->getKScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getKScaleBuffer() offset:0 atIndex:8];
            }
            int gridZ = mQkQsplit ? group_size : 1;
            int qkGroups = mBatch * (mNumHead / group_size) * seqLenPiece;
            int maxLocalSize = ALIMAX(32, ((int)mKernel_qk_softmax.maxTotalThreadsPerThreadgroup / 32) * 32);
            int localSize = qkGroups <= 8 ? ALIMIN(maxLocalSize, ALIMAX(128, ROUND_UP(mKvSeqLen, 32))) :
                            ALIMIN(maxLocalSize, ALIMAX(64, ROUND_UP(UP_DIV(mKvSeqLen, 6), 32)));
            if (mQkQsplit) {
                // Half-width TGs: total threads match the non-split path while
                // TG count doubles. The narrow kv/6 formula is a net loss here.
                localSize = ALIMIN(maxLocalSize, ALIMAX(128, ROUND_UP(UP_DIV(mKvSeqLen, 2), 32)));
            }
            auto gl = std::make_pair(MTLSizeMake(mBatch * (mNumHead / group_size), seqLenPiece, gridZ), MTLSizeMake(localSize, 1, 1));
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
            const AttentionSoftmaxParam softmaxParam = _makeSoftmaxParam(seqLenPiece);
            [encoder setComputePipelineState:mKernel_softmax];
            // [mBatch, mNumHead, mSeqLen, mKvSeqLen]
            MetalBackend::setTensor(mTempQK.get(), encoder, 0);
            // [mBatch, mNumHead, mSeqLen, ROUND_UP(mKvSeqLen, mKvAlignNum)]
            MetalBackend::setTensor(mTempSoftMax.get(), encoder, 1);
            [encoder setBytes:&softmaxParam length:sizeof(AttentionSoftmaxParam) atIndex:2];
            if (mCausalBound) {
                // Both setBytes: each dispatch keeps its own seq_idx regardless of
                // what later pieces of the same command buffer encode.
                [encoder setBytes:&seq_idx length:sizeof(seq_idx) atIndex:3];
            }

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
            [encoder setBytes:&qkvParam length:sizeof(AttentionQKVParam) atIndex:4];
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
    if (mKVCache && !mIsKVShared) {
        mKVCacheManager->setPastLength(mKVCacheManager->kvLength() + mCurrentKvLen);
    }
    // Replay bookkeeping: capture the structural fingerprint + raw scale-buffer
    // identities so onReplayUpdate can detect path switches / KV reallocations.
    mLastEncodeSig = _pathSignature();
    mLastKScaleBuffer = (mKVCache && mKVCacheManager) ? mKVCacheManager->getKScaleBuffer() : nil;
    mLastVScaleBuffer = (mKVCache && mKVCacheManager) ? mKVCacheManager->getVScaleBuffer() : nil;
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
    // else (prefill, disk KV, FA, split, shared layers) bails and re-encodes.
    // Checks here use last-encode state; the signature below is authoritative.
    if (!mKVCache || mIsKVShared || mKvInDisk || mFlashAttnPrefill || mFaTcPrefill || mFaSgPrefill ||
        mQseqSplitNum != 1) {
        return false;
    }
    // Recompute per-token path state exactly as onEncode would; idempotent on
    // fallback since handleKVAllocMemory's onRealloc is a no-op once capacity fits.
    _computePathFlags(inputs);
    if (!mShortSeq || _pathSignature() != mLastEncodeSig) {
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
    // Structural layout must match the recording before patching by position.
    const int expectEvents = mSdpaSinglePass ? 2 : (mDecodeQkSoftmax ? 3 : 4);
    if ((int)mReplayEvents.size() != expectEvents) {
        return false;
    }
    for (const auto& e : mReplayEvents) {
        if (e.type != MetalReplayEvent::Dispatch) {
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
    int seqLenPiece = UP_DIV(mSeqLen, mQseqSplitNum);

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
    if (mSdpaSinglePass) {
        // events[1] = fused sdpa: grid (1, B*head groups) is kv-length
        // independent; kv reaches the kernel via the param rewrite.
    } else if (mDecodeQkSoftmax) {
        // events[1] = fused qk_softmax (tg width tracks kv); events[2] = qkv
        // (grid is kv-independent).
        auto& fused = mReplayEvents[1];
        int qkGroups = mBatch * (mNumHead / group_size) * seqLenPiece;
        int maxLocalSize = ALIMAX(32, ((int)fused.pipeline.maxTotalThreadsPerThreadgroup / 32) * 32);
        int localSize = qkGroups <= 8 ? ALIMIN(maxLocalSize, ALIMAX(128, ROUND_UP(mKvSeqLen, 32))) :
                        ALIMIN(maxLocalSize, ALIMAX(64, ROUND_UP(UP_DIV(mKvSeqLen, 6), 32)));
        if (mQkQsplit) {
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
