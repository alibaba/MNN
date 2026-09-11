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
    if (mHasTensorMask) {
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
        auto buildSdpa = [&](int nsg, bool partial, const char* fn) -> id<MTLComputePipelineState> {
            std::vector<std::string> keys = {std::string("decode_splitkv_sdpa_") + fn, ftype, group_str,
                                             "HEAD_DIM_" + head_dim_str, "NSG_" + std::to_string(nsg)};
            if (mSdpaQhPerTg > 1) {
                keys.emplace_back("QHTG_" + std::to_string(mSdpaQhPerTg));
            }
            if (partial) {
                keys.emplace_back("SPLIT_KV_PARTIAL");
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
                // keys[0..4] plus the optional QHTG entry are structural (they map
                // to valued macros, not flags); the rest are plain =1 flags.
                size_t flagStart = 5;
                if (mSdpaQhPerTg > 1) {
                    [dic setValue:@(std::to_string(mSdpaQhPerTg).c_str()) forKey:@"SDPA_QH_PER_TG"];
                    flagStart = 6;
                }
                for (size_t j = flagStart; j < keys.size(); ++j) {
                    [dic setValue:@"1" forKey:@(keys[j].c_str())];
                }
                option.preprocessorMacros = dic;
                pipeline = mtbn->makeComputePipelineWithSourceOption(gDecodeSplitKV, fn, option);
                if (nil != pipeline) {
                    rt->insertPipeline(keys, pipeline);
                }
            }
            return pipeline;
        };
        // Degrade NSG until the compiled pipeline can actually host 32*NSG threads.
        int nsg = mSdpaNsg;
        const bool twoPass = mSdpaNtg > 1;
        id<MTLComputePipelineState> pipeline = nil;
        while (nsg >= 4) {
            pipeline = buildSdpa(nsg, twoPass, "decode_splitkv");
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
        if (nil != pipeline && twoPass) {
            mKernel_sdpaReduce = buildSdpa(nsg, true, "decode_splitkv_reduce");
            if (nil == mKernel_sdpaReduce) {
                MNN_ERROR("MNN::Metal SDPA: reduce pipeline unavailable, using single pass\n");
                mSdpaNtg = 1;
                pipeline = buildSdpa(nsg, false, "decode_splitkv");
            }
        }
        if (nil == pipeline) {
            MNN_ERROR("MNN::Metal SDPA: no viable NSG, falling back to legacy decode path\n");
            mSdpaSinglePass = false;
            // the single-pass early return skipped mTempQK/mTempSoftMax; re-run
            handleKVAllocMemory();
        } else {
            mSdpaNsg = nsg;
            mKernel_sdpa = pipeline;
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
        const bool faTcQReg = !MetalEnv::get().faTcQRegDisabled;
        if (faTcQReg) {
            keys.emplace_back("FATC_Q_REG");
        }
        const bool faTcQkK32 = MetalEnv::get().faTcQkK32;
        if (faTcQkK32) {
            keys.emplace_back("FATC_QK_K32");
        }
        const bool faTcOCt = MetalEnv::get().faTcOCt;
        if (faTcOCt) {
            keys.emplace_back("FATC_O_CT");
        }
        // K/V read in place as matmul2d tensor handles; needs the persistent-CT
        // PV block.
        const bool faTcKvDevTensor = MetalEnv::get().faTcKvDevTensor && faTcOCt;
        if (faTcKvDevTensor) {
            keys.emplace_back("FATC_KV_DEV_TENSOR");
        }
        const bool faTcQRev = MetalEnv::get().faTcQRev;
        if (faTcQRev) {
            keys.emplace_back("FATC_QREV");
        }
        // Persistent QK left-input CTs replace the q_reg array, and the shader
        // only spells the fill out for FATC_TDK <= 4, i.e. the wide-K QK.
        const bool faTcQCt = MetalEnv::get().faTcQCt && faTcQReg && faTcQkK32;
        if (faTcQCt) {
            keys.emplace_back("FATC_Q_CT");
        }
        // Wide-K PV only exists in the persistent-CT PV block.
        const bool faTcPvK32 = MetalEnv::get().faTcPvK32 && faTcOCt;
        if (faTcPvK32) {
            keys.emplace_back("FATC_PV_K32");
        }
        // Head_dim d-split. Only the persistent-CT PV block honors d_base, and a
        // slice must stay a whole multiple of the 32-wide O tile.
        int faTcDSplit = MetalEnv::get().faTcDSplit;
        if (faTcDSplit < 0) {
            faTcDSplit = (mHeadDim == 256) ? 2 : 1;
        }
        if (!faTcOCt || (mHeadDim % (32 * faTcDSplit)) != 0) {
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
            if (faTcQReg) {
                [dic setValue:@"1" forKey:@"FATC_Q_REG"];
            }
            if (faTcQkK32) {
                [dic setValue:@"1" forKey:@"FATC_QK_K32"];
            }
            if (faTcOCt) {
                [dic setValue:@"1" forKey:@"FATC_O_CT"];
            }
            if (faTcKvDevTensor) {
                [dic setValue:@"1" forKey:@"FATC_KV_DEV_TENSOR"];
            }
            if (faTcQRev) {
                [dic setValue:@"1" forKey:@"FATC_QREV"];
            }
            if (faTcQCt) {
                [dic setValue:@"1" forKey:@"FATC_Q_CT"];
            }
            if (faTcPvK32) {
                [dic setValue:@"1" forKey:@"FATC_PV_K32"];
            }
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
                          mSeqLen, mHeadDim, faTcQkK32 ? 32 : 16, faTcPvK32 ? 32 : 16,
                          faTcDSplit);
            }
        }
    }
    if (mFaSgPrefill) {
        std::string head_dim_str = std::to_string(mHeadDim);
        const bool fasgQReg = MetalEnv::get().fasgQReg;
        // NSG8 widens the q tile from 32 to 64 rows, which doubles Qs. FA-SG only runs
        // at head_dim 64 or 128, so the widened tile peaks at 26112 bytes and
        // always fits Metal's threadgroup memory limit.
        const bool fasgNsg8 = MetalEnv::get().fasgNsg8;
        // The batched V loop is unrolled over head_dim fragments, so a batch width
        // that does not divide their count would index past mO.
        const int fasgVb = MetalEnv::get().fasgLoadBatch;
        const int fasgLoadBatch = (fasgVb > 0 && (mHeadDim / 8) % fasgVb == 0) ? fasgVb : 0;
        mFaSgNsg = fasgNsg8 ? 8 : 4;
        mFaSgBq  = mFaSgNsg * 8;
        std::vector<std::string> keys = {"prefill_flash_attn_sg", ftype, "HEAD_DIM_" + head_dim_str};
        if (mHasTensorMask) {
            keys.emplace_back("HAS_MASK");
        }
        if (mOutputC4) {
            keys.emplace_back("ATTENTION_C4");
        }
        if (fasgQReg) {
            keys.emplace_back("FASG_Q_REG");
        }
        if (fasgNsg8) {
            keys.emplace_back("FASG_NSG8");
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
            if (fasgQReg) {
                [dic setValue:@"1" forKey:@"FASG_Q_REG"];
            }
            if (fasgNsg8) {
                [dic setValue:@"1" forKey:@"FASG_NSG8"];
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
    if (env.attnQSplit > 0) {
        split = env.attnQSplit;
    } else if (bytesPerQRow > 0) {
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

void AttentionBufExecution::handleKVAllocMemory() {
    constexpr auto allocType = Backend::DYNAMIC_IN_EXECUTION;
    if (!mKVCache) {
        mKvSeqLen = mCurrentKvLen;
        mKvMaxLen = ROUND_UP(mKvSeqLen, mKvAlignNum);
        mQseqSplitNum = _resolveQseqSplit();

        int keySize = mKvMaxLen * mBatch * mKvNumHead * mHeadDim;
        int valueSize = mBatch * mKvNumHead * mHeadDim * mKvMaxLen;
        // mTempK/mTempV are setTensor-bound, so re-creating them here would
        // leave a recorded encode-replay holding a freed Tensor* (see the
        // lifetime invariant in MetalReplay.hpp). Safe only because this block
        // is !mKVCache-only and onReplayUpdate bails on !mKVCache before
        // metalReplayValidate runs. Keep that order, and if a mKVCache path
        // ever reallocates these, guard it explicitly.
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
    mQseqSplitNum = _resolveQseqSplit();

    // Flash-attn prefill path is self-contained: online softmax accumulator lives
    // in threadgroup memory and never materializes the full QK / softmax tensors.
    // Skipping these scratch buffers is the whole point of using flash-attn for
    // long context — mTempQK alone is O(B * H * seq * kv_max) which reaches TB
    // scale at 512K prompts.
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
    // A mask input with dims>=2 is a per-element (tensor) mask that must be read
    // position by position. A dims<2 float mask is the scalar causal sentinel
    // (see llm.cpp gen_attention_mask / CPUAttention) -- not a tensor mask; its
    // causal meaning is resolved into mCausalLayout in _computePathFlags.
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
void AttentionBufExecution::_computePathFlags(const std::vector<Tensor*>& inputs) {
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto rt = (MetalRuntime*)mtbn->runtime();

    int group_size = mNumHead / mKvNumHead;

    // Causal-fast-path gate. Enable causal-tri / causal-bound / FA / faTc ONLY
    // when no per-element mask is in play: either the mask input is absent, or it
    // is the standard-causal scalar sentinel -- a scalar float mask (dims<2) whose
    // value is ~0 (matching CPUAttention.cpp:574-587's shape().empty() &&
    // ptr[0]<1e-6). Both LLM decode/prefill and embedding models feed this scalar
    // sentinel (embedding via gen_attention_mask, kv_cache=true), so both take the
    // fast path. kv-cache is also required because the per-tile causal masking
    // these opts rely on is DEFAULT_MASK, only compiled in compilerShader's
    // `else if (mKVCache)` branch.
    //   - absent mask / scalar sentinel(0), plus kv-cache => causal fast path
    //   - tensor mask (SWA/arbitrary)                     => honor per-element
    //   - non-zero scalar / no kv-cache                   => no fast path
    //     (correctness still holds: DEFAULT_MASK causal when mKVCache, else dense)
    // Value read is from shared-storage contents, safe here (_computePathFlags
    // runs at encode time, mask data already written). Replaces the old
    // MNN_METAL_QK_CAUSAL_TRI env. Invariant: mCausalLayout==true => mKVCache==true.
    bool scalarCausalSentinel = inputs.size() <= 3;
    if (!mHasTensorMask && inputs.size() > 3 &&
        inputs[3]->dimensions() < 2 && inputs[3]->getType() == halide_type_of<float>()) {
        auto alloc = (MetalRuntimeAllocator::MetalBufferAlloc*)inputs[3]->deviceId();
        if (alloc != nullptr) {
            auto buf = (id<MTLBuffer>)alloc->getBuffer();
            const float* mptr = (const float*)((uint8_t*)buf.contents +
                                 TensorUtils::getDescribeOrigin(inputs[3])->offset);
            scalarCausalSentinel = (mptr[0] < 1e-6f && mptr[0] > -1e-6f);
        }
    }
    mCausalLayout = scalarCausalSentinel && mKVCache;

    // Single-pass SDPA auto threshold. decode_splitkv beats both fallbacks
    // (fused qk_softmax and the three-stage decode_qk path) down to very small
    // kv, while the fallbacks regress with the row-major V cache (per-token
    // strided reads). Only kv==1 stays on the fallbacks; all-kv splitkv
    // regressed.
    {
        const int sDecodeFusedThresh = 2;
        bool trivialMask = mHasTensorMask && mIsAddMask && mSeqLen == 1 && inputs[3]->elementSize() == 1;
        const int totalKv = (mKVCache && mKVCacheManager != nullptr ? mKVCacheManager->kvLength() : 0) + mCurrentKvLen;

        // Single-pass fused decode SDPA: decode_splitkv
        // runs a single threadgroup per q head (ntg=1), no reduce dispatch, final
        // output written by the kernel itself. Default auto-on
        // (MNN_METAL_DECODE_SDPA); =0 disables (fused qk_softmax at kv<=cap /
        // three-stage decode_qk beyond it take over); =N>1 overrides the kv
        // threshold. Only kv below the threshold stays on the fallback paths.
        mSdpaSinglePass = false;
        const int sdpaEnv = MetalEnv::get().decodeSdpa;
        if (sdpaEnv > 0) {
            const int sdpaThresh = (sdpaEnv == 1) ? sDecodeFusedThresh : sdpaEnv;
            // q heads per threadgroup. Must divide group_size so every q head in
            // the threadgroup reads the same kv head. MNN_METAL_DECODE_SDPA_QH_PER_TG
            // overrides; 0 = auto. Resolved before nsg, because it sets grid.y and
            // the nsg tier below is a function of the threadgroup count.
            const int groupSize = (mKvNumHead > 0) ? (mNumHead / mKvNumHead) : 1;
            mSdpaQhPerTg = MetalEnv::get().decodeSdpaQhPerTg;
            if (mSdpaQhPerTg <= 0) {
                // Auto: share each KV row across as many q heads as the dispatch
                // can afford. Raising qh divides KV read requests by qh, but it
                // also divides the threadgroup count by qh, and those pull
                // opposite ways.
                // Below 8 threadgroups the dispatch is parallelism-starved and
                // nothing redeems it. In the 8..15 range grouping costs real
                // parallelism, so it only pays while there is still redundancy
                // left to remove afterwards: residual group_size/qh >= 2 means
                // request traffic is still the binding constraint, whereas
                // residual 1 means unique-KV DRAM traffic already is (qh already
                // near DRAM peak, so a wider qh buys nothing and only sheds
                // threadgroups).
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
            mSdpaNsg = MetalEnv::get().decodeSdpaNsg;
            if (mSdpaNsg == 0) {
                // grid.y = batch*head_num/qh, so that count is the whole
                // dispatch's parallelism and nsg only sets each threadgroup's
                // width (32*nsg threads). Fewer threadgroups therefore want a
                // wider one: the decode attention op test finds a roughly
                // constant threadgroups*nsg product, and each step off it costs
                // real time.
                const int tgCount = ALIMAX(mBatch * mNumHead / mSdpaQhPerTg, 1);
                if (mtbn->isSupportTensorApi()) {
                    // Tensor-API tier: nsg32 measured best; the threadgroup-count
                    // sweep from the non-tensor tier has not been repeated here,
                    // so keep the measured constant rather than extrapolating.
                    mSdpaNsg = 32;
                } else {
                    // The target threadgroups*nsg product follows the device's
                    // memory-bandwidth tier: 256 for M4 base, 512 for M4 Pro.
                    const bool highBandwidthM4 = rt->isHighBandwidthM4();
                    const int product = highBandwidthM4 ? 512 : 256;
                    mSdpaNsg = ALIMIN(ALIMAX(product / tgCount, 4), 32);
                    // M4 base retains its short-KV cap. On M4 Pro, production
                    // shapes stay wide; only the measured high-register-pressure
                    // qh2*hd256 corner narrows below 256 tokens.
                    const bool shortKvNeedsNarrow =
                        (!highBandwidthM4 && totalKv < 512 && tgCount < 16) ||
                        (highBandwidthM4 && totalKv < 256 && tgCount < 16 &&
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
        }
        // 2-pass split-KV threadgroup count. Capped so each threadgroup still owns
        // a full NSG-wide token stripe; beyond that the extra threadgroups sit idle
        // and only add reduce work. Resolved here and then frozen for the
        // recorded encode-replay, so a later kv growth leaves the cap stale --
        // harmless, since an idle threadgroup contributes a zero-weight partial.
        mSdpaNtg = 1;
        if (mSdpaSinglePass) {
            const int ntgEnv = MetalEnv::get().decodeSdpaNtg;
            int ntgWant = 1;
            if (ntgEnv == 0) {
                // Auto is off: env-only until re-measured. The band that shipped
                // with this path was calibrated when every q head owned a
                // threadgroup. mSdpaQhPerTg now attacks the same long-ctx KV
                // bottleneck from the other side and already picks qh>1 there, so
                // the dispatch it was measured against no longer exists.
            } else {
                ntgWant = ntgEnv;
            }
            if (ntgWant > 1) {
                const int maxNtg = ALIMAX(totalKv / mSdpaNsg, 1);
                mSdpaNtg = ALIMIN(ntgWant, maxNtg);
            }
        }
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
    //   - head_dim in {64, 128, 256}   (256 for memory-bound long context)
    //   - GQA group_size in {1, 2, 4, 8}
    //   - prefill length >= 128 (short seqs already fast via existing paths)
    //
    // At head_dim=256 the kernel is compute-bound and slightly slower than
    // the three-kernel path in isolation, but at long context the fused path
    // skips the O(seq^2 * B * H) mTempQK / mTempSoftMax scratch allocations,
    // which dominates peak memory.  Trade is acceptable for long-context /
    // constrained-device runs.
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
                        && mCausalLayout
                        && !mKvInDisk
                        && (mHeadDim == 64 || mHeadDim == 128 || mHeadDim == 256)
                        && (group_size == 1 || group_size == 2 || group_size == 4 || group_size == 8)
                        && !mShortSeq
                        && mSeqLen >= 128;
        mFlashAttnPrefill = enableFlashAttn && eligible;
        // M4-class demotion: the three-kernel path with CAUSAL_TRI +
        // CAUSAL_BOUND beats the FA kernel and the gap grows with seq because the
        // bounded softmax skips O(seq^2)/2 of QK-write + softmax read/write
        // bandwidth that FA does not. Prefer three-kernel on non-tensor-API
        // M4/A-series devices whenever causal-tri can engage; env
        // MNN_ENABLE_FLASH_ATTN_PREFILL=1 still forces FA.
        // Long context used to be excluded here (kv <= 8192) because the
        // three-kernel scratch grows as B*H*seq*kv. _resolveQseqSplit now caps
        // that scratch by splitting the q sequence, which also runs faster than
        // FA at those shapes, so the cutoff is gone.
        if (mFlashAttnPrefill && !envForceOn) {
            bool boundUsable = mCausalLayout && !mKvInDisk && mKvSeqLen >= mSeqLen;
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
            if ((causalTriUsable || causalBoundUsable) && m4Class) {
                mFlashAttnPrefill = false;
            }
        }
        static bool _fa_log_once = false;
        if (mFlashAttnPrefill && !_fa_log_once) {
            _fa_log_once = true;
            MNN_PRINT("[MetalAttention] flash-attn-prefill kernel active (seq=%d, head_dim=%d, group=%d, mask=%d, outC4=%d, quant_k=%d, quant_v=%d).\n",
                      mSeqLen, mHeadDim, group_size, (int)mHasTensorMask, (int)mOutputC4,
                      (int)mQuantKey, (int)mQuantValue);
        }
    }

    // Fused prefill attention on the Metal tensor API (prefill_flash_attn_tc),
    // env-gated (MNN_METAL_PREFILL_FA_TENSORAPI, default on for causal models).
    // Registers-resident S/O, so the O(n^2) score matrix is never materialized.
    // Must be decided BEFORE handleKVAllocMemory() so the mTempQK / mTempSoftMax
    // scratch is skipped (same constraint as mFlashAttnPrefill).
    // Scope: fp16, head_dim 64/128/256, causal semantics (ADD mask or kv-cache
    // default causal), fp16 KV.
    {
        mFaTcPrefill = false;
        int faEnvMode = MetalEnv::get().prefillFaTensorApi;
        if (faEnvMode < 0) {
            // Unset -> follow the data-driven causal layout: enable for standard
            // causal masks (scalar sentinel / kv-cache), off for arbitrary masks.
            // This kernel hard-codes causal with no opt-out of its own, so it must
            // never see a non-causal mask. Further gated below by
            // isSupportTensorCoopInput() (M5+), so this is a no-op on M4/M3.
            faEnvMode = mCausalLayout ? 1 : 0;
        }
        // Arbitrary masks (mCausalLayout==false) must never reach faTc even when
        // MNN_METAL_PREFILL_FA_TENSORAPI=1 force-enables it.
        const bool faCausal = mCausalLayout;
        const bool faCommon = mtbn->useFp16InsteadFp32() && faCausal && !mKvInDisk &&
                              !mQuantKey && !mQuantValue && mKvSeqLen >= mSeqLen &&
                              (mHeadDim == 64 || mHeadDim == 128 || mHeadDim == 256);
        if (faEnvMode == 1 && !mFaTcUnavailable) {
            // matmul2d input cooperative tensors are single-simdgroup only.
            mFaTcPrefill = mtbn->isSupportTensorCoopInput() && faCommon && mSeqLen >= 64;
        }
        if (mFaTcPrefill) {
            mFlashAttnPrefill = false;
        }
    }

    // M4 fused prefill (prefill_flash_attn_sg). Scores stay in simdgroup
    // fragments; mTempQK/mTempSoftMax are not allocated. Must be decided
    // BEFORE handleKVAllocMemory(). Generic auto-on remains seq>=1024; the
    // verified M4 Pro 32q/8kv/head_dim128 shape starts at seq512. Shorter
    // prefills stay on three-stage. MNN_METAL_PREFILL_FA_SG=1 force-enables and
    // =0 disables the path. Legacy FA force-on keeps that path instead.
    {
        mFaSgPrefill = false;
        int sgEnv = MetalEnv::get().prefillFaSg;
        const bool sgCommon = mtbn->useFp16InsteadFp32() && mCausalLayout && !mKvInDisk &&
                              !mQuantKey && !mQuantValue && mKvSeqLen >= mSeqLen &&
                              (mHeadDim == 64 || mHeadDim == 128) && !mShortSeq &&
                              mSeqLen >= 64 && supportSimdMatrix &&
                              (group_size == 1 || group_size == 2 || group_size == 4 || group_size == 8);
        const bool m4Pro4BSeq512 = rt->isHighBandwidthM4() && mSeqLen >= 512 &&
                                      mNumHead == 32 && mKvNumHead == 8 && mHeadDim == 128;
        const bool autoM4 = (sgEnv < 0) && !mtbn->isSupportTensorCoopInput() &&
                            (mSeqLen >= 1024 || m4Pro4BSeq512);
        const bool wantSg = (sgEnv == 1) || autoM4;
        const bool legacyFaForced = MetalEnv::get().flashAttnPrefill == 1;
        if (wantSg && sgCommon && !mFaTcPrefill && !mFaSgUnavailable && !legacyFaForced) {
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

    if (mSdpaNtg > 1) {
        constexpr auto allocType = Backend::DYNAMIC_IN_EXECUTION;
        const int rows = mBatch * mNumHead * mSdpaNtg;
        // The shader reads/writes these as `device float*`. A float-typed tensor
        // would be allocated at sizeof(half) per element in fp16 mode, so size
        // them in raw bytes.
        const int outBytes = rows * mHeadDim * (int)sizeof(float);
        const int smBytes = rows * 2 * (int)sizeof(float);
        // Only re-create on a size change: these are setTensor-bound, and a
        // recorded encode-replay would otherwise hold a freed Tensor* (same
        // lifetime invariant as mTempK/mTempV in handleKVAllocMemory).
        if (nullptr == mSdpaPartialOut || mSdpaPartialOut->elementSize() != outBytes) {
            mSdpaPartialOut.reset(Tensor::createDevice<uint8_t>({outBytes}));
        }
        if (nullptr == mSdpaPartialSm || mSdpaPartialSm->elementSize() != smBytes) {
            mSdpaPartialSm.reset(Tensor::createDevice<uint8_t>({smBytes}));
        }
        auto res = backend()->onAcquireBuffer(mSdpaPartialOut.get(), allocType) &&
                   backend()->onAcquireBuffer(mSdpaPartialSm.get(), allocType);
        if (!res) {
            MNN_ERROR("MNN::Metal: OUT_OF_MEMORY for split-KV partials, using single pass\n");
            mSdpaNtg = 1;
        } else {
            backend()->onReleaseBuffer(mSdpaPartialOut.get(), allocType);
            backend()->onReleaseBuffer(mSdpaPartialSm.get(), allocType);
        }
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
    // Gated on mCausalLayout (standard causal mask), the simdgroup-matrix tile
    // path, in-memory KV, and kv >= q so the diagonal offset D is non-negative.
    {
        mQkCausalTri = mCausalLayout && !mShortSeq && (mQkSimdMatrix || mQkTensorMatrix) &&
                       !mFlashAttnPrefill && !mFaTcPrefill && !mFaSgPrefill && !mKvInDisk &&
                       mKvSeqLen >= mSeqLen;
        // CAUSAL_BOUND is path-agnostic: activates on both simd-matrix (M4 and
        // below) and tensor-API (M5+) three-kernel prefill paths, so long as we
        // are not on the FA path and the causal-mask semantics hold.
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
    sig |= (uint32_t)mFaTcPrefill << 29;
    // SDPA_QH_PER_TG changes both the compiled kernel and the dispatch grid.
    sig |= (uint32_t)((mSdpaQhPerTg == 8 ? 3 : (mSdpaQhPerTg == 4 ? 2 : (mSdpaQhPerTg == 2 ? 1 : 0))) & 0x3) << 30;
    return sig;
}

void AttentionBufExecution::_writeCopyParam(const Tensor* key, const Tensor* value) {
    auto copyp = (CopyParam*)mParamCopy.contents;
    /*
     Key -> K-Cache :   [mBatch, mKvSeqLen, mKvNumHead, mHeadDim] -> [mKvMaxLen, mBatch, mKvNumHead, mHeadDim]
     Value -> V-Cache : [mBatch, mKvSeqLen, mKvNumHead, mHeadDim] -> [mKvMaxLen, mBatch, mKvNumHead, mHeadDim]
     */
    copyp->head_count = mKvNumHead * mHeadDim;
    // current new kv_len
    copyp->kv_seq_len = key->shape()[1];
    copyp->max_kv_len = mKvMaxLen;
    int pastLength = mKVCache ? mKVCacheManager->kvLength() : 0;
    copyp->dst_k_offset = pastLength * copyp->head_count;
    copyp->dst_v_offset = pastLength * copyp->head_count;
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
}

void AttentionBufExecution::_writeQKVParam(const std::vector<Tensor*>& inputs, int seqLenPiece) {
    int group_size = mNumHead / mKvNumHead;
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
    param->mask_batch = mHasTensorMask ? inputs[3]->length(0) : 1;
    param->mask_head_num = (mHasTensorMask && inputs[3]->dimensions() > 3) ? inputs[3]->length(1) : 1;
    // q/k are the mask's two trailing dims at any rank: [b,h,q,k] / [b,q,k] / [q,k].
    param->mask_q_len = (mHasTensorMask && inputs[3]->dimensions() > 2) ? inputs[3]->length(inputs[3]->dimensions() - 2) : 1;
    param->mask_k_len = (mHasTensorMask && inputs[3]->dimensions() > 0) ? inputs[3]->length(inputs[3]->dimensions() - 1) : 1;
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

void AttentionBufExecution::_writeSoftmaxParam(int seqLenPiece) {
    // [mBatch, mNumHead, mSeqLen, mKvSeqLen]
    int inside = 1;
    int outside = mBatch * mNumHead * seqLenPiece;
    int axis = mKvSeqLen;
    int axis_align = ROUND_UP(axis, mKvAlignNum);
    auto softmax = (int*)mParamSoftmax.contents;
    // Inside, axis, outside, plane(invalid)
    softmax[0] = inside;
    softmax[1] = axis;
    softmax[2] = outside;
    softmax[3] = axis_align;
    // CAUSAL_BOUND fields (ignored by non-causal softmax variants). causal_base
    // stays piece-independent: this buffer is shared by every q piece encoded
    // into one command buffer, so a per-piece value written here would be
    // overwritten before the earlier dispatches run. The piece offset is bound
    // per dispatch as seq_idx instead.
    softmax[4] = seqLenPiece;
    softmax[5] = (mKvSeqLen - mSeqLen) + 1;
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
    {
        _writeCopyParam(key, value);
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
        encoder = mtbn->profileNextSubpass(mSdpaSinglePass ? "sdpa_fused" : (mShortSeq ? "qk_short" : (mDecodeQkSoftmax ? "qk_softmax_fused" : "qk")));
    }
#endif

    // Update Parameters
    int seqLenPiece = UP_DIV(mSeqLen, mQseqSplitNum);
    _writeQKVParam(inputs, seqLenPiece);

    for (int seq_idx = 0; seq_idx < mQseqSplitNum; seq_idx++) {
        if (mFaTcPrefill) {
            // Fused prefill on the tensor API: one dispatch, S and O in registers.
            [encoder setComputePipelineState:mKernel_faTc];
            MetalBackend::setTensor(query, encoder, 0);
            MetalBackend::setTensor(outputs[0], encoder, 1);
            MetalBackend::setTensor(tempTensorK, encoder, 2);
            MetalBackend::setTensor(tempTensorV, encoder, 3);
            [encoder setBuffer:mParamQKV offset:0 atIndex:4];
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
            [encoder setBuffer:mParamQKV offset:0 atIndex:4];
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
            [encoder setBuffer:mParamQKV offset:0 atIndex:4];
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
            // Fused decode SDPA. mSdpaNtg == 1: one kernel, no reduce, no
            // partials. mSdpaNtg > 1: pass 1 writes per-threadgroup (S, m) and
            // unnormalized O partials, pass 2 recombines them.
            const int ntg = mSdpaNtg;
            const bool twoPass = ntg > 1;
            [encoder setComputePipelineState:mKernel_sdpa];
            MetalBackend::setTensor(query, encoder, 0);
            // Bound even in two-pass mode: pass 1 still declares buffer 1, it
            // just writes the partials instead.
            MetalBackend::setTensor(outputs[0], encoder, 1);
            MetalBackend::setTensor(tempTensorK, encoder, 2);
            MetalBackend::setTensor(tempTensorV, encoder, 3);
            [encoder setBuffer:mParamQKV offset:0 atIndex:4];
            [encoder setBytes:&ntg length:sizeof(ntg) atIndex:5];
            if (twoPass) {
                MetalBackend::setTensor(mSdpaPartialOut.get(), encoder, 6);
                MetalBackend::setTensor(mSdpaPartialSm.get(), encoder, 7);
            }
            if (mQuantKey && mKVCacheManager->getKScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getKScaleBuffer() offset:0 atIndex:8];
            }
            if (mQuantValue && mKVCacheManager->getVScaleBuffer() != nil) {
                [encoder setBuffer:mKVCacheManager->getVScaleBuffer() offset:0 atIndex:9];
            }
            // Pass 1: one threadgroup per (q-head group, kv slice). Grouping
            // divides grid.y; the split-KV slices multiply grid.x.
            const int gridGroups = mBatch * (mNumHead / mSdpaQhPerTg);
            [encoder dispatchThreadgroups:MTLSizeMake(ntg, gridGroups, 1)
                    threadsPerThreadgroup:MTLSizeMake(32 * mSdpaNsg, 1, 1)];
            if (twoPass) {
                // The reduce is indexed per individual q-head row, not per
                // group: pass 1 publishes one partial row per q head it owns.
                const int gridRows = mBatch * mNumHead;
                [encoder setComputePipelineState:mKernel_sdpaReduce];
                MetalBackend::setTensor(outputs[0], encoder, 1);
                [encoder setBuffer:mParamQKV offset:0 atIndex:4];
                [encoder setBytes:&ntg length:sizeof(ntg) atIndex:5];
                MetalBackend::setTensor(mSdpaPartialOut.get(), encoder, 6);
                MetalBackend::setTensor(mSdpaPartialSm.get(), encoder, 7);
                [encoder dispatchThreadgroups:MTLSizeMake(1, gridRows, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            // No trailing profileNextSubpass here: the subtag was already set to
            // "sdpa_fused" after the copy flush, and profileOpEncoded settles the
            // encoder under that name (a trailing call would add a same-named
            // empty encoder and dilute the average).
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
            [encoder setBuffer:mParamQKV offset:0 atIndex:4];
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
            _writeSoftmaxParam(seqLenPiece);
            [encoder setComputePipelineState:mKernel_softmax];
            // [mBatch, mNumHead, mSeqLen, mKvSeqLen]
            MetalBackend::setTensor(mTempQK.get(), encoder, 0);
            // [mBatch, mNumHead, mSeqLen, ROUND_UP(mKvSeqLen, mKvAlignNum)]
            MetalBackend::setTensor(mTempSoftMax.get(), encoder, 1);
            [encoder setBuffer:mParamSoftmax offset:0 atIndex:2];
            if (mCausalBound) {
                // setBytes copies into the encoder, so unlike mParamSoftmax this
                // survives the later pieces of the same command buffer.
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
    // Replay is only maintained for the in-memory KV-cache decode paths with a
    // single seq piece. Anything else (prefill, disk KV, flash-attn, split
    // pieces) bails: the execution drops the recording and re-encodes normally.
    // (Checks here use last-encode state; the signature check below is the
    // authoritative one — it recomputes every flag for THIS token first.)
    if (!mKVCache || mKvInDisk || mFlashAttnPrefill || mFaTcPrefill || mFaSgPrefill || mQseqSplitNum != 1) {
        return false;
    }
    // Recompute all per-token path state (kv length, split-kv / fused-decode
    // switches, simd flags) exactly as onEncode would. Idempotent when the
    // caller falls back to onEncode after a bail: handleKVAllocMemory's
    // onRealloc is a no-op once capacity suffices.
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
    // KV expansion (possibly just performed by _computePathFlags above)
    // DESTROYS the old cache tensors (mPastKey.reset), leaving dangling tensor
    // pointers in the recorded bindings. Detect the swap by tensor identity —
    // buffer identity is NOT reliable (pool suballocation can keep the same
    // MTLBuffer). The stored pointers are compared, never dereferenced.
    if (mKVCacheManager->getKeyTensor() != mLastKTensor ||
        mKVCacheManager->getValueTensor() != mLastVTensor) {
        return false;
    }
    // Structural layout must match the recording before patching by position.
    const int expectEvents = mSdpaSinglePass ? (mSdpaNtg > 1 ? 3 : 2) : (mDecodeQkSoftmax ? 3 : 4);
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

    // Rewrite param-buffer contents consumed by the recorded dispatches.
    _writeCopyParam(inputs[1], inputs[2]);
    _writeQKVParam(inputs, seqLenPiece);
    _writeSoftmaxParam(seqLenPiece);

    auto patchIntBytes = [](MetalReplayEvent& e, int index, int value) {
        for (auto& by : e.bytesArgs) {
            if (by.first == index && by.second.size() == sizeof(int)) {
                memcpy(by.second.data(), &value, sizeof(int));
            }
        }
    };
    // events[0] is the KV copy: its grid is kv-length independent.
    if (mSdpaSinglePass) {
        // events[1] = fused sdpa: grid (ntg, B*heads) and the ntg bytes arg are
        // both kv-length independent (ntg is frozen at resize); kv reaches the
        // kernel via the param rewrite. events[2], when ntg > 1, is the reduce
        // pass with the equally kv-independent grid (1, B*heads).
    } else if (mDecodeQkSoftmax) {
        // events[1] = fused qk_softmax: threadgroup width tracks kv.
        // events[2] = qkv: grid is kv-length independent.
        // (QK_QSPLIT: recorded grid already carries z=group_size; only the
        // qkGroups count feeding the width heuristic changes.)
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
        return new AttentionBufExecution(backend, param->kv_cache(), param->output_c4(), param->attnScale(), quantParam);
    }
};
REGISTER_METAL_OP_TRANSFORMER_CREATOR(AttentionBufCreator, OpType_Attention);

} // namespace MNN
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif
