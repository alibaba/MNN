//
//  MetalLinearAttention.mm
//  MNN
//
//  Created by MNN on 2026/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalLinearAttention.hpp"
#import "MNNMetalContext.h"
#import "MetalLinearAttentionShader.hpp"
#import "MetalEnv.hpp"
#import "core/TensorUtils.hpp"

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {

// Must match LinearAttnParam in MetalLinearAttentionShader.hpp
struct LinearAttnParam {
    int batch;
    int conv_dim;
    int seq_len;
    int kernel_size;
    int conv_state_size;
    int num_k_heads;
    int num_v_heads;
    int head_k_dim;
    int head_v_dim;
    int key_dim;
    int val_dim;
    int gqa_factor;
    int use_l2norm;
    int qkv_c4;
    int gate_c4;
    int beta_c4;
    int output_c4;
    float q_scale;
    int commit_len;    // speculative rollback: pending tokens to replay before this block
    int pending_seq;   // length of the saved pending block
    int lazy_mode;     // 1: do not persist state after the new block (spec verify)
    // gate/beta chain fold (gate_c4 & 2 / beta_c4 & 2): per-head constants
    // -exp(A_log)[h] and dt_bias[h]; capped at kMaxFoldHeads heads.
    float gate_coef[64];
    float gate_bias[64];
};
// Capacity of the fold constant arrays above; a model with more v-heads cannot
// use the fold (checked at create time).
static constexpr int kMaxFoldHeads = (int)(sizeof(LinearAttnParam::gate_coef) / sizeof(float));

static void linearAttentionDims(const Tensor* qkv, int& batch, int& convDim, int& seqLen) {
    if (TensorUtils::getDescribe(qkv)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        batch = 1;
        seqLen = qkv->length(0);
        convDim = qkv->length(1);
        return;
    }
    batch = qkv->length(0);
    convDim = qkv->length(1);
    seqLen = qkv->length(2);
}

// (Re)allocate a STATIC device buffer; only the rare block-size change reallocates.
static bool reacquireStatic(Backend* bn, std::shared_ptr<Tensor>& tensor, int elements) {
    if (tensor.get() != nullptr) {
        bn->onReleaseBuffer(tensor.get(), Backend::STATIC);
    }
    tensor.reset(Tensor::createDevice<float>({ALIMAX(elements, 1)}));
    return bn->onAcquireBuffer(tensor.get(), Backend::STATIC);
}

MetalLinearAttention::MetalLinearAttention(Backend *backend, const MNN::Op* op)
    : MetalExecution(backend) {
    auto param = op->main_as_LinearAttentionParam();
    mAttentionType = param->attn_type()->str();
    mNumKHeads = param->num_k_heads();
    mNumVHeads = param->num_v_heads();
    mHeadKDim = param->head_k_dim();
    mHeadVDim = param->head_v_dim();
    mUseQKL2Norm = param->use_qk_l2norm();
    // short_conv never touches gate/beta, so the fold is a no-op there.
    mGateFold = param->gate_fold() && mAttentionType != "short_conv";
    if (mGateFold) {
        // The folded graph no longer contains the gate chain, so neither of the
        // rejections below has a fallback: running unfolded would consume the
        // raw `a` projection as the decay gate.
        if (param->gate_coef() == nullptr || param->gate_bias() == nullptr ||
            (int)param->gate_coef()->size() != mNumVHeads || (int)param->gate_bias()->size() != mNumVHeads) {
            MNN_ERROR("MetalLinearAttention: gate_fold set but gate_coef/gate_bias missing or wrong size\n");
            mValid = false;
            return;
        }
        if (mNumVHeads > kMaxFoldHeads) {
            MNN_ERROR("MetalLinearAttention: gate_fold needs num_v_heads <= %d, got %d\n", kMaxFoldHeads, mNumVHeads);
            mValid = false;
            return;
        }
        mGateCoef.assign(param->gate_coef()->begin(), param->gate_coef()->end());
        mGateBias.assign(param->gate_bias()->begin(), param->gate_bias()->end());
    }
    mStateCache.reset(new MetalStateCache);

    auto mtbn = static_cast<MetalBackend *>(backend);
    mMeta = (KVMeta*)(mtbn->getMetaPtr());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mParamBuffer = [context newDeviceBuffer:sizeof(LinearAttnParam) access:CPUWriteOnly];
    mParamBufferFlush = [context newDeviceBuffer:sizeof(LinearAttnParam) access:CPUWriteOnly];

    // Compile shader pipelines
    MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
    auto rt = (MetalRuntime *)mtbn->runtime();
    bool useFp16 = mtbn->useFp16InsteadFp32();
    if (useFp16) {
        option.preprocessorMacros = @{@"MNN_METAL_FLOAT16_STORAGE" : @"1"};
    }

    if (mAttentionType == "short_conv") {
        std::vector<std::string> commonKeys;
        if (useFp16) {
            commonKeys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
        }
        auto buildShortPipeline = [&](const char* kernel) -> id<MTLComputePipelineState> {
            auto keys = commonKeys;
            keys.insert(keys.begin(), kernel);
            id<MTLComputePipelineState> pipeline = rt->findPipeline(keys);
            if (nil == pipeline) {
                pipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnShortConv, kernel, option);
                rt->insertPipeline(keys, pipeline);
            }
            return pipeline;
        };
        mShortConvPipeline = buildShortPipeline("linear_attn_short_conv_nosilu");
        mShortConvStateUpdatePipeline = buildShortPipeline("linear_attn_short_conv_state_update");
        mShortConvOutputPipeline = buildShortPipeline("linear_attn_short_conv_output");
        return;
    }

    // ── Conv + SiLU pipeline (used every forward) ──────────────────────
    {
        std::vector<std::string> keys = {"linear_attn_conv_silu"};
        if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
        mConvSiluPipeline = rt->findPipeline(keys);
        if (nil == mConvSiluPipeline) {
            mConvSiluPipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnConvSilu, "linear_attn_conv_silu", option);
            rt->insertPipeline(keys, mConvSiluPipeline);
        }
    }
    {
        std::vector<std::string> keys = {"linear_attn_conv_silu_state_decode"};
        if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
        mConvSiluStateDecodePipeline = rt->findPipeline(keys);
        if (nil == mConvSiluStateDecodePipeline) {
            mConvSiluStateDecodePipeline = mtbn->makeComputePipelineWithSourceOption(
                gLinearAttnConvSilu, "linear_attn_conv_silu_state_decode", option);
            if (nil != mConvSiluStateDecodePipeline) {
                rt->insertPipeline(keys, mConvSiluStateDecodePipeline);
            }
        }
    }
    {
        std::vector<std::string> keys = {"linear_attn_conv_state_update"};
        if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
        mConvStateUpdatePipeline = rt->findPipeline(keys);
        if (nil == mConvStateUpdatePipeline) {
            mConvStateUpdatePipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnConvSilu, "linear_attn_conv_state_update", option);
            rt->insertPipeline(keys, mConvStateUpdatePipeline);
        }
    }
    {
        std::vector<std::string> keys = {"linear_attn_conv_state_commit"};
        if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
        mConvCommitPipeline = rt->findPipeline(keys);
        if (nil == mConvCommitPipeline) {
            mConvCommitPipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnConvSilu, "linear_attn_conv_state_commit", option);
            rt->insertPipeline(keys, mConvCommitPipeline);
        }
    }
    {
        std::vector<std::string> keys = {"linear_attn_qkvraw_save"};
        if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
        mQKVRawSavePipeline = rt->findPipeline(keys);
        if (nil == mQKVRawSavePipeline) {
            mQKVRawSavePipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnConvSilu, "linear_attn_qkvraw_save", option);
            rt->insertPipeline(keys, mQKVRawSavePipeline);
        }
    }
    // ── QKV prep pipelines: scalar (baseline) and simdgroup (short prefill) ──
    {
        std::vector<std::string> keys = {"linear_attn_qkv_prep"};
        if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
        mQKVPrepPipeline = rt->findPipeline(keys);
        if (nil == mQKVPrepPipeline) {
            mQKVPrepPipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnGatedDeltaRule, "linear_attn_qkv_prep", option);
            rt->insertPipeline(keys, mQKVPrepPipeline);
        }
    }
    // Simdgroup-optimized QKV prep: used for short prefill (2 <= L < 16).
    // Uses compile-time head dims for register-resident buffers.
    if (rt->supportSimdGroupReduce()) {
        constexpr int kMaxHeadDim = 512;
        if (mHeadKDim > 0 && mHeadKDim <= kMaxHeadDim &&
            mHeadVDim > 0 && mHeadVDim <= kMaxHeadDim) {
            const int simdItersK = (mHeadKDim + 31) / 32;
            const int simdItersV = (mHeadVDim + 31) / 32;
            MTLCompileOptions *qkvOpt = [[MTLCompileOptions alloc] init];
            NSMutableDictionary *qkvMacros = [NSMutableDictionary dictionary];
            if (useFp16) qkvMacros[@"MNN_METAL_FLOAT16_STORAGE"] = @"1";
            qkvMacros[@"HEAD_K_DIM"]   = [NSString stringWithFormat:@"%d", mHeadKDim];
            qkvMacros[@"HEAD_V_DIM"]   = [NSString stringWithFormat:@"%d", mHeadVDim];
            qkvMacros[@"SIMD_ITERS_K"] = [NSString stringWithFormat:@"%d", simdItersK];
            qkvMacros[@"SIMD_ITERS_V"] = [NSString stringWithFormat:@"%d", simdItersV];
            qkvOpt.preprocessorMacros = qkvMacros;

            std::string qkvKey = "QKV_PREP_SG_dk" + std::to_string(mHeadKDim) +
                                 "_dv" + std::to_string(mHeadVDim);
            std::vector<std::string> keys = {"linear_attn_qkv_prep_sg", qkvKey};
            if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
            mQKVPrepSGPipeline = rt->findPipeline(keys);
            if (nil == mQKVPrepSGPipeline) {
                mQKVPrepSGPipeline = mtbn->makeComputePipelineWithSourceOption(
                    gLinearAttnQKVPrepSG, "linear_attn_qkv_prep_sg", qkvOpt);
                if (nil != mQKVPrepSGPipeline) rt->insertPipeline(keys, mQKVPrepSGPipeline);
            }
        }
    }

    // ── Simdgroup-based delta / decode pipelines ────────────────────────
    mUseSimdGroupOpt = rt->supportSimdGroupReduce();
    if (mUseSimdGroupOpt) {
        int simdIters = (mHeadKDim + 31) / 32;
        NSString *simdItersStr = [NSString stringWithFormat:@"%d", simdIters];
        MTLCompileOptions *sgOption = [[MTLCompileOptions alloc] init];
        NSMutableDictionary *sgMacros = [NSMutableDictionary dictionary];
        if (useFp16) sgMacros[@"MNN_METAL_FLOAT16_STORAGE"] = @"1";
        sgMacros[@"SIMD_ITERS"] = simdItersStr;
        sgOption.preprocessorMacros = sgMacros;

        std::string simdItersKey = "SIMD_ITERS_" + std::to_string(simdIters);
        // Simdgroup delta rule (used by short prefill unfused path)
        {
            std::vector<std::string> keys = {"linear_attn_gated_delta_rule_sg", simdItersKey};
            if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
            mGatedDeltaRuleSGPipeline = rt->findPipeline(keys);
            if (nil == mGatedDeltaRuleSGPipeline) {
                mGatedDeltaRuleSGPipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnGatedDeltaRuleSG, "linear_attn_gated_delta_rule_sg", sgOption);
                rt->insertPipeline(keys, mGatedDeltaRuleSGPipeline);
            }
        }
        {
            std::vector<std::string> keys = {"linear_attn_verify_fused_sg", simdItersKey};
            if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
            mVerifyFusedSGPipeline = rt->findPipeline(keys);
            if (nil == mVerifyFusedSGPipeline) {
                mVerifyFusedSGPipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnGatedDeltaRuleSG, "linear_attn_verify_fused_sg", sgOption);
                rt->insertPipeline(keys, mVerifyFusedSGPipeline);
            }
        }
        // dk==128 vectorized variant (ftype4 lanes, register state)
        if (mHeadKDim == 128) {
            std::vector<std::string> keys = {"linear_attn_gated_delta_rule_sg_v4", simdItersKey};
            if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
            mGatedDeltaRuleSGV4Pipeline = rt->findPipeline(keys);
            if (nil == mGatedDeltaRuleSGV4Pipeline) {
                mGatedDeltaRuleSGV4Pipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnGatedDeltaRuleSG, "linear_attn_gated_delta_rule_sg_v4", sgOption);
                if (nil != mGatedDeltaRuleSGV4Pipeline) {
                    rt->insertPipeline(keys, mGatedDeltaRuleSGV4Pipeline);
                }
            }
        }
        // Master baseline fused decode kernel (fallback for fused_sg_align)
        {
            std::vector<std::string> keys = {"linear_attn_fused_sg", simdItersKey};
            if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
            mGatedDeltaRuleFusedSGPipeline = rt->findPipeline(keys);
            if (nil == mGatedDeltaRuleFusedSGPipeline) {
                mGatedDeltaRuleFusedSGPipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnFusedSG, "linear_attn_fused_sg", sgOption);
                rt->insertPipeline(keys, mGatedDeltaRuleFusedSGPipeline);
            }
        }
        // fused_sg_align: register-resident state + D_K_ALIGNED for decode.
        // Preferred over fused_sg for decode (L=1) when available; used as the
        // fallback when the TG-shared-QK variant below is unavailable.
        int dkAligned = (mHeadKDim % 32 == 0) ? 1 : 0;
        NSString *dkAlignedStr = [NSString stringWithFormat:@"%d", dkAligned];
        std::string dkAlignedKey = "D_K_ALIGNED_" + std::to_string(dkAligned);
        {
            MTLCompileOptions *alignOpt = [[MTLCompileOptions alloc] init];
            NSMutableDictionary *alignMacros = [NSMutableDictionary dictionary];
            if (useFp16) alignMacros[@"MNN_METAL_FLOAT16_STORAGE"] = @"1";
            mFusedSGAlignSimds = 8;
            alignMacros[@"SIMD_ITERS"]  = simdItersStr;
            alignMacros[@"D_K_ALIGNED"] = dkAlignedStr;
            alignMacros[@"ALIGN_SIMDS_PER_TG"] =
                [NSString stringWithFormat:@"%d", mFusedSGAlignSimds];
            alignOpt.preprocessorMacros = alignMacros;

            std::vector<std::string> keys = {"linear_attn_fused_sg_align", simdItersKey, dkAlignedKey,
                                              "ALIGN_SIMDS_PER_TG_" + std::to_string(mFusedSGAlignSimds)};
            if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
            mFusedSGAlignPipeline = rt->findPipeline(keys);
            if (nil == mFusedSGAlignPipeline) {
                mFusedSGAlignPipeline = mtbn->makeComputePipelineWithSourceOption(
                    gLinearAttnFusedSGAlign, "linear_attn_fused_sg_align", alignOpt);
                if (nil != mFusedSGAlignPipeline) rt->insertPipeline(keys, mFusedSGAlignPipeline);
            }
        }
        // fused_sg_tg: TG-shared-QK on top of fused_sg_align.
        // Requires d_v % SIMDS_PER_TG (== 4) == 0 so that all 4 SGs of a TG
        // share the same (b, h). Preferred over fused_sg_align for decode.
        mFusedSGTGSimds = 4;
        if (mHeadKDim > 0 && (mHeadVDim % mFusedSGTGSimds == 0)) {
            MTLCompileOptions *tgOpt = [[MTLCompileOptions alloc] init];
            NSMutableDictionary *tgMacros = [NSMutableDictionary dictionary];
            if (useFp16) tgMacros[@"MNN_METAL_FLOAT16_STORAGE"] = @"1";
            tgMacros[@"SIMD_ITERS"]   = simdItersStr;
            tgMacros[@"D_K_ALIGNED"]  = dkAlignedStr;
            tgMacros[@"HEAD_K_DIM"]   = [NSString stringWithFormat:@"%d", mHeadKDim];
            tgMacros[@"SIMDS_PER_TG"] = [NSString stringWithFormat:@"%d", mFusedSGTGSimds];
            tgOpt.preprocessorMacros = tgMacros;

            std::string tgDkKey = "DK_" + std::to_string(mHeadKDim);
            std::vector<std::string> keys = {"linear_attn_fused_sg_tg", simdItersKey,
                                              dkAlignedKey, tgDkKey,
                                              "SIMDS_PER_TG_" + std::to_string(mFusedSGTGSimds)};
            if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
            mFusedSGTGPipeline = rt->findPipeline(keys);
            if (nil == mFusedSGTGPipeline) {
                mFusedSGTGPipeline = mtbn->makeComputePipelineWithSourceOption(
                    gLinearAttnFusedSGTG, "linear_attn_fused_sg_tg", tgOpt);
                if (nil != mFusedSGTGPipeline) rt->insertPipeline(keys, mFusedSGTGPipeline);
            }
        }
        // ── Tensor-op chunk-64 prefill, without sequence-sized scratch ──
        // A/P reuse conv_out's V storage after V is backed up into the output;
        // the recurrent scan overwrites that backup with final results. Two
        // 64x64 matrices require at least 128 V channels.
        mUseFlashChunk = rt->supportTensorOps() && mHeadKDim > 0 && mHeadKDim <= 128 &&
                         mHeadKDim % 16 == 0 && mHeadVDim >= 128 &&
                         mHeadVDim % mFlashDvBlock == 0;
        if (mUseFlashChunk) {
            MTLCompileOptions *flashOption = [[MTLCompileOptions alloc] init];
            NSMutableDictionary *flashMacros = [NSMutableDictionary dictionary];
            if (useFp16) flashMacros[@"MNN_METAL_FLOAT16_STORAGE"] = @"1";
            flashMacros[@"CK_DK"] = [NSString stringWithFormat:@"%d", mHeadKDim];
            flashMacros[@"CK_DV"] = [NSString stringWithFormat:@"%d", mHeadVDim];
            flashMacros[@"CK_CHUNK"] = @"64";
            flashMacros[@"CK_W"] = [NSString stringWithFormat:@"%d", mFlashDvBlock];
            flashOption.preprocessorMacros = flashMacros;

            std::string flashKey = "CHUNK64_INPLACE_" + std::to_string(mHeadKDim) + "_" +
                                   std::to_string(mHeadVDim) + "_DVB" +
                                   std::to_string(mFlashDvBlock);
            {
                std::vector<std::string> keys = {"linear_attn_chunk64_prep_inplace", flashKey};
                if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
                mFlashChunkPrepPipeline = rt->findPipeline(keys);
                if (nil == mFlashChunkPrepPipeline) {
                    mFlashChunkPrepPipeline = mtbn->makeComputePipelineWithSourceOption(
                        gLinearAttnChunk64Inplace, "linear_attn_chunk64_prep_inplace", flashOption);
                    if (nil != mFlashChunkPrepPipeline) {
                        rt->insertPipeline(keys, mFlashChunkPrepPipeline);
                    }
                }
            }
            {
                std::vector<std::string> keys = {"linear_attn_chunk64_recurrent_inplace", flashKey};
                if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
                mFlashChunkScanPipeline = rt->findPipeline(keys);
                if (nil == mFlashChunkScanPipeline) {
                    mFlashChunkScanPipeline = mtbn->makeComputePipelineWithSourceOption(
                        gLinearAttnChunk64Inplace, "linear_attn_chunk64_recurrent_inplace", flashOption);
                    if (nil != mFlashChunkScanPipeline) {
                        rt->insertPipeline(keys, mFlashChunkScanPipeline);
                    }
                }
            }
            const int requiredThreads = mFlashSimdsPerTG * 32;
            if (nil == mFlashChunkPrepPipeline || nil == mFlashChunkScanPipeline ||
                mFlashChunkPrepPipeline.maxTotalThreadsPerThreadgroup < requiredThreads ||
                mFlashChunkScanPipeline.maxTotalThreadsPerThreadgroup < requiredThreads) {
                mFlashChunkPrepPipeline = nil;
                mFlashChunkScanPipeline = nil;
                mUseFlashChunk = false;
            }
        }
        // ── simdgroup_matrix flash chunk prefill (non-tensor-API devices) ──
        // Same chunked algorithm as flash_chunk but with 8x8 fp32 MMA tiles,
        // for M4-class Macs / iPhone where MPP tensor ops are unavailable.
        // Gated to head_k_dim == 128: at dk=64 the scalar fused_chunk_sg
        // baseline (CHUNK_BT=32) measures faster across all L, while dk=128
        // measures +15% e2e prefill (Qwen3.5 0.8B/2B, M4 Pro).
        mUseFlashChunkSGMM = !mUseFlashChunk && rt->supportSimdGroupMatrix() &&
                             mHeadKDim == 128 &&
                             mHeadVDim >= mSgmmDvBlock && mHeadVDim % mSgmmDvBlock == 0;
        if (mUseFlashChunkSGMM) {
            MTLCompileOptions *sgmmOption = [[MTLCompileOptions alloc] init];
            NSMutableDictionary *sgmmMacros = [NSMutableDictionary dictionary];
            if (useFp16) sgmmMacros[@"MNN_METAL_FLOAT16_STORAGE"] = @"1";
            sgmmMacros[@"CHUNK_BT"] = @"16";
            sgmmMacros[@"HEAD_K_DIM"] = [NSString stringWithFormat:@"%d", mHeadKDim];
            sgmmMacros[@"HEAD_V_DIM"] = [NSString stringWithFormat:@"%d", mHeadVDim];
            sgmmMacros[@"DV_BLOCK"] = [NSString stringWithFormat:@"%d", mSgmmDvBlock];
            sgmmMacros[@"SIMDS_PER_TG"] = [NSString stringWithFormat:@"%d", mSgmmSimdsPerTG];
            sgmmOption.preprocessorMacros = sgmmMacros;

            std::string sgmmKey = "SGMM_" + std::to_string(mHeadKDim) + "_" +
                                  std::to_string(mHeadVDim) + "_BT16_DVB" +
                                  std::to_string(mSgmmDvBlock) + "_SG" +
                                  std::to_string(mSgmmSimdsPerTG);
            std::vector<std::string> keys = {"linear_attn_flash_chunk_sgmm", sgmmKey};
            if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
            mFlashChunkSGMMPipeline = rt->findPipeline(keys);
            if (nil == mFlashChunkSGMMPipeline) {
                mFlashChunkSGMMPipeline = mtbn->makeComputePipelineWithSourceOption(
                    gLinearAttnFlashChunkSGMM, "linear_attn_flash_chunk_sgmm", sgmmOption);
                if (nil != mFlashChunkSGMMPipeline) {
                    rt->insertPipeline(keys, mFlashChunkSGMMPipeline);
                }
            }
            if (nil == mFlashChunkSGMMPipeline ||
                mFlashChunkSGMMPipeline.maxTotalThreadsPerThreadgroup < mSgmmSimdsPerTG * 32) {
                mFlashChunkSGMMPipeline = nil;
                mUseFlashChunkSGMM = false;
            }
        }
        // ── Fused chunked prefill fallback ────────────────────────────
        const int simdsPerTG = 4;
        mUseFusedChunkSG = (mHeadVDim % simdsPerTG == 0) && (mHeadVDim >= simdsPerTG);
        if (mUseFusedChunkSG) {
            const int chunkBT = (mHeadKDim >= 128) ? 16 : 32;
            MTLCompileOptions *chunkOption = [[MTLCompileOptions alloc] init];
            NSMutableDictionary *chunkMacros = [NSMutableDictionary dictionary];
            if (useFp16) chunkMacros[@"MNN_METAL_FLOAT16_STORAGE"] = @"1";
            chunkMacros[@"SIMD_ITERS"]   = simdItersStr;
            chunkMacros[@"CHUNK_BT"]     = [NSString stringWithFormat:@"%d", chunkBT];
            chunkMacros[@"HEAD_K_DIM"]   = [NSString stringWithFormat:@"%d", mHeadKDim];
            chunkMacros[@"HEAD_V_DIM"]   = [NSString stringWithFormat:@"%d", mHeadVDim];
            chunkMacros[@"SIMDS_PER_TG"] = [NSString stringWithFormat:@"%d", simdsPerTG];
            chunkOption.preprocessorMacros = chunkMacros;

            std::string chunkKey = "CHUNK_" + std::to_string(mHeadKDim) + "_" +
                                   std::to_string(mHeadVDim) + "_BT" + std::to_string(chunkBT);
            std::vector<std::string> keys = {"linear_attn_fused_chunk_sg", chunkKey};
            if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
            mFusedChunkSGPipeline = rt->findPipeline(keys);
            if (nil == mFusedChunkSGPipeline) {
                mFusedChunkSGPipeline = mtbn->makeComputePipelineWithSourceOption(
                    gLinearAttnFusedChunkSG, "linear_attn_fused_chunk_sg", chunkOption);
                if (nil != mFusedChunkSGPipeline) rt->insertPipeline(keys, mFusedChunkSGPipeline);
            }
            if (nil == mFusedChunkSGPipeline) {
                mUseFusedChunkSG = false;
            } else {
                mChunkTGThreads = simdsPerTG * 32;
            }
        }
    }
    // Scalar delta rule fallback (when SG-reduce is unavailable)
    {
        std::vector<std::string> keys = {"linear_attn_gated_delta_rule"};
        if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
        mGatedDeltaRulePipeline = rt->findPipeline(keys);
        if (nil == mGatedDeltaRulePipeline) {
            mGatedDeltaRulePipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnGatedDeltaRule, "linear_attn_gated_delta_rule", option);
            rt->insertPipeline(keys, mGatedDeltaRulePipeline);
        }
    }
}

ErrorCode MetalLinearAttention::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto qkv = inputs[0];
    int batch = 0, convDim = 0, seqLen = 0;
    linearAttentionDims(qkv, batch, convDim, seqLen);
    mLastSeqLen = seqLen;
    // gate/beta chain fold: (re-)register the request for onResizeEnd
    // matching. Results persist across the per-token forced re-resize (the
    // chain executions and tensors are untouched then); a change of the
    // gate/beta tensors invalidates them.
    if (mAttentionType == "gated_delta_rule") {
        if (mGateFold) {
            // Export-time fold: inputs[1]=raw_a, inputs[2]=raw_b, constants
            // from LinearAttentionParam. Register for onResizeEnd handling —
            // the STATIC re-home must NOT happen here in onResize: the
            // pipeline's resize sweep releases a consumer's input memory when
            // its useCount exhausts (Pipeline.cpp _releaseTensor), freeing a
            // STATIC home acquired this early before encode runs.
            if (mFoldReq.rawA != inputs[1] || mFoldReq.rawB != inputs[2]) {
                mFoldReq = MetalBackend::LinearAttnFoldRequest();
                mFoldReq.rawA = inputs[1];
                mFoldReq.rawB = inputs[2];
                mFoldReq.gateCoef = mGateCoef;
                mFoldReq.gateBias = mGateBias;
                mFoldReq.exportFold = true;
            }
            mFoldReq.numHeads = mNumVHeads;
            static_cast<MetalBackend*>(backend())->registerLinearAttnFold(&mFoldReq);
        } else {
            if (mFoldReq.gate != inputs[1] || mFoldReq.beta != inputs[2]) {
                mFoldReq = MetalBackend::LinearAttnFoldRequest();
                mFoldReq.gate = inputs[1];
                mFoldReq.beta = inputs[2];
            }
            mFoldReq.numHeads = mNumVHeads;
            static_cast<MetalBackend*>(backend())->registerLinearAttnFold(&mFoldReq);
        }
    }
    int K_conv = inputs[3]->length(2);
    int convStateSize = K_conv - 1;
    int H = mNumVHeads;
    int dk = mHeadKDim;
    int dv = mHeadVDim;
    int convChannels = mAttentionType == "short_conv" ? mHeadVDim : convDim;
    bool needRecurrentState = mAttentionType != "short_conv";

    // ─── Persistent state buffers (STATIC): allocate once, shared via onClone ───
    auto mtbn = static_cast<MetalBackend *>(backend());
    int bytesPerElement = mtbn->useFp16InsteadFp32() ? 2 : 4;
    const bool needConvStateInit = mStateCache->mConvState.get() == nullptr;
    const bool needRecurrentStateInit = needRecurrentState && mStateCache->mRecurrentState.get() == nullptr;
    if (needConvStateInit || needRecurrentStateInit) {
        // First time: allocate and zero-initialize
        if (needConvStateInit) {
            int convStateTotal = ALIMAX(batch * convChannels * convStateSize, 1);
            mStateCache->mConvState.reset(Tensor::createDevice<float>({convStateTotal}));
            bool success = backend()->onAcquireBuffer(mStateCache->mConvState.get(), Backend::STATIC);
            if (!success) return OUT_OF_MEMORY;
            auto convDevice = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)mStateCache->mConvState->deviceId())->getBuffer();
            auto convPtr = (uint8_t*)convDevice.contents + TensorUtils::getDescribeOrigin(mStateCache->mConvState.get())->offset;
            ::memset(convPtr, 0, convStateTotal * bytesPerElement);
        }

        if (needRecurrentStateInit) {
            mStateCache->mRecurrentState.reset(Tensor::createDevice<float>({batch, H, dk, dv}));
            bool success = backend()->onAcquireBuffer(mStateCache->mRecurrentState.get(), Backend::STATIC);
            if (!success) return OUT_OF_MEMORY;
            auto rnnDevice = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)mStateCache->mRecurrentState->deviceId())->getBuffer();
            auto rnnPtr = (uint8_t*)rnnDevice.contents + TensorUtils::getDescribeOrigin(mStateCache->mRecurrentState.get())->offset;
            ::memset(rnnPtr, 0, batch * H * dk * dv * bytesPerElement);
        }
    } else if (seqLen > 1) {
        // Prefill: reset state for new sequence, UNLESS:
        // 1. Loading from prefix cache (PendingRead), or
        // 2. Reusing KV from previous inference (reuse_kv=true, i.e. previous != remove)
        bool loadingFromDisk = (mMeta != nullptr && mMeta->file_flag == KVMeta::PendingRead && mMeta->file_name.size() > 0);
        bool reusingKV = (mMeta != nullptr && mMeta->previous != mMeta->remove);
        if (!loadingFromDisk && !reusingKV) {
            mStateCache->mPendingLen = 0;
            if (mStateCache->mConvState.get() != nullptr) {
                auto convDevice = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)mStateCache->mConvState->deviceId())->getBuffer();
                auto convPtr = (uint8_t*)convDevice.contents + TensorUtils::getDescribeOrigin(mStateCache->mConvState.get())->offset;
                ::memset(convPtr, 0, mStateCache->mConvState->elementSize() * bytesPerElement);
            }
            if (mStateCache->mRecurrentState.get() != nullptr) {
                auto rnnDevice = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)mStateCache->mRecurrentState->deviceId())->getBuffer();
                auto rnnPtr = (uint8_t*)rnnDevice.contents + TensorUtils::getDescribeOrigin(mStateCache->mRecurrentState.get())->offset;
                ::memset(rnnPtr, 0, mStateCache->mRecurrentState->elementSize() * bytesPerElement);
            }
        }
    }
    // Decode (seqLen == 1): keep existing state untouched

    // Sized to the verify block and allocated on first use: a plain AR session never pays for them.
    const bool specVerify = mMeta != nullptr && mMeta->spec_block > 0 && needRecurrentState;
    if (specVerify) {
        if (nil == mVerifyFusedSGPipeline) {
            // Fail loudly: without the fused kernel a verify block would commit rejected tokens.
            MNN_ERROR("MetalLinearAttention: speculative verify needs the simdgroup verify kernel\n");
            return NOT_SUPPORT;
        }
        if (seqLen != mMeta->spec_block) {
            MNN_ERROR("MetalLinearAttention: seq_len %d does not match spec_block %d\n", seqLen, mMeta->spec_block);
            return INPUT_DATA_ERROR;
        }
        if (mStateCache->mPendingCap != seqLen) {
            auto* sc = mStateCache.get();
            const int cap = seqLen;
            auto bn = backend();
            bool ok = reacquireStatic(bn, sc->mPendingQKVRaw, batch * convDim * cap)
                   && reacquireStatic(bn, sc->mPendingK,      batch * cap * H * dk)
                   && reacquireStatic(bn, sc->mPendingV,      batch * cap * H * dv)
                   && reacquireStatic(bn, sc->mPendingGate,   batch * cap * H)
                   && reacquireStatic(bn, sc->mPendingBeta,   batch * cap * H)
                   && reacquireStatic(bn, sc->mPendingK2,     batch * cap * H * dk)
                   && reacquireStatic(bn, sc->mPendingV2,     batch * cap * H * dv)
                   && reacquireStatic(bn, sc->mPendingGate2,  batch * cap * H)
                   && reacquireStatic(bn, sc->mPendingBeta2,  batch * cap * H);
            if (!ok) {
                sc->mPendingCap = 0;
                return OUT_OF_MEMORY;
            }
            sc->mPendingCap = cap;
            sc->mPendingLen = 0;
            sc->mPendingIdx = 0;
        }
    }

    // Pipeline force-resizes LinearAttention every decode token. Keep the
    // mConvOut Tensor object alive when its shape is unchanged so encode-replay
    // recordings never hold a dangling Tensor*; buffer/offset drift after an
    // allocator re-plan is caught by metalReplayValidate.
    const bool convOutChanged = mConvOut.get() == nullptr ||
                                mConvOut->length(0) != batch ||
                                mConvOut->length(1) != convChannels ||
                                mConvOut->length(2) != seqLen;
    if (convOutChanged) {
        mConvOut.reset(Tensor::createDevice<float>({batch, convChannels, seqLen}));
        mResizeGeneration++;
    }
    bool success = backend()->onAcquireBuffer(mConvOut.get(), Backend::DYNAMIC);


    // Fused decode kernels loop over L internally and read conv_out directly,
    // so short prefill (2 <= L < 16) shares the decode path: skips qkv_prep
    // and the mQ/mK/mV round-trip. Longer prefill takes a chunked kernel.
    bool fusedDecode      = mUseSimdGroupOpt && seqLen < 16;
    bool fusedLongPrefill = (seqLen >= 64 && mUseFlashChunk) ||
                            (seqLen >= 16 && mUseFlashChunkSGMM) ||
                            (seqLen >= 32 && mUseFusedChunkSG);
    // Register-state scan prefill (qkv_prep + delta_rule_sg_v4) replaces the
    // chunked kernels on non-tensor-API devices: +10~24% prefill vs sgmm on
    // M4 Pro. Tensor-API devices keep the chunk64 flash path (untested there).
    bool scanPrefill = mUseSimdGroupOpt && !mUseFlashChunk &&
                       mGatedDeltaRuleSGV4Pipeline != nil && seqLen >= 16;
    if (scanPrefill) {
        fusedLongPrefill = false;
    }
    // A verify block uses the fused verify kernel, which also reads conv_out directly.
    bool needQKV = mAttentionType != "short_conv" && !specVerify && !fusedDecode && !fusedLongPrefill;
    if (needQKV) {
        // Same reasoning as mConvOut above: keep the Tensor objects while their
        // shape is unchanged so a recording never holds a freed Tensor*, and
        // bump the generation when they really are re-allocated.
        const bool qkvChanged = mQ.get() == nullptr ||
                                mQ->length(0) != batch || mQ->length(1) != seqLen ||
                                mQ->length(2) != H || mQ->length(3) != dk ||
                                mV->length(3) != dv;
        if (qkvChanged) {
            mQ.reset(Tensor::createDevice<float>({batch, seqLen, H, dk}));
            mK.reset(Tensor::createDevice<float>({batch, seqLen, H, dk}));
            mV.reset(Tensor::createDevice<float>({batch, seqLen, H, dv}));
            mResizeGeneration++;
        }
        success = success && backend()->onAcquireBuffer(mQ.get(), Backend::DYNAMIC);
        success = success && backend()->onAcquireBuffer(mK.get(), Backend::DYNAMIC);
        success = success && backend()->onAcquireBuffer(mV.get(), Backend::DYNAMIC);
    }
    if (!success) return OUT_OF_MEMORY;

    if (needQKV) {
        backend()->onReleaseBuffer(mV.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mK.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mQ.get(), Backend::DYNAMIC);
    }
    backend()->onReleaseBuffer(mConvOut.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

void MetalLinearAttention::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    mRecordedGeneration = mResizeGeneration;
    // onResize() may be skipped when shapes are unchanged. Ensure state is reset here too.
    int resetBatch = 0, resetConvDim = 0, resetSeqLen = 0;
    linearAttentionDims(inputs[0], resetBatch, resetConvDim, resetSeqLen);
    if (resetSeqLen > 1 && mMeta != nullptr && mMeta->previous == mMeta->remove) {
        bool loadingFromDisk = (mMeta->file_flag == KVMeta::PendingRead && mMeta->file_name.size() > 0);
        if (!loadingFromDisk) {
            mStateCache->mPendingLen = 0;
            auto mtbn = static_cast<MetalBackend *>(backend());
            int bytesPerElement = mtbn->useFp16InsteadFp32() ? 2 : 4;
            if (mStateCache->mConvState.get() != nullptr) {
                auto convDevice = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)mStateCache->mConvState->deviceId())->getBuffer();
                auto convPtr = (uint8_t*)convDevice.contents + TensorUtils::getDescribeOrigin(mStateCache->mConvState.get())->offset;
                ::memset(convPtr, 0, mStateCache->mConvState->elementSize() * bytesPerElement);
            }
            if (mStateCache->mRecurrentState.get() != nullptr) {
                auto rnnDevice = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)mStateCache->mRecurrentState->deviceId())->getBuffer();
                auto rnnPtr = (uint8_t*)rnnDevice.contents + TensorUtils::getDescribeOrigin(mStateCache->mRecurrentState.get())->offset;
                ::memset(rnnPtr, 0, mStateCache->mRecurrentState->elementSize() * bytesPerElement);
            }
        }
    }

    auto qkv = inputs[0];
    int batch = 0, convDim = 0, seqLen = 0;
    linearAttentionDims(qkv, batch, convDim, seqLen);
    int K_conv = inputs[3]->length(2);
    int convStateSize = K_conv - 1;
    int H = mNumVHeads;
    int dk = mHeadKDim;
    int dv = mHeadVDim;
    int key_dim = mNumKHeads * dk;
    int val_dim = mNumVHeads * dv;
    int gqa_factor = (mNumVHeads > mNumKHeads) ? (mNumVHeads / mNumKHeads) : 1;
    Tensor* attentionOutput = outputs[0];

    // Update param buffer
    auto paramPtr = (LinearAttnParam *)mParamBuffer.contents;
    paramPtr->batch = batch;
    paramPtr->conv_dim = convDim;
    paramPtr->seq_len = seqLen;
    paramPtr->kernel_size = K_conv;
    paramPtr->conv_state_size = convStateSize;
    paramPtr->num_k_heads = mNumKHeads;
    paramPtr->num_v_heads = mNumVHeads;
    paramPtr->head_k_dim = dk;
    paramPtr->head_v_dim = dv;
    paramPtr->key_dim = key_dim;
    paramPtr->val_dim = val_dim;
    paramPtr->gqa_factor = gqa_factor;
    paramPtr->use_l2norm = mUseQKL2Norm ? 1 : 0;
    paramPtr->qkv_c4 = TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 ? 1 : 0;
    // gate/beta chain fold: bind the raw projections and let the kernels do
    // the chain math (bit 2 of gate_c4/beta_c4); constants travel in cst.
    const Tensor* gateSrc = mFoldReq.gateFolded ? mFoldReq.rawA : inputs[1];
    const Tensor* betaSrc = mFoldReq.betaFolded ? mFoldReq.rawB : inputs[2];
    paramPtr->gate_c4 =
        (TensorUtils::getDescribe(gateSrc)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 ? 1 : 0) |
        (mFoldReq.gateFolded ? 2 : 0);
    paramPtr->beta_c4 =
        (TensorUtils::getDescribe(betaSrc)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 ? 1 : 0) |
        (mFoldReq.betaFolded ? 2 : 0);
    ::memset(paramPtr->gate_coef, 0, sizeof(paramPtr->gate_coef));
    ::memset(paramPtr->gate_bias, 0, sizeof(paramPtr->gate_bias));
    if (mFoldReq.gateFolded) {
        ::memcpy(paramPtr->gate_coef, mFoldReq.gateCoef.data(),
                 ALIMIN((int)mFoldReq.gateCoef.size(), kMaxFoldHeads) * sizeof(float));
        ::memcpy(paramPtr->gate_bias, mFoldReq.gateBias.data(),
                 ALIMIN((int)mFoldReq.gateBias.size(), kMaxFoldHeads) * sizeof(float));
    }
    paramPtr->output_c4 = TensorUtils::getDescribe(attentionOutput)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 ? 1 : 0;
    paramPtr->q_scale = 1.0f / sqrtf((float)dk);

    // lazyMode defers this block's state update; commitLen replays the previous block's accepted prefix.
    const bool lazyMode = mMeta != nullptr && mMeta->spec_block > 0 &&
                          mStateCache->mPendingCap == seqLen && nil != mVerifyFusedSGPipeline;
    int pendingLen = mStateCache->mPendingLen;
    int commitLen  = 0;
    if (pendingLen > 0 && mMeta != nullptr) {
        commitLen = ALIMAX(0, ALIMIN(pendingLen - (int)mMeta->remove, pendingLen));
    }
    paramPtr->commit_len  = commitLen;
    paramPtr->pending_seq = pendingLen > 0 ? pendingLen : 1;
    paramPtr->lazy_mode   = lazyMode ? 1 : 0;

    if (mAttentionType == "short_conv") {
        int total = batch * mHeadVDim * seqLen;
        NSUInteger threadGroupSize = MIN((NSUInteger)256, mShortConvPipeline.maxTotalThreadsPerThreadgroup);
        threadGroupSize = MIN(threadGroupSize, (NSUInteger)total);

        [encoder setComputePipelineState:mShortConvPipeline];
        MetalBackend::setTensor(inputs[0], encoder, 0);
        MetalBackend::setTensor(mStateCache->mConvState.get(), encoder, 1);
        MetalBackend::setTensor(inputs[3], encoder, 2);
        MetalBackend::setTensor(mConvOut.get(), encoder, 3);
        [encoder setBuffer:mParamBuffer offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake((total + threadGroupSize - 1) / threadGroupSize, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];

        if (convStateSize > 0) {
            int stateTotal = batch * mHeadVDim * convStateSize;
            NSUInteger stateThreadGroupSize =
                MIN((NSUInteger)256, mShortConvStateUpdatePipeline.maxTotalThreadsPerThreadgroup);
            stateThreadGroupSize = MIN(stateThreadGroupSize, (NSUInteger)stateTotal);
            [encoder setComputePipelineState:mShortConvStateUpdatePipeline];
            MetalBackend::setTensor(inputs[0], encoder, 0);
            MetalBackend::setTensor(mStateCache->mConvState.get(), encoder, 1);
            [encoder setBuffer:mParamBuffer offset:0 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake((stateTotal + stateThreadGroupSize - 1) / stateThreadGroupSize,
                                                       1, 1)
                    threadsPerThreadgroup:MTLSizeMake(stateThreadGroupSize, 1, 1)];
        }

        [encoder setComputePipelineState:mShortConvOutputPipeline];
        MetalBackend::setTensor(inputs[0], encoder, 0);
        MetalBackend::setTensor(mConvOut.get(), encoder, 1);
        MetalBackend::setTensor(attentionOutput, encoder, 2);
        [encoder setBuffer:mParamBuffer offset:0 atIndex:3];
        NSUInteger outputThreadGroupSize =
            MIN((NSUInteger)256, mShortConvOutputPipeline.maxTotalThreadsPerThreadgroup);
        outputThreadGroupSize = MIN(outputThreadGroupSize, (NSUInteger)total);
        [encoder dispatchThreadgroups:MTLSizeMake((total + outputThreadGroupSize - 1) / outputThreadGroupSize, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(outputThreadGroupSize, 1, 1)];
        return;
    }

    // Advance the conv state over the accepted prefix of the pending block.
    auto encodeConvCommit = [&](id<MTLBuffer> param) {
        [encoder setComputePipelineState:mConvCommitPipeline];
        MetalBackend::setTensor(mStateCache->mPendingQKVRaw.get(), encoder, 0);
        MetalBackend::setTensor(mStateCache->mConvState.get(), encoder, 1);
        [encoder setBuffer:param offset:0 atIndex:2];
        int total = batch * convDim;
        NSUInteger tg = MIN((NSUInteger)256, mConvCommitPipeline.maxTotalThreadsPerThreadgroup);
        tg = MIN(tg, (NSUInteger)total);
        [encoder dispatchThreadgroups:MTLSizeMake((total + tg - 1) / tg, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    };
    // Ping-pong: the prologue reads the previous block's set while the new block goes to the other.
    auto encodeVerifyFused = [&](id<MTLBuffer> param, int writeIdx) {
        auto* sc = mStateCache.get();
        const int readIdx = sc->mPendingIdx;
        [encoder setComputePipelineState:mVerifyFusedSGPipeline];
        MetalBackend::setTensor(mConvOut.get(), encoder, 0);
        MetalBackend::setTensor(inputs[1], encoder, 1);
        MetalBackend::setTensor(inputs[2], encoder, 2);
        MetalBackend::setTensor(sc->mRecurrentState.get(), encoder, 3);
        MetalBackend::setTensor(attentionOutput, encoder, 4);
        [encoder setBuffer:param offset:0 atIndex:5];
        MetalBackend::setTensor(writeIdx ? sc->mPendingK2.get()    : sc->mPendingK.get(),    encoder, 6);
        MetalBackend::setTensor(writeIdx ? sc->mPendingV2.get()    : sc->mPendingV.get(),    encoder, 7);
        MetalBackend::setTensor(writeIdx ? sc->mPendingGate2.get() : sc->mPendingGate.get(), encoder, 8);
        MetalBackend::setTensor(writeIdx ? sc->mPendingBeta2.get() : sc->mPendingBeta.get(), encoder, 9);
        MetalBackend::setTensor(readIdx  ? sc->mPendingK2.get()    : sc->mPendingK.get(),    encoder, 10);
        MetalBackend::setTensor(readIdx  ? sc->mPendingV2.get()    : sc->mPendingV.get(),    encoder, 11);
        MetalBackend::setTensor(readIdx  ? sc->mPendingGate2.get() : sc->mPendingGate.get(), encoder, 12);
        MetalBackend::setTensor(readIdx  ? sc->mPendingBeta2.get() : sc->mPendingBeta.get(), encoder, 13);
        const int simdgroupsPerTG = 4;
        int totalSimdgroups = batch * H * dv;
        [encoder dispatchThreadgroups:MTLSizeMake((totalSimdgroups + simdgroupsPerTG - 1) / simdgroupsPerTG, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(simdgroupsPerTG * 32, 1, 1)];
    };

    // A pending block reaching a non-verify forward must commit before anything reads the state.
    if (!lazyMode && pendingLen > 0) {
        if (commitLen > 0 && nil != mVerifyFusedSGPipeline) {
            auto flushPtr = (LinearAttnParam *)mParamBufferFlush.contents;
            *flushPtr = *paramPtr;
            flushPtr->seq_len   = 0;
            flushPtr->lazy_mode = 0;
            encodeConvCommit(mParamBufferFlush);
            encodeVerifyFused(mParamBufferFlush, 1 - mStateCache->mPendingIdx);
        }
        mStateCache->mPendingLen = 0;
        pendingLen = 0;
    }

    // Must precede the conv, which reads the state as left padding for the new block.
    if (lazyMode && commitLen > 0) {
        encodeConvCommit(mParamBuffer);
    }

    const bool fuseDecodeConvState =
        seqLen == 1 && convStateSize > 0 && !lazyMode && mConvSiluStateDecodePipeline != nil &&
        getenv("MNN_METAL_DISABLE_LINEAR_ATTN_CONV_STATE_FUSION") == nullptr;

    // ── Fixed head: Conv1D + SiLU (always run) ────────────────────────
    {
        id<MTLComputePipelineState> convPipeline =
            fuseDecodeConvState ? mConvSiluStateDecodePipeline : mConvSiluPipeline;
        [encoder setComputePipelineState:convPipeline];
        MetalBackend::setTensor(inputs[0], encoder, 0);                              // qkv
        MetalBackend::setTensor(mStateCache->mConvState.get(), encoder, 1);          // conv_state
        MetalBackend::setTensor(inputs[3], encoder, 2);                              // conv_weight
        MetalBackend::setTensor(mConvOut.get(), encoder, 3);                         // conv_out
        [encoder setBuffer:mParamBuffer offset:0 atIndex:4];

        int totalConvSilu = fuseDecodeConvState ? batch * convDim : batch * convDim * seqLen;
        NSUInteger threadGroupSize = MIN((NSUInteger)256, convPipeline.maxTotalThreadsPerThreadgroup);
        threadGroupSize = MIN(threadGroupSize, (NSUInteger)totalConvSilu);
        [encoder dispatchThreadgroups:MTLSizeMake((totalConvSilu + threadGroupSize - 1) / threadGroupSize, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
    }
    // Save the new block's raw qkv instead of updating the conv state in place.
    if (lazyMode) {
        [encoder setComputePipelineState:mQKVRawSavePipeline];
        MetalBackend::setTensor(inputs[0], encoder, 0);
        MetalBackend::setTensor(mStateCache->mPendingQKVRaw.get(), encoder, 1);
        [encoder setBuffer:mParamBuffer offset:0 atIndex:2];
        int totalSave = batch * convDim * seqLen;
        NSUInteger tgS = MIN((NSUInteger)256, mQKVRawSavePipeline.maxTotalThreadsPerThreadgroup);
        tgS = MIN(tgS, (NSUInteger)totalSave);
        [encoder dispatchThreadgroups:MTLSizeMake((totalSave + tgS - 1) / tgS, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tgS, 1, 1)];
    }
    if (convStateSize > 0 && !fuseDecodeConvState && !lazyMode) {
        [encoder setComputePipelineState:mConvStateUpdatePipeline];
        MetalBackend::setTensor(inputs[0], encoder, 0);
        MetalBackend::setTensor(mStateCache->mConvState.get(), encoder, 1);
        [encoder setBuffer:mParamBuffer offset:0 atIndex:2];

        int totalUpdate = batch * convDim;
        NSUInteger threadGroupSize = MIN((NSUInteger)256, mConvStateUpdatePipeline.maxTotalThreadsPerThreadgroup);
        threadGroupSize = MIN(threadGroupSize, (NSUInteger)totalUpdate);
        [encoder dispatchThreadgroups:MTLSizeMake((totalUpdate + threadGroupSize - 1) / threadGroupSize, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
    }

    // ── Variable tail: pick optimal path by seqLen ────────────────────
    // Priority within each branch: prefer *_align/flash > baseline > fallback.
    const bool scanPrefill = mUseSimdGroupOpt && !mUseFlashChunk &&
                             mGatedDeltaRuleSGV4Pipeline != nil && seqLen >= 16;
    if (lazyMode) {
        auto* sc = mStateCache.get();
        int writeIdx = 1 - sc->mPendingIdx;
        encodeVerifyFused(mParamBuffer, writeIdx);
        sc->mPendingLen = seqLen;
        sc->mPendingIdx = writeIdx;
    } else if (mUseSimdGroupOpt && seqLen < 16) {
        // ── Decode (L=1) and short prefill (2<=L<16) — the fused kernels loop
        //    over L internally. Priority: fused_sg_tg (decode only) >
        //    fused_sg_align > fused_sg. ──
        id<MTLComputePipelineState> decodePipe = mGatedDeltaRuleFusedSGPipeline;
        const bool preferTG = H < 16 && seqLen == 1;
        if (mFusedSGTGPipeline != nil && preferTG) {
            decodePipe = mFusedSGTGPipeline;
        } else if (mFusedSGAlignPipeline != nil) {
            decodePipe = mFusedSGAlignPipeline;
        }
        [encoder setComputePipelineState:decodePipe];
        MetalBackend::setTensor(mConvOut.get(), encoder, 0);                              // conv_out
        MetalBackend::setTensor(gateSrc, encoder, 1);                                  // gate
        MetalBackend::setTensor(betaSrc, encoder, 2);                                  // beta
        MetalBackend::setTensor(mStateCache->mRecurrentState.get(), encoder, 3);          // recurrent_state
        MetalBackend::setTensor(attentionOutput, encoder, 4);                             // attn_out
        [encoder setBuffer:mParamBuffer offset:0 atIndex:5];
        const int simdgroupsPerTG = decodePipe == mFusedSGTGPipeline ? mFusedSGTGSimds :
                                    (decodePipe == mFusedSGAlignPipeline ? mFusedSGAlignSimds : 4);
        int totalSimdgroups = batch * H * dv;
        NSUInteger threadGroupSize = simdgroupsPerTG * 32;
        int numThreadgroups = (totalSimdgroups + simdgroupsPerTG - 1) / simdgroupsPerTG;
        [encoder dispatchThreadgroups:MTLSizeMake(numThreadgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
    } else if (mUseSimdGroupOpt && seqLen >= 64 && mUseFlashChunk) {
        // ── Chunk-parallel A/P prep, followed by chunk-sequential scan ──
        const int numChunks = (seqLen + 63) / 64;
        [encoder setComputePipelineState:mFlashChunkPrepPipeline];
        MetalBackend::setTensor(mConvOut.get(), encoder, 0);
        MetalBackend::setTensor(gateSrc, encoder, 1);
        MetalBackend::setTensor(betaSrc, encoder, 2);
        MetalBackend::setTensor(attentionOutput, encoder, 3);
        [encoder setBuffer:mParamBuffer offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(numChunks, batch * H, 1)
                threadsPerThreadgroup:MTLSizeMake(32, mFlashSimdsPerTG, 1)];

        [encoder setComputePipelineState:mFlashChunkScanPipeline];
        MetalBackend::setTensor(mConvOut.get(), encoder, 0);
        MetalBackend::setTensor(gateSrc, encoder, 1);
        MetalBackend::setTensor(betaSrc, encoder, 2);
        MetalBackend::setTensor(mStateCache->mRecurrentState.get(), encoder, 3);
        MetalBackend::setTensor(attentionOutput, encoder, 4);
        [encoder setBuffer:mParamBuffer offset:0 atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(dv / mFlashDvBlock, batch * H, 1)
                threadsPerThreadgroup:MTLSizeMake(32, mFlashSimdsPerTG, 1)];
    } else if (mUseSimdGroupOpt && seqLen >= 16 && mUseFlashChunkSGMM && !scanPrefill) {
        // ── simdgroup_matrix chunk prefill (L>=16, non-tensor-API) ────
        [encoder setComputePipelineState:mFlashChunkSGMMPipeline];
        MetalBackend::setTensor(mConvOut.get(), encoder, 0);
        MetalBackend::setTensor(gateSrc, encoder, 1);
        MetalBackend::setTensor(betaSrc, encoder, 2);
        MetalBackend::setTensor(mStateCache->mRecurrentState.get(), encoder, 3);
        MetalBackend::setTensor(attentionOutput, encoder, 4);
        [encoder setBuffer:mParamBuffer offset:0 atIndex:5];

        int numThreadgroups = batch * H * (dv / mSgmmDvBlock);
        NSUInteger threadGroupSize = (NSUInteger)(mSgmmSimdsPerTG * 32);
        [encoder dispatchThreadgroups:MTLSizeMake(numThreadgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
    } else if (mUseSimdGroupOpt && seqLen >= 32 && mUseFusedChunkSG && !scanPrefill) {
        // ── Long prefill (L>=32) w/o tensor_ops: fused_chunk_sg ──────
        [encoder setComputePipelineState:mFusedChunkSGPipeline];
        MetalBackend::setTensor(mConvOut.get(), encoder, 0);
        MetalBackend::setTensor(gateSrc, encoder, 1);
        MetalBackend::setTensor(betaSrc, encoder, 2);
        MetalBackend::setTensor(mStateCache->mRecurrentState.get(), encoder, 3);
        MetalBackend::setTensor(attentionOutput, encoder, 4);
        [encoder setBuffer:mParamBuffer offset:0 atIndex:5];

        int simdsPerTG = mChunkTGThreads / 32;
        int numThreadgroups = batch * H * (dv / simdsPerTG);
        [encoder dispatchThreadgroups:MTLSizeMake(numThreadgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)mChunkTGThreads, 1, 1)];
    } else {
        // ── Short prefill (2<=L<16), or long prefill w/o SG fused path:
        //     unfused two-stage: qkv_prep_sg (or scalar) + delta_rule_sg (or scalar) ──

        // Kernel: QKV prep — prefer simdgroup version
        if (mQKVPrepSGPipeline != nil) {
            [encoder setComputePipelineState:mQKVPrepSGPipeline];
            MetalBackend::setTensor(mConvOut.get(), encoder, 0);
            MetalBackend::setTensor(mQ.get(), encoder, 1);
            MetalBackend::setTensor(mK.get(), encoder, 2);
            MetalBackend::setTensor(mV.get(), encoder, 3);
            [encoder setBuffer:mParamBuffer offset:0 atIndex:4];

            int total = batch * seqLen * H;
            const int simdgroupsPerTG = 4;
            int numTG = (total + simdgroupsPerTG - 1) / simdgroupsPerTG;
            [encoder dispatchThreadgroups:MTLSizeMake(numTG, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(simdgroupsPerTG * 32, 1, 1)];
        } else {
            [encoder setComputePipelineState:mQKVPrepPipeline];
            MetalBackend::setTensor(mConvOut.get(), encoder, 0);
            MetalBackend::setTensor(mQ.get(), encoder, 1);
            MetalBackend::setTensor(mK.get(), encoder, 2);
            MetalBackend::setTensor(mV.get(), encoder, 3);
            [encoder setBuffer:mParamBuffer offset:0 atIndex:4];

            int totalPrep = batch * seqLen * H;
            NSUInteger threadGroupSize = MIN((NSUInteger)256, mQKVPrepPipeline.maxTotalThreadsPerThreadgroup);
            threadGroupSize = MIN(threadGroupSize, (NSUInteger)totalPrep);
            [encoder dispatchThreadgroups:MTLSizeMake((totalPrep + threadGroupSize - 1) / threadGroupSize, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        }

        // Kernel: Delta rule
        if (mUseSimdGroupOpt) {
            id<MTLComputePipelineState> deltaPipe =
                (mGatedDeltaRuleSGV4Pipeline != nil) ? mGatedDeltaRuleSGV4Pipeline
                                                     : mGatedDeltaRuleSGPipeline;
            [encoder setComputePipelineState:deltaPipe];
            MetalBackend::setTensor(mQ.get(), encoder, 0);
            MetalBackend::setTensor(mK.get(), encoder, 1);
            MetalBackend::setTensor(mV.get(), encoder, 2);
            MetalBackend::setTensor(gateSrc, encoder, 3);
            MetalBackend::setTensor(betaSrc, encoder, 4);
            MetalBackend::setTensor(mStateCache->mRecurrentState.get(), encoder, 5);
            MetalBackend::setTensor(attentionOutput, encoder, 6);
            [encoder setBuffer:mParamBuffer offset:0 atIndex:7];

            const int simdgroupsPerTG = 4;
            int totalSimdgroups = batch * H * dv;
            NSUInteger threadGroupSize = simdgroupsPerTG * 32;
            int numThreadgroups = (totalSimdgroups + simdgroupsPerTG - 1) / simdgroupsPerTG;
            [encoder dispatchThreadgroups:MTLSizeMake(numThreadgroups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        } else {
            [encoder setComputePipelineState:mGatedDeltaRulePipeline];
            MetalBackend::setTensor(mQ.get(), encoder, 0);
            MetalBackend::setTensor(mK.get(), encoder, 1);
            MetalBackend::setTensor(mV.get(), encoder, 2);
            MetalBackend::setTensor(gateSrc, encoder, 3);
            MetalBackend::setTensor(betaSrc, encoder, 4);
            MetalBackend::setTensor(mStateCache->mRecurrentState.get(), encoder, 5);
            MetalBackend::setTensor(attentionOutput, encoder, 6);
            [encoder setBuffer:mParamBuffer offset:0 atIndex:7];

            int total = batch * H * dv;
            NSUInteger threadGroupSize = MIN((NSUInteger)256, mGatedDeltaRulePipeline.maxTotalThreadsPerThreadgroup);
            threadGroupSize = MIN(threadGroupSize, (NSUInteger)total);
            [encoder dispatchThreadgroups:MTLSizeMake((total + threadGroupSize - 1) / threadGroupSize, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        }
    }
}

bool MetalLinearAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto tmp = new MetalLinearAttention(bn, op);
    // Share persistent state buffers between prefill and decode Executions
    tmp->mStateCache = mStateCache;
    *dst = tmp;
    MNN_METAL_PROFILE_REGISTER_CLONE(bn, op, *dst);
    return true;
}

class MetalLinearAttentionCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op,
                                Backend *backend, const std::vector<Tensor *> &outputs) const {
        return new MetalLinearAttention(backend, op);
    }
};
REGISTER_METAL_OP_TRANSFORMER_CREATOR(MetalLinearAttentionCreator, OpType_LinearAttention);

} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_METAL_ENABLED */
