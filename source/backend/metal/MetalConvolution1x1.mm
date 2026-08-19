//
//  MetalConvolution1x1.mm
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalConvolution1x1.hpp"
#import "backend/metal/MetalEnv.hpp"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"
#import "backend/metal/MetalSharedGather.hpp"
#import "core/KVMeta.hpp"
#import "ConvSimdGroupShader.hpp"

#if MNN_METAL_ENABLED

#if MNN_METAL_OP_PROFILE
#define CONV1X1_SET_TAG(name) mProfileTag = (name)
#else
#define CONV1X1_SET_TAG(name) do {} while(0)
#endif

namespace MNN {

static bool isQ4W16BlockSlices(int blockSlices) {
    return blockSlices == 8 || blockSlices == 16 || blockSlices == 32 || blockSlices == 64;
}

// Lanes per quant block for the Q4 16-byte decode GEMV (GEMV_QBLOCK_W16_MID).
// A simdgroup owns `blocksPerSimdgroup` blocks. With `mid` lanes on a block,
// 32/mid blocks are in flight and each lane walks `pairsPerBlock/mid` uint4
// loads. One uint4 covers two C4 slices.
// Pick the mid that minimises the loads on the critical path, preferring the
// smallest on ties (longest contiguous run per lane).
static int chooseQ4W16Mid(int blocksPerSimdgroup, int pairsPerBlock) {
    int best = 1, bestCost = 1 << 30;
    const int blocks = blocksPerSimdgroup > 1 ? blocksPerSimdgroup : 1;
    for (int mid : {1, 2, 4, 8}) {
        if (mid > pairsPerBlock || pairsPerBlock % mid != 0) {
            continue;
        }
        const int slots = 32 / mid;
        const int iters = UP_DIV(blocks, slots);
        const int cost  = iters * (pairsPerBlock / mid);
        if (cost < bestCost) {
            bestCost = cost;
            best = mid;
        }
    }
    return best;
}

bool MetalConvolution1x1::isValid(const Convolution2D *conv, const Tensor *input) {
    auto common = conv->common();
    auto kx = common->kernelX(), ky = common->kernelY();
    auto dx = common->dilateX(), dy = common->dilateY();
    auto sx = common->strideX(), sy = common->strideY();
    auto px = common->padX(), py = common->padY();
    return kx == 1 && ky == 1 && dx == 1 && dy == 1 && px == 0 && py == 0 && sx == 1 && sy == 1;
}

MetalConvolution1x1::MetalConvolution1x1(Backend *backend, const MNN::Op *op) : MetalConvolutionCommon(backend, op, nullptr) {
    auto conv2D = op->main_as_Convolution2D();
    bool ldInt8Weight = false;
    if(static_cast<MetalBackend*>(backend)->getMemoryMode() == BackendConfig::Memory_Low) {
        if (conv2D->quanParameter() && (conv2D->external() || conv2D->quanParameter()->buffer())) {
            // quant type equal to 3 means fp16, fallback to float weight
            if(conv2D->quanParameter()->type() != 3 && conv2D->quanParameter()->type() != 8) {
            	ldInt8Weight = true;
            }
        }
    }
    loadWeight(op, ldInt8Weight);
}

MetalConvolution1x1::MetalConvolution1x1(Backend *backend, const MNN::Op *op,
                                         std::shared_ptr<MNN::Tensor> weight,
                                         std::shared_ptr<MNN::Tensor> bias,
                                         std::shared_ptr<MNN::Tensor> dequantScale,
                                         int dequantBits, float scaleCoef)
    : MetalConvolutionCommon(backend, op, bias) {
    mWeight = weight;
    mBias = bias;
    mDequantScaleBias = dequantScale;
    mDequantBits = dequantBits;
    mScaleCoef = scaleCoef;
}

bool MetalConvolution1x1::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    if (op->type() == OpType_GatherV2) {
        // SharedGather path: reuse quantized weight and dequant resources
        auto dequantScale = getDequantScale();
        if (!dequantScale ||
            (mDequantBits != 2 && mDequantBits != 3 && mDequantBits != 4 && mDequantBits != 8)) {
            // Quantized weight is required for SharedGather
            return false;
        }
        auto conv2D = mOp->main_as_Convolution2D();
        int oc = conv2D->common()->outputCount();
        *dst = new MetalSharedGather(bn, oc, mWeight, dequantScale, mDequantBits, mScaleCoef);
        MNN_METAL_PROFILE_REGISTER_CLONE(bn, op, *dst);
        return true;
    }
    *dst = new MetalConvolution1x1(bn, op, mWeight, mBias, mDequantScaleBias,
                                   mDequantBits, mScaleCoef);
    MNN_METAL_PROFILE_REGISTER_CLONE(bn, op, *dst);
    return true;
}

bool MetalConvolution1x1::setupGateUpFusion(MetalConvolution1x1* peer, const Tensor* peerOutput) {
    if (!mIs2sgDecode || !peer->mIs2sgDecode ||
        mDequantBits != peer->mDequantBits || mBlockSize != peer->mBlockSize) {
        return false;
    }
    // Leader = gate (this), Follower = up (peer)
    mIsGateUpLeader = true;
    mGateUpPeer = peer;
    mGateUpPeerOutput = peerOutput;
    peer->mIsGateUpFollower = true;

    // Build fused pipeline with GATE_UP_FUSED macro
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();

    // Store up's scale_coef separately: gate uses cst.scale_coef (via buffer(2)),
    // but up needs its own tensor-specific coefficient. Without this, up's dequant
    // is scaled by gate's coefficient and any range mismatch drifts decode into
    // garbage on models like Qwen3.5-2B.
    mGateUpSegBuffer = backend->getConstBuffer(sizeof(float));
    ((float *)mGateUpSegBuffer.contents)[0] = peer->mScaleCoef;
    MetalRuntime* rt = (MetalRuntime *)backend->runtime();

    std::string ftype4 = backend->useFp16InsteadFp32() ? "half4" : "float4";
    std::vector<std::string> keys = {ftype4, "MNN_METAL_FLOAT32_COMPUTER"};
    if (backend->useFp16InsteadFp32()) {
        keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
    }
    if (mDequantBits == 2) {
        keys.emplace_back("conv1x1_wquant_2");
    } else if (mDequantBits == 3) {
        keys.emplace_back("conv1x1_wquant_3");
    } else if (mDequantBits == 4) {
        keys.emplace_back("conv1x1_wquant_4");
    } else if (mDequantBits == 8) {
        keys.emplace_back("conv1x1_wquant_8");
    }
    keys.emplace_back("conv1x1_wquant_sg_reduce");
    keys.emplace_back("conv1x1_gemv_g4m1_2sg_wquant_sg");
    keys.emplace_back("GATE_UP_FUSED");
    // Extend the generalized Q4 W16 decode specialization onto the fused GEMV.
    // Gate and up share the block-input and must compile the same block shape.
    const bool w16 = mQ4W16BlockSlices > 0 && mQ4W16BlockSlices == peer->mQ4W16BlockSlices &&
                     mDequantBits == 4;
    const int w16Mid = w16 ? chooseQ4W16Mid(mBlockSize, mQ4W16BlockSlices / 2) : 0;
    if (w16) {
        keys.emplace_back("GEMV_QBLOCK_W16");
        keys.emplace_back("W16_BLOCK_SLICES_" + std::to_string(mQ4W16BlockSlices));
        keys.emplace_back("W16_MID_" + std::to_string(w16Mid));
    }

    mGateUpFusedPipeline = rt->findPipeline(keys);
    if (nil == mGateUpFusedPipeline && !rt->pipelineCompileFailed(keys)) {
        std::string ftype = backend->useFp16InsteadFp32() ? "half" : "float";
        std::string ftype2 = backend->useFp16InsteadFp32() ? "half2" : "float2";
        std::string ftype2x4 = backend->useFp16InsteadFp32() ? "half2x4" : "float2x4";
        std::string ftype4x4 = backend->useFp16InsteadFp32() ? "half4x4" : "float4x4";

        MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
        auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
        [dic setValue:@(ftype.c_str()) forKey:@"ftype"];
        [dic setValue:@(ftype2.c_str()) forKey:@"ftype2"];
        [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
        [dic setValue:@(ftype2x4.c_str()) forKey:@"ftype2x4"];
        [dic setValue:@(ftype4x4.c_str()) forKey:@"ftype4x4"];
        [dic setValue:@"1" forKey:@"MNN_METAL_FLOAT32_COMPUTER"];
        if (backend->useFp16InsteadFp32()) {
            [dic setValue:@"1" forKey:@"MNN_METAL_FLOAT16_STORAGE"];
        }
        if (mDequantBits == 2) {
            [dic setValue:@"1" forKey:@"W_QUANT_2"];
        } else if (mDequantBits == 3) {
            [dic setValue:@"1" forKey:@"W_QUANT_3"];
        } else if (mDequantBits == 4) {
            [dic setValue:@"1" forKey:@"W_QUANT_4"];
        } else if (mDequantBits == 8) {
            [dic setValue:@"1" forKey:@"W_QUANT_8"];
        }
        [dic setValue:@"1" forKey:@"GATE_UP_FUSED"];
        if (w16) {
            [dic setValue:@"1" forKey:@"GEMV_QBLOCK_W16"];
            [dic setValue:@(mQ4W16BlockSlices).stringValue forKey:@"GEMV_QBLOCK_W16_BLOCK_SLICES"];
            [dic setValue:@(w16Mid).stringValue forKey:@"GEMV_QBLOCK_W16_MID"];
        }
        option.preprocessorMacros = dic;

        std::string sgrWqStr = std::string(gBasicConvPrefix) + gConv1x1WqSgReduce;
        mGateUpFusedPipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", option);
        rt->insertPipeline(keys, mGateUpFusedPipeline);
    }

    if (nil == mGateUpFusedPipeline) {
        // Compilation failed, revert fusion
        mIsGateUpLeader = false;
        mGateUpPeer = nullptr;
        mGateUpSegBuffer = nil;
        peer->mIsGateUpFollower = false;
        return false;
    }

    // Update grid: add z=2 dimension for gate/up selection. The fused pipeline
    // is 2sg-kernel based — force 64 threads (plain pipeline may be split-K g8
    // with a 128-thread group).
    auto gridSize = mThreads.first;
    mThreads.first = MTLSizeMake(gridSize.width, gridSize.height, 2);
    mThreads.second = MTLSizeMake(64, 1, 1);

    return true;
}

bool MetalConvolution1x1::setupQKVFusion(MetalConvolution1x1* peerK, const Tensor* peerKOutput,
                                         MetalConvolution1x1* peerV, const Tensor* peerVOutput,
                                         MetalConvolution1x1* peerW, const Tensor* peerWOutput) {
    if (!mIs2sgDecode || !peerK->mIs2sgDecode || !peerV->mIs2sgDecode) {
        return false;
    }
    if (peerW != nullptr && !peerW->mIs2sgDecode) {
        return false;
    }
    // No stacking with other fusion roles (buffer indices 6-9/14 collide with
    // GATE_UP_FUSED; LN_FUSED pipeline would lack the QKV macro).
    if (mIsGateUpLeader || mIsGateUpFollower || mHasLNFusion ||
        peerK->mIsGateUpLeader || peerK->mIsGateUpFollower || peerK->mHasLNFusion ||
        peerV->mIsGateUpLeader || peerV->mIsGateUpFollower || peerV->mHasLNFusion) {
        return false;
    }
    if (peerW != nullptr &&
        (peerW->mIsGateUpLeader || peerW->mIsGateUpFollower || peerW->mHasLNFusion)) {
        return false;
    }
    // The fused kernel shares the leader's cst for everything except
    // output_slice and scale_coef, so quant layout and activation must match.
    if (peerK->mDequantBits != mDequantBits || peerV->mDequantBits != mDequantBits ||
        peerK->mBlockSize != mBlockSize || peerV->mBlockSize != mBlockSize ||
        peerK->mActivationType != mActivationType || peerV->mActivationType != mActivationType) {
        return false;
    }
    if (peerW != nullptr &&
        (peerW->mDequantBits != mDequantBits || peerW->mBlockSize != mBlockSize ||
         peerW->mActivationType != mActivationType)) {
        return false;
    }

    auto backend = static_cast<MetalBackend *>(this->backend());
    MetalRuntime* rt = (MetalRuntime *)backend->runtime();

    std::string ftype4 = backend->useFp16InsteadFp32() ? "half4" : "float4";
    std::vector<std::string> keys = {ftype4, "MNN_METAL_FLOAT32_COMPUTER"};
    if (backend->useFp16InsteadFp32()) {
        keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
    }
    if (mDequantBits == 2) {
        keys.emplace_back("conv1x1_wquant_2");
    } else if (mDequantBits == 3) {
        keys.emplace_back("conv1x1_wquant_3");
    } else if (mDequantBits == 4) {
        keys.emplace_back("conv1x1_wquant_4");
    } else if (mDequantBits == 8) {
        keys.emplace_back("conv1x1_wquant_8");
    }
    keys.emplace_back("conv1x1_wquant_sg_reduce");
    keys.emplace_back("conv1x1_gemv_g4m1_2sg_wquant_sg");
    keys.emplace_back("QKV_FUSED");
    // Compact projection grids improve Qwen3.5-2B decode on Mac M5 Pro, but
    // showed no end-to-end gain on iPad M5. Keep the portable default on the
    // legacy rectangular grid and make compact mode an explicit opt-in.
    const bool compactGrid = MetalEnv::get().qkvCompactGridEnabled;
    if (compactGrid) {
        keys.emplace_back("QKV_COMPACT_GRID");
    }
    if (peerW != nullptr) {
        keys.emplace_back("QKV_FUSED_P4");
    }
    // Generalized Q4 W16 on the fused QKV GEMV. Every member must carry the
    // same compile-time quant-block shape.
    const bool w16 = mQ4W16BlockSlices > 0 && mQ4W16BlockSlices == peerK->mQ4W16BlockSlices &&
                     mQ4W16BlockSlices == peerV->mQ4W16BlockSlices &&
                     (peerW == nullptr || mQ4W16BlockSlices == peerW->mQ4W16BlockSlices) &&
                     mDequantBits == 4;
    const int w16Mid = w16 ? chooseQ4W16Mid(mBlockSize, mQ4W16BlockSlices / 2) : 0;
    if (w16) {
        keys.emplace_back("GEMV_QBLOCK_W16");
        keys.emplace_back("W16_BLOCK_SLICES_" + std::to_string(mQ4W16BlockSlices));
        keys.emplace_back("W16_MID_" + std::to_string(w16Mid));
    }

    mQKVFusedPipeline = rt->findPipeline(keys);
    if (nil == mQKVFusedPipeline && !rt->pipelineCompileFailed(keys)) {
        std::string ftype = backend->useFp16InsteadFp32() ? "half" : "float";
        std::string ftype2 = backend->useFp16InsteadFp32() ? "half2" : "float2";
        std::string ftype2x4 = backend->useFp16InsteadFp32() ? "half2x4" : "float2x4";
        std::string ftype4x4 = backend->useFp16InsteadFp32() ? "half4x4" : "float4x4";

        MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
        auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
        [dic setValue:@(ftype.c_str()) forKey:@"ftype"];
        [dic setValue:@(ftype2.c_str()) forKey:@"ftype2"];
        [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
        [dic setValue:@(ftype2x4.c_str()) forKey:@"ftype2x4"];
        [dic setValue:@(ftype4x4.c_str()) forKey:@"ftype4x4"];
        [dic setValue:@"1" forKey:@"MNN_METAL_FLOAT32_COMPUTER"];
        if (backend->useFp16InsteadFp32()) {
            [dic setValue:@"1" forKey:@"MNN_METAL_FLOAT16_STORAGE"];
        }
        if (mDequantBits == 2) {
            [dic setValue:@"1" forKey:@"W_QUANT_2"];
        } else if (mDequantBits == 3) {
            [dic setValue:@"1" forKey:@"W_QUANT_3"];
        } else if (mDequantBits == 4) {
            [dic setValue:@"1" forKey:@"W_QUANT_4"];
        } else if (mDequantBits == 8) {
            [dic setValue:@"1" forKey:@"W_QUANT_8"];
        }
        [dic setValue:@"1" forKey:@"QKV_FUSED"];
        if (compactGrid) {
            [dic setValue:@"1" forKey:@"QKV_COMPACT_GRID"];
        }
        if (peerW != nullptr) {
            [dic setValue:@"1" forKey:@"QKV_FUSED_P4"];
        }
        if (w16) {
            [dic setValue:@"1" forKey:@"GEMV_QBLOCK_W16"];
            [dic setValue:@(mQ4W16BlockSlices).stringValue forKey:@"GEMV_QBLOCK_W16_BLOCK_SLICES"];
            [dic setValue:@(w16Mid).stringValue forKey:@"GEMV_QBLOCK_W16_MID"];
        }
        option.preprocessorMacros = dic;

        std::string sgrWqStr = std::string(gBasicConvPrefix) + gConv1x1WqSgReduce;
        mQKVFusedPipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", option);
        rt->insertPipeline(keys, mQKVFusedPipeline);
    }
    if (nil == mQKVFusedPipeline) {
        return false;
    }

    mIsQKVLeader = true;
    mQKVPeerK = peerK;
    mQKVPeerV = peerV;
    mQKVPeerW = peerW;
    mQKVPeerKOutput = peerKOutput;
    mQKVPeerVOutput = peerVOutput;
    mQKVPeerWOutput = peerWOutput;
    mQKVCompactGrid = compactGrid;
    peerK->mIsQKVFollower = true;
    peerV->mIsQKVFollower = true;
    if (peerW != nullptr) {
        peerW->mIsQKVFollower = true;
    }

    // The fused dispatch writes k/v outputs at the leader's (earlier) position.
    // Ops between the leader and the k/v consumers (q/k-norm Cast/Raster, RoPE,
    // attention in-execution scratch) may share the followers' dynamic-pool
    // regions — their lifetimes don't overlap in the UNfused schedule — and
    // would clobber the early writes (observed as decode garbage once KV growth
    // reshuffled the pool). Re-home both outputs to static memory, which the
    // dynamic pool can never alias; consumers bind tensor addresses at encode
    // time and follow automatically. Called after the allocator's compute(), so
    // the assignment sticks. The few KB of static memory per projection are not
    // reclaimed on later resizes (decode module resizes once; bounded waste).
    bool rehomed = backend->onAcquireBuffer(peerKOutput, Backend::STATIC) &&
                   backend->onAcquireBuffer(peerVOutput, Backend::STATIC);
    if (rehomed && peerW != nullptr) {
        rehomed = backend->onAcquireBuffer(peerWOutput, Backend::STATIC);
    }
    if (!rehomed) {
        mIsQKVLeader = false;
        mQKVPeerK = nullptr;
        mQKVPeerV = nullptr;
        mQKVPeerW = nullptr;
        mQKVPeerKOutput = nullptr;
        mQKVPeerVOutput = nullptr;
        mQKVPeerWOutput = nullptr;
        mQKVCompactGrid = false;
        peerK->mIsQKVFollower = false;
        peerV->mIsQKVFollower = false;
        if (peerW != nullptr) {
            peerW->mIsQKVFollower = false;
        }
        return false;
    }

    // Followers' per-projection scale_coef + output_slice (leader's come from cst).
    auto kSlice = ((Param*)peerK->mConstBuffer.contents)->output_slice;
    auto vSlice = ((Param*)peerV->mConstBuffer.contents)->output_slice;
    auto wSlice = peerW != nullptr ? ((Param*)peerW->mConstBuffer.contents)->output_slice : 0;
    mQKVSegBuffer = backend->getConstBuffer(6 * sizeof(float));
    auto seg = (float *)mQKVSegBuffer.contents;
    seg[0] = peerK->mScaleCoef;
    seg[1] = peerV->mScaleCoef;
    seg[2] = (float)kSlice;
    seg[3] = (float)vSlice;
    seg[4] = peerW != nullptr ? peerW->mScaleCoef : 0.0f;
    seg[5] = (float)wSlice;

    // Compact mode concatenates each projection's exact threadgroup range on
    // grid.x. The shader maps the flat index back to projection + local row.
    // The rollback path retains the legacy max-grid.x x projection-count
    // rectangle. Fused pipeline is 2sg-kernel based — force 64 threads (plain
    // pipeline may be split-K g8 with a 128-thread group).
    auto leaderSlice = ((Param*)mConstBuffer.contents)->output_slice;
    if (compactGrid) {
        NSUInteger compactGridX = (NSUInteger)(UP_DIV(leaderSlice, 2) + UP_DIV(kSlice, 2) +
                                                UP_DIV(vSlice, 2) + UP_DIV(wSlice, 2));
        mThreads.first = MTLSizeMake(compactGridX, mThreads.first.height, 1);
    } else {
        auto maxSlice = ALIMAX(ALIMAX(leaderSlice, wSlice), ALIMAX(kSlice, vSlice));
        NSUInteger maxGridX = (NSUInteger)UP_DIV(maxSlice, 2);
        mThreads.first = MTLSizeMake(maxGridX, mThreads.first.height, peerW != nullptr ? 4 : 3);
    }
    mThreads.second = MTLSizeMake(64, 1, 1);

    return true;
}

bool MetalConvolution1x1::setupLNFusion(const Tensor* hiddenInput, const Tensor* residualInput,
                                        const Tensor* residualOutput, std::shared_ptr<Tensor> gamma, float eps) {
    if (!mIs2sgDecode) {
        return false;
    }

    mLNHiddenInput = hiddenInput;
    mLNResidualInput = residualInput;
    mLNResidualOutput = residualOutput;
    mLNGamma = gamma;
    mHasLNFusion = true;

    auto backend = static_cast<MetalBackend *>(this->backend());
    MetalRuntime* rt = (MetalRuntime *)backend->runtime();
    mLNEpsBuffer = backend->getConstBuffer(sizeof(float));
    *((float *)mLNEpsBuffer.contents) = eps;

    std::string ftype = backend->useFp16InsteadFp32() ? "half" : "float";
    std::string ftype2 = backend->useFp16InsteadFp32() ? "half2" : "float2";
    std::string ftype4 = backend->useFp16InsteadFp32() ? "half4" : "float4";
    std::string ftype2x4 = backend->useFp16InsteadFp32() ? "half2x4" : "float2x4";
    std::string ftype4x4 = backend->useFp16InsteadFp32() ? "half4x4" : "float4x4";

    std::vector<std::string> keys = {ftype4, "MNN_METAL_FLOAT32_COMPUTER"};
    if (backend->useFp16InsteadFp32()) {
        keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
    }
    if (mDequantBits == 2) {
        keys.emplace_back("conv1x1_wquant_2");
    } else if (mDequantBits == 3) {
        keys.emplace_back("conv1x1_wquant_3");
    } else if (mDequantBits == 4) {
        keys.emplace_back("conv1x1_wquant_4");
    } else if (mDequantBits == 8) {
        keys.emplace_back("conv1x1_wquant_8");
    }
    keys.emplace_back("conv1x1_wquant_sg_reduce");
    keys.emplace_back("conv1x1_gemv_g4m1_2sg_wquant_sg");
    if (mIsGateUpLeader) {
        keys.emplace_back("GATE_UP_FUSED");
    }
    if (mIsQKVLeader) {
        keys.emplace_back("QKV_FUSED");
        if (mQKVCompactGrid) {
            keys.emplace_back("QKV_COMPACT_GRID");
        }
        if (mQKVPeerW != nullptr) {
            keys.emplace_back("QKV_FUSED_P4");
        }
    }
    keys.emplace_back("LN_FUSED");
    // Generalized Q4 W16 on the LN-folded fused GEMV. Every fused member must
    // carry the same compile-time quant-block shape.
    bool w16 = mQ4W16BlockSlices > 0 && mDequantBits == 4;
    if (mIsGateUpLeader &&
        (mGateUpPeer == nullptr || mGateUpPeer->mQ4W16BlockSlices != mQ4W16BlockSlices)) {
        w16 = false;
    }
    if (mIsQKVLeader) {
        if (mQKVPeerK == nullptr || mQKVPeerK->mQ4W16BlockSlices != mQ4W16BlockSlices ||
            mQKVPeerV == nullptr || mQKVPeerV->mQ4W16BlockSlices != mQ4W16BlockSlices ||
            (mQKVPeerW != nullptr && mQKVPeerW->mQ4W16BlockSlices != mQ4W16BlockSlices)) {
            w16 = false;
        }
    }
    const int w16Mid = w16 ? chooseQ4W16Mid(mBlockSize, mQ4W16BlockSlices / 2) : 0;
    if (w16) {
        keys.emplace_back("GEMV_QBLOCK_W16");
        keys.emplace_back("W16_BLOCK_SLICES_" + std::to_string(mQ4W16BlockSlices));
        keys.emplace_back("W16_MID_" + std::to_string(w16Mid));
    }

    mLNFusedPipeline = rt->findPipeline(keys);
    if (nil == mLNFusedPipeline && !rt->pipelineCompileFailed(keys)) {
        MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
        auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
        [dic setValue:@(ftype.c_str()) forKey:@"ftype"];
        [dic setValue:@(ftype2.c_str()) forKey:@"ftype2"];
        [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
        [dic setValue:@(ftype2x4.c_str()) forKey:@"ftype2x4"];
        [dic setValue:@(ftype4x4.c_str()) forKey:@"ftype4x4"];
        [dic setValue:@"1" forKey:@"MNN_METAL_FLOAT32_COMPUTER"];
        if (backend->useFp16InsteadFp32()) {
            [dic setValue:@"1" forKey:@"MNN_METAL_FLOAT16_STORAGE"];
        }
        if (mDequantBits == 2) {
            [dic setValue:@"1" forKey:@"W_QUANT_2"];
        } else if (mDequantBits == 3) {
            [dic setValue:@"1" forKey:@"W_QUANT_3"];
        } else if (mDequantBits == 4) {
            [dic setValue:@"1" forKey:@"W_QUANT_4"];
        } else if (mDequantBits == 8) {
            [dic setValue:@"1" forKey:@"W_QUANT_8"];
        }
        if (mIsGateUpLeader) {
            [dic setValue:@"1" forKey:@"GATE_UP_FUSED"];
        }
        if (mIsQKVLeader) {
            [dic setValue:@"1" forKey:@"QKV_FUSED"];
            if (mQKVCompactGrid) {
                [dic setValue:@"1" forKey:@"QKV_COMPACT_GRID"];
            }
            if (mQKVPeerW != nullptr) {
                [dic setValue:@"1" forKey:@"QKV_FUSED_P4"];
            }
        }
        [dic setValue:@"1" forKey:@"LN_FUSED"];
        if (w16) {
            [dic setValue:@"1" forKey:@"GEMV_QBLOCK_W16"];
            [dic setValue:@(mQ4W16BlockSlices).stringValue forKey:@"GEMV_QBLOCK_W16_BLOCK_SLICES"];
            [dic setValue:@(w16Mid).stringValue forKey:@"GEMV_QBLOCK_W16_MID"];
        }
        option.preprocessorMacros = dic;

        std::string sgrWqStr = std::string(gBasicConvPrefix) + gConv1x1WqSgReduce;
        mLNFusedPipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", option);
        rt->insertPipeline(keys, mLNFusedPipeline);
    }

    if (nil == mLNFusedPipeline) {
        mHasLNFusion = false;
        return false;
    }
    // LN-fused pipeline is 2sg-kernel based — force 64 threads (the plain
    // sole-consumer path dispatches mLNFusedPipeline with mThreads, which may
    // be split-K g8's 128-thread group).
    mThreads.second = MTLSizeMake(64, 1, 1);
    return true;
}

void MetalConvolution1x1::bindLNBuffers(id<MTLComputeCommandEncoder> encoder) {
    MetalBackend::setTensor(mLNResidualInput, encoder, 20);
    MetalBackend::setTensor(mLNGamma.get(), encoder, 21);
    MetalBackend::setTensor(mLNResidualOutput, encoder, 22);
    [encoder setBuffer:mLNEpsBuffer offset:0 atIndex:23];
}

ErrorCode MetalConvolution1x1::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MetalConvolutionCommon::onResize(inputs, outputs);

    mQ4W16BlockSlices = 0;

    // Reset Gate/Up fusion state on each resize
    mIs2sgDecode = false;
    mIsGateUpLeader = false;
    mIsGateUpFollower = false;
    mGateUpPeer = nullptr;
    mGateUpFusedPipeline = nil;
    mGateUpSegBuffer = nil;

    // Reset QKV fusion state on each resize
    mIsQKVLeader = false;
    mIsQKVFollower = false;
    mQKVPeerK = nullptr;
    mQKVPeerV = nullptr;
    mQKVPeerW = nullptr;
    mQKVPeerKOutput = nullptr;
    mQKVPeerVOutput = nullptr;
    mQKVPeerWOutput = nullptr;
    mQKVFusedPipeline = nil;
    mQKVSegBuffer = nil;
    mQKVCompactGrid = false;


    mHasLNFusion = false;
    mLNFusedPipeline = nil;
    mLNHiddenInput = nullptr;
    mLNResidualInput = nullptr;
    mLNResidualOutput = nullptr;
    mLNGamma = nullptr;
    mLNEpsBuffer = nil;

    // prepare
    // For C4NHW4 format, NHW can be fuse to W
    auto input = inputs[0];
    auto output = outputs[0];
    int is = input->batch();
    for (int i=2; i<input->dimensions(); ++i) {
        is *= input->length(i);
    }
    int ic  = input->channel();
    int ic_4  = UP_DIV(input->channel(), 4);
    int ow  = is;
    int oh  = 1;
    int os  = ow;
    int ob  = 1;
    auto oc  = output->channel();
    auto oc_4  = UP_DIV(output->channel(), 4);
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto dequantScale = getDequantScale();
    int blockSize = 1;
    if (dequantScale) {
        int bytes = sizeof(float);
        if(backend->useFp16InsteadFp32()) {
            bytes = sizeof(__fp16);
        }
        blockSize = (int)(dequantScale->usize() / bytes / oc_4 / 2 / 4);
    }
    // create const buffer
    mConstBuffer = backend->getConstBuffer(sizeof(Param));
    auto param = (Param *)mConstBuffer.contents;
    param->input_size = is;
    param->input_slice = ic_4;
    param->output_width = ow;
    param->output_height = oh;
    param->output_size = os;
    param->output_slice = oc_4;
    param->output_channel = oc;
    param->batch = ob;
    param->block_size = blockSize;
    param->activation = mActivationType;
    param->scale_coef = mScaleCoef;
    mBlockSize = blockSize;
    int area = ob * ow * oh;
    // basic marco info
    std::string ftype = "float";
    std::string ftype2 = "float2";
    std::string ftype4 = "float4";
    std::string ftype2x4 = "float2x4";
    std::string ftype4x4 = "float4x4";
    if (backend->useFp16InsteadFp32()) {
        ftype = "half";
        ftype2 = "half2";
        ftype4 = "half4";
        ftype2x4 = "half2x4";
        ftype4x4 = "half4x4";
    }

    MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
    auto baseDic = [NSMutableDictionary dictionaryWithCapacity:0];
    [baseDic setValue:@(ftype.c_str()) forKey:@"ftype"];
    [baseDic setValue:@(ftype2.c_str()) forKey:@"ftype2"];
    [baseDic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
    [baseDic setValue:@(ftype2x4.c_str()) forKey:@"ftype2x4"];
    [baseDic setValue:@(ftype4x4.c_str()) forKey:@"ftype4x4"];
    [baseDic setValue:@"1" forKey:@"MNN_METAL_FLOAT32_COMPUTER"];
    if (backend->useFp16InsteadFp32()) {
        [baseDic setValue:@"1" forKey:@"MNN_METAL_FLOAT16_STORAGE"];
    }
    std::vector<std::string> baseKeys = {ftype4, "MNN_METAL_FLOAT32_COMPUTER"};

    MetalRuntime* rt = (MetalRuntime *)backend->runtime();
    std::string basicShaderPrefix = gBasicConvPrefix;

    // if M is small, dequant weight in shader
    // if device not support simdgroup matrix, only support dequant in shader
    bool dequantInShader = (area < 64) || !(rt->supportSimdGroupMatrix());
    // Decode (area==1) has native W_QUANT_2/3 paths in the GEMV kernels (2sg, and
    // g16 for lm_head). The multi-token in-shader kernels (g4mN / sg-matrix gemm)
    // have no true W2/3 branches, so for prefill we route through the outer-dequant
    // + fp gemm path instead, which has a real W_QUANT_2/3 dequant in
    // conv1x1_w_dequant. The outer-dequant path itself uses simdgroup-matrix; only
    // override when the device supports it, otherwise stay on the in-shader path
    // (g8/g16 cover all areas in-shader there).
    if ((mDequantBits == 2 || mDequantBits == 3) && area > 1 && rt->supportSimdGroupMatrix()) {
        dequantInShader = false;
    }
    // Tensor API vs in-shader sg_matrix path for prefill (area > 1) with Q4/Q8.
    //
    // Default on M5+ (tensor API devices): route to outer-dequant + tensor API
    // GEMM (conv1x1_gemm_32x64_split_k_sg with USE_METAL_TENSOR_OPS).
    //
    // Measurement (M5, Qwen3-4B pp512, W4-block32, fp16, 4 threads):
    //   default (tensor API):  698 tok/s
    //   forced in-shader:      361 tok/s  (-48% — massive regression)
    //   llama.cpp Q4_0:       1039 tok/s (MNN gap: 33% behind)
    //
    // Tested the in-shader sg_matrix route on M5 and it
    // regresses catastrophically because the in-shader Q4 sg_matrix kernels
    // (conv1x1_gemm_32x16_wquant_sg / 16x32_wquant_sg / 32x64_wquant_split_k_sg)
    // do NOT use tensor API — they're pure SIMD-matrix. For pp512-scale
    // workloads on M5 the tensor API path is still faster, just not as fast
    // as llama.cpp's Metal kernels.
    //
    // The real gap vs llama.cpp is inside conv1x1_gemm_32x64_split_k_sg's
    // tensor API path — reproducing llama.cpp's efficiency requires kernel-
    // level changes (better matmul2d tiling, weight prefetching, etc.), not
    // dispatcher-level rerouting.
    //
    if (backend->isSupportTensorApi() && area > 1 && (mDequantBits == 4 || mDequantBits == 8)) {
        // On tensor-API devices (M5+) always force outer-dequant + tensor API.
        dequantInShader = false;
    }
    // On non-tensor-API devices (M4 and below), choose in-shader vs outer-dequant
    // based on weight size AND prompt length. In-shader dequant re-unpacks the Q4
    // weights once per M-tile (unpack count ~ area/32), so it only wins for large
    // weights at short area; outer-dequant pays a fixed double-pass instead.
    // M4 calibration (EXP11, Qwen3-4B): pp256 in-shader +1.2%, pp512 parity,
    // pp768 outdeq +2.4%, pp2048 outdeq +5.3%; Qwen3.5-2B pp2048 outdeq +3.0%.
    // ⚠️ M3 has a regression precedent with this heuristic — re-validate there
    // before relying on the 512 boundary outside M4.
    // Env MNN_METAL_PREFILL_INSHADER_DEQUANT_SGMATRIX=1 forces on, =0 forces off.
    if (!backend->isSupportTensorApi() && rt->supportSimdGroupMatrix() && area > 1 &&
        (mDequantBits == 4 || mDequantBits == 8)) {
        const int kForceInShader = MetalEnv::get().prefillInshaderDequant;
        if (kForceInShader == 1) {
            dequantInShader = true;
        } else if (kForceInShader == -1) {
            dequantInShader = false;
        } else if ((size_t)ic * oc > 4 * 1024 * 1024 && area < 512) {
            dequantInShader = true;
        }
    }
    mPreDequantWeight = false;
    mUseFusedDecode = false;

#ifdef MNN_LOW_MEMORY
    if (dequantScale && dequantInShader) {
        //printf("inner dequant MNK: %d %d %d %d\n", area, oc, ic, blockSize);

        std::string sgmWqShader  = gConv1x1WqSgMatrix;
        std::string sgrWqShader  = gConv1x1WqSgReduce;

        NSMutableDictionary *dic = [baseDic mutableCopy];
        if(mDequantBits == 2) {
            [dic setValue:@"1" forKey:@"W_QUANT_2"];
        } else if(mDequantBits == 3) {
            [dic setValue:@"1" forKey:@"W_QUANT_3"];
        } else if(mDequantBits == 4) {
            [dic setValue:@"1" forKey:@"W_QUANT_4"];
        } else if(mDequantBits == 8) {
            [dic setValue:@"1" forKey:@"W_QUANT_8"];
        }
        // Q4 block 32/64/128/256 decode specialization for the standalone GEMV
        // kernels. The C4 slices per quant block become a compile-time constant
        // (8/16/32/64), enabling 16-byte weight reads without tail checks.
        int q4W16BlockSlices = 0;
        if (area == 1 && mDequantBits == 4 && blockSize > 0 && ic_4 % blockSize == 0) {
            const int candidate = ic_4 / blockSize;
            if (isQ4W16BlockSlices(candidate)) {
                q4W16BlockSlices = candidate;
            }
        }
        const bool q4W16 = q4W16BlockSlices > 0;
        // Record the layout so a fusion leader can extend W16 onto the fused
        // decode-GEMV pipeline built later in setup{QKV,GateUp,LN}Fusion.
        mQ4W16BlockSlices = q4W16BlockSlices;
        if (q4W16) {
            [dic setValue:@"1" forKey:@"GEMV_QBLOCK_W16"];
            [dic setValue:@(q4W16BlockSlices).stringValue forKey:@"GEMV_QBLOCK_W16_BLOCK_SLICES"];
            // Non-split-K pipelines: the simdgroup owns every block of the row.
            [dic setValue:@(chooseQ4W16Mid(blockSize, q4W16BlockSlices / 2)).stringValue
                   forKey:@"GEMV_QBLOCK_W16_MID"];
            [dic setValue:@"1" forKey:@"GEMV_QBLOCK_W16_LMHEAD"];
        }
        option.preprocessorMacros = dic;

        NSUInteger gid_x = UP_DIV(ow * oh, 4);
        NSUInteger gid_y = oc_4;
        NSUInteger gid_z = ob;
        std::string name = "conv1x1_g1z4_w8";
        mPipeline = [context pipelineWithName:@"conv1x1_g1z4_w8" fp16:backend->useFp16InsteadFp32()];

        if (mDequantBits == 2 || mDequantBits == 3 || mDequantBits == 4 || mDequantBits == 8) {
            // TODO: define short_seq more accurately
            int short_seq = 16;

            if(mDequantBits == 2) {
                baseKeys.emplace_back("conv1x1_wquant_2");
            } else if(mDequantBits == 3) {
                baseKeys.emplace_back("conv1x1_wquant_3");
            } else if(mDequantBits == 4) {
                baseKeys.emplace_back("conv1x1_wquant_4");
            } else if(mDequantBits == 8) {
                baseKeys.emplace_back("conv1x1_wquant_8");
            }
            if (q4W16) {
                baseKeys.emplace_back("GEMV_QBLOCK_W16");
                baseKeys.emplace_back("W16_BLOCK_SLICES_" + std::to_string(q4W16BlockSlices));
                baseKeys.emplace_back("W16_MID_" +
                                      std::to_string(chooseQ4W16Mid(blockSize, q4W16BlockSlices / 2)));
                baseKeys.emplace_back("GEMV_QBLOCK_W16_LMHEAD");
            }
            // W_QUANT_2/3 on non-simdgroup-matrix devices: the outer-dequant GEMM
            // (sg-matrix based) and the g1z4 fallback below are both unusable, so
            // g8/g16 must cover all areas in-shader, not just area <= short_seq.
            const bool w23NoMatrix = (mDequantBits == 2 || mDequantBits == 3) && !rt->supportSimdGroupMatrix();
            if(rt->supportSimdGroupReduce() && (area <= short_seq || w23NoMatrix)) {
                baseKeys.emplace_back("conv1x1_wquant_sg_reduce");

                std::string sgrWqStr = basicShaderPrefix + sgrWqShader;
                // g4mN kernels now have true W_QUANT_2/3 branches.
                if(area > 1 && (mDequantBits == 2 || mDequantBits == 3 || mDequantBits == 4 || mDequantBits == 8)) {
                    auto keys = baseKeys;
                    int piece = 1;
                    // memory bound not so seriously, can add more thread to reduce computation in each thread
                    float ratio = 1.0 * ic_4 / 2048.0 * oc / 2048.0;
                    bool heavyMemory = ratio > 1.0;
                    if(area > 5 && !heavyMemory) {
                        if(area % 2 != 0) {
                            keys.emplace_back("MNN_METAL_SRC_PROTECT");
                            [dic setValue:@"1" forKey:@"MNN_METAL_SRC_PROTECT"];;
                            option.preprocessorMacros = dic;
                        }
                        area = UP_DIV(area, 2);
                        piece = 2;
                    }
//                    MNN_PRINT("Conv1x1 Oc:%d Ic:%d\n", oc, ic_4*4);
                    std::string kernel_name = "conv1x1_gemv_g4m" + std::to_string(area) + "_wquant_sg";
                    keys.emplace_back(kernel_name);
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), kernel_name.c_str(), option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 4), piece, 1), MTLSizeMake(32, 1, 1));
                } else if(oc > 16384 && oc_4 % 2 == 0) {
                    // lm_head path. Baseline g16 = 2 simdgroups per TG,
                    // threadgroup size 64, each TG covers 16 OC (2 SG x 8 OC/SG).
                    // Variants explored and retired (see skills/metal-optimize):
                    // 4SG (halved grid) — e2e neutral with 7x worse stddev on M5;
                    // G16_OC4 (4 oc_4 rows/SG) — kernel -4.8% on M5 but e2e neutral.
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemv_g16_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g16_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 16), area, 1),
                                              MTLSizeMake(64, 1, 1));
                } else if(area == 1) {
                    // Decode GEMV. Per-kernel bandwidth profiling (M4 Pro,
                    // Qwen3-0.6B Q4) showed the 2sg kernel is latency-limited on
                    // small projections: one simdgroup streams a whole row, so
                    // 0.5-1.8MB matrices reach only 88-137 GB/s while the 87MB
                    // lm_head (g16 path) hits 252 GB/s. SPLIT_K_2 keeps the same
                    // pre-scaling inner loop but runs 4 simdgroups per
                    // threadgroup — two K-halves per row combined via threadgroup
                    // memory — doubling in-flight reads per row.
                    // (Routing to the g8 kernel instead was tried first: its
                    // nibble-unpack inner loop is slower and lost 5% e2e.)
                    // MNN_METAL_GEMV_SPLITK=0 restores the legacy 2sg kernel.
                    //
                    // Legacy 2sg lane partitioning notes (WIDE_MIDDLE knob, M5
                    // regression) kept in the shader comment block.
                    const int sSplitK = MetalEnv::get().gemvSplitK;
                    const bool splitkUsable = (oc % 8 == 0) && (blockSize % 2 == 0);
                    if (sSplitK > 0 && splitkUsable) {
                        auto keys = baseKeys;
                        keys.emplace_back("conv1x1_gemv_g4m1_2sg_wquant_sg");
                        keys.emplace_back("SPLIT_K_2");
                        // Split-K halves the blocks a simdgroup owns, so the
                        // lane split is re-chosen for that pipeline.
                        const int skMid = q4W16 ? chooseQ4W16Mid(blockSize / 2, q4W16BlockSlices / 2) : 0;
                        if (q4W16) {
                            keys.emplace_back("W16_SK_MID_" + std::to_string(skMid));
                        }
                        auto pipeline = rt->findPipeline(keys);
                        if (nil == pipeline) {
                            NSMutableDictionary *skDic = [dic mutableCopy];
                            [skDic setValue:@"1" forKey:@"SPLIT_K_2"];
                            if (q4W16) {
                                [skDic setValue:@(skMid).stringValue forKey:@"GEMV_QBLOCK_W16_MID"];
                            }
                            MTLCompileOptions *skOption = [[MTLCompileOptions alloc] init];
                            skOption.preprocessorMacros = skDic;
                            pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", skOption);
                            rt->insertPipeline(keys, pipeline);
                        }
                        mPipeline = pipeline; CONV1X1_SET_TAG("splitk2_gemv_g4m1_2sg_wquant_sg");
                        mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 8), 1, 1), MTLSizeMake(128, 1, 1));
                    } else {
                        auto keys = baseKeys;
                        keys.emplace_back("conv1x1_gemv_g4m1_2sg_wquant_sg");
                        auto pipeline = rt->findPipeline(keys);
                        if (nil == pipeline) {
                            pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", option);
                            rt->insertPipeline(keys, pipeline);
                        }
                        mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                        // 2 simdgroups per threadgroup, each handles 4 OC independently
                        mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 8), 1, 1), MTLSizeMake(64, 1, 1));
                    }
                    // Fusion leaders (gate/up, qkv, LN) build their own 2sg-based
                    // pipelines and force a 64-thread dispatch in their setup.
                    mIs2sgDecode = true;
                } else {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemv_g8_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g8_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
//                    MNN_PRINT("g8  ic: %d oc: %d\n", input->channel(), oc);
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 8), area, 1), MTLSizeMake(128, 1, 1));
                }
                return NO_ERROR;
            } else if(rt->supportSimdGroupMatrix()  && area > short_seq && oc > 8 && (ic_4 % 8 == 0 || ic_4 % 2 == 0)) {
                baseKeys.emplace_back("conv1x1_wquant_sg_matrix");
                std::string sgmWqStr = basicShaderPrefix + sgmWqShader;

                // Generally threadgroup memory >= 16KB
                auto smem_size = [[context device] maxThreadgroupMemoryLength];
                // choose different tile for different computation
                if(ic_4 % 8 != 0) {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_8x16_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), "conv1x1_gemm_8x16_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 8), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
                } else if(area >= 128 && oc >= 512 && area * oc > 512 * 2048 && smem_size >= 8192) {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_32x64_wquant_split_k_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), "conv1x1_gemm_32x64_wquant_split_k_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 32), UP_DIV(oc, 64), 1), MTLSizeMake(128, 1, 1));

                } else if(area >= 32 && area * oc > 128 * 2048) {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_32x16_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), "conv1x1_gemm_32x16_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 32), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
                } else if(oc > 512 && area * oc > 128 * 2048) {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_16x32_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), "conv1x1_gemm_16x32_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 16), UP_DIV(oc, 32), 1), MTLSizeMake(32, 1, 1));
                } else if(area < 16) {
                    // TODO: define useMatrix more accurate
                    bool useMatrix = area > 6 && oc > 2048 && ic*2 < oc;
                    if(useMatrix) {
                        auto keys = baseKeys;
                        int oc_block = (oc > 4096) ? 32 : 16;
                        std::string kernel_name = "conv1x1_gemm_8x" + std::to_string(oc_block) + "_wquant_sg";

                        keys.emplace_back(kernel_name);
                        auto pipeline = rt->findPipeline(keys);
                        if (nil == pipeline) {
                            pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), kernel_name.c_str(), option);
                            rt->insertPipeline(keys, pipeline);
                        }
                        mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                        mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 8), UP_DIV(oc, oc_block), 1), MTLSizeMake(32, 1, 1));
                    } else {
                        std::string sgrWqStr = basicShaderPrefix + sgrWqShader;

                        auto keys = baseKeys;
                        std::string kernel_name = "conv1x1_gemv_g4m" + std::to_string(area) + "_wquant_sg";
                        keys.emplace_back(kernel_name);
                        auto pipeline = rt->findPipeline(keys);
                        if (nil == pipeline) {
                            pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), kernel_name.c_str(), option);
                            rt->insertPipeline(keys, pipeline);
                        }
                        mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                        mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 4), 1, 1), MTLSizeMake(32, 1, 1));
                    }
                } else {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_16x16_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), "conv1x1_gemm_16x16_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
//                                    MNN_PRINT("gemm M: %d N: %d\n", area, oc);
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 16), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
                }
                return NO_ERROR;
            } else if(mDequantBits == 4) {
                mPipeline = [context pipelineWithName:@"conv1x1_g1z4_w4" fp16:backend->useFp16InsteadFp32()];
                name = "conv1x1_g1z4_w4";
            } else if(mDequantBits == 8) {
                mPipeline = [context pipelineWithName:@"conv1x1_g1z4_w8" fp16:backend->useFp16InsteadFp32()];
                name = "conv1x1_g1z4_w8";
            } else {
                // W_QUANT_2/3 without simdGroupReduce: no usable kernel exists
                // (g1z4_w4/w8 would misread the packed 2/3-bit buffer, and the
                // outer-dequant GEMM requires simdgroup matrix).
                MNN_ERROR("metal W_QUANT_%d conv1x1 requires simdgroup reduce support!\n", mDequantBits);
                return NOT_SUPPORT;
            }
        } else {
            MNN_ERROR("metal conv weight quant not support %d bits yet!\n", mDequantBits);
        }
        NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                        (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                        mConstBuffer, (((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId()))->getBuffer(),
                        ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(),
                        (((MetalRuntimeAllocator::MetalBufferAlloc *)dequantScale->deviceId()))->getBuffer(),
                        nil];
        const Tensor* weight = mWeight.get();
        const Tensor* bias = mBias.get();
        int buffer_offset[] = {
            TensorUtils::getDescribeOrigin(input)->offset,
            TensorUtils::getDescribeOrigin(output)->offset,
            0,
            TensorUtils::getDescribeOrigin(weight)->offset,
            TensorUtils::getDescribeOrigin(bias)->offset,
            TensorUtils::getDescribeOrigin(dequantScale.get())->offset,
            0};

        MetalRuntime *rt = (MetalRuntime *)backend->runtime();
        auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name offsets:buffer_offset  queue:backend->queue()];
        mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
        CONV1X1_SET_TAG(name);
        return NO_ERROR;
    }
#endif

    std::string sgmWfpShader = std::string(gConv1x1WfpSgMatrix) + gConv1x1WfpSgMatrixM64;
    std::string sgrWfpShader = gConv1x1WfpSgReduce;

    // Dequant using single shader
    if (dequantScale) {
        baseKeys.emplace_back("conv1x1_dequant_weight_outter");
        std::string sgmWfpStr = basicShaderPrefix + sgmWfpShader;

        mPreDequantWeight = true;
        auto specMeta = (KVMeta *)backend->getMetaPtr();
        const bool specBlock = (specMeta != nullptr && specMeta->spec_block > 0);
        {
            NSMutableDictionary *dic = [baseDic mutableCopy];

            auto keys = baseKeys;
            keys.emplace_back("conv1x1_w_dequant");
            if(mDequantBits == 2) {
                [dic setValue:@"1" forKey:@"W_QUANT_2"];
                keys.emplace_back("W_QUANT_2");
            } else if(mDequantBits == 3) {
                [dic setValue:@"1" forKey:@"W_QUANT_3"];
                keys.emplace_back("W_QUANT_3");
            } else if(mDequantBits == 4) {
                [dic setValue:@"1" forKey:@"W_QUANT_4"];
                keys.emplace_back("W_QUANT_4");
            } else if(mDequantBits == 8) {
                [dic setValue:@"1" forKey:@"W_QUANT_8"];
                keys.emplace_back("W_QUANT_8");
            }
            if(ic % 16 != 0) {
                [dic setValue:@"1" forKey:@"W_ALIGN_K16_PROTECT"];
                keys.emplace_back("W_ALIGN_K16_PROTECT");
            }
            option.preprocessorMacros = dic;

            // Fused quant GEMM: the fused kernel unpacks quantized weights
            // in-kernel, skipping both the dequant pre-pass dispatch and the
            // mTempWeight allocation.
            // Enabled when proven correct and profitable: W2/W3/W4/W8, tensor-API
            // capable device, prefill (area >= 64), or a speculative block (else a
            // block-sized decode re-dequantizes the whole weight every encode).
            // MNN_METAL_W4W8_OUTER_DEQUANT_GEMM_TENSORAPI=1 forces the outer-dequant
            // baseline instead (A/B + emergency rollback; see
            // skills/metal-optimize/env-registry.md).
            // W3 is additionally gated by weight size: its 6-scalar-load +
            // hi-mask unpack costs more ALU per M-tile than W2/W4's wide loads,
            // which goes net-negative when fp16 layer weights stay L2-resident
            // (M5: 0.6B W3 pp2048 fused -3.6% both orders, pp512 neutral) but
            // wins once weights exceed L2 (4B W3 +10~11% pp512/pp2048). 4M
            // elements ~ fp16 8MB, same boundary as the in-shader dequant gate;
            // splits the calibrated points (0.6B max conv 3.1M, 4B min 6.5M).
            auto specMeta = (KVMeta *)backend->getMetaPtr();
            const bool specBlock = (specMeta != nullptr && specMeta->spec_block > 0);
            const bool fusedQ4 = !MetalEnv::get().w4w8OuterDequantGemm &&
                                 (mDequantBits == 2 || mDequantBits == 3 ||
                                  mDequantBits == 4 || mDequantBits == 8) &&
                                 backend->isSupportTensorApi() &&
                                 (area >= 64 || specBlock) &&
                                 (mDequantBits != 3 ||
                                  (int64_t)oc * ic >= (int64_t)4 * 1024 * 1024);

            // M_TILE=64 variant (requires tensor API — implicitly M5+).
            // Measured on M5, Qwen3-4B, Metal fp16, 4 threads, 3-rep A/B:
            //   pp512  M32 851 t/s  -> M64 901 t/s  (+5.9%)
            //   pp2048 M32 715 t/s  -> M64 764 t/s  (+6.8%)
            mFusedQ4M64 = fusedQ4 && mDequantBits == 4 && area >= 128;

            mFusedQ4 = fusedQ4;

            if (!fusedQ4) {
                int bytes = backend->useFp16InsteadFp32() ? 2 : 4;
                const int tempSize = ROUND_UP(oc, 4) * ROUND_UP(ic, 32) * bytes;
                // Size depends only on oc/ic/bytes, so create the Tensor once and
                // keep the object for the execution's lifetime: a recorded
                // encode-replay holds raw Tensor* to it, and destroying it would
                // leave that recording dangling. Only the buffer range is
                // acquired per resize; metalReplayValidate catches an address
                // change on its own.
                if (mTempWeight == nullptr || mTempWeight->elementSize() != tempSize) {
                    mTempWeight.reset(Tensor::createDevice<uint8_t>(std::vector<int>{tempSize}));
                }
                backend->onAcquireBuffer(mTempWeight.get(), Backend::DYNAMIC);
                backend->onReleaseBuffer(mTempWeight.get(), Backend::DYNAMIC);

                auto pipeline = rt->findPipeline(keys);
                if (nil == pipeline) {
                    pipeline = backend->makeComputePipelineWithSourceOption(sgmWfpStr.c_str(), "conv1x1_w_dequant", option);
                    rt->insertPipeline(keys, pipeline);
                }
                mDequantPipeline = pipeline;

                mDequantThreads = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(UP_DIV(oc, 1),  UP_DIV(ic, 16), 1)];
            } else {
                mDequantPipeline = nil;
                // mTempWeight is deliberately kept: see above.
            }
        }

        {
            auto keys = baseKeys;
            const char* gemmKernelName = "conv1x1_gemm_32x64_split_k_sg";
            bool sgMatrixM64 = false;
            // K-split x4 recovers TG-starved speculative-verify shapes; TG-rich ones
            // (lm_head) stay out. MNN_METAL_FUSED_Q4_KSPLIT: unset auto, =0 off, =1 on.
            const auto& metalEnv = MetalEnv::get();
            const bool fusedQ4SpecBlock = mFusedQ4 && mDequantPipeline == nil &&
                                      mDequantBits == 4 && !mFusedQ4M64 && specBlock;
            mUseFusedKsplit = false;
            if (fusedQ4SpecBlock && metalEnv.fusedQ4Ksplit >= 0) {
                mUseFusedKsplit = metalEnv.fusedQ4Ksplit == 1 ||
                                  (area <= 32 && UP_DIV(oc, 64) <= 48 && blockSize >= 4);
            }
            // M8-native tile for small-M shapes the K-split gate skips (a padded M32
            // tile runs at quarter occupancy). MNN_METAL_FUSED_Q4_M8=0 forces it off.
            const bool useFusedM8 = fusedQ4SpecBlock && !mUseFusedKsplit &&
                                    area > 1 && area <= 8 && metalEnv.fusedQ4M8 >= 0;
            if (mFusedQ4) {
                if (mUseFusedKsplit) {
                    // Stack the M8 tile on K-split when the block is narrow enough;
                    // MNN_METAL_FUSED_Q4_KSPLIT_M8=0 keeps the M32 tile.
                    mKsplitM8 = area <= 8 && metalEnv.fusedQ4KsplitM8 >= 0;
                    gemmKernelName = mKsplitM8 ? "conv1x1_fused_q4_gemm_stage_ksplit_m8"
                                               : "conv1x1_fused_q4_gemm_stage_ksplit";
                    keys.emplace_back(gemmKernelName);
                } else if (useFusedM8) {
                    gemmKernelName = "conv1x1_fused_q4_gemm_stage_m8";
                    keys.emplace_back("conv1x1_fused_q4_gemm_stage_m8");
                } else if (mFusedQ4M64) {
                    gemmKernelName = "conv1x1_fused_q4_gemm_stage_m64";
                    keys.emplace_back("conv1x1_fused_q4_gemm_stage_m64");
                } else {
                    gemmKernelName = "conv1x1_fused_q4_gemm_stage";
                    keys.emplace_back("conv1x1_fused_q4_gemm_stage");
                }
            } else if (!backend->isSupportTensorApi() &&
                       ((MetalRuntime*)backend->runtime())->preferM64Gemm() && area >= 128) {
                // sg_matrix M=64 tile, device-tiered via architecture.name
                // (M4-class Macs only, see MetalBackend.mm; env removed 2026-07-31):
                // halves grid.x / weight DRAM traffic; fp16 weights from the
                // outer-dequant pre-pass, same bindings as the 32x64 kernel.
                gemmKernelName = "conv1x1_gemm_64x64_split_k_sg";
                keys.emplace_back("conv1x1_gemm_64x64_split_k_sg");
                sgMatrixM64 = true;
            } else {
                keys.emplace_back("conv1x1_gemm_32x64_split_k_sg");
            }

            NSMutableDictionary *dic = [baseDic mutableCopy];
            if (ic_4 % 8 != 0) {
                [dic setValue:@"1" forKey:@"MNN_METAL_SRC_PROTECT"];
                keys.emplace_back("MNN_METAL_SRC_PROTECT");
            }
            if(backend->isSupportTensorApi() == true) {
                [dic setValue:@"1" forKey:@"USE_METAL_TENSOR_OPS"];
                keys.emplace_back("USE_METAL_TENSOR_OPS");
                if(ic > oc && ic > 2048 && (ic / blockSize) % 64 == 0 && !mFusedQ4) {
                    // LOOP_K64 branch only exists for conv1x1_gemm_32x64_split_k_sg.
                    // Fused-stage kernel is always K=32 tile.
                    [dic setValue:@"1" forKey:@"LOOP_K64"];
                    keys.emplace_back("LOOP_K64");
                }
            }
            // Fused-stage kernel is compiled with W_QUANT_{2,3,4,8}.
            // (kernel body is guarded by `#if defined(W_QUANT_2) || defined(W_QUANT_3)
            //  || defined(W_QUANT_4) || defined(W_QUANT_8)`).
            if (mFusedQ4) {
                switch (mDequantBits) {
                    case 2:
                        [dic setValue:@"1" forKey:@"W_QUANT_2"];
                        keys.emplace_back("W_QUANT_2");
                        break;
                    case 3:
                        [dic setValue:@"1" forKey:@"W_QUANT_3"];
                        keys.emplace_back("W_QUANT_3");
                        break;
                    case 4:
                        [dic setValue:@"1" forKey:@"W_QUANT_4"];
                        keys.emplace_back("W_QUANT_4");
                        // The M8/K-split kernels are guarded by W_QUANT_4 && FUSED_Q4_REAL_UNPACK;
                        // without the macro the library compiles but pipeline lookup finds no function.
                        [dic setValue:@"1" forKey:@"FUSED_Q4_REAL_UNPACK"];
                        keys.emplace_back("FUSED_Q4_REAL_UNPACK");
                        break;
                    default:
                        [dic setValue:@"1" forKey:@"W_QUANT_8"];
                        keys.emplace_back("W_QUANT_8");
                        break;
                }
            }
            option.preprocessorMacros = dic;

            auto pipeline = rt->findPipeline(keys);
            if (nil == pipeline) {
                pipeline = backend->makeComputePipelineWithSourceOption(sgmWfpStr.c_str(), gemmKernelName, option);
                rt->insertPipeline(keys, pipeline);
            }
            mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
            if (mUseFusedKsplit) {
                // grid gains z=4 K-partitions; the reduce pass sums the fp32
                // partials and applies bias + activation.
                mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, mKsplitM8 ? 8 : 32), UP_DIV(oc, 64), 4), MTLSizeMake(128, 1, 1));
                auto rkeys = keys;
                rkeys.emplace_back("conv1x1_fused_q4_ksplit_reduce");
                auto rpipeline = rt->findPipeline(rkeys);
                if (nil == rpipeline) {
                    rpipeline = backend->makeComputePipelineWithSourceOption(sgmWfpStr.c_str(), "conv1x1_fused_q4_ksplit_reduce", option);
                    rt->insertPipeline(rkeys, rpipeline);
                }
                mKsplitReducePipeline = rpipeline;
                // fp32 partials [KS=4, oc_4, area, 4] = 4 * oc_4 * area * 16 bytes.
                int totalOut = oc_4 * area;
                mKsplitPartial.reset(Tensor::createDevice<uint8_t>(std::vector<int>{4 * totalOut * 16}));
                backend->onAcquireBuffer(mKsplitPartial.get(), Backend::DYNAMIC);
                backend->onReleaseBuffer(mKsplitPartial.get(), Backend::DYNAMIC);
                mKsplitReduceThreads = std::make_pair(MTLSizeMake(UP_DIV(totalOut, 128), 1, 1), MTLSizeMake(128, 1, 1));
            } else {
                const int mTile = useFusedM8 ? 8 : ((mFusedQ4M64 || sgMatrixM64) ? 64 : 32);
                mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, mTile), UP_DIV(oc, 64), 1), MTLSizeMake(128, 1, 1));
            }
            //printf("out dequant MNK: %d %d %d %d\n", area, oc, ic, blockSize);
        }

        return NO_ERROR;
    }

    option.preprocessorMacros = baseDic;

    if(rt->supportSimdGroupMatrix()) {
        std::string sgmWfpStr = basicShaderPrefix + sgmWfpShader;

        baseKeys.emplace_back("conv1x1_float_sg_matrix");
        // total computation not too small
        if(area >= 16 && ic_4 >= 4 && ic_4 % 2 == 0 && oc_4 >= 4 && area * ic_4 * oc_4 >= 64 * 64 * 64) {
            // Enough threads
            if(area * oc_4 / ic_4 >= 1024) {
                auto keys = baseKeys;
                keys.emplace_back("conv1x1_gemm_32x16_sg");
                auto pipeline = rt->findPipeline(keys);
                if (nil == pipeline) {
                    pipeline = backend->makeComputePipelineWithSourceOption(sgmWfpStr.c_str(), "conv1x1_gemm_32x16_sg", option);
                    rt->insertPipeline(keys, pipeline);
                }
                mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 32), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
            } else {
                auto keys = baseKeys;
                keys.emplace_back("conv1x1_gemm_16x16_sg");
                auto pipeline = rt->findPipeline(keys);
                if (nil == pipeline) {
                    pipeline = backend->makeComputePipelineWithSourceOption(sgmWfpStr.c_str(), "conv1x1_gemm_16x16_sg", option);
                    rt->insertPipeline(keys, pipeline);
                }
                mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 16), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
            }
            return NO_ERROR;
        }
    }
    if(rt->supportSimdGroupReduce()) {
        std::string sgrWfpStr = basicShaderPrefix + sgrWfpShader;

        baseKeys.emplace_back("conv1x1_float_sg_reduce");
        // do input_channel reduce
        auto magic_num = 4.0; // total threads pretty small and loop pretty large
        if(ic_4 >= 32 && ic_4 % 2 == 0 && 1.0 * area * oc_4 / ic_4 < magic_num) {
            auto keys = baseKeys;
            keys.emplace_back("conv1x1_z4_sg");
            auto pipeline = rt->findPipeline(keys);
            if (nil == pipeline) {
                pipeline = backend->makeComputePipelineWithSourceOption(sgrWfpStr.c_str(), "conv1x1_z4_sg", option);
                rt->insertPipeline(keys, pipeline);
            }
            mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
            mThreads = std::make_pair(MTLSizeMake(ow * oh, oc_4, ob), MTLSizeMake(32, 1, 1));
            return NO_ERROR;
        }
    }
//    printf("lora: %d %d %d %d %d\n", ob, oh, ow, oc, input->channel());
    if(rt->getTuneLevel() == Never) {
        if (ow * oh >= 128) {
            NSUInteger gid_x = UP_DIV(ow * oh, 8);
            NSUInteger gid_y = oc_4;
            NSUInteger gid_z = ob;

            mPipeline = [context pipelineWithName:@"conv1x1_g1z8" fp16:backend->useFp16InsteadFp32()];

            NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                            (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                            mConstBuffer, (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId()))->getBuffer(), ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(), nil];

            const Tensor* weight = mWeight.get();
            const Tensor* bias = mBias.get();
            int buffer_offset[] = {TensorUtils::getDescribeOrigin(input)->offset, TensorUtils::getDescribeOrigin(output)->offset, 0, TensorUtils::getDescribeOrigin(weight)->offset, TensorUtils::getDescribeOrigin(bias)->offset, 0};
            std::string name = "conv1x1_g1z8";
            MetalRuntime *rt = (MetalRuntime *)backend->runtime();
            auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name offsets: buffer_offset queue:backend->queue()];
            mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
            CONV1X1_SET_TAG(name);
        } else {
            NSUInteger gid_x = UP_DIV(ow * oh, 4);
            NSUInteger gid_y = oc_4;
            NSUInteger gid_z = ob;

            mPipeline = [context pipelineWithName:@"conv1x1_g1z4" fp16:backend->useFp16InsteadFp32()];

            NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                            (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                            mConstBuffer, (((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId()))->getBuffer(), ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(), nil];
            const Tensor* weight = mWeight.get();
            const Tensor* bias = mBias.get();
            int buffer_offset[] = {TensorUtils::getDescribeOrigin(input)->offset, TensorUtils::getDescribeOrigin(output)->offset, 0,  TensorUtils::getDescribeOrigin(weight)->offset, TensorUtils::getDescribeOrigin(bias)->offset, 0};
            std::string name = "conv1x1_g1z4";
            MetalRuntime *rt = (MetalRuntime *)backend->runtime();
            auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name offsets: buffer_offset queue:backend->queue()];
            mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
            CONV1X1_SET_TAG(name);
            //printf("conv1x1_z4, %d %d %d %d\n", ow, oh, oc_4, ic_4);
        }
    } else {
        NSString* shaderName[] = {@"conv1x1_g1z8", @"conv1x1_g1z4", @"conv1x1_w4h4",  @"conv1x1_w2c2", @"conv1x1_w4c2"};
        int itemW[] = {8, 4, 16, 2, 4};
        int itemC[] = {4, 4, 4, 8, 8};
        int actual_kernel = 5;
        if (oc_4 % 2 != 0) {
            // Don't unrool c for avoid memory exceed
            actual_kernel = 3;
        }
        std::pair<NSUInteger, int> min_cost(INT_MAX, 0);//(min_time, min_index)

        NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                        (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                        mConstBuffer, (((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId()))->getBuffer(), ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(), nil];
        const Tensor* weight = mWeight.get();
        const Tensor* bias = mBias.get();
        int buffer_offset[] = {TensorUtils::getDescribeOrigin(input)->offset, TensorUtils::getDescribeOrigin(output)->offset, 0, TensorUtils::getDescribeOrigin(weight)->offset, TensorUtils::getDescribeOrigin(bias)->offset, 0};

        for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
            id<MTLComputePipelineState> pipeline = [context pipelineWithName:shaderName[knl_idx] fp16:backend->useFp16InsteadFp32()];
            NSUInteger gid_x = UP_DIV(ow, itemW[knl_idx]);
            NSUInteger gid_y = UP_DIV(oc, itemC[knl_idx]);
            NSUInteger gid_z = 1;

            std::string name = [shaderName[knl_idx] UTF8String];
            auto ret = [context getGridAndThreadgroup:pipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name offsets:buffer_offset queue:backend->queue()];

            if(min_cost.first > std::get<2>(ret)) {
                min_cost.first = std::get<2>(ret);
                min_cost.second = knl_idx;
                mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
            }
            //printf("conv1x1 idx:%d, global:%d %d %d, local:%d %d %d, min_cost:%d\n", knl_idx, (int)retTune.second.first.width, (int)retTune.second.first.height, (int)retTune.second.first.depth, (int)retTune.second.second.width, (int)retTune.second.second.height, (int)retTune.second.second.depth, (int)retTune.first);
        }
        //printf("conv1x1 idx:%d, min_cost:%d\n", (int)min_cost.second, (int)min_cost.first);
        mPipeline = [context pipelineWithName:shaderName[min_cost.second] fp16:backend->useFp16InsteadFp32()];
        CONV1X1_SET_TAG(std::string([shaderName[min_cost.second] UTF8String]));
    }

    return NO_ERROR;
}

void MetalConvolution1x1::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    // Gate/Up follower: the leader already dispatched this projection
    if (mIsGateUpFollower) {
#if MNN_METAL_OP_PROFILE
        static_cast<MetalBackend *>(backend())->profileDropCurrentSample();
#endif
        return;
    }
    // QKV follower: the leader already dispatched this projection
    if (mIsQKVFollower) {
#if MNN_METAL_OP_PROFILE
        static_cast<MetalBackend *>(backend())->profileDropCurrentSample();
#endif
        return;
    }
#if MNN_METAL_OP_PROFILE
    // Report kernel-variant tag so the profile output can distinguish shader paths
    // (e.g. Convolution/gemm_32x64_split_k_sg vs Convolution/gemv_g4m1_2sg_wquant_sg).
    {
        std::string subtag = mProfileTag;
        if (mIsGateUpLeader) {
            subtag = "gate_up_fused_" + subtag;
        } else if (mIsQKVLeader) {
            subtag = "qkv_fused_" + subtag;
        } else if (mPreDequantWeight) {
            subtag = "outdeq+" + subtag;
        }
        static_cast<MetalBackend *>(backend())->setProfileSubtag(subtag);
    }
#endif

    auto input = inputs[0];
    auto output = outputs[0];

    // Gate/Up leader: dispatch fused kernel covering both gate and up projections
    if (mIsGateUpLeader && mGateUpPeer && nil != (mHasLNFusion ? mLNFusedPipeline : mGateUpFusedPipeline) && mGateUpPeerOutput) {
        [encoder setComputePipelineState:(mHasLNFusion ? mLNFusedPipeline : mGateUpFusedPipeline)];
        // buffer(0): input (shared by gate and up) — with LN fusion, use hidden input
        {
            auto inTensor = mHasLNFusion ? mLNHiddenInput : input;
            [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)inTensor->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(inTensor)->offset atIndex:0];
        }
        // buffer(1): gate output (this)
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(output)->offset atIndex:1];
        // buffer(2): gate params (also used by up since dimensions are identical)
        [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
        // buffer(3): gate weight
        MetalBackend::setTensor(mWeight.get(), encoder, 3);
        // buffer(4): gate bias
        MetalBackend::setTensor(mBias.get(), encoder, 4);
        // buffer(5): gate dequant scale
        MetalBackend::setTensor(getDequantScale().get(), encoder, 5);
        // buffer(6): up output
        MetalBackend::setTensor(mGateUpPeerOutput, encoder, 6);
        // buffer(7): up weight
        MetalBackend::setTensor(mGateUpPeer->getWeight().get(), encoder, 7);
        // buffer(8): up bias
        MetalBackend::setTensor(mGateUpPeer->getBias().get(), encoder, 8);
        // buffer(9): up dequant scale
        MetalBackend::setTensor(mGateUpPeer->getDequantScale().get(), encoder, 9);
        // buffer(14): {up_scale_coef} - per-tensor coefficient used by up branch
        [encoder setBuffer:mGateUpSegBuffer offset:0 atIndex:14];
        if (mHasLNFusion) {
            bindLNBuffers(encoder);
        }
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
        return;
    }

    // QKV leader: dispatch fused kernel covering q, k and v projections
    if (mIsQKVLeader && mQKVPeerK && mQKVPeerV && nil != (mHasLNFusion ? mLNFusedPipeline : mQKVFusedPipeline) && mQKVPeerKOutput && mQKVPeerVOutput) {
        [encoder setComputePipelineState:(mHasLNFusion ? mLNFusedPipeline : mQKVFusedPipeline)];
        // buffer(0): input (shared by q/k/v) — with LN fusion, the raw hidden input
        {
            auto inTensor = mHasLNFusion ? mLNHiddenInput : input;
            [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)inTensor->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(inTensor)->offset atIndex:0];
        }
        // buffer(1): q output (this)
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(output)->offset atIndex:1];
        // buffer(2): q params (input dims shared; k/v output_slice via seg)
        [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
        MetalBackend::setTensor(mWeight.get(), encoder, 3);
        MetalBackend::setTensor(mBias.get(), encoder, 4);
        MetalBackend::setTensor(getDequantScale().get(), encoder, 5);
        // buffers(6-9): k projection
        MetalBackend::setTensor(mQKVPeerKOutput, encoder, 6);
        MetalBackend::setTensor(mQKVPeerK->getWeight().get(), encoder, 7);
        MetalBackend::setTensor(mQKVPeerK->getBias().get(), encoder, 8);
        MetalBackend::setTensor(mQKVPeerK->getDequantScale().get(), encoder, 9);
        // buffers(10-13): v projection
        MetalBackend::setTensor(mQKVPeerVOutput, encoder, 10);
        MetalBackend::setTensor(mQKVPeerV->getWeight().get(), encoder, 11);
        MetalBackend::setTensor(mQKVPeerV->getBias().get(), encoder, 12);
        MetalBackend::setTensor(mQKVPeerV->getDequantScale().get(), encoder, 13);
        // buffer(14): {k_coef, v_coef, k_oslice, v_oslice[, w_coef, w_oslice]}
        [encoder setBuffer:mQKVSegBuffer offset:0 atIndex:14];
        // buffers(15-18): optional 4th projection (QKV_FUSED_P4)
        if (mQKVPeerW != nullptr && mQKVPeerWOutput != nullptr) {
            MetalBackend::setTensor(mQKVPeerWOutput, encoder, 15);
            MetalBackend::setTensor(mQKVPeerW->getWeight().get(), encoder, 16);
            MetalBackend::setTensor(mQKVPeerW->getBias().get(), encoder, 17);
            MetalBackend::setTensor(mQKVPeerW->getDequantScale().get(), encoder, 18);
        }
        if (mHasLNFusion) {
            bindLNBuffers(encoder);
        }
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
        return;
    }

    // Plain single-conv LN fusion (merged gate/up projection consumers): the
    // standalone LN_FUSED kernel variant computes RMSNorm in-kernel; no
    // leader/follower pairing involved.
    if (mHasLNFusion && !mIsGateUpLeader && nil != mLNFusedPipeline && mLNHiddenInput != nullptr) {
        [encoder setComputePipelineState:mLNFusedPipeline];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)mLNHiddenInput->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(mLNHiddenInput)->offset atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(output)->offset atIndex:1];
        [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
        MetalBackend::setTensor(mWeight.get(), encoder, 3);
        MetalBackend::setTensor(mBias.get(), encoder, 4);
        MetalBackend::setTensor(getDequantScale().get(), encoder, 5);
        bindLNBuffers(encoder);
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
        return;
    }

    if(mPreDequantWeight) {
        // Fused path: mDequantPipeline is nil and mTempWeight was never
        // allocated. Dispatch only the fused GEMM which reads quantized weight
        // from buffer(3) directly. buffer(6) is bound to mWeight as a harmless
        // alias — the fused kernel body never reads buffer(6), but
        // binding *something* keeps the Metal validation layer happy in debug
        // builds.
        const bool fused = (mDequantPipeline == nil) && mFusedQ4;

#if MNN_METAL_OP_PROFILE
        // In profile mode, split the two sub-passes (weight dequant + gemm) into
        // independent command buffers so each shows up as its own profile row.
        if (!fused) {
            static_cast<MetalBackend*>(backend())->setProfileSubtag("outdeq_wdq");
        }
#endif
        // pre dequant weight pipeline (legacy outer-dequant path)
        if (!fused) {
            [encoder setComputePipelineState:mDequantPipeline];
            MetalBackend::setTensor(mWeight.get(), encoder, 0);
            MetalBackend::setTensor(mTempWeight.get(), encoder, 1);
            [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
            MetalBackend::setTensor(getDequantScale().get(), encoder, 3);
            [encoder dispatchThreadgroups:mDequantThreads.first threadsPerThreadgroup:mDequantThreads.second];
#if MNN_METAL_OP_PROFILE
            {
                auto* mtbn = static_cast<MetalBackend*>(backend());
                encoder = mtbn->profileNextSubpass(std::string("outdeq_gemm_") + mProfileTag);
            }
#endif
        }
#if MNN_METAL_OP_PROFILE
        if (fused) {
            static_cast<MetalBackend*>(backend())->setProfileSubtag(std::string("fused_gemm_") + mProfileTag);
        }
#endif
        // convolution pipeline
        {
            [encoder setComputePipelineState:mPipeline];
            [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(input)->offset atIndex:0];
            [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(output)->offset atIndex:1];
            [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
            if (mUseFusedKsplit) {
                // Two-pass: the ksplit GEMM writes fp32 partials over buffer(1); the
                // reduce pass sums them, adds bias/activation, writes the real output.
                MetalBackend::setTensor(mKsplitPartial.get(), encoder, 1);
                MetalBackend::setTensor(mWeight.get(), encoder, 3);
                MetalBackend::setTensor(mBias.get(), encoder, 4);
                MetalBackend::setTensor(mDequantScaleBias.get(), encoder, 5);
                MetalBackend::setTensor(mWeight.get(), encoder, 6);
                [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [encoder setComputePipelineState:mKsplitReducePipeline];
                MetalBackend::setTensor(mKsplitPartial.get(), encoder, 0);
                [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(output)->offset atIndex:1];
                MetalBackend::setTensor(mBias.get(), encoder, 2);
                [encoder setBuffer:mConstBuffer offset:0 atIndex:3];
                [encoder dispatchThreadgroups:mKsplitReduceThreads.first threadsPerThreadgroup:mKsplitReduceThreads.second];
                return;
            }
            if (mFusedQ4) {
                // Fused kernel bindings: buffer(3) = quantized weight,
                // buffer(4) = bias, buffer(5) = dequantScale, buffer(6) =
                // placeholder alias of mWeight (never read by the fused
                // kernel; mTempWeight is not allocated).
                MetalBackend::setTensor(mWeight.get(), encoder, 3);
                MetalBackend::setTensor(mBias.get(), encoder, 4);
                MetalBackend::setTensor(getDequantScale().get(), encoder, 5);
                MetalBackend::setTensor(mWeight.get(), encoder, 6);
            } else {
                // Legacy conv1x1_gemm_32x64_split_k_sg: buffer(3)=fp16 dequanted
                // weight (mTempWeight), buffer(5)=dequantScale (used for LOOP_K64
                // W_QUANT_4/8 variants only).
                MetalBackend::setTensor(mTempWeight.get(), encoder, 3);
                MetalBackend::setTensor(mBias.get(), encoder, 4);
                MetalBackend::setTensor(getDequantScale().get(), encoder, 5);
            }
            [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
        }
    } else if (mUseFusedDecode) {
        // Fused weight+scale decode path: single buffer contains interleaved scale/bias/weights
        [encoder setComputePipelineState:mPipeline];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(input)->offset atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(output)->offset atIndex:1];
        [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
        MetalBackend::setTensor(mFusedWeightScale.get(), encoder, 3);
        MetalBackend::setTensor(mBias.get(), encoder, 4);
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
    } else {
        [encoder setComputePipelineState:mPipeline];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(input)->offset atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(output)->offset atIndex:1];
        [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
        MetalBackend::setTensor(mWeight.get(), encoder, 3);
        MetalBackend::setTensor(mBias.get(), encoder, 4);
        auto dequantScale = getDequantScale();
        if (dequantScale) {
            MetalBackend::setTensor(dequantScale.get(), encoder, 5);
        }
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
    }
#ifdef MNN_METAL_DEBUG_INFO
    if(!static_cast<MetalBackend*>(backend())->useFp16InsteadFp32()) {
        {
            static_cast<MetalBackend*>(backend())->flushEncoder();
            static_cast<MetalBackend*>(backend())->commit_net();
            static_cast<MetalBackend*>(backend())->wait();

            auto buffer = static_cast<MetalBackend*>(backend())->getBuffer(input);
            auto ptr = (float*)((int8_t*)buffer.first.contents + buffer.second);
            for(int i=0; i<64; i++) {
                printf("%f ", ptr[i]);
            }
            printf("\n\n");
        }
        {
            auto buffer = static_cast<MetalBackend*>(backend())->getBuffer(mWeight.get());
            auto ptr = (int8_t*)((int8_t*)buffer.first.contents + buffer.second);
            for(int i=0; i<64; i++) {
                printf("%d ", ptr[i]);
            }
            printf("\n\n");
        }
        {
            auto buffer = static_cast<MetalBackend*>(backend())->getBuffer(getDequantScale().get());
            auto ptr = (float*)((int8_t*)buffer.first.contents + buffer.second);
            for(int i=0; i<64; i++) {
                printf("%f ", ptr[i]);
            }
            printf("\n\n");
        }

        {
            auto buffer = static_cast<MetalBackend*>(backend())->getBuffer(output);
            auto ptr = (float*)((int8_t*)buffer.first.contents + buffer.second);
            for(int i=0; i<64; i++) {
                printf("%f ", ptr[i]);
            }
            printf("\n\n");
        }
    }
#endif
}
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
