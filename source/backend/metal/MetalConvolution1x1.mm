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

// Must match the shader's SIMD_GROUP_WIDTH.
static constexpr int kMetalSimdGroupWidth = 32;

// Apple GPU device-memory cache line.
static constexpr int kMetalCacheLineBytes = 128;

// QKV seg buffer layout, shared with the shader's qkv_seg.
static constexpr int kQKVSegFloats = 13;

bool MetalConvolution1x1::isValid(const Convolution2D *conv, const Tensor *input) {
    auto common = conv->common();
    auto kx = common->kernelX(), ky = common->kernelY();
    auto dx = common->dilateX(), dy = common->dilateY();
    auto sx = common->strideX(), sy = common->strideY();
    auto px = common->padX(), py = common->padY();
    return kx == 1 && ky == 1 && dx == 1 && dy == 1 && px == 0 && py == 0 && sx == 1 && sy == 1;
}

MetalConvolution1x1::MetalConvolution1x1(Backend *backend, const MNN::Op *op) : MetalConvolutionCommon(backend, op, nullptr) {
    bool ldInt8Weight = false;
#ifdef MNN_LOW_MEMORY
    auto conv2D = op->main_as_Convolution2D();
    if(static_cast<MetalBackend*>(backend)->getMemoryMode() == BackendConfig::Memory_Low) {
        if (conv2D->quanParameter() && (conv2D->external() || conv2D->quanParameter()->buffer())) {
            // quant type equal to 3 means fp16, fallback to float weight
            if(conv2D->quanParameter()->type() != 3 && conv2D->quanParameter()->type() != 8) {
            	ldInt8Weight = true;
            }
        }
    }
#endif
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
        auto dequantScale = getDequantScale();
        if (!dequantScale ||
            (mDequantBits != 2 && mDequantBits != 3 && mDequantBits != 4 && mDequantBits != 8)) {
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

struct Conv1x1GpuTier {
    bool prefersOc16Sg8K2;
    bool hasOc8Sg4K2Budget;
    bool proTier;
    bool aSeriesM4Gen;
    bool belowM4Gen;
};

static Conv1x1GpuTier conv1x1GpuTier(const MetalRuntime* rt) {
    const MetalGpuInfo& gpu = rt->gpuInfo();
    const bool baseTier = (gpu.tier == MetalGpuInfo::TierBase);
    Conv1x1GpuTier t;
    t.prefersOc16Sg8K2    = baseTier && gpu.gen == 16;
    t.hasOc8Sg4K2Budget = baseTier && gpu.gen >= 16;
    t.proTier             = gpu.tier >= MetalGpuInfo::TierPro;
    t.aSeriesM4Gen        = gpu.gen >= 16 && gpu.aSeries;
    t.belowM4Gen          = gpu.gen < 16 && !t.proTier;
    return t;
}

// Lanes sweeping K inside one quant block; split-K keeps a 2-lane floor
// unless a single lane already covers its half in one pass.
static int gemvLanesPerBlock(int inputDepthQuad, int blockCount, bool splitK) {
    if (blockCount <= 0) {
        return 0;
    }
    const int quadsPerBlock = UP_DIV(inputDepthQuad, blockCount);
    const int legacyLanes    = std::min(kMetalSimdGroupWidth, std::max(quadsPerBlock / 4, 1));
    if (!splitK) {
        return legacyLanes;
    }
    const int blocksPerSimdgroup    = std::max(blockCount / 2, 1);
    const int oneBlockPerLaneGroup  = std::max(kMetalSimdGroupWidth / blocksPerSimdgroup, 1);
    const int pairLaneGroups        = kMetalSimdGroupWidth / 2;
    const bool shortSecondPass      = blocksPerSimdgroup <= pairLaneGroups + pairLaneGroups / 2;
    const int floorLanes            = shortSecondPass ? 1 : 2;
    const int chosenLanes =
        std::max(std::min(legacyLanes, oneBlockPerLaneGroup), std::min(legacyLanes, floorLanes));
    int pow2Lanes = 1;
    while (pow2Lanes * 2 <= chosenLanes) {
        pow2Lanes *= 2;
    }
    return pow2Lanes;
}

// Lanes per quant block that minimize idle lane-group slots across a split-K
// simdgroup's half of the K sweep.
static int gemvSplitKFillLanes(int blockCount) {
    const int blocksOwned = std::max(blockCount / 2, 1);
    int bestLanes = 2;
    int bestSlots = 0;
    for (int lanes = 2; lanes <= 4; lanes *= 2) {
        const int blocksInFlight = kMetalSimdGroupWidth / lanes;
        const int slots = UP_DIV(blocksOwned, blocksInFlight) * blocksInFlight;
        if (bestSlots == 0 || slots < bestSlots) {
            bestSlots = slots;
            bestLanes = lanes;
        }
    }
    return bestLanes;
}

// Quant quads of one weight row that share a cache line. A quad carries 4
// output channels x 4 K elements, i.e. 2 * dequantBits bytes. Rounded down to a
// power of two (W3: 21 -> 16) so the kernel's lane groups tile the simdgroup.
static int gemvQuadsPerCacheLine(int dequantBits) {
    const int quads = std::max(kMetalCacheLineBytes / std::max(2 * dequantBits, 1), 1);
    int pow2Quads = 1;
    while (pow2Quads * 2 <= quads) {
        pow2Quads *= 2;
    }
    return pow2Quads;
}

// Raises lanes until a lane group's run of consecutive quant quads covers a
// cache line, so weight loads stay on the bandwidth roofline.
static int gemvCacheLineLanes(int inputDepthQuad, int blockCount, int dequantBits, int lanes) {
    if (blockCount <= 0 || dequantBits <= 0) {
        return lanes;
    }
    const int quadsPerBlock = UP_DIV(inputDepthQuad, blockCount);
    const int lineQuads     = gemvQuadsPerCacheLine(dequantBits);
    if (quadsPerBlock <= lineQuads) {
        return lanes;
    }
    return std::min(kMetalSimdGroupWidth, std::max(lanes, lineQuads));
}

// Lanes per quant block for the plain (no split-K) 2sg decode GEMV. Returns 0
// to keep the kernel's own default.
static int gemvPlainLanesPerBlock(int inputDepthQuad, int blockCount, int dequantBits) {
    const int defaultLanes = gemvLanesPerBlock(inputDepthQuad, blockCount, /*splitK=*/false);
    if (defaultLanes <= 0) {
        return 0;
    }
    const int lanes = gemvCacheLineLanes(inputDepthQuad, blockCount, dequantBits, defaultLanes);
    return lanes == defaultLanes ? 0 : lanes;
}

// oc16Sg8K2 is only picked when lane-group occupancy stays above ~80% and
// the K sweep (or weight size) can repay the reduction barrier.
static bool qkvOc16Sg8K2Admissible(int blockCount, int inputDepthQuad, int totalOutputChannels) {
    if (blockCount % 2 != 0) {
        return false;
    }
    const int lanesPerBlock = gemvLanesPerBlock(inputDepthQuad, blockCount, /*splitK=*/true);
    if (lanesPerBlock <= 0) {
        return false;
    }
    const int blocksOwned    = blockCount / 2;
    const int blocksInFlight = kMetalSimdGroupWidth / lanesPerBlock;
    const int passes         = UP_DIV(blocksOwned, blocksInFlight);
    if (blocksOwned * 5 < blocksInFlight * passes * 4) {
        return false;
    }
    constexpr int kMinSplitKQuadsPerSG = 256;
    if (inputDepthQuad / 2 >= kMinSplitKQuadsPerSG) {
        return true;
    }
    constexpr int kShortKSplitMaxTotalOutputChannels = 4096;
    return totalOutputChannels <= kShortKSplitMaxTotalOutputChannels;
}

static void copyTensorBytes(id<MTLBuffer> dst, size_t dstOff, const std::shared_ptr<Tensor>& src, size_t bytes) {
    auto alloc = (MetalRuntimeAllocator::MetalBufferAlloc *)src->deviceId();
    memcpy((uint8_t *)dst.contents + dstOff,
           (uint8_t *)alloc->getBuffer().contents + TensorUtils::getDescribeOrigin(src.get())->offset, bytes);
}

// SwiGLU K-split factor on top of oc16Sg2K1: picks the widest factor within
// kMaxPairs whose lanes cover a pair's blocks in one pass, else the narrowest.
static int gateUpSiluSplitK(int blockCount, int quadsPerBlock) {
    constexpr int kMaxThreadgroupThreads = 1024;
    constexpr int kMaxPairs              = 6;
    const int legacyLanes = std::max(quadsPerBlock / 4, 1);
    const int minOwned    = std::max(2 * kMetalSimdGroupWidth / std::max(quadsPerBlock, 1), 1);
    const int maxOwned    = kMetalSimdGroupWidth / legacyLanes;
    int widest    = 0;
    int narrowest = 0;
    for (int factor = 2; factor * 2 * kMetalSimdGroupWidth <= kMaxThreadgroupThreads; ++factor) {
        if (blockCount % factor != 0) {
            continue;
        }
        const int blocksOwned = blockCount / factor;
        if (blocksOwned < minOwned || blocksOwned > maxOwned) {
            continue;
        }
        if (narrowest == 0) {
            narrowest = factor;
        }
        if (factor <= kMaxPairs) {
            widest = factor;
        }
    }
    return widest > 0 ? widest : narrowest;
}

bool MetalConvolution1x1::setupGateUpFusion(MetalConvolution1x1* peer, const Tensor* peerOutput,
                                            const Tensor* siluOutput) {
    if (!mIs2sgDecode || !peer->mIs2sgDecode) {
        return false;
    }
    // Leader = gate (this), Follower = up (peer)
    mIsGateUpLeader = true;
    mGateUpPeer = peer;
    mGateUpPeerOutput = peerOutput;
    peer->mIsGateUpFollower = true;

    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();

    // up needs its own scaleCoef; gate's comes from cst. Allocated once and
    // rewritten in place, like mConstBuffer.
    if (nil == mGateUpSegBuffer) {
        mGateUpSegBuffer = backend->getConstBuffer(sizeof(float));
    }
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
    // oc16Sg2K1 is forced on devices without tensor-API, and SwiGLU reuses its
    // body for the gate/up epilogue so it is also forced when mGateUpSilu is on.
    const bool useOc16Sg2K1 = !backend->isSupportTensorApi();
    mGateUpSilu = siluOutput != nullptr;
    mGateUpSiluOutput = mGateUpSilu ? siluOutput : nullptr;
    const int gateUpDepthQuad = ((Param*)mConstBuffer.contents)->outputDepthQuad;
    const int peerDepthQuad = ((Param*)peer->mConstBuffer.contents)->outputDepthQuad;
    const int gateUpQuadsPerBlock = UP_DIV(((Param*)mConstBuffer.contents)->inputDepthQuad, mBlockCount);
    const bool useOc16Sg4K1 = !useOc16Sg2K1 && !mGateUpSilu && gateUpQuadsPerBlock >= 32 &&
                              gateUpDepthQuad % 4 == 0 && peerDepthQuad % 4 == 0;
    mUseOc16Sg4K1 = useOc16Sg4K1;
    if (useOc16Sg4K1) {
        keys.emplace_back("oc16Sg4K1");
    } else if (useOc16Sg2K1 || mGateUpSilu) {
        keys.emplace_back("oc16Sg2K1");
    }
    // Split-K gate: raw channels whole in equal-size quant blocks (any block size)
    // and 8-wide output pairs, so C4-padded tails stay out.
    const int gateUpInputChannels = mOp->main_as_Convolution2D()->common()->inputCount();
    const int gateUpOutputChannels = ((Param*)mConstBuffer.contents)->outputChannel;
    const Conv1x1GpuTier gpuTier = conv1x1GpuTier(rt);
    const bool gateUpFactorSkRoute =
        (gpuTier.hasOc8Sg4K2Budget ||
         (gpuTier.aSeriesM4Gen && !backend->isSupportTensorApi())) &&
        backend->useFp16InsteadFp32() && mGateUpSilu &&
        gateUpOutputChannels % 8 == 0 &&
        (mDequantBits == 4 || mDequantBits == 8) && peer->mDequantBits == mDequantBits && mBlockCount > 0 &&
        gateUpInputChannels % mBlockCount == 0 &&
        peer->mBlockCount == mBlockCount &&
        peer->mOp->main_as_Convolution2D()->common()->inputCount() == gateUpInputChannels &&
        ((Param*)peer->mConstBuffer.contents)->outputChannel == gateUpOutputChannels;
    const int gateUpSkQuadsFloor = (gpuTier.belowM4Gen || gpuTier.proTier) ? 32 : 4;
    mGateUpSiluKSplit = 0;
    if (gateUpFactorSkRoute) {
        mGateUpSiluKSplit = gateUpSiluSplitK(mBlockCount, gateUpQuadsPerBlock);
    }
    if (mGateUpSiluKSplit == 0 && mGateUpSilu && (gateUpDepthQuad % 2 == 0) &&
        gateUpQuadsPerBlock >= gateUpSkQuadsFloor && (mBlockCount & (mBlockCount - 1)) == 0) {
        mGateUpSiluKSplit = 2;
    }
    if (mGateUpSiluKSplit > 0) {
        keys.emplace_back("oc16Sg2K1_siluK" + std::to_string(mGateUpSiluKSplit));
    }
    if (mGateUpSilu) {
        keys.emplace_back("GATE_UP_SILU");
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
        if (useOc16Sg4K1) {
            [dic setValue:@"4" forKey:@"GEMV_QUADS_PER_TG"];
        } else if (useOc16Sg2K1 || mGateUpSilu) {
            [dic setValue:@"1" forKey:@"GEMV_SG_OC_QUADS_2"];
        }
        if (mGateUpSiluKSplit > 0) {
            [dic setValue:@(mGateUpSiluKSplit).stringValue forKey:@"GEMV_SG_OC_QUADS_2_KSPLIT"];
        }
        if (mGateUpSilu) {
            [dic setValue:@"1" forKey:@"GATE_UP_SILU"];
        }
        option.preprocessorMacros = dic;

        std::string sgrWqStr = std::string(gBasicConvPrefix) + gConv1x1WqSgReduce;
        mGateUpFusedPipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", option);
        rt->insertPipeline(keys, mGateUpFusedPipeline);
    }

    if (nil == mGateUpFusedPipeline ||
        (mGateUpSiluKSplit > 0 &&
         mGateUpFusedPipeline.maxTotalThreadsPerThreadgroup < (NSUInteger)(64 * mGateUpSiluKSplit))) {
        // Compilation failed or the split-K threadgroup does not fit, revert fusion.
        // mGateUpSegBuffer is kept so its identity stays stable for any live recording.
        mIsGateUpLeader = false;
        mGateUpPeer = nullptr;
        mGateUpSilu = false;
        mGateUpSiluOutput = nullptr;
        mGateUpSiluKSplit = 0;
        mUseOc16Sg4K1 = false;
        peer->mIsGateUpFollower = false;
        return false;
    }

    // Grid adds z=2 for gate/up selection; the fused pipeline is 2sg-kernel based,
    // so force 64 threads regardless of the plain pipeline's group size.
    auto gridSize = mThreads.first;
    NSUInteger gw = gridSize.width;
    NSUInteger gz = 2;
    int tgThreads = 64;
    if (useOc16Sg4K1) {
        gw = (NSUInteger)UP_DIV(ALIMAX(gateUpDepthQuad, peerDepthQuad), 4);
        tgThreads = 128;
    } else if (useOc16Sg2K1) {
        gw = (NSUInteger)UP_DIV(gateUpDepthQuad, 4);
        if (mGateUpSilu) {
            // A simdgroup covers one quad of both matrices, so the gate/up split
            // moves out of z and into x.
            gw = (NSUInteger)UP_DIV(gateUpDepthQuad, 2);
            gz = 1;
        }
    } else if (mGateUpSilu) {
        gw = (NSUInteger)UP_DIV(gateUpDepthQuad, 2);
        gz = 1;
    }
    if (mGateUpSiluKSplit > 0) {
        tgThreads = 64 * mGateUpSiluKSplit;
    }
    mThreads.first = MTLSizeMake(gw, gridSize.height, gz);
    mThreads.second = MTLSizeMake(tgThreads, 1, 1);

    return true;
}

bool MetalConvolution1x1::setupQKVFusion(const Tensor* selfOutput,
                                         MetalConvolution1x1* peerK, const Tensor* peerKOutput,
                                         MetalConvolution1x1* peerV, const Tensor* peerVOutput,
                                         MetalConvolution1x1* peerW, const Tensor* peerWOutput) {
    if (!mIs2sgDecode || !peerK->mIs2sgDecode || !peerV->mIs2sgDecode) {
        return false;
    }
    if (peerW != nullptr && !peerW->mIs2sgDecode) {
        return false;
    }
    // No stacking with other fusion roles: buffer indices 6-9/14 collide.
    if (mIsGateUpLeader || mIsGateUpFollower || mHasLNFusion ||
        peerK->mIsGateUpLeader || peerK->mIsGateUpFollower || peerK->mHasLNFusion ||
        peerV->mIsGateUpLeader || peerV->mIsGateUpFollower || peerV->mHasLNFusion) {
        return false;
    }
    if (peerW != nullptr &&
        (peerW->mIsGateUpLeader || peerW->mIsGateUpFollower || peerW->mHasLNFusion)) {
        return false;
    }
    // The fused kernel shares the leader's cst except outputDepthQuad and
    // scaleCoef, so quant layout and activation must match.
    if (peerK->mDequantBits != mDequantBits || peerV->mDequantBits != mDequantBits ||
        peerK->mBlockCount != mBlockCount || peerV->mBlockCount != mBlockCount ||
        peerK->mActivationType != mActivationType || peerV->mActivationType != mActivationType) {
        return false;
    }
    if (peerW != nullptr &&
        (peerW->mDequantBits != mDequantBits || peerW->mBlockCount != mBlockCount ||
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
    if (peerW != nullptr) {
        keys.emplace_back("QKV_FUSED_P4");
    }
    const int numProj = peerW != nullptr ? 4 : 3;
    const int projDepthQuad[4] = {
        ((Param*)mConstBuffer.contents)->outputDepthQuad,
        ((Param*)peerK->mConstBuffer.contents)->outputDepthQuad,
        ((Param*)peerV->mConstBuffer.contents)->outputDepthQuad,
        peerW != nullptr ? ((Param*)peerW->mConstBuffer.contents)->outputDepthQuad : 0,
    };
    // Each K half must be a whole number of quant blocks, and no projection may
    // early-return before the reduction barrier.
    const bool splitKAllowed = mBlockCount % 2 == 0;
    const Conv1x1GpuTier gpuTier = conv1x1GpuTier(rt);
    const int qkvInputChannels = mOp->main_as_Convolution2D()->common()->inputCount();
    const int qkvInputDepthQuad = ((Param*)mConstBuffer.contents)->inputDepthQuad;
    const int qkvQuadsPerBlock = UP_DIV(qkvInputDepthQuad, mBlockCount);
    const bool qkvAllowOc8Sg4K2 = gpuTier.hasOc8Sg4K2Budget && !gpuTier.prefersOc16Sg8K2 &&
        backend->useFp16InsteadFp32() && splitKAllowed &&
        (mDequantBits == 4 || mDequantBits == 8) && mBlockCount > 0 && qkvInputChannels % mBlockCount == 0;
    const int qkvOutputChannels[4] = {
        ((Param*)mConstBuffer.contents)->outputChannel,
        ((Param*)peerK->mConstBuffer.contents)->outputChannel,
        ((Param*)peerV->mConstBuffer.contents)->outputChannel,
        peerW != nullptr ? ((Param*)peerW->mConstBuffer.contents)->outputChannel : 0,
    };
    const bool qkvAllowOc16Sg8K2 = gpuTier.prefersOc16Sg8K2 && backend->useFp16InsteadFp32() &&
        (mDequantBits == 4 || mDequantBits == 8) && mBlockCount > 0 && qkvInputChannels % mBlockCount == 0;
    int qkvTotalOutputChannels = 0;
    for (int p = 0; p < numProj; ++p) {
        qkvTotalOutputChannels += qkvOutputChannels[p];
    }
    const bool proAllowOc16Sg8K2 = gpuTier.proTier && backend->useFp16InsteadFp32() &&
        (mDequantBits == 4 || mDequantBits == 8) &&
        qkvOc16Sg8K2Admissible(mBlockCount, ((Param*)mConstBuffer.contents)->inputDepthQuad,
                               qkvTotalOutputChannels);
    bool useOc16Sg8K2 = splitKAllowed && ((qkvAllowOc16Sg8K2 && !qkvAllowOc8Sg4K2) || proAllowOc16Sg8K2);
    for (int p = 0; p < numProj; ++p) {
        useOc16Sg8K2 = useOc16Sg8K2 && projDepthQuad[p] % 4 == 0;
    }
    // oc16Sg2K1 (GEMV_SG_OC_QUADS_2) pairs with the quadsPerTG=4 grid below; dropping
    // either half leaves half of every projection's rows unwritten.
    const bool useOc16Sg2K1 = !useOc16Sg8K2 && !backend->isSupportTensorApi();
    bool useOc16Sg4K1 = !useOc16Sg8K2 && !useOc16Sg2K1 && !qkvAllowOc8Sg4K2 && qkvQuadsPerBlock >= 32;
    for (int p = 0; p < numProj; ++p) {
        useOc16Sg4K1 = useOc16Sg4K1 && projDepthQuad[p] % 4 == 0;
    }
    // oc8Sg4K2 (plain split-K): 2 quads/TG over 4 simdgroups. Tensor-API only in
    // practice, since !useOc16Sg8K2 already implies useOc16Sg2K1 otherwise.
    bool useOc8Sg4K2 = !useOc16Sg8K2 && !useOc16Sg4K1 && !useOc16Sg2K1 && splitKAllowed &&
                   (qkvAllowOc8Sg4K2 || mBlockCount <= MetalEnv::get().qkvSplitKMaxBlocks);
    for (int p = 0; p < numProj; ++p) {
        useOc8Sg4K2 = useOc8Sg4K2 && projDepthQuad[p] % 2 == 0;
    }
    mQKVUseOc16Sg2K1 = useOc16Sg2K1;
    mQKVSplitK = useOc16Sg8K2 || useOc8Sg4K2;
    mQKVUseOc16Sg8K2 = useOc16Sg8K2;
    mUseOc16Sg4K1 = useOc16Sg4K1;
    if (useOc16Sg4K1) {
        keys.emplace_back("oc16Sg4K1");
    }
    if (useOc16Sg2K1) {
        keys.emplace_back("oc16Sg2K1");
    }
    mQKVSkLanesPerBlock = !mQKVSplitK ? 0
        : gemvLanesPerBlock(qkvInputDepthQuad, mBlockCount, true);
    const int qkvSkLanesPerBlock = mQKVSkLanesPerBlock;
    // The grid below and the macro must both come from this one value.
    const int quadsPerTG = (useOc16Sg2K1 || useOc16Sg8K2 || useOc16Sg4K1) ? 4 : 2;
    if (mQKVSplitK) {
        keys.emplace_back(quadsPerTG == 4 ? "oc16Sg8K2" : "oc8Sg4K2");
        if (qkvSkLanesPerBlock > 0) {
            keys.emplace_back("lpb" + std::to_string(qkvSkLanesPerBlock));
        }
    }
    // The packed grid is a compile-time variant, so it belongs in the cache key and
    // the macro dictionary.
    int projGridX[4] = {0, 0, 0, 0};
    int packedGridX = 0;
    int maxGridX = 0;
    for (int p = 0; p < numProj; ++p) {
        projGridX[p] = UP_DIV(projDepthQuad[p], quadsPerTG);
        packedGridX += projGridX[p];
        maxGridX = ALIMAX(maxGridX, projGridX[p]);
    }
    // Packing only differs from the rectangular grid when the members differ in
    // outputChannel.
    mQKVPackedGrid = packedGridX < numProj * maxGridX;
    if (mQKVPackedGrid) {
        keys.emplace_back("QKV_PACKED_GRID");
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
        if (peerW != nullptr) {
            [dic setValue:@"1" forKey:@"QKV_FUSED_P4"];
        }
        if (useOc16Sg2K1) {
            [dic setValue:@"1" forKey:@"GEMV_SG_OC_QUADS_2"];
        }
        if (useOc16Sg4K1) {
            [dic setValue:@"4" forKey:@"GEMV_QUADS_PER_TG"];
        } else if (mQKVSplitK) {
            [dic setValue:@(quadsPerTG).stringValue forKey:@"GEMV_QUADS_PER_TG"];
            [dic setValue:@"2" forKey:@"GEMV_SPLIT_K"];
            if (qkvSkLanesPerBlock > 0) {
                [dic setValue:@(qkvSkLanesPerBlock).stringValue forKey:@"GEMV_LANES_PER_BLOCK"];
            }
        }
        if (mQKVPackedGrid) {
            [dic setValue:@"1" forKey:@"QKV_PACKED_GRID"];
        }
        option.preprocessorMacros = dic;

        std::string sgrWqStr = std::string(gBasicConvPrefix) + gConv1x1WqSgReduce;
        mQKVFusedPipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", option);
        rt->insertPipeline(keys, mQKVFusedPipeline);
    }
    if (nil == mQKVFusedPipeline) {
        mQKVSplitK = false;
        mQKVUseOc16Sg8K2 = false;
        mQKVSkLanesPerBlock = 0;
        mUseOc16Sg4K1 = false;
        return false;
    }

    mIsQKVLeader = true;
    mQKVPeerK = peerK;
    mQKVPeerV = peerV;
    mQKVPeerW = peerW;
    mQKVPeerKOutput = peerKOutput;
    mQKVPeerVOutput = peerVOutput;
    mQKVPeerWOutput = peerWOutput;
    peerK->mIsQKVFollower = true;
    peerV->mIsQKVFollower = true;
    if (peerW != nullptr) {
        peerW->mIsQKVFollower = true;
    }

    // Followers' outputs must land in static memory: the fused dispatch writes
    // them at the leader's position, where the dynamic pool may alias them.
    const Tensor* projOutputs[4] = {selfOutput, peerKOutput, peerVOutput, peerWOutput};
    size_t mergedOutOff[4] = {0, 0, 0, 0};
    mQKVMergedOut = false;
    // One holder keeps every projection reachable from buffer(1) with 16B-aligned starts.
    size_t total = 0;
    for (int p = 0; p < numProj; ++p) {
        mergedOutOff[p] = total;
        total += backend->getTensorSizeInBytes(projOutputs[p]);
    }
    if (nullptr == mQKVMergedOutHolder ||
        backend->getTensorSizeInBytes(mQKVMergedOutHolder.get()) < total) {
        mQKVMergedOutHolder.reset(Tensor::createDevice<int8_t>({(int)total}));
        if (!backend->onAcquireBuffer(mQKVMergedOutHolder.get(), Backend::STATIC)) {
            mQKVMergedOutHolder = nullptr;
        }
    }
    bool rehomed = false;
    if (nullptr != mQKVMergedOutHolder) {
        for (int p = 0; p < numProj; ++p) {
            MetalBackend::aliasTensor(projOutputs[p], mQKVMergedOutHolder.get(), mergedOutOff[p]);
        }
        mQKVMergedOut = true;
        rehomed = true;
    }
    // No per-member fallback: the pipeline already writes every projection at the
    // leader's base, so roll the whole fusion back.
    if (!rehomed) {
        mIsQKVLeader = false;
        mQKVMergedOut = false;
        mQKVPackedGrid = false;
        mQKVSplitK = false;
        mQKVUseOc16Sg8K2 = false;
        mQKVSkLanesPerBlock = 0;
        mUseOc16Sg4K1 = false;
        mQKVPeerK = nullptr;
        mQKVPeerV = nullptr;
        mQKVPeerW = nullptr;
        mQKVPeerKOutput = nullptr;
        mQKVPeerVOutput = nullptr;
        mQKVPeerWOutput = nullptr;
        peerK->mIsQKVFollower = false;
        peerV->mIsQKVFollower = false;
        if (peerW != nullptr) {
            peerW->mIsQKVFollower = false;
        }
        return false;
    }

    // Followers' scaleCoef + outputDepthQuad and the packed grid bases. Allocated
    // once and rewritten in place, like mConstBuffer.
    if (nil == mQKVSegBuffer) {
        mQKVSegBuffer = backend->getConstBuffer(kQKVSegFloats * sizeof(float));
    }
    auto seg = (float *)mQKVSegBuffer.contents;
    seg[0] = peerK->mScaleCoef;
    seg[1] = peerV->mScaleCoef;
    seg[2] = (float)projDepthQuad[1];
    seg[3] = (float)projDepthQuad[2];
    seg[4] = peerW != nullptr ? peerW->mScaleCoef : 0.0f;
    seg[5] = (float)projDepthQuad[3];
    // seg[6..8]: packed-grid base of projections 1..3.
    seg[6] = (float)projGridX[0];
    seg[7] = (float)(projGridX[0] + projGridX[1]);
    seg[8] = (float)(projGridX[0] + projGridX[1] + projGridX[2]);

    // Merge q/k/v weights + dequant scales into one buffer to cut the concurrent
    // DRAM streams. W4 only, and 4-member groups are excluded.
    const bool doMerge = mDequantBits == 4 && peerW == nullptr;
    mQKVMergedCount = 0;
    if (doMerge) {
        const int nm = peerW == nullptr ? 3 : 4;
        const std::shared_ptr<Tensor> wt[4] = {mWeight, peerK->getWeight(), peerV->getWeight(),
                                              peerW != nullptr ? peerW->getWeight() : nullptr};
        const std::shared_ptr<Tensor> st[4] = {getDequantScale(), peerK->getDequantScale(), peerV->getDequantScale(),
                                              peerW != nullptr ? peerW->getDequantScale() : nullptr};
        size_t bytes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        size_t off = 0;
        for (int p = 0; p < nm; ++p) {
            bytes[p]     = TensorUtils::getRawSize(wt[p].get()) * wt[p]->getType().bytes();
            bytes[4 + p] = TensorUtils::getRawSize(st[p].get()) * st[p]->getType().bytes();
        }
        for (int i = 0; i < 8; ++i) {
            if (bytes[i] == 0) {
                continue;
            }
            off = (off + 15) & ~(size_t)15;
            mQKVMergedOff[i] = off;
            off += bytes[i];
        }
        if (nil == mQKVMergedBuffer || mQKVMergedSize < off) {
            auto ctx = (__bridge MNNMetalContext *)backend->context();
            mQKVMergedBuffer = [ctx.device newBufferWithLength:off options:MTLResourceStorageModeShared];
            mQKVMergedSize = off;
        }
        for (int p = 0; p < nm; ++p) {
            copyTensorBytes(mQKVMergedBuffer, mQKVMergedOff[p], wt[p], bytes[p]);
            copyTensorBytes(mQKVMergedBuffer, mQKVMergedOff[4 + p], st[p], bytes[4 + p]);
        }
        mQKVMergedCount = nm;
    } else {
        mQKVMergedBuffer = nil;
    }

    // seg[9..12]: each projection's output offset inside the merged allocation, in
    // the ftype4 units the shader's `out` is typed with.
    for (int p = 0; p < 4; ++p) {
        seg[9 + p] = 0.0f;
    }
    if (mQKVMergedOut) {
        const size_t outElem = backend->useFp16InsteadFp32() ? 8 : 16;
        for (int p = 0; p < numProj; ++p) {
            seg[9 + p] = (float)(mergedOutOff[p] / outElem);
        }
    }

    if (mQKVPackedGrid) {
        // x runs over the projections' threadgroup ranges laid end to end.
        mThreads.first = MTLSizeMake((NSUInteger)packedGridX, mThreads.first.height, 1);
    } else {
        // grid.x covers the largest projection; z selects the projection, and
        // out-of-range simdgroups early-return in the shader.
        mThreads.first = MTLSizeMake((NSUInteger)maxGridX, mThreads.first.height, numProj);
    }
    const int tgSimds = useOc16Sg2K1 ? 2 : quadsPerTG * (mQKVSplitK ? 2 : 1);
    mThreads.second = MTLSizeMake(32 * tgSimds, 1, 1);

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
    // Allocated once and rewritten in place, like mConstBuffer.
    if (nil == mLNEpsBuffer) {
        mLNEpsBuffer = backend->getConstBuffer(sizeof(float));
    }
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
    // Macros below must match what setupGateUpFusion/setupQKVFusion sized the grid for.
    if (mIsGateUpLeader) {
        keys.emplace_back("GATE_UP_FUSED");
        if (mGateUpSilu) {
            keys.emplace_back("GATE_UP_SILU");
        }
    }
    if (mIsQKVLeader) {
        keys.emplace_back("QKV_FUSED");
        if (mQKVPeerW != nullptr) {
            keys.emplace_back("QKV_FUSED_P4");
        }
        if (mQKVPackedGrid) {
            keys.emplace_back("QKV_PACKED_GRID");
        }
    }
    keys.emplace_back("LN_FUSED");
    const bool useOc16Sg2K1 = mIsQKVLeader ? mQKVUseOc16Sg2K1 : !backend->isSupportTensorApi();
    const bool qkvSplitK = mIsQKVLeader && mQKVSplitK;
    const bool useOc16Sg8K2 = mIsQKVLeader && mQKVUseOc16Sg8K2;
    const int lnStageQuads = UP_DIV(hiddenInput->channel(), 4);
    // lnStageFits is the threadgroup memory ceiling (float4 worst case);
    // lnStageWorth is the narrower range staging was tuned for.
    const bool lnStageFits  = lnStageQuads * 16 <= 8192;
    const bool lnStageWorth = lnStageQuads * 16 <= 4096;
    const bool lnStage = useOc16Sg8K2 && lnStageFits && lnStageWorth;
    if (lnStage) {
        keys.emplace_back("LN_STAGE_" + std::to_string(lnStageQuads));
    }
    if (useOc16Sg2K1 || mGateUpSilu) {
        keys.emplace_back("oc16Sg2K1");
    }
    if (mUseOc16Sg4K1) {
        keys.emplace_back("oc16Sg4K1");
    }
    if (mGateUpSiluKSplit > 0) {
        keys.emplace_back("oc16Sg2K1_siluK" + std::to_string(mGateUpSiluKSplit));
    }
    const int qkvSkLanesPerBlock = qkvSplitK ? mQKVSkLanesPerBlock : 0;
    if (qkvSplitK) {
        keys.emplace_back(useOc16Sg8K2 ? "oc16Sg8K2" : "oc8Sg4K2");
        if (qkvSkLanesPerBlock > 0) {
            keys.emplace_back("lpb" + std::to_string(qkvSkLanesPerBlock));
        }
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
            if (mGateUpSilu) {
                [dic setValue:@"1" forKey:@"GATE_UP_SILU"];
            }
        }
        if (mIsQKVLeader) {
            [dic setValue:@"1" forKey:@"QKV_FUSED"];
            if (mQKVPeerW != nullptr) {
                [dic setValue:@"1" forKey:@"QKV_FUSED_P4"];
            }
            if (mQKVPackedGrid) {
                [dic setValue:@"1" forKey:@"QKV_PACKED_GRID"];
            }
        }
        [dic setValue:@"1" forKey:@"LN_FUSED"];
        if (lnStage) {
            [dic setValue:@"1" forKey:@"LN_STAGE"];
            [dic setValue:@(lnStageQuads).stringValue forKey:@"LN_STAGE_QUADS"];
        }
        if (mUseOc16Sg4K1) {
            [dic setValue:@"4" forKey:@"GEMV_QUADS_PER_TG"];
        } else if (useOc16Sg2K1 || mGateUpSilu) {
            [dic setValue:@"1" forKey:@"GEMV_SG_OC_QUADS_2"];
        }
        if (mGateUpSiluKSplit > 0) {
            [dic setValue:@(mGateUpSiluKSplit).stringValue forKey:@"GEMV_SG_OC_QUADS_2_KSPLIT"];
        }
        if (qkvSplitK) {
            [dic setValue:(useOc16Sg8K2 ? @"4" : @"2") forKey:@"GEMV_QUADS_PER_TG"];
            [dic setValue:@"2" forKey:@"GEMV_SPLIT_K"];
            if (qkvSkLanesPerBlock > 0) {
                [dic setValue:@(qkvSkLanesPerBlock).stringValue forKey:@"GEMV_LANES_PER_BLOCK"];
            }
        }
        option.preprocessorMacros = dic;

        std::string sgrWqStr = std::string(gBasicConvPrefix) + gConv1x1WqSgReduce;
        mLNFusedPipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", option);
        rt->insertPipeline(keys, mLNFusedPipeline);
    }

    if (nil == mLNFusedPipeline ||
        (mGateUpSiluKSplit > 0 &&
         mLNFusedPipeline.maxTotalThreadsPerThreadgroup < (NSUInteger)(64 * mGateUpSiluKSplit))) {
        mHasLNFusion = false;
        return false;
    }
    // LN-fused pipeline is 2sg-kernel based, so force 64 threads and widen it for
    // the split-K variants.
    NSUInteger lnThreads = 64;
    if (mGateUpSiluKSplit > 0) {
        lnThreads = 64 * mGateUpSiluKSplit;
    } else if (mUseOc16Sg4K1) {
        lnThreads = 128;
    } else if (useOc16Sg8K2) {
        lnThreads = 256;
    } else if (qkvSplitK) {
        lnThreads = 128;
    }
    mThreads.second = MTLSizeMake(lnThreads, 1, 1);
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

    // Reset fusion state on each resize; const buffers (mGateUpSegBuffer /
    // mQKVSegBuffer / mLNEpsBuffer) stay alive so recordings keep seeing them.
    mIs2sgDecode = false;
    mIsGateUpLeader = false;
    mIsGateUpFollower = false;
    mGateUpPeer = nullptr;
    mGateUpFusedPipeline = nil;
    mGateUpSilu = false;
    mGateUpSiluOutput = nullptr;

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
    mQKVPackedGrid = false;
    mUseOc16Sg4K1 = false;


    mHasLNFusion = false;
    mLNFusedPipeline = nil;
    mLNHiddenInput = nullptr;
    mLNResidualInput = nullptr;
    mLNResidualOutput = nullptr;
    mLNGamma = nullptr;

    // prepare
    // For C4NHW4 format, NHW can be fuse to W
    auto input = inputs[0];
    auto output = outputs[0];
    int is = input->batch();
    for (int i=2; i<input->dimensions(); ++i) {
        is *= input->length(i);
    }
    int ic  = input->channel();
    int icDiv4  = UP_DIV(input->channel(), 4);
    int ow  = is;
    int oh  = 1;
    int os  = ow;
    int ob  = 1;
    auto oc  = output->channel();
    auto ocDiv4  = UP_DIV(output->channel(), 4);
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto dequantScale = getDequantScale();
    int blockCount = 1;
    if (dequantScale) {
        int bytes = sizeof(float);
        if(backend->useFp16InsteadFp32()) {
            bytes = sizeof(__fp16);
        }
        blockCount = (int)(dequantScale->usize() / bytes / ocDiv4 / 2 / 4);
    }
    // Allocate once and rewrite in place: this buffer is bound raw, so encode-replay
    // cannot see a swap and a live recording would keep emitting the previous Param.
    if (nil == mConstBuffer) {
        mConstBuffer = backend->getConstBuffer(sizeof(Param));
    }
    auto param = (Param *)mConstBuffer.contents;
    param->inputSize = is;
    param->inputDepthQuad = icDiv4;
    param->outputWidth = ow;
    param->outputHeight = oh;
    param->outputSize = os;
    param->outputDepthQuad = ocDiv4;
    param->outputChannel = oc;
    param->batch = ob;
    param->blockCount = blockCount;
    param->activation = mActivationType;
    param->scaleCoef = mScaleCoef;
    mBlockCount = blockCount;
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

    // Tile kernels step whole K-quad groups inside a quant block: 2 for the 8x16/32x16/16x32
    // wquant tiles, 4 for the outer dequant pre-pass, 8 (32 K) for 16x16/32x64 and fused Q4.
    const int quadsPerBlock = UP_DIV(icDiv4, blockCount);
    auto blockFitsQuads = [blockCount, quadsPerBlock](int quads) {
        return blockCount <= 1 || quadsPerBlock % quads == 0;
    };
    // The outer-dequant sg_matrix M=64 fp tile has no per-block loop.
    const bool outerSgM64 = !backend->isSupportTensorApi() && rt->supportSimdGroupMatrix() && area >= 128;

    // Small M or no simdgroup matrix: dequant in shader.
    bool dequantInShader = (area < 64) || !(rt->supportSimdGroupMatrix());
    // Multi-token in-shader kernels lack W2/3 paths; prefill goes through
    // outer-dequant + fp gemm when simdgroup matrix is available.
    if ((mDequantBits == 2 || mDequantBits == 3) && area > 1 && rt->supportSimdGroupMatrix()) {
        dequantInShader = false;
    }
    // Tensor-API devices run Q4/Q8 prefill through outer-dequant + tensor GEMM.
    if (backend->isSupportTensorApi() && area > 1 && (mDequantBits == 4 || mDequantBits == 8)) {
        dequantInShader = false;
    }
    // In-shader dequant re-unpacks weights per M-tile; only wins for large
    // weights at short area on non-tensor-API devices.
    if (!backend->isSupportTensorApi() && rt->supportSimdGroupMatrix() && area > 1 &&
        (mDequantBits == 4 || mDequantBits == 8)) {
        if ((size_t)ic * oc > 4 * 1024 * 1024 && area < 512) {
            dequantInShader = true;
        }
    }
    if (!blockFitsQuads(outerSgM64 ? 4 : 8)) {
        dequantInShader = true;
    }
    mPreDequantWeight = false;

#ifdef MNN_LOW_MEMORY
    if (dequantScale && dequantInShader) {
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
        const Conv1x1GpuTier gpuTier = conv1x1GpuTier(rt);
        const int rawIc = mOp->main_as_Convolution2D()->common()->inputCount();
        // Raw channels divide evenly into quant blocks, so none straddle the
        // C4-padded tail; required by the tuned decode GEMV route.
        const bool wqEvenBlockFp16 = backend->useFp16InsteadFp32() &&
            (mDequantBits == 4 || mDequantBits == 8) &&
            blockCount > 0 && rawIc % blockCount == 0;
        // Threadgroup budget the decode GEMV grid is compared against (area==1).
        const int tgBudget = gpuTier.proTier ? 160 : (gpuTier.hasOc8Sg4K2Budget ? 80 : 0);
        // Decode keeps lm_head on the 2sg route; split-K g16 is prefill-only.
        const int lmheadSplitKMode = area != 1 ? 2 : 1;
        // Vocabulary-sized outputs that may take the g16 lm_head kernels.
        const bool lmHeadScale = oc > 16384;
        const bool g16LmHeadRoute = lmHeadScale && ocDiv4 % 2 == 0 &&
                                    !(lmheadSplitKMode == 1 && (oc % 8 == 0) && (blockCount % 2 == 0));
        option.preprocessorMacros = dic;

        NSUInteger gid_x = UP_DIV(ow * oh, 4);
        NSUInteger gid_y = ocDiv4;
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
            // The in-shader tile GEMMs lack W2/3 and need an even quad count per quant
            // block, so those shapes take g8/g16 at every area.
            const bool gemvAllArea = mDequantBits == 2 || mDequantBits == 3 ||
                                     (rt->supportSimdGroupMatrix() && !blockFitsQuads(2));
            if(rt->supportSimdGroupReduce() && (area <= short_seq || gemvAllArea)) {
                baseKeys.emplace_back("conv1x1_wquant_sg_reduce");

                std::string sgrWqStr = basicShaderPrefix + sgrWqShader;
                // memory bound not so seriously, can add more thread to reduce computation in each thread
                float ratio = 1.0 * icDiv4 / 2048.0 * oc / 2048.0;
                bool heavyMemory = ratio > 1.0;
                // g4mN instantiations only exist up to g4m16; larger areas fall
                // through to the all-area g8 kernel.
                const bool g4mNUsable = area <= 16 || (!heavyMemory && area <= 32);
                if(area > 1 && g4mNUsable) {
                    auto keys = baseKeys;
                    int piece = 1;
                    if(area > 5 && !heavyMemory) {
                        if(area % 2 != 0) {
                            keys.emplace_back("MNN_METAL_SRC_PROTECT");
                            [dic setValue:@"1" forKey:@"MNN_METAL_SRC_PROTECT"];;
                            option.preprocessorMacros = dic;
                        }
                        area = UP_DIV(area, 2);
                        piece = 2;
                    }
                    std::string kernel_name = "conv1x1_gemv_g4m" + std::to_string(area) + "_wquant_sg";
                    keys.emplace_back(kernel_name);
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), kernel_name.c_str(), option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 4), piece, 1), MTLSizeMake(32, 1, 1));
                } else if(g16LmHeadRoute) {
                    // lm_head path: g16 = 2 simdgroups per TG, 64 threads, 16 OC per TG.
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
                    // Pick between oc16Sg4K1 (whole K) and oc8Sg4K2 (split-K, 2x grid):
                    // split-K while its grid fits the TG budget, else fall back.
                    const bool oc16Sg4K1Window = quadsPerBlock >= 32 && quadsPerBlock < 96;
                    const bool tgBudgetRoute = tgBudget > 0 && wqEvenBlockFp16 && quadsPerBlock >= 32 &&
                        oc % 16 == 0;
                    const bool proTgBudgetRoute = tgBudgetRoute && gpuTier.proTier;
                    const int oc16Sg4K1Threadgroups = oc / 16;
                    const bool oc8Sg4K2FitsBudget = tgBudgetRoute && 2 * oc16Sg4K1Threadgroups <= tgBudget;
                    const int oc16Sg4K1GridLimit = gpuTier.proTier ? 2 * tgBudget : tgBudget;
                    const bool oc16Sg4K1ExceedsLimit = tgBudgetRoute && oc16Sg4K1Threadgroups > oc16Sg4K1GridLimit;
                    bool useOc8Sg4K2 = oc % 8 == 0 && blockCount % 2 == 0 &&
                        (oc8Sg4K2FitsBudget || oc16Sg4K1ExceedsLimit || (!oc16Sg4K1Window && !proTgBudgetRoute));
                    if (gpuTier.belowM4Gen && oc % 16 == 0 && quadsPerBlock <= 96) {
                        useOc8Sg4K2 = false;
                    }
                    if (useOc8Sg4K2) {
                        int lanesPerBlock = gemvLanesPerBlock(icDiv4, blockCount, true);
                        if (oc8Sg4K2FitsBudget && oc16Sg4K1Window && 4 * oc16Sg4K1Threadgroups >= tgBudget) {
                            lanesPerBlock = 4;
                        } else if (oc16Sg4K1ExceedsLimit && gpuTier.proTier) {
                            lanesPerBlock = gemvSplitKFillLanes(blockCount);
                        }
                        lanesPerBlock = gemvCacheLineLanes(icDiv4, blockCount, mDequantBits, lanesPerBlock);
                        auto keys = baseKeys;
                        keys.emplace_back("conv1x1_gemv_g4m1_2sg_wquant_sg");
                        keys.emplace_back("oc8Sg4K2");
                        if (lanesPerBlock > 0) {
                            keys.emplace_back("lpb" + std::to_string(lanesPerBlock));
                        }
                        auto pipeline = rt->findPipeline(keys);
                        if (nil == pipeline) {
                            NSMutableDictionary *skDic = [dic mutableCopy];
                            [skDic setValue:@"2" forKey:@"GEMV_QUADS_PER_TG"];
                            [skDic setValue:@"2" forKey:@"GEMV_SPLIT_K"];
                            if (lanesPerBlock > 0) {
                                [skDic setValue:@(std::to_string(lanesPerBlock).c_str())
                                         forKey:@"GEMV_LANES_PER_BLOCK"];
                            }
                            MTLCompileOptions *skOption = [[MTLCompileOptions alloc] init];
                            skOption.preprocessorMacros = skDic;
                            pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", skOption);
                            rt->insertPipeline(keys, pipeline);
                        }
                        mPipeline = pipeline;
                        CONV1X1_SET_TAG("splitk2_gemv_g4m1_2sg_wquant_sg");
                        mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 8), 1, 1),
                                                  MTLSizeMake(128, 1, 1));
                    } else {
                        const bool useOc16Sg4K1 = (oc % 16 == 0) &&
                                                  (quadsPerBlock >= 32 || gpuTier.belowM4Gen);
                        const int tgQuads = useOc16Sg4K1 ? 4 : 2;
                        const int plainLanes = gemvPlainLanesPerBlock(icDiv4, blockCount, mDequantBits);
                        auto keys = baseKeys;
                        keys.emplace_back("conv1x1_gemv_g4m1_2sg_wquant_sg");
                        if (useOc16Sg4K1) {
                            keys.emplace_back("oc16Sg4K1");
                        }
                        if (plainLanes > 0) {
                            keys.emplace_back("lpb" + std::to_string(plainLanes));
                        }
                        auto pipeline = rt->findPipeline(keys);
                        if (nil == pipeline) {
                            if (useOc16Sg4K1 || plainLanes > 0) {
                                NSMutableDictionary *dic4 = [dic mutableCopy];
                                if (useOc16Sg4K1) {
                                    [dic4 setValue:@"4" forKey:@"GEMV_QUADS_PER_TG"];
                                }
                                if (plainLanes > 0) {
                                    [dic4 setValue:@(std::to_string(plainLanes).c_str())
                                            forKey:@"GEMV_LANES_PER_BLOCK"];
                                }
                                MTLCompileOptions *opt4 = [[MTLCompileOptions alloc] init];
                                opt4.preprocessorMacros = dic4;
                                pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", opt4);
                            } else {
                                pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", option);
                            }
                            rt->insertPipeline(keys, pipeline);
                        }
                        mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                        mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 4 * tgQuads), 1, 1),
                                                  MTLSizeMake(32 * tgQuads, 1, 1));
                    }
                    // Fusion leaders build their own 2sg-based pipelines and force a
                    // 64-thread dispatch in their setup.
                    mIs2sgDecode = true;
                } else {
                    // All-area fallback for gemvAllArea shapes beyond the g4mN range.
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemv_g8_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g8_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 8), area, 1), MTLSizeMake(128, 1, 1));
                }
                return NO_ERROR;
            } else if(rt->supportSimdGroupMatrix()  && area > short_seq && oc > 8 && (icDiv4 % 8 == 0 || icDiv4 % 2 == 0)) {
                baseKeys.emplace_back("conv1x1_wquant_sg_matrix");
                std::string sgmWqStr = basicShaderPrefix + sgmWqShader;

                // Generally threadgroup memory >= 16KB
                auto smem_size = [[context device] maxThreadgroupMemoryLength];
                // choose different tile for different computation; 16x16 and 32x64 step 8 quads per
                // quant block, so other block sizes stay on 8x16.
                if(icDiv4 % 8 != 0 || !blockFitsQuads(8)) {
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
                } else {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_16x16_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), "conv1x1_gemm_16x16_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
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
                // W_QUANT_2/3 without simdGroupReduce: no usable kernel exists.
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

            // Fused quant GEMM unpacks weights in-kernel, skipping the dequant
            // pre-pass and mTempWeight; W3 is additionally gated by weight size.
            const bool tensorApiFusedQuant =
                                 (mDequantBits == 2 || mDequantBits == 3 ||
                                  mDequantBits == 4 || mDequantBits == 8) &&
                                 backend->isSupportTensorApi() &&
                                 (area >= 64 || specBlock) &&
                                 (mDequantBits != 3 ||
                                  (int64_t)oc * ic >= (int64_t)4 * 1024 * 1024);

            // M_TILE=64 variant halves grid.x and DRAM weight traffic, trading
            // higher register/smem pressure (needs tensor API).
            mFusedQuantM64 = tensorApiFusedQuant &&
                             (mDequantBits == 4 || mDequantBits == 8) && area >= 64;

            mTensorApiFusedQuant = tensorApiFusedQuant;

            if (!tensorApiFusedQuant) {
                int bytes = backend->useFp16InsteadFp32() ? 2 : 4;
                const int tempSize = ROUND_UP(oc, 4) * ROUND_UP(ic, 32) * bytes;
                // Keep the Tensor object alive: encode-replay holds a raw
                // pointer to it. Only the buffer range is reacquired per resize.
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
            bool fusedQuantDoubleBuf = false;
            // K-split x4 recovers TG-starved speculative-verify shapes.
            const bool fusedQ4SpecBlock = mTensorApiFusedQuant && mDequantPipeline == nil &&
                                      mDequantBits == 4 && !mFusedQuantM64 && specBlock;
            mUseFusedKsplit = false;
            mPrefillSiluOn = false;
            mPrefillDualOn = false;
            if (fusedQ4SpecBlock) {
                mUseFusedKsplit = area <= 32 && UP_DIV(oc, 64) <= 48 && blockCount >= 4;
            }
            // M8-native tile for the small-M shapes the K-split gate skips.
            const bool useFusedM8 = fusedQ4SpecBlock && !mUseFusedKsplit && area > 1 && area <= 8;
            if (mTensorApiFusedQuant) {
                if (mUseFusedKsplit) {
                    // Stack the M8 tile on K-split when the block is narrow enough.
                    mKsplitM8 = area <= 8;
                    gemmKernelName = mKsplitM8 ? "conv1x1_fused_q4_gemm_stage_ksplit_m8"
                                               : "conv1x1_fused_q4_gemm_stage_ksplit";
                    keys.emplace_back(gemmKernelName);
                } else if (useFusedM8) {
                    gemmKernelName = "conv1x1_fused_q4_gemm_stage_m8";
                    keys.emplace_back("conv1x1_fused_q4_gemm_stage_m8");
                } else if (mFusedQuantM64) {
                    gemmKernelName = "conv1x1_fused_q4_gemm_stage_m64";
                    keys.emplace_back(gemmKernelName);
                    // Double-buffered operand staging: one barrier per K8 window.
                    // fp16 only, fp32 would exceed the 32 KB threadgroup limit.
                    fusedQuantDoubleBuf = backend->useFp16InsteadFp32();
                    if (fusedQuantDoubleBuf) {
                        keys.emplace_back("FQ_DOUBLE_BUF");
                    }
                    // Fold the group's MUL_SILU into this epilogue, gated by the
                    // env-adjustable min area.
                    mPrefillSiluOn = (mDequantBits == 4 || mDequantBits == 8) && nullptr != mPrefillSiluGate &&
                                     area >= MetalEnv::get().fusedQ4GateUpDualMinArea;
                    // The dual supersedes the fold: the gate tile stays in a second
                    // accumulator and is never written.
                    mPrefillDualOn = (mDequantBits == 4 || mDequantBits == 8) && nullptr != mPrefillDualPeer &&
                                     area >= MetalEnv::get().fusedQ4GateUpDualMinArea;
                    if (mPrefillDualOn) {
                        mPrefillSiluOn = false;
                        keys.emplace_back("FQ4_GATEUP_DUAL");
                        if (nil == mPrefillDualCoef) {
                            mPrefillDualCoef = backend->getConstBuffer(sizeof(float));
                        }
                        ((float *)mPrefillDualCoef.contents)[0] = mPrefillDualPeer->scaleCoef();
                    } else if (mPrefillSiluOn) {
                        keys.emplace_back("FQ4_SILU_MUL");
                    }
                } else {
                    gemmKernelName = "conv1x1_fused_q4_gemm_stage";
                    keys.emplace_back("conv1x1_fused_q4_gemm_stage");
                }
            } else if (outerSgM64) {
                // sg_matrix M=64 tile: halves grid.x and weight DRAM traffic, same
                // bindings as the 32x64 kernel.
                gemmKernelName = "conv1x1_gemm_64x64_split_k_sg";
                keys.emplace_back("conv1x1_gemm_64x64_split_k_sg");
                sgMatrixM64 = true;
            } else {
                keys.emplace_back("conv1x1_gemm_32x64_split_k_sg");
            }

            NSMutableDictionary *dic = [baseDic mutableCopy];
            // The fused-stage K=32 tile spans 8 K4 groups.
            if (icDiv4 % 8 != 0) {
                [dic setValue:@"1" forKey:@"MNN_METAL_SRC_PROTECT"];
                keys.emplace_back("MNN_METAL_SRC_PROTECT");
            }
            if(backend->isSupportTensorApi() == true) {
                [dic setValue:@"1" forKey:@"USE_METAL_TENSOR_OPS"];
                keys.emplace_back("USE_METAL_TENSOR_OPS");
                if(ic > oc && ic > 2048 && (ic / blockCount) % 64 == 0 && !mTensorApiFusedQuant) {
                    // LOOP_K64 only exists for conv1x1_gemm_32x64_split_k_sg.
                    [dic setValue:@"1" forKey:@"LOOP_K64"];
                    keys.emplace_back("LOOP_K64");
                }
            }
            if (mTensorApiFusedQuant) {
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
                        // The M8/K-split kernels are guarded by W_QUANT_4 &&
                        // FUSED_Q4_REAL_UNPACK.
                        [dic setValue:@"1" forKey:@"FUSED_Q4_REAL_UNPACK"];
                        keys.emplace_back("FUSED_Q4_REAL_UNPACK");
                        break;
                    default:
                        [dic setValue:@"1" forKey:@"W_QUANT_8"];
                        keys.emplace_back("W_QUANT_8");
                        break;
                }
            }
            if (fusedQuantDoubleBuf) {
                [dic setValue:@"1" forKey:@"FQ_DOUBLE_BUF"];
            }
            if (mPrefillSiluOn) {
                [dic setValue:@"1" forKey:@"FQ4_SILU_MUL"];
            }
            if (mPrefillDualOn) {
                [dic setValue:@"1" forKey:@"FQ4_GATEUP_DUAL"];
            }
            option.preprocessorMacros = dic;

            auto pipeline = rt->findPipeline(keys);
            if (nil == pipeline) {
                pipeline = backend->makeComputePipelineWithSourceOption(sgmWfpStr.c_str(), gemmKernelName, option);
                rt->insertPipeline(keys, pipeline);
            }
            mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
            if (mUseFusedKsplit) {
                // grid gains z=4 K-partitions; the reduce pass sums the partials.
                mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, mKsplitM8 ? 8 : 32), UP_DIV(oc, 64), 4), MTLSizeMake(128, 1, 1));
                auto rkeys = keys;
                rkeys.emplace_back("conv1x1_fused_q4_ksplit_reduce");
                auto rpipeline = rt->findPipeline(rkeys);
                if (nil == rpipeline) {
                    rpipeline = backend->makeComputePipelineWithSourceOption(sgmWfpStr.c_str(), "conv1x1_fused_q4_ksplit_reduce", option);
                    rt->insertPipeline(rkeys, rpipeline);
                }
                mKsplitReducePipeline = rpipeline;
                // fp32 partials [KS=4, ocDiv4, area, 4] = 4 * ocDiv4 * area * 16 bytes.
                int totalOut = ocDiv4 * area;
                mKsplitPartial.reset(Tensor::createDevice<uint8_t>(std::vector<int>{4 * totalOut * 16}));
                backend->onAcquireBuffer(mKsplitPartial.get(), Backend::DYNAMIC);
                backend->onReleaseBuffer(mKsplitPartial.get(), Backend::DYNAMIC);
                mKsplitReduceThreads = std::make_pair(MTLSizeMake(UP_DIV(totalOut, 128), 1, 1), MTLSizeMake(128, 1, 1));
            } else {
                const int mTile = useFusedM8 ? 8 : ((mFusedQuantM64 || sgMatrixM64) ? 64 : 32);
                // The dual halves N per matrix, so it needs twice the N tiles.
                const int nTile = mPrefillDualOn ? 32 : 64;
                mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, mTile), UP_DIV(oc, nTile), 1),
                                          MTLSizeMake(128, 1, 1));
            }
        }

        return NO_ERROR;
    }

    option.preprocessorMacros = baseDic;

    if(rt->supportSimdGroupMatrix()) {
        std::string sgmWfpStr = basicShaderPrefix + sgmWfpShader;

        baseKeys.emplace_back("conv1x1_float_sg_matrix");
        // total computation not too small
        if(area >= 16 && icDiv4 >= 4 && icDiv4 % 2 == 0 && ocDiv4 >= 4 && area * icDiv4 * ocDiv4 >= 64 * 64 * 64) {
            // Enough threads
            if(area * ocDiv4 / icDiv4 >= 1024) {
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
        if(icDiv4 >= 32 && icDiv4 % 2 == 0 && 1.0 * area * ocDiv4 / icDiv4 < magic_num) {
            auto keys = baseKeys;
            keys.emplace_back("conv1x1_z4_sg");
            auto pipeline = rt->findPipeline(keys);
            if (nil == pipeline) {
                pipeline = backend->makeComputePipelineWithSourceOption(sgrWfpStr.c_str(), "conv1x1_z4_sg", option);
                rt->insertPipeline(keys, pipeline);
            }
            mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
            mThreads = std::make_pair(MTLSizeMake(ow * oh, ocDiv4, ob), MTLSizeMake(32, 1, 1));
            return NO_ERROR;
        }
    }
    if(rt->getTuneLevel() == Never) {
        if (ow * oh >= 128) {
            NSUInteger gid_x = UP_DIV(ow * oh, 8);
            NSUInteger gid_y = ocDiv4;
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
            NSUInteger gid_y = ocDiv4;
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
        }
    } else {
        NSString* shaderName[] = {@"conv1x1_g1z8", @"conv1x1_g1z4", @"conv1x1_w4h4",  @"conv1x1_w2c2", @"conv1x1_w4c2"};
        int itemW[] = {8, 4, 16, 2, 4};
        int itemC[] = {4, 4, 4, 8, 8};
        int actual_kernel = 5;
        if (ocDiv4 % 2 != 0) {
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
        }
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
        // buffer(1): gate output, or the group's SwiGLU output when the epilogue
        // folds MUL_SILU in
        {
            auto outTensor = mGateUpSilu ? mGateUpSiluOutput : output;
            [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)outTensor->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(outTensor)->offset atIndex:1];
        }
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
        // buffer(2): q params (input dims shared; k/v outputDepthQuad via seg)
        [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
        if (mQKVMergedBuffer != nil) {
            [encoder setBuffer:mQKVMergedBuffer offset:mQKVMergedOff[0] atIndex:3];
            [encoder setBuffer:mQKVMergedBuffer offset:mQKVMergedOff[4] atIndex:5];
            [encoder setBuffer:mQKVMergedBuffer offset:mQKVMergedOff[1] atIndex:7];
            [encoder setBuffer:mQKVMergedBuffer offset:mQKVMergedOff[5] atIndex:9];
            [encoder setBuffer:mQKVMergedBuffer offset:mQKVMergedOff[2] atIndex:11];
            [encoder setBuffer:mQKVMergedBuffer offset:mQKVMergedOff[6] atIndex:13];
        } else {
            MetalBackend::setTensor(mWeight.get(), encoder, 3);
            MetalBackend::setTensor(getDequantScale().get(), encoder, 5);
            MetalBackend::setTensor(mQKVPeerK->getWeight().get(), encoder, 7);
            MetalBackend::setTensor(mQKVPeerK->getDequantScale().get(), encoder, 9);
            MetalBackend::setTensor(mQKVPeerV->getWeight().get(), encoder, 11);
            MetalBackend::setTensor(mQKVPeerV->getDequantScale().get(), encoder, 13);
        }
        MetalBackend::setTensor(mBias.get(), encoder, 4);
        // buffers(6-9): k projection
        MetalBackend::setTensor(mQKVPeerKOutput, encoder, 6);
        MetalBackend::setTensor(mQKVPeerK->getBias().get(), encoder, 8);
        // buffers(10-13): v projection
        MetalBackend::setTensor(mQKVPeerVOutput, encoder, 10);
        MetalBackend::setTensor(mQKVPeerV->getBias().get(), encoder, 12);
        // buffer(14): {k_coef, v_coef, kOutQuad, vOutQuad[, w_coef, wOutQuad]}
        [encoder setBuffer:mQKVSegBuffer offset:0 atIndex:14];
        // buffers(15-18): optional 4th projection (QKV_FUSED_P4)
        if (mQKVPeerW != nullptr && mQKVPeerWOutput != nullptr) {
            MetalBackend::setTensor(mQKVPeerWOutput, encoder, 15);
            MetalBackend::setTensor(mQKVPeerW->getBias().get(), encoder, 17);
            if (mQKVMergedCount == 4) {
                [encoder setBuffer:mQKVMergedBuffer offset:mQKVMergedOff[3] atIndex:16];
                [encoder setBuffer:mQKVMergedBuffer offset:mQKVMergedOff[7] atIndex:18];
            } else {
                MetalBackend::setTensor(mQKVPeerW->getWeight().get(), encoder, 16);
                MetalBackend::setTensor(mQKVPeerW->getDequantScale().get(), encoder, 18);
            }
        }
        if (mHasLNFusion) {
            bindLNBuffers(encoder);
        }
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
        return;
    }

    // Plain single-conv LN fusion: the standalone LN_FUSED kernel computes RMSNorm
    // in-kernel, no leader/follower pairing.
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
        // Fused path: mDequantPipeline is nil, so dispatch only the fused GEMM which
        // reads quantized weight from buffer(3) directly.
        const bool fused = (mDequantPipeline == nil) && mTensorApiFusedQuant;

#if MNN_METAL_OP_PROFILE
        // Split the two sub-passes into independent command buffers so each shows up
        // as its own profile row.
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
                // Two-pass: the ksplit GEMM writes fp32 partials, the reduce pass sums
                // them into the real output.
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
            if (mTensorApiFusedQuant) {
                // buffer(6) is a placeholder alias of mWeight, never read by the fused
                // kernel.
                MetalBackend::setTensor(mWeight.get(), encoder, 3);
                MetalBackend::setTensor(mBias.get(), encoder, 4);
                MetalBackend::setTensor(getDequantScale().get(), encoder, 5);
                MetalBackend::setTensor(mWeight.get(), encoder, 6);
                if (mPrefillDualOn) {
                    // Peer (gate) operands mirror 3/4/5 at 7/8/9, with the gate's own
                    // scale coefficient at 10.
                    MetalBackend::setTensor(mPrefillDualPeer->getWeight().get(), encoder, 7);
                    MetalBackend::setTensor(mPrefillDualPeer->getBias().get(), encoder, 8);
                    MetalBackend::setTensor(mPrefillDualPeer->getDequantScale().get(), encoder, 9);
                    [encoder setBuffer:mPrefillDualCoef offset:0 atIndex:10];
                } else if (mPrefillSiluOn) {
                    MetalBackend::setTensor(mPrefillSiluGate, encoder, 7);
                }
            } else {
                // Legacy conv1x1_gemm_32x64_split_k_sg: buffer(3) is the dequanted
                // weight (mTempWeight).
                MetalBackend::setTensor(mTempWeight.get(), encoder, 3);
                MetalBackend::setTensor(mBias.get(), encoder, 4);
                MetalBackend::setTensor(getDequantScale().get(), encoder, 5);
            }
            [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
        }
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
