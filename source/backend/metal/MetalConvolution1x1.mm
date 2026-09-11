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

static bool isQ4W16QuadsPerBlock(int quadsPerBlock) {
    return quadsPerBlock == 8 || quadsPerBlock == 16 || quadsPerBlock == 32 || quadsPerBlock == 64;
}

// Mirrors the shader's SIMD_GROUP_WIDTH: both lane-choosing helpers below
// partition one simdgroup, which sweeps one K half.
static constexpr int kMetalSimdGroupWidth = 32;

// Lanes per quant block for the Q4 16-byte decode GEMV (GEMV_QBLOCK_W16_LANES_PER_BLOCK).
// A simdgroup owns `blocksPerSimdgroup` quant blocks. Putting L lanes on one
// block puts `kMetalSimdGroupWidth / L` blocks in flight and gives each lane
// `uint4LoadsPerBlock / L` uint4 loads to walk. One uint4 covers two C4 quads.
//
// Counting loads only ranks the candidates into ties: with 8 uint4 loads per
// block every L in {2, 4, 8} costs the same whenever the owned block count is a
// multiple of the in-flight count. The tie-break below is empirical, not
// derived: L=4 is the optimum at 8, 16 and 24 blocks per simdgroup, ties with
// L=2 at 32, and only loses at 48, where its 6 iterations fall behind L=2's 3.
// From 64 blocks up every candidate ties and L=2 wins the whole tie zone,
// measured 4-9% of weight throughput above L=1 at 64/96/128/160/192/224/256
// blocks; leaving the tie to candidate order would hand that zone to L=1.
//
// Once per 16 blocks L=8 uniquely minimizes the loads instead: owned ≡ 12
// (mod 16) puts L=2 one load behind (20 vs 19 at 76, b152's split-K half).
// Measured at 76, L=8 loses ~3% of weight throughput to L=2 -- 19 rounds with
// only 4 blocks in flight is latency-bound next to L=2's 5 rounds of 16 -- but
// at 20 (4b's fused qkv) the unique L=8 minimum is the measured optimum, so the
// L=2 override only applies from the tie zone's 64 blocks up. The one-load gap
// exists only at 8 uint4 loads per block (block 64), so the rule is inert for
// other block sizes.
//
// Do not try to fold that back into the cost model: the empirical preference is
// for *more* iterations at 16 blocks (L=4 over L=2) and *fewer* at 48 (L=2 over
// L=4), so any per-iteration penalty gets one of the two wrong. The preference
// is for the 4-lanes-per-block grouping itself, fading into 2 beyond it.
//
// The above is the M4-class calibration. On tensor-API hardware two of those
// tie-breaks flip, and not at the same owned count for every kernel family, so
// the caller tags the pipeline (wideSplitK) and the device gate below applies
// per family: the wide split-K QKV kernel (8 simdgroups, 4 quads/TG) prefers
// L=8 at 8 blocks per simdgroup but keeps L=4 at 16, while the 2-simdgroup
// non-split kernel prefers L=8 at 16 blocks but keeps L=4 at 32. Both flips
// only fire when L=8 ties the load minimum, so unique-minimum shapes (e.g.
// the 20-block split-K half) are untouched. Measured on tensor-API hardware
// where the wide kernel gains 4-9% and the non-split kernel 10% of weight
// throughput; non-tensor-API devices keep the M4 rules above unchanged.
static constexpr int kQ4W16PreferredLanesPerBlock = 4;
static constexpr int kQ4W16PreferredMaxBlockIters = 4;
static constexpr int kQ4W16TieLanesPerBlock = 2;
static constexpr int kQ4W16NearTieMinBlocks = 64;
static constexpr int kQ4W16WideLanesPerBlock = 8;
// Blocks per simdgroup at which each family flips its tie preference to L=8.
static constexpr int kQ4W16M5WideSplitBlocks = 8;
static constexpr int kQ4W16M5PlainBlocks = 16;

static int chooseQ4W16LanesPerBlock(int blocksPerSimdgroup, int uint4LoadsPerBlock, bool wideSplitK,
                                    bool tensorApiDevice) {
    int bestLanes = 1, bestLoadsPerLane = 1 << 30;
    int preferredBlockIters = 0;
    int tieLanesLoads = 1 << 30;
    int wideLanesLoads = 1 << 30;
    bool preferredTied = false;
    bool tieLanesTied = false;
    const int blocksOwned = blocksPerSimdgroup > 1 ? blocksPerSimdgroup : 1;
    const int forcedLanes = MetalEnv::get().gemvW16LanesPerBlock;
    for (int candidate : {1, 2, 4, 8}) {
        if (candidate > uint4LoadsPerBlock || uint4LoadsPerBlock % candidate != 0) {
            continue;
        }
        if (forcedLanes != 0) {
            if (candidate == forcedLanes) {
                return candidate;
            }
            continue;
        }
        const int blocksInFlight = kMetalSimdGroupWidth / candidate;
        const int blockIters     = UP_DIV(blocksOwned, blocksInFlight);
        const int loadsPerLane   = blockIters * (uint4LoadsPerBlock / candidate);
        if (loadsPerLane < bestLoadsPerLane) {
            bestLoadsPerLane = loadsPerLane;
            bestLanes = candidate;
            preferredTied = false;
            tieLanesTied = false;
        }
        if (candidate == kQ4W16TieLanesPerBlock) {
            tieLanesLoads = loadsPerLane;
        }
        if (candidate == kQ4W16WideLanesPerBlock) {
            wideLanesLoads = loadsPerLane;
        }
        if (candidate == kQ4W16PreferredLanesPerBlock && loadsPerLane == bestLoadsPerLane) {
            preferredTied = true;
            preferredBlockIters = blockIters;
        } else if (candidate == kQ4W16TieLanesPerBlock && loadsPerLane == bestLoadsPerLane) {
            tieLanesTied = true;
        }
    }
    // M5-class tie-break, before the M4 preference below (which would hand both
    // flipped shapes to L=4). Gated on L=8 actually tying the load minimum, so
    // forced-lanes envs and unique-minimum shapes bypass it unchanged.
    if (tensorApiDevice && bestLanes != kQ4W16WideLanesPerBlock && wideLanesLoads == bestLoadsPerLane &&
        blocksOwned == (wideSplitK ? kQ4W16M5WideSplitBlocks : kQ4W16M5PlainBlocks)) {
        return kQ4W16WideLanesPerBlock;
    }
    if (preferredTied && preferredBlockIters <= kQ4W16PreferredMaxBlockIters) {
        return kQ4W16PreferredLanesPerBlock;
    }
    if (bestLanes == 8 && tieLanesLoads == bestLoadsPerLane + 1 && blocksOwned >= kQ4W16NearTieMinBlocks) {
        return kQ4W16TieLanesPerBlock;
    }
    if (tieLanesTied) {
        return kQ4W16TieLanesPerBlock;
    }
    return bestLanes;
}

// Lanes for the gate/up dual-stream W16 branch: mirrors the generic body's
// runtime widening (max(quadsPerBlock/4, 32/(blockCount/SK))) so the compiled
// lane count equals what laneInBlock/blockSlot derive at runtime, and the uint4
// pair loop additionally needs lanesPerBlock <= quadsPerBlock/2 (one whole pair
// per lane). MNN_METAL_GEMV_W16_MID overrides for sweeps. Returns 0 when the
// branch cannot apply.
static int gateUpW16DsLanes(int quadsPerBlock, int dsBlocksOwned) {
    if (quadsPerBlock <= 0 || dsBlocksOwned <= 0) {
        return 0;
    }
    int lanes = ALIMAX(quadsPerBlock / 4, kMetalSimdGroupWidth / dsBlocksOwned);
    lanes = ALIMIN(lanes, kMetalSimdGroupWidth);
    if (MetalEnv::get().gemvW16LanesPerBlock != 0) {
        lanes = MetalEnv::get().gemvW16LanesPerBlock;
    }
    return lanes <= quadsPerBlock / 2 ? lanes : 0;
}

// QKV seg buffer layout, shared with the shader's qkv_seg:
// [0..1] k/v scaleCoef, [2..3] k/v outputDepthQuad, [4..5] 4th projection's
// scaleCoef + outputDepthQuad, [6..8] packed-grid base of projections 1..3,
// [9..12] each projection's output offset inside the merged
// output allocation, in ftype4 units (QKV_MERGED_OUT only; slot 9 is the
// leader's and is 0).
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

// Lane split for the 2sg decode GEMV: the returned count of lanes sweeps K
// inside one quant block, and the remaining lanes / lanesPerBlock groups walk
// different blocks concurrently.
//
// The historical value is quadsPerBlock/4, tuned when a simdgroup covered every
// block of a row. Splitting K halves that count, so on shapes with few, fat blocks
// a lane ends up sweeping several blocks and re-pays the per-block scale/bias
// loads on each one -- measurably so, because the inner sweep is too short to
// hide them. Size the in-flight count to the blocks the simdgroup actually
// covers, so a lane group owns exactly one, but never drop below two lanes on
// the inner reduction: once there are more blocks than lanes a group already
// sweeps more than one block and narrowing further would trade away the inner
// parallelism for nothing.
//
// Returns 0 when the shape should keep the legacy expression.
//
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
    const int chosenLanes =
        std::max(std::min(legacyLanes, oneBlockPerLaneGroup), std::min(legacyLanes, 2));
    int pow2Lanes = 1;   // the in-flight block count has to divide the simdgroup evenly
    while (pow2Lanes * 2 <= chosenLanes) {
        pow2Lanes *= 2;
    }
    return pow2Lanes;
}

static void copyTensorBytes(id<MTLBuffer> dst, size_t dstOff, const std::shared_ptr<Tensor>& src, size_t bytes) {
    auto alloc = (MetalRuntimeAllocator::MetalBufferAlloc *)src->deviceId();
    memcpy((uint8_t *)dst.contents + dstOff,
           (uint8_t *)alloc->getBuffer().contents + TensorUtils::getDescribeOrigin(src.get())->offset, bytes);
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

    // Build fused pipeline with GATE_UP_FUSED macro
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();

    // Store up's scaleCoef separately: gate uses cst.scaleCoef (via buffer(2)),
    // but up needs its own tensor-specific coefficient. Without this, up's dequant
    // is scaled by gate's coefficient and any range mismatch drifts decode into
    // garbage on models like Qwen3.5-2B.
    // Allocated once and rewritten in place, like mConstBuffer (see onResize).
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
    const bool dualStream = !backend->isSupportTensorApi();
    // The SwiGLU epilogue reuses the dual-stream body, so it needs
    // GEMV_2OCQUAD_PER_SG. On tensor-API devices (whose fused pipelines are
    // otherwise single-stream) this one pipeline is compiled with that macro as
    // well: its grid — one quad of both matrices per simdgroup, grid.z 1 — is
    // exactly the single-stream fused grid, so only the macros change.
    mGateUpSilu = siluOutput != nullptr;
    mGateUpSiluOutput = mGateUpSilu ? siluOutput : nullptr;
    // 4-simdgroup non-split-K path for fused GateUp on tensor-API devices,
    // analogous to the standalone use4sg route: doubles in-flight weight reads
    // per threadgroup without the split-K barrier. Requires both projections
    // to have oc divisible by 16 and blockCount >= 32.
    const int gateUpDepthQuad = ((Param*)mConstBuffer.contents)->outputDepthQuad;
    const int peerDepthQuad = ((Param*)peer->mConstBuffer.contents)->outputDepthQuad;
    const bool use4sg = !dualStream && !mGateUpSilu && mBlockCount >= 32 &&
                        gateUpDepthQuad % 4 == 0 && peerDepthQuad % 4 == 0;
    mUse4sg = use4sg;
    if (use4sg) {
        keys.emplace_back("tg4");
    } else if (dualStream || mGateUpSilu) {
        keys.emplace_back("GEMV_2OCQUAD_PER_SG");
    }
    // K-split inside the SwiGLU dual-stream body: four K quarters over 8
    // simdgroups/TG. Even outputDepthQuad keeps the grid exact so no simdgroup
    // skips the reduce barrier; a power-of-two block count keeps the per-pair
    // block range and the derived lanes-per-block both exact powers of two.
    mGateUpDualStreamSplitK = 0;
    if (mGateUpSilu && (gateUpDepthQuad % 2 == 0) && mBlockCount >= 4 && (mBlockCount & (mBlockCount - 1)) == 0) {
        mGateUpDualStreamSplitK = 2;
    }
    const int skOverride = MetalEnv::get().gateUpSplitK;
    if (mGateUpSilu && skOverride >= 0) {
        // An override the shape cannot divide evenly would break the exact grid
        // (no early returns before the reduce barrier), so fall back to no split.
        mGateUpDualStreamSplitK =
            (skOverride > 0 && (gateUpDepthQuad % 2 == 0) && mBlockCount >= skOverride &&
             mBlockCount % skOverride == 0)
                ? skOverride
                : 0;
    }
    if (mGateUpDualStreamSplitK > 0) {
        keys.emplace_back("GEMV_2OCQUAD_PER_SG_SPLIT_K_" + std::to_string(mGateUpDualStreamSplitK));
    }
    // Extend the generalized Q4 W16 decode specialization onto the fused GEMV.
    // Gate and up share the block-input and must compile the same block shape.
    const bool w16 = mQ4W16QuadsPerBlock > 0 && mQ4W16QuadsPerBlock == peer->mQ4W16QuadsPerBlock &&
                     mDequantBits == 4;
    const int w16BlocksPerSimdgroup =
        mGateUpDualStreamSplitK > 0 ? mBlockCount / mGateUpDualStreamSplitK : mBlockCount;
    const int w16DsLanes = w16 && mGateUpSilu && mGateUpDualStreamSplitK > 0 &&
                           !MetalEnv::get().gemvW16DsDisabled
                               ? gateUpW16DsLanes(mQ4W16QuadsPerBlock,
                                                  mBlockCount / mGateUpDualStreamSplitK)
                               : 0;
    const bool w16Ds = w16DsLanes > 0;
    const int w16LanesPerBlock = w16Ds ? w16DsLanes
                                       : (w16 ? chooseQ4W16LanesPerBlock(w16BlocksPerSimdgroup,
                                                                         mQ4W16QuadsPerBlock / 2, false,
                                                                         backend->isSupportTensorApi())
                                              : 0);
    if (w16) {
        keys.emplace_back("GEMV_QBLOCK_W16");
        keys.emplace_back("W16_QUADS_PER_BLOCK_" + std::to_string(mQ4W16QuadsPerBlock));
        keys.emplace_back("W16_LANES_PER_BLOCK_" + std::to_string(w16LanesPerBlock));
        if (w16Ds) {
            keys.emplace_back("GEMV_QBLOCK_W16_DS");
        }
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
        if (use4sg) {
            [dic setValue:@"4" forKey:@"GEMV_QUADS_PER_TG"];
        } else if (dualStream || mGateUpSilu) {
            [dic setValue:@"1" forKey:@"GEMV_2OCQUAD_PER_SG"];
        }
        if (mGateUpDualStreamSplitK > 0) {
            [dic setValue:@(mGateUpDualStreamSplitK).stringValue forKey:@"GEMV_2OCQUAD_PER_SG_SPLIT_K"];
        }
        if (w16) {
            [dic setValue:@"1" forKey:@"GEMV_QBLOCK_W16"];
            [dic setValue:@(mQ4W16QuadsPerBlock).stringValue forKey:@"GEMV_QBLOCK_W16_QUADS_PER_BLOCK"];
            [dic setValue:@(w16LanesPerBlock).stringValue forKey:@"GEMV_QBLOCK_W16_LANES_PER_BLOCK"];
            if (w16Ds) {
                [dic setValue:@"1" forKey:@"GEMV_QBLOCK_W16_DS"];
            }
        }
        if (mGateUpSilu) {
            [dic setValue:@"1" forKey:@"GATE_UP_SILU"];
        }
        option.preprocessorMacros = dic;

        std::string sgrWqStr = std::string(gBasicConvPrefix) + gConv1x1WqSgReduce;
        mGateUpFusedPipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", option);
        rt->insertPipeline(keys, mGateUpFusedPipeline);
    }

    if (nil == mGateUpFusedPipeline) {
        // Compilation failed, revert fusion. mGateUpSegBuffer is kept (nothing
        // binds it with mIsGateUpLeader false) so its identity stays stable for
        // any live recording, see the mConstBuffer note in onResize.
        mIsGateUpLeader = false;
        mGateUpPeer = nullptr;
        mGateUpSilu = false;
        mGateUpSiluOutput = nullptr;
        mGateUpDualStreamSplitK = 0;
        mUse4sg = false;
        peer->mIsGateUpFollower = false;
        return false;
    }

    // Update grid: add z=2 dimension for gate/up selection. The fused pipeline
    // is 2sg-kernel based — force 64 threads (plain pipeline may be split-K g8
    // with a 128-thread group). The use4sg variant doubles to 4 simdgroups /
    // 128 threads.
    auto gridSize = mThreads.first;
    NSUInteger gw = gridSize.width;
    NSUInteger gz = 2;
    int tgThreads = 64;
    if (use4sg) {
        gw = (NSUInteger)UP_DIV(ALIMAX(gateUpDepthQuad, peerDepthQuad), 4);
        tgThreads = 128;
    } else if (dualStream) {
        gw = (NSUInteger)UP_DIV(gateUpDepthQuad, 4);
        if (mGateUpSilu) {
            // A simdgroup now covers one quad of both matrices rather than two
            // quads of one, so the gate/up split moves out of z and into x. The
            // trade is exact: the threadgroup count and the four quad-dots each
            // threadgroup issues are the same as the unfused grid.
            gw = (NSUInteger)UP_DIV(gateUpDepthQuad, 2);
            gz = 1;
        }
    } else if (mGateUpSilu) {
        // Single-stream devices: the SILU pipeline is GEMV_2OCQUAD_PER_SG +
        // GATE_UP_SILU, one quad of both matrices per simdgroup — same
        // threadgroup count as the plain single-stream fused grid, with the
        // gate/up z dimension gone.
        gw = (NSUInteger)UP_DIV(gateUpDepthQuad, 2);
        gz = 1;
    }
    if (mGateUpDualStreamSplitK > 0) {
        tgThreads = 64 * mGateUpDualStreamSplitK;
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
    // outputDepthQuad and scaleCoef, so quant layout and activation must match.
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
    // Generalized Q4 W16 on the fused QKV GEMV. Every member must carry the
    // same compile-time quant-block shape.
    const bool w16 = mQ4W16QuadsPerBlock > 0 && mQ4W16QuadsPerBlock == peerK->mQ4W16QuadsPerBlock &&
                     mQ4W16QuadsPerBlock == peerV->mQ4W16QuadsPerBlock &&
                     (peerW == nullptr || mQ4W16QuadsPerBlock == peerW->mQ4W16QuadsPerBlock) &&
                     mDequantBits == 4;
    const int numProj = peerW != nullptr ? 4 : 3;
    const int projDepthQuad[4] = {
        ((Param*)mConstBuffer.contents)->outputDepthQuad,
        ((Param*)peerK->mConstBuffer.contents)->outputDepthQuad,
        ((Param*)peerV->mConstBuffer.contents)->outputDepthQuad,
        peerW != nullptr ? ((Param*)peerW->mConstBuffer.contents)->outputDepthQuad : 0,
    };
    // Either K split needs an even per-stream block count so each half is a whole
    // number of quant blocks, and no projection may leave a simdgroup
    // early-returning before the reduction barrier -- so every projection's
    // quad count has to divide that shape's quads/TG.
    const bool splitKAllowed = mBlockCount % 2 == 0;
    // Wide keeps the dual-stream grid's 4 quads/TG on 8 simdgroups, so the
    // threadgroup count -- and with it the LN prologue's redundant input reads --
    // stays where the unsplit shape had it. The W16 gate below is a routing
    // choice, not a correctness one (the shader's geometry is W16-agnostic):
    // wide is single-stream, so it only beats the dual-stream shape it displaces
    // when the single-stream inner loop is the W16 one.
    //
    // No quant-block ceiling here: the ceiling below was tuned against the
    // narrow 2-quad shape, and wide behaves the opposite way on the shapes
    // where the two differ. Paired op runs, fused QKV, min of 5 rounds:
    // 40 blocks 3.08-3.10 vs 3.35 ms unsplit (4/4 pairs), 32 blocks neutral.
    // The 16-block shapes take the narrow split below instead: wide needs
    // 4 quads/TG per projection, and extending the 4-simdgroup gate below down
    // to 16 blocks measured as noise on both the LN-fused and no-LN variants.
    // Non-tensor devices keep the wide split there, where the single-stream
    // W16 loop displaces the dual-stream shape it was measured against.
    bool skWide = splitKAllowed && w16 && (mBlockCount >= 32 || !backend->isSupportTensorApi());
    for (int p = 0; p < numProj; ++p) {
        skWide = skWide && projDepthQuad[p] % 4 == 0;
    }
    // GEMV_2OCQUAD_PER_SG pairs with the quadsPerTG=4 grid below; dropping either half
    // leaves half of every projection's rows unwritten (2026-08-19 rebase-merge
    // regression: key/dic lost while the grid kept 4 quads/TG).
    const bool dualStream = !skWide && !backend->isSupportTensorApi();
    // 4-simdgroup non-split-K path for fused QKV on tensor-API devices,
    // analogous to the standalone use4sg route: doubles in-flight weight reads
    // per threadgroup without the split-K barrier.
    bool use4sg = !skWide && !dualStream && mBlockCount >= 32;
    for (int p = 0; p < numProj; ++p) {
        use4sg = use4sg && projDepthQuad[p] % 4 == 0;
    }
    // Narrow split: 2 quads/TG over 4 simdgroups. Halving quads/TG doubles the
    // threadgroup count, but it carries no W16 requirement, so it covers the
    // shapes wide cannot take. The quant-block ceiling belongs here: with the
    // W16 body the split only pays back its barrier while a simdgroup's K range
    // stays short. Paired runs, fused QKV decode, 28-layer totals on M5, min of
    // 3 rounds: 16 blocks wins on every shape it gates -- 0.6b qkv 0.590 vs
    // 0.633 ms, 3.5-0.8b qkv 0.158 vs 0.166, 3.5-0.8b linear_in 0.724 vs 0.749
    // (2/2 pairs each). 32 blocks and up route to skWide instead. An earlier
    // A/B put the ceiling at 8; it was run before the harness stopped aliasing
    // its per-layer output buffers, which serialized the dispatches and hid the
    // parallelism the split buys.
    bool skPlain = !skWide && !use4sg && !dualStream && splitKAllowed &&
                   mBlockCount <= MetalEnv::get().qkvSplitKMaxBlocks;
    for (int p = 0; p < numProj; ++p) {
        skPlain = skPlain && projDepthQuad[p] % 2 == 0;
    }
    mQKVDualStream = dualStream;
    mQKVSplitK = skWide || skPlain;
    mQKVSplitKWide = skWide;
    mUse4sg = use4sg;
    if (use4sg) {
        keys.emplace_back("tg4");
    }
    if (dualStream) {
        keys.emplace_back("GEMV_2OCQUAD_PER_SG");
    }
    // Lane split for the non-W16 body, which the split shapes narrow to half a
    // row's blocks per simdgroup.
    const int qkvSkLanesPerBlock =
        mQKVSplitK ? gemvLanesPerBlock(((Param*)mConstBuffer.contents)->inputDepthQuad, mBlockCount, true) : 0;
    // Threadgroup shape as two numbers; see the shader's geometry block. The
    // grid below and the macro must come from this one value, or the host
    // dispatches a wider grid than the kernel writes.
    const int quadsPerTG = (dualStream || skWide || use4sg) ? 4 : 2;
    if (mQKVSplitK) {
        keys.emplace_back("qtg" + std::to_string(quadsPerTG));
        keys.emplace_back("sk2");
        if (qkvSkLanesPerBlock > 0) {
            keys.emplace_back("lpb" + std::to_string(qkvSkLanesPerBlock));
        }
    }
    // Split-K halves the blocks a simdgroup owns, so the lane split is
    // re-chosen for that pipeline. Only the single-stream body is W16-specialized.
    const int w16LanesPerBlock = w16 ? chooseQ4W16LanesPerBlock(mQKVSplitK ? mBlockCount / 2 : mBlockCount,
                                                               mQ4W16QuadsPerBlock / 2, skWide,
                                                               backend->isSupportTensorApi())
                                     : 0;
    if (w16) {
        keys.emplace_back("GEMV_QBLOCK_W16");
        keys.emplace_back("W16_QUADS_PER_BLOCK_" + std::to_string(mQ4W16QuadsPerBlock));
        keys.emplace_back("W16_LANES_PER_BLOCK_" + std::to_string(w16LanesPerBlock));
    }
    // Grid geometry, decided before the pipeline is built: the packed grid is a
    // compile-time variant (the shader derives the projection from x instead of
    // z), so it has to be part of the cache key and the macro dictionary.
    int projGridX[4] = {0, 0, 0, 0};
    int packedGridX = 0;
    int maxGridX = 0;
    for (int p = 0; p < numProj; ++p) {
        projGridX[p] = UP_DIV(projDepthQuad[p], quadsPerTG);
        packedGridX += projGridX[p];
        maxGridX = ALIMAX(maxGridX, projGridX[p]);
    }
    // Packing only differs from the rectangular grid when the members differ in
    // outputChannel; equal-sized groups (dense q/k/v) keep the simpler form.
    // MNN_METAL_QKV_PACKED_GRID forces either shape (both are correct here).
    const int pgEnv = MetalEnv::get().qkvPackedGrid;
    mQKVPackedGrid = pgEnv > 0 ? true : (pgEnv < 0 ? false : (packedGridX < numProj * maxGridX));
    if (mQKVPackedGrid) {
        keys.emplace_back("QKV_PACKED_GRID");
    }
    // Lay the group's member output tensors into one allocation, so the fused
    // body writes through the single bound pointer at buffer(1) plus a
    // per-projection offset, instead of selecting one of N bound output pointers
    // in a branch. Measured as a win on the post-cache-cliff shapes; the aliasing
    // itself is done below, once the pipeline is known to exist.
    const bool mergedOut = MetalEnv::get().qkvMergedOut >= 0;
    if (mergedOut) {
        keys.emplace_back("QKV_MERGED_OUT");
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
        if (dualStream) {
            [dic setValue:@"1" forKey:@"GEMV_2OCQUAD_PER_SG"];
        }
        if (use4sg) {
            [dic setValue:@"4" forKey:@"GEMV_QUADS_PER_TG"];
        } else if (mQKVSplitK) {
            [dic setValue:@(quadsPerTG).stringValue forKey:@"GEMV_QUADS_PER_TG"];
            [dic setValue:@"2" forKey:@"GEMV_SPLIT_K"];
            if (qkvSkLanesPerBlock > 0) {
                [dic setValue:@(qkvSkLanesPerBlock).stringValue forKey:@"GEMV_LANES_PER_BLOCK"];
            }
        }
        if (w16) {
            [dic setValue:@"1" forKey:@"GEMV_QBLOCK_W16"];
            [dic setValue:@(mQ4W16QuadsPerBlock).stringValue forKey:@"GEMV_QBLOCK_W16_QUADS_PER_BLOCK"];
            [dic setValue:@(w16LanesPerBlock).stringValue forKey:@"GEMV_QBLOCK_W16_LANES_PER_BLOCK"];
        }
        if (mQKVPackedGrid) {
            [dic setValue:@"1" forKey:@"QKV_PACKED_GRID"];
        }
        if (mergedOut) {
            [dic setValue:@"1" forKey:@"QKV_MERGED_OUT"];
        }
        option.preprocessorMacros = dic;

        std::string sgrWqStr = std::string(gBasicConvPrefix) + gConv1x1WqSgReduce;
        mQKVFusedPipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", option);
        rt->insertPipeline(keys, mQKVFusedPipeline);
    }
    if (nil == mQKVFusedPipeline) {
        mQKVSplitK = false;
        mQKVSplitKWide = false;
        mUse4sg = false;
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

    // The fused dispatch writes k/v outputs at the leader's (earlier) position.
    // Ops between the leader and the k/v consumers (q/k-norm Cast/Raster, RoPE,
    // attention in-execution scratch) may share the followers' dynamic-pool
    // regions — their lifetimes don't overlap in the UNfused schedule — and
    // would clobber the early writes (observed as decode garbage once KV growth
    // reshuffled the pool). Both forms below move the followers' outputs into
    // static memory, which the dynamic pool can never alias; consumers bind
    // tensor addresses at encode time and follow automatically. Called after the
    // allocator's compute(), so the assignment sticks. The few KB of static
    // memory per group are not reclaimed on later resizes (decode module resizes
    // once; bounded waste).
    const Tensor* projOutputs[4] = {selfOutput, peerKOutput, peerVOutput, peerWOutput};
    size_t mergedOutOff[4] = {0, 0, 0, 0};
    mQKVMergedOut = false;
    bool rehomed = false;
    if (mergedOut) {
        // One holder owns the whole span and the members alias into it, so the
        // shader reaches every projection from buffer(1). getTensorSizeInBytes is
        // a multiple of 16, which keeps each section start aligned for the ftype4
        // the shader indexes with, in both the fp16 and fp32 builds.
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
        if (nullptr != mQKVMergedOutHolder) {
            for (int p = 0; p < numProj; ++p) {
                MetalBackend::aliasTensor(projOutputs[p], mQKVMergedOutHolder.get(), mergedOutOff[p]);
            }
            mQKVMergedOut = true;
            rehomed = true;
        }
        // No fallback to the per-member form on failure: the pipeline above was
        // already compiled with QKV_MERGED_OUT and would write every projection
        // at the leader's base. Roll the whole fusion back instead.
    } else {
        rehomed = backend->onAcquireBuffer(peerKOutput, Backend::STATIC) &&
                  backend->onAcquireBuffer(peerVOutput, Backend::STATIC);
        if (rehomed && peerW != nullptr) {
            rehomed = backend->onAcquireBuffer(peerWOutput, Backend::STATIC);
        }
    }
    if (!rehomed) {
        mIsQKVLeader = false;
        mQKVMergedOut = false;
        mQKVPackedGrid = false;
        mQKVSplitK = false;
        mQKVSplitKWide = false;
        mUse4sg = false;
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

    // Followers' per-projection scaleCoef + outputDepthQuad (leader's come from cst),
    // plus the packed grid's per-projection bases.
    // Allocated once and rewritten in place, like mConstBuffer (see onResize).
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
    // seg[6..8]: base of projections 1..3 in the packed grid, so the shader can
    // recover (projection, local x) from the flat index.
    seg[6] = (float)projGridX[0];
    seg[7] = (float)(projGridX[0] + projGridX[1]);
    seg[8] = (float)(projGridX[0] + projGridX[1] + projGridX[2]);

    // Merge q/k/v weights + dequant scales into one contiguous buffer: otherwise
    // the dispatch streams three weight and three scale allocations, and the
    // concurrent DRAM streams cost ~15% of weight throughput (0.0221 vs 0.0188
    // ms/layer, 0.6B decode qkv, min of 4 paired runs on M5). W4 only -- W2/W3/W8
    // use different bytes-per-cell layouts. Each tensor's full logical byte size
    // is copied (the shader's addressing is unchanged from the separate bindings;
    // the scale layout in particular carries a quant header and padding), with
    // section starts 16B-aligned. Allocated once and rewritten in place, so a
    // recording that survives a resize keeps binding a live buffer.
    // 4-member groups are excluded by default: measured neutral on the
    // Qwen3.5 4-wide shapes (wide W16 body streams one dominant member anyway),
    // and the extra copy is memory the 3-member win does not justify there.
    // MNN_METAL_QKV_MERGE overrides both the 3-member default and that exclusion.
    const int mergeEnv = MetalEnv::get().qkvMerge;
    const bool doMerge = mDequantBits == 4 && mergeEnv >= 0 && (peerW == nullptr || mergeEnv > 0);
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

    // seg[9..12]: where each projection's output sits inside the merged output
    // allocation, in the ftype4 units the shader's `out` is typed with. Slot 9 is
    // the leader's and is always 0: buffer(1) is bound at the leader's own start,
    // which the layout above put at the base of the allocation.
    for (int p = 0; p < 4; ++p) {
        seg[9 + p] = 0.0f;
    }
    if (mQKVMergedOut) {
        const size_t outElem = backend->useFp16InsteadFp32() ? 8 : 16;
        for (int p = 0; p < numProj; ++p) {
            seg[9 + p] = (float)(mergedOutOff[p] / outElem);
        }
    }

    // Fused pipeline is 2sg-kernel based, so the threadgroup is sized from the
    // shader's tg_simds rather than the plain pipeline's (which may be a split-K
    // g8 with 128 threads).
    if (mQKVPackedGrid) {
        // x runs over the projections' threadgroup ranges laid end to end, so a
        // group whose members differ in outputChannel no longer launches
        // maxGridX threadgroups for each of them. This avoids the rectangular
        // grid launching threadgroups that only early-return on the smaller
        // projections.
        mThreads.first = MTLSizeMake((NSUInteger)packedGridX, mThreads.first.height, 1);
    } else {
        // grid.x covers the largest projection; z selects the projection.
        // Out-of-range simdgroups on the smaller projections early-return in the
        // shader.
        mThreads.first = MTLSizeMake((NSUInteger)maxGridX, mThreads.first.height, numProj);
    }
    const int tgSimds = dualStream ? 2 : quadsPerTG * (mQKVSplitK ? 2 : 1);
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
    // Allocated once and rewritten in place, like mConstBuffer (see onResize).
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
    if (mIsGateUpLeader) {
        keys.emplace_back("GATE_UP_FUSED");
        // setupGateUpFusion already shaped the grid for the SwiGLU epilogue; the
        // LN variant of the same kernel has to be compiled to match it.
        if (mGateUpSilu) {
            keys.emplace_back("GATE_UP_SILU");
        }
    }
    if (mIsQKVLeader) {
        keys.emplace_back("QKV_FUSED");
        if (mQKVPeerW != nullptr) {
            keys.emplace_back("QKV_FUSED_P4");
        }
        // setupQKVFusion already fixed the grid shape; the LN variant of the
        // same kernel has to be compiled to match it.
        if (mQKVPackedGrid) {
            keys.emplace_back("QKV_PACKED_GRID");
        }
        // The member outputs are already aliased into one allocation, so this
        // variant can use the same single-pointer store. Unlike the grid shape
        // this one is not a correctness constraint -- out_k/out_v/out_w are bound
        // at the aliased tensors' own offsets, so the branch-selected form still
        // lands in the right place -- but it is where the win is.
        if (mQKVMergedOut) {
            keys.emplace_back("QKV_MERGED_OUT");
        }
    }
    keys.emplace_back("LN_FUSED");
    const bool dualStream = mIsQKVLeader ? mQKVDualStream : !backend->isSupportTensorApi();
    // setupQKVFusion already committed to the grid shape; the wide split keeps
    // its 4 quads/TG but spends 8 simdgroups on them where the narrow split
    // uses 4 over 2, and the sq_sum sweep spreads over all of them.
    const bool qkvSplitK = mIsQKVLeader && mQKVSplitK;
    const bool skWide = mIsQKVLeader && mQKVSplitKWide;
    // Only the wide split re-reads the input often enough (4 quad-dots per
    // threadgroup) to pay for the staging barrier; the gate/up
    // GEMV_2OCQUAD_PER_SG shape re-reads it only twice, so it has its own env
    // gate (gateUpLnStage) and is off by default.
    const int lnStageQuads = UP_DIV(hiddenInput->channel(), 4);
    // Two separate budgets. lnStageFits is the threadgroup memory ceiling: the
    // staged vector is ftype4 (half4 in fp16 mode, float4 otherwise) so *16 is
    // the float4 worst case, and it also bounds a forced-on override.
    // lnStageWorth is narrower because staging stops being a reliable win once
    // hidden grows. Note the cost/benefit ratio is NOT shape-dependent in the
    // obvious way: with the packed grid every threadgroup produces exactly
    // quadsPerTG output quads, so staging always costs
    // lnStageQuads/quadsPerTG writes per output quad no matter how lopsided the
    // fused members are. Empirically the sign at hidden=2048 just varies by
    // shape, and the driver is not identified (M5 base, Q4 g64, tight
    // alternating pairs, staging on vs off):
    //   Qwen3.5-2B linear_in [6144,2048,16,16]  -1.8%  (4/4 pairs, on is worse)
    //   Qwen3.5-2B qkv       [2048,512,512,2048] -0.5% (4/4 pairs, on is worse)
    //   Qwen3-1.7B qkv       [2048,1024,1024]    +0.4% (5/6 pairs, on is better)
    // So auto only keeps staging in the hidden <= 1024 range where it was
    // originally tuned and still measures consistently positive; hidden=2048
    // gives up 1.7B's 0.4% to take linear_in's 1.8%. MNN_METAL_LN_STAGE is the
    // tri-state override for A/B-ing the rest.
    const bool lnStageFits  = lnStageQuads * 16 <= 8192;
    const bool lnStageWorth = lnStageQuads * 16 <= 4096;
    const int lnStageOverride = MetalEnv::get().lnStage;
    const bool lnStage = lnStageOverride >= 0 && lnStageFits &&
                         (skWide || (mIsGateUpLeader && MetalEnv::get().gateUpLnStage)) &&
                         (lnStageOverride > 0 || lnStageWorth);
    if (lnStage) {
        keys.emplace_back("LN_STAGE_" + std::to_string(lnStageQuads));
        if (MetalEnv::get().lnStageFp32) {
            keys.emplace_back("LN_STAGE_FP32");
        }
    }
    // The SILU-folded gate/up pipeline is GEMV_2OCQUAD_PER_SG even on tensor-API devices
    // (see setupGateUpFusion); its LN variant has to be compiled to match.
    if (dualStream || mGateUpSilu) {
        keys.emplace_back("GEMV_2OCQUAD_PER_SG");
    }
    // setupGateUpFusion / setupQKVFusion already committed to the 4-simdgroup
    // non-split-K shape; the LN variant of the same kernel has to be compiled
    // to match.
    if (mUse4sg) {
        keys.emplace_back("tg4");
    }
    // setupGateUpFusion already sized the grid and threadgroup for the K split;
    // the LN variant of the same kernel has to be compiled to match.
    if (mGateUpDualStreamSplitK > 0) {
        keys.emplace_back("GEMV_2OCQUAD_PER_SG_SPLIT_K_" + std::to_string(mGateUpDualStreamSplitK));
    }
    // setupQKVFusion already sized the grid and threadgroup for the K split; the
    // LN variant of the same kernel has to be compiled to match.
    const int qkvSkLanesPerBlock =
        qkvSplitK ? gemvLanesPerBlock(((Param*)mConstBuffer.contents)->inputDepthQuad, mBlockCount, true) : 0;
    if (qkvSplitK) {
        keys.emplace_back("qtg" + std::to_string(skWide ? 4 : 2));
        keys.emplace_back("sk2");
        if (qkvSkLanesPerBlock > 0) {
            keys.emplace_back("lpb" + std::to_string(qkvSkLanesPerBlock));
        }
    }
    // Generalized Q4 W16 on the LN-folded fused GEMV. Every fused member must
    // carry the same compile-time quant-block shape.
    bool w16 = mQ4W16QuadsPerBlock > 0 && mDequantBits == 4;
    if (mIsGateUpLeader &&
        (mGateUpPeer == nullptr || mGateUpPeer->mQ4W16QuadsPerBlock != mQ4W16QuadsPerBlock)) {
        w16 = false;
    }
    if (mIsQKVLeader) {
        if (mQKVPeerK == nullptr || mQKVPeerK->mQ4W16QuadsPerBlock != mQ4W16QuadsPerBlock ||
            mQKVPeerV == nullptr || mQKVPeerV->mQ4W16QuadsPerBlock != mQ4W16QuadsPerBlock ||
            (mQKVPeerW != nullptr && mQKVPeerW->mQ4W16QuadsPerBlock != mQ4W16QuadsPerBlock)) {
            w16 = false;
        }
    }
    // Match setupGateUpFusion's lane pick so the compiled
    // GEMV_QBLOCK_W16_LANES_PER_BLOCK equals the dual-stream body's runtime
    // lane widening (both follow blockCount / split factor).
    const int w16BlocksPerSimdgroup =
        mIsGateUpLeader && mGateUpDualStreamSplitK > 0 ? mBlockCount / mGateUpDualStreamSplitK
        : qkvSplitK ? mBlockCount / 2 : mBlockCount;
    const int w16DsLanes = w16 && mIsGateUpLeader && mGateUpSilu && mGateUpDualStreamSplitK > 0 &&
                           !MetalEnv::get().gemvW16DsDisabled
                               ? gateUpW16DsLanes(mQ4W16QuadsPerBlock,
                                                  mBlockCount / mGateUpDualStreamSplitK)
                               : 0;
    const bool w16Ds = w16DsLanes > 0;
    const int w16LanesPerBlock =
        w16Ds ? w16DsLanes
              : (w16 ? chooseQ4W16LanesPerBlock(w16BlocksPerSimdgroup, mQ4W16QuadsPerBlock / 2,
                                                skWide, backend->isSupportTensorApi())
                     : 0);
    if (w16) {
        keys.emplace_back("GEMV_QBLOCK_W16");
        keys.emplace_back("W16_QUADS_PER_BLOCK_" + std::to_string(mQ4W16QuadsPerBlock));
        keys.emplace_back("W16_LANES_PER_BLOCK_" + std::to_string(w16LanesPerBlock));
        if (w16Ds) {
            keys.emplace_back("GEMV_QBLOCK_W16_DS");
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
            if (mQKVMergedOut) {
                [dic setValue:@"1" forKey:@"QKV_MERGED_OUT"];
            }
        }
        [dic setValue:@"1" forKey:@"LN_FUSED"];
        if (lnStage) {
            [dic setValue:@"1" forKey:@"LN_STAGE"];
            [dic setValue:@(lnStageQuads).stringValue forKey:@"LN_STAGE_QUADS"];
            if (MetalEnv::get().lnStageFp32) {
                [dic setValue:@"1" forKey:@"LN_STAGE_FP32"];
            }
        }
        if (mUse4sg) {
            [dic setValue:@"4" forKey:@"GEMV_QUADS_PER_TG"];
        } else if (dualStream || mGateUpSilu) {
            [dic setValue:@"1" forKey:@"GEMV_2OCQUAD_PER_SG"];
        }
        if (mGateUpDualStreamSplitK > 0) {
            [dic setValue:@(mGateUpDualStreamSplitK).stringValue forKey:@"GEMV_2OCQUAD_PER_SG_SPLIT_K"];
        }
        if (qkvSplitK) {
            [dic setValue:(skWide ? @"4" : @"2") forKey:@"GEMV_QUADS_PER_TG"];
            [dic setValue:@"2" forKey:@"GEMV_SPLIT_K"];
            if (qkvSkLanesPerBlock > 0) {
                [dic setValue:@(qkvSkLanesPerBlock).stringValue forKey:@"GEMV_LANES_PER_BLOCK"];
            }
        }
        if (w16) {
            [dic setValue:@"1" forKey:@"GEMV_QBLOCK_W16"];
            [dic setValue:@(mQ4W16QuadsPerBlock).stringValue forKey:@"GEMV_QBLOCK_W16_QUADS_PER_BLOCK"];
            [dic setValue:@(w16LanesPerBlock).stringValue forKey:@"GEMV_QBLOCK_W16_LANES_PER_BLOCK"];
            if (w16Ds) {
                [dic setValue:@"1" forKey:@"GEMV_QBLOCK_W16_DS"];
            }
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
    // be split-K g8's 128-thread group). GEMV_2OCQUAD_PER_SG_SPLIT_K wants 2 * factor, and the
    // QKV split wants 8 simdgroups when wide, 4 otherwise.
    NSUInteger lnThreads = 64;
    if (mGateUpDualStreamSplitK > 0) {
        lnThreads = 64 * mGateUpDualStreamSplitK;
    } else if (mUse4sg) {
        lnThreads = 128;
    } else if (skWide) {
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

    mQ4W16QuadsPerBlock = 0;

    // The fusion state resets below deliberately keep the const buffers
    // (mGateUpSegBuffer / mQKVSegBuffer / mLNEpsBuffer, like mConstBuffer): a
    // recording must keep seeing the same buffer across resizes, see the note
    // at the mConstBuffer allocation. Only the flags are cleared, and none of
    // them is decided by a buffer being nil -- the encode path binds these
    // solely under mIsGateUpLeader / mIsQKVLeader / mHasLNFusion, and whichever
    // setup re-establishes the fusion rewrites the contents first.

    // Reset Gate/Up fusion state on each resize
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
    mUse4sg = false;


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
    int blockCount = 1;
    if (dequantScale) {
        int bytes = sizeof(float);
        if(backend->useFp16InsteadFp32()) {
            bytes = sizeof(__fp16);
        }
        blockCount = (int)(dequantScale->usize() / bytes / oc_4 / 2 / 4);
    }
    // Const buffer: allocate once, then rewrite in place on every later resize.
    // Handing out a fresh buffer per resize would go unnoticed by encode-replay:
    // this is bound raw (no MetalBackend::setTensor annotation), so neither
    // metalReplayValidate nor _replayHashIO can see the swap, and a live
    // recording would keep emitting the PREVIOUS resize's Param. Today decode
    // recomputes identical values so nothing breaks, but any token-dependent
    // field added to Param would silently read one resize stale.
    // Dropping the recording instead is not an option: LLM decode resizes every
    // token, so it would never re-arm. Rewriting in place is safe because
    // MetalBackend::onResizeBegin fenced our own in-flight GPU work, and every
    // Param field is assigned below (no stale field survives the reuse).
    if (nil == mConstBuffer) {
        mConstBuffer = backend->getConstBuffer(sizeof(Param));
    }
    auto param = (Param *)mConstBuffer.contents;
    param->inputSize = is;
    param->inputDepthQuad = ic_4;
    param->outputWidth = ow;
    param->outputHeight = oh;
    param->outputSize = os;
    param->outputDepthQuad = oc_4;
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
    // On tensor-API devices route to outer-dequant + tensor API GEMM
    // (conv1x1_gemm_32x64_split_k_sg with USE_METAL_TENSOR_OPS). The in-shader
    // Q4 sg_matrix kernels (conv1x1_gemm_32x16_wquant_sg / 16x32_wquant_sg /
    // 32x64_wquant_split_k_sg) are pure SIMD-matrix -- no tensor API -- and
    // regress badly at prefill scale, so they are only an explicit A/B lever.
    // The residual cost lives inside the K loop: the activation smem store is
    // bank-conflicted -- its address (4*ml + r) * 8 + kl in ftype4 units has a
    // bank index independent of ml, so all 16 ml lanes of a simdgroup hit the
    // same banks, and FQ4_SMEM_PAD pads the staged row stride to break it. The
    // packed-weight DRAM read plus nibble unpack is the other item: FQ4_UNORM
    // replaces the per-uint shift/mask/int-to-float chain with two masks plus
    // unpack_unorm4x8, folding the /255 into the per-lane scale.
    //
    if (backend->isSupportTensorApi() && area > 1 && (mDequantBits == 4 || mDequantBits == 8)) {
        // On tensor-API devices (M5+) always force outer-dequant + tensor API,
        // unless the env explicitly asks for the classic simdgroup-matrix
        // in-shader-dequant kernel (A/B lever; the tensor path normally wins).
        dequantInShader = MetalEnv::get().prefillInshaderDequant == 1;
    }
    // On non-tensor-API devices (M4 and below), choose in-shader vs outer-dequant
    // based on weight size AND prompt length. In-shader dequant re-unpacks the Q4
    // weights once per M-tile (unpack count ~ area/32), so it only wins for large
    // weights at short area; outer-dequant pays a fixed double-pass instead.
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
        //printf("inner dequant MNK: %d %d %d %d\n", area, oc, ic, blockCount);

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
        // kernels. The C4 quads per quant block become a compile-time constant
        // (8/16/32/64), enabling 16-byte weight reads without tail checks.
        // The g16 lm_head pipeline deliberately has no W16 body: it is the one
        // consumer where the specialization measures as a loss.
        // Hoisted out of the branch below so the routing can pick the mode once.
        const int lmheadSplitKOverride = MetalEnv::get().lmheadSplitK;
        // Four-model lm_head calibration: g16 split-K wins only for the wide-vocab,
        // short-K shape; the g4 route wins on the other measured shapes.
        const int lmheadSplitKMode =
            lmheadSplitKOverride >= 0 ? lmheadSplitKOverride
                                      : (area != 1 || (blockCount < 32 && oc > 200000) ? 2 : 1);
        const bool g16LmHeadRoute = oc > 16384 && oc_4 % 2 == 0 &&
                                    !(lmheadSplitKMode == 1 && (oc % 8 == 0) && (blockCount % 2 == 0));
        const bool g16SplitKRoute = g16LmHeadRoute && area == 1 && lmheadSplitKMode == 2 &&
                                    oc % 16 == 0 && blockCount % 2 == 0;
        int q4W16QuadsPerBlock = 0;
        if (area == 1 && mDequantBits == 4 && blockCount > 0 && ic_4 % blockCount == 0 &&
            !MetalEnv::get().gemvW16Disabled) {
            const int candidate = ic_4 / blockCount;
            if (isQ4W16QuadsPerBlock(candidate)) {
                q4W16QuadsPerBlock = candidate;
            }
        }
        const bool q4W16 = q4W16QuadsPerBlock > 0;
        // Non-split-K pipelines: the simdgroup owns every block of the row.
        const int q4W16LanesPerBlock =
            q4W16 ? chooseQ4W16LanesPerBlock(blockCount, q4W16QuadsPerBlock / 2, false,
                                             backend->isSupportTensorApi())
                  : 0;
        // Record the layout so a fusion leader can extend W16 onto the fused
        // decode-GEMV pipeline built later in setup{QKV,GateUp,LN}Fusion.
        mQ4W16QuadsPerBlock = q4W16QuadsPerBlock;
        if (q4W16) {
            [dic setValue:@"1" forKey:@"GEMV_QBLOCK_W16"];
            [dic setValue:@(q4W16QuadsPerBlock).stringValue forKey:@"GEMV_QBLOCK_W16_QUADS_PER_BLOCK"];
            [dic setValue:@(q4W16LanesPerBlock).stringValue forKey:@"GEMV_QBLOCK_W16_LANES_PER_BLOCK"];
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
                baseKeys.emplace_back("W16_QUADS_PER_BLOCK_" + std::to_string(q4W16QuadsPerBlock));
                baseKeys.emplace_back("W16_LANES_PER_BLOCK_" + std::to_string(q4W16LanesPerBlock));
            }
            // W_QUANT_2/3 on non-simdgroup-matrix devices: the outer-dequant GEMM
            // (sg-matrix based) and the g1z4 fallback below are both unusable, so
            // g8/g16 must cover all areas in-shader, not just area <= short_seq.
            const bool w23NoMatrix = (mDequantBits == 2 || mDequantBits == 3) && !rt->supportSimdGroupMatrix();
            if(rt->supportSimdGroupReduce() && (area <= short_seq || w23NoMatrix)) {
                baseKeys.emplace_back("conv1x1_wquant_sg_reduce");

                std::string sgrWqStr = basicShaderPrefix + sgrWqShader;
                // memory bound not so seriously, can add more thread to reduce computation in each thread
                float ratio = 1.0 * ic_4 / 2048.0 * oc / 2048.0;
                bool heavyMemory = ratio > 1.0;
                // g4mN kernels now have true W_QUANT_2/3 branches, but instantiations
                // only exist up to g4m16 (area <= 16 direct, or <= 32 with the piece=2
                // halving below). Larger areas — reachable only via w23NoMatrix — must
                // fall through to the all-area g8 kernel.
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
                } else if(g16LmHeadRoute) {
                    // lm_head path. Baseline g16 = 2 simdgroups per TG,
                    // threadgroup size 64, each TG covers 16 OC (2 SG x 8 OC/SG).
                    // Variants explored and retired (see skills/metal-optimize):
                    // 4SG (halved grid) — e2e neutral with 7x worse stddev on M5;
                    // G16_OC4 (4 oc_4 rows/SG) — kernel -4.8% on M5 but e2e neutral.
                    if (g16SplitKRoute) {
                        // G16_SPLIT_K: split-K inside the g16 kernel. 4 simdgroups
                        // per TG; each SG pair halves one simdgroup's quant-block
                        // range and combines via threadgroup memory. The grid is
                        // exact (oc % 16 == 0), which the kernel's barrier relies on.
                        auto keys = baseKeys;
                        keys.emplace_back("conv1x1_gemv_g16_wquant_sg");
                        keys.emplace_back("G16_SPLIT_K");
                        auto pipeline = rt->findPipeline(keys);
                        if (nil == pipeline) {
                            NSMutableDictionary *skDic = [dic mutableCopy];
                            [skDic setValue:@"1" forKey:@"G16_SPLIT_K"];
                            MTLCompileOptions *skOption = [[MTLCompileOptions alloc] init];
                            skOption.preprocessorMacros = skDic;
                            pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g16_wquant_sg", skOption);
                            rt->insertPipeline(keys, pipeline);
                        }
                        mPipeline = pipeline;
                        CONV1X1_SET_TAG("g16splitk_gemv_g16_wquant_sg");
                        mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 16), area, 1),
                                                  MTLSizeMake(128, 1, 1));
                        return NO_ERROR;
                    }
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
                    // Decode GEMV. The 2sg kernel is latency-limited on small
                    // projections: one simdgroup streams a whole row, so small
                    // matrices fall far short of DRAM peak while the large lm_head
                    // (g16 path) saturates it. Splitting K keeps the same
                    // pre-scaling inner loop but runs 4 simdgroups per
                    // threadgroup -- two K-halves per row combined via threadgroup
                    // memory -- doubling in-flight reads per row.
                    // (Routing to the g8 kernel instead was tried first: its
                    // nibble-unpack inner loop is slower and lost 5% e2e.)
                    // Split-K only buys in-flight reads while a row's quant
                    // blocks cannot keep all 32 lanes of a simdgroup busy. Once
                    // blockCount reaches 32, halving each simdgroup's K range
                    // idles half the lanes and the threadgroup reduce is pure
                    // overhead. M5, 3/3 paired passes on speed/Qwen3DecodeGemv:
                    // o_proj (ic 2048, 32 blocks) 0.412..0.417 -> 0.388..0.397
                    // ms (-5.5%), down_proj (ic 3072, 48 blocks) 0.577..0.581 ->
                    // 0.549..0.567 (-3.0%).
                    // The ic=1024 projections split-K was originally tuned on
                    // (16 blocks) stay on the split path, where it is neutral.
                    // From 96 blocks up the picture flips back with the post
                    // tie-break-fix lanes: the split-K residual is positive
                    // (+2..13% of weight throughput at oc 2560/4096) and e2e
                    // decode gains on Qwen3.5-2B (down_proj b96); the owned=76
                    // half of b152 stays on L=2 via the chooser's near-tie rule,
                    // which keeps the split neutral there.
                    const bool splitkUsable =
                        (oc % 8 == 0) && (blockCount % 2 == 0) &&
                        (blockCount < 32 || blockCount >= 96);
                    if (splitkUsable) {
                        const int lanesPerBlock = gemvLanesPerBlock(ic_4, blockCount, true);
                        auto keys = baseKeys;
                        keys.emplace_back("conv1x1_gemv_g4m1_2sg_wquant_sg");
                        keys.emplace_back("qtg2");
                        keys.emplace_back("sk2");
                        // Split-K halves the blocks a simdgroup owns, so the
                        // lane split is re-chosen for that pipeline.
                        const int skLanesPerBlock =
                            q4W16 ? chooseQ4W16LanesPerBlock(blockCount / 2, q4W16QuadsPerBlock / 2, false,
                                                             backend->isSupportTensorApi())
                                  : 0;
                        if (lanesPerBlock > 0) {
                            keys.emplace_back("lpb" + std::to_string(lanesPerBlock));
                        }
                        if (q4W16) {
                            keys.emplace_back("W16_SK_LANES_PER_BLOCK_" + std::to_string(skLanesPerBlock));
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
                            if (q4W16) {
                                [skDic setValue:@(skLanesPerBlock).stringValue
                                         forKey:@"GEMV_QBLOCK_W16_LANES_PER_BLOCK"];
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
                        // blockCount >= 32: the 2-simdgroup kernel (64 threads)
                        // keeps all 32 lanes busy but only 2 simdgroups per TG.
                        // A 4-simdgroup variant (128 threads) doubles the
                        // in-flight weight reads per TG without the split-K
                        // threadgroup reduction barrier.
                        const bool use4sg = (blockCount >= 32 && oc % 16 == 0);
                        auto keys = baseKeys;
                        keys.emplace_back("conv1x1_gemv_g4m1_2sg_wquant_sg");
                        if (use4sg) {
                            keys.emplace_back("tg4");
                        }
                        auto pipeline = rt->findPipeline(keys);
                        if (nil == pipeline) {
                            if (use4sg) {
                                NSMutableDictionary *dic4 = [dic mutableCopy];
                                [dic4 setValue:@"4" forKey:@"GEMV_QUADS_PER_TG"];
                                MTLCompileOptions *opt4 = [[MTLCompileOptions alloc] init];
                                opt4.preprocessorMacros = dic4;
                                pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", opt4);
                            } else {
                                pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", option);
                            }
                            rt->insertPipeline(keys, pipeline);
                        }
                        mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                        if (use4sg) {
                            mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 16), 1, 1), MTLSizeMake(128, 1, 1));
                        } else {
                            mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 8), 1, 1), MTLSizeMake(64, 1, 1));
                        }
                    }
                    // Fusion leaders (gate/up, qkv, LN) build their own 2sg-based
                    // pipelines and force a 64-thread dispatch in their setup.
                    mIs2sgDecode = true;
                } else {
                    // All-area fallback: w23NoMatrix shapes whose area exceeds the
                    // g4mN instantiation range (g4m2..g4m16) land here.
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
            // but wins once weights exceed L2. The 4M-element threshold (~ fp16
            // 8MB) matches the in-shader dequant gate.
            auto specMeta = (KVMeta *)backend->getMetaPtr();
            const bool specBlock = (specMeta != nullptr && specMeta->spec_block > 0);
            const bool tensorApiFusedQuant = !MetalEnv::get().w4w8OuterDequantGemm &&
                                 (mDequantBits == 2 || mDequantBits == 3 ||
                                  mDequantBits == 4 || mDequantBits == 8) &&
                                 backend->isSupportTensorApi() &&
                                 (area >= 64 || specBlock) &&
                                 (mDequantBits != 3 ||
                                  (int64_t)oc * ic >= (int64_t)4 * 1024 * 1024);

            // M_TILE=64 variant (requires tensor API). The M=64 tile halves
            // grid.x and DRAM weight traffic across TGs, but doubles per-thread
            // register/smem pressure. It amortises well once the batch is large
            // enough to spread the DRAM savings over many rows; on short prefill
            // the extra pressure costs occupancy without enough weight-reuse to
            // compensate. The auto gate turns it on whenever the fused-quant
            // tensor path is taken with bits==4 and area>=64.
            //
            // MNN_METAL_FUSED_Q4_M64 tri-state overrides the default:
            //   unset      = default on (M64 whenever tensorApiFusedQuant && bits==4 && area>=64),
            //   0          = force off (always M32),
            //   1          = force on (identical to default).
            {
                const int fusedQ4M64Env = MetalEnv::get().fusedQ4M64;
                // envTriState: unset = 0 (auto), "0" = -1 (off), "1" = 1 (on).
                if (fusedQ4M64Env < 0) {
                    mFusedQ4M64 = false;
                } else {
                    mFusedQ4M64 = tensorApiFusedQuant && mDequantBits == 4 && area >= 64;
                }
            }

            mTensorApiFusedQuant = tensorApiFusedQuant;

            if (!tensorApiFusedQuant) {
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
            bool fusedQ4SmemPad = false;
            bool fusedQ4Unorm = false;
            bool fusedQ4DoubleBuf = false;
            // K-split x4 recovers TG-starved speculative-verify shapes; TG-rich ones
            // (lm_head) stay out. MNN_METAL_FUSED_Q4_KSPLIT: unset auto, =0 off, =1 on.
            const auto& metalEnv = MetalEnv::get();
            const bool fusedQ4SpecBlock = mTensorApiFusedQuant && mDequantPipeline == nil &&
                                      mDequantBits == 4 && !mFusedQ4M64 && specBlock;
            mUseFusedKsplit = false;
            mPrefillSiluOn = false;
            mPrefillDualOn = false;
            if (fusedQ4SpecBlock && metalEnv.fusedQ4Ksplit >= 0) {
                mUseFusedKsplit = metalEnv.fusedQ4Ksplit == 1 ||
                                  (area <= 32 && UP_DIV(oc, 64) <= 48 && blockCount >= 4);
            }
            // M8-native tile for small-M shapes the K-split gate skips (a padded M32
            // tile runs at quarter occupancy). MNN_METAL_FUSED_Q4_M8=0 forces it off.
            const bool useFusedM8 = fusedQ4SpecBlock && !mUseFusedKsplit &&
                                    area > 1 && area <= 8 && metalEnv.fusedQ4M8 >= 0;
            if (mTensorApiFusedQuant) {
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
                    keys.emplace_back(gemmKernelName);
                    // Padded staged-operand stride, default on; set the env to 0
                    // to fall back to the unpadded stride.
                    fusedQ4SmemPad = MetalEnv::get().fusedQ4SmemPad >= 0;
                    if (fusedQ4SmemPad) {
                        keys.emplace_back("FQ4_SMEM_PAD");
                    }
                    // unpack_unorm4x8 nibble unpack, default on; env 0 restores
                    // the scalar shift/mask path.
                    fusedQ4Unorm = MetalEnv::get().fusedQ4Unorm >= 0;
                    if (fusedQ4Unorm) {
                        keys.emplace_back("FQ4_UNORM");
                    }
                    // Double-buffered operand staging, default on (A/B-verified:
                    // one barrier per K8 window instead of two, with the next
                    // window's dequant/stores overlapping the current window's
                    // tensor matmul); env 0 restores the single buffer. fp16
                    // only: at fp32 the two operand sets alone would need 40 KB
                    // of threadgroup memory with the padded stride, over the
                    // 32 KB device limit.
                    fusedQ4DoubleBuf = MetalEnv::get().fusedQ4DoubleBuf >= 0 && backend->useFp16InsteadFp32();
                    if (fusedQ4DoubleBuf) {
                        keys.emplace_back("FQ4_DOUBLE_BUF");
                    }
                    // Fold the group's MUL_SILU into this epilogue when the
                    // owner offered a gate tensor. Only the baseline m64 kernel
                    // carries the branch; the offer is otherwise declined and
                    // the owner keeps its separate MUL_SILU dispatch.
                    //
                    // Gated by the env-adjustable min area (default 128): the
                    // dropped pass is only tens of microseconds, but measured
                    // 2.9% net win on Qwen3.5-2B s=128 for the dual below.
                    mPrefillSiluOn = nullptr != mPrefillSiluGate && area >= MetalEnv::get().fusedQ4GateUpDualMinArea &&
                                     MetalEnv::get().fusedQ4SiluMul >= 0;
                    // The dual supersedes the fold: instead of reading back a
                    // materialized gate tile it keeps the gate tile in a second
                    // accumulator, so the gate tensor is never written at all.
                    // Halving N to 32 per matrix is what keeps the two
                    // accumulators at the same 32 floats/thread one N=64
                    // destination already costs.
                    mPrefillDualOn = nullptr != mPrefillDualPeer && area >= MetalEnv::get().fusedQ4GateUpDualMinArea &&
                                     MetalEnv::get().fusedQ4GateUpDual >= 0;
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
            } else if (!backend->isSupportTensorApi() &&
                       ((MetalRuntime*)backend->runtime())->preferM64Gemm() && area >= 128) {
                // sg_matrix M=64 tile, device-tiered via architecture.name
                // (M4-class Macs only, see MetalBackend.mm): halves grid.x /
                // weight DRAM traffic; fp16 weights from the outer-dequant
                // pre-pass, same bindings as the 32x64 kernel.
                gemmKernelName = "conv1x1_gemm_64x64_split_k_sg";
                keys.emplace_back("conv1x1_gemm_64x64_split_k_sg");
                sgMatrixM64 = true;
            } else {
                keys.emplace_back("conv1x1_gemm_32x64_split_k_sg");
            }

            NSMutableDictionary *dic = [baseDic mutableCopy];
            // The fused-stage K=32 tile spans 8 K4 groups.
            if (ic_4 % 8 != 0) {
                [dic setValue:@"1" forKey:@"MNN_METAL_SRC_PROTECT"];
                keys.emplace_back("MNN_METAL_SRC_PROTECT");
            }
            if(backend->isSupportTensorApi() == true) {
                [dic setValue:@"1" forKey:@"USE_METAL_TENSOR_OPS"];
                keys.emplace_back("USE_METAL_TENSOR_OPS");
                if(ic > oc && ic > 2048 && (ic / blockCount) % 64 == 0 && !mTensorApiFusedQuant) {
                    // LOOP_K64 branch only exists for conv1x1_gemm_32x64_split_k_sg.
                    // Fused-stage kernel is always K=32 tile.
                    [dic setValue:@"1" forKey:@"LOOP_K64"];
                    keys.emplace_back("LOOP_K64");
                }
            }
            // Fused-stage kernel is compiled with W_QUANT_{2,3,4,8}.
            // (kernel body is guarded by `#if defined(W_QUANT_2) || defined(W_QUANT_3)
            //  || defined(W_QUANT_4) || defined(W_QUANT_8)`).
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
            if (fusedQ4SmemPad) {
                [dic setValue:@"1" forKey:@"FQ4_SMEM_PAD"];
            }
            if (fusedQ4Unorm) {
                [dic setValue:@"1" forKey:@"FQ4_UNORM"];
            }
            if (fusedQ4DoubleBuf) {
                [dic setValue:@"1" forKey:@"FQ4_DOUBLE_BUF"];
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
                // The dual halves N per matrix, so it needs twice the N tiles.
                const int nTile = mPrefillDualOn ? 32 : 64;
                mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, mTile), UP_DIV(oc, nTile), 1),
                                          MTLSizeMake(128, 1, 1));
            }
            //printf("out dequant MNK: %d %d %d %d\n", area, oc, ic, blockCount);
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
        // buffer(1): gate output (this), or the group's SwiGLU output when the
        // epilogue folds MUL_SILU in -- gate and up then never reach memory.
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
        const bool fused = (mDequantPipeline == nil) && mTensorApiFusedQuant;

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
            if (mTensorApiFusedQuant) {
                // Fused kernel bindings: buffer(3) = quantized weight,
                // buffer(4) = bias, buffer(5) = dequantScale, buffer(6) =
                // placeholder alias of mWeight (never read by the fused
                // kernel; mTempWeight is not allocated).
                MetalBackend::setTensor(mWeight.get(), encoder, 3);
                MetalBackend::setTensor(mBias.get(), encoder, 4);
                MetalBackend::setTensor(getDequantScale().get(), encoder, 5);
                MetalBackend::setTensor(mWeight.get(), encoder, 6);
                if (mPrefillDualOn) {
                    // Peer (gate) operands mirror 3/4/5 at 7/8/9; the gate's own
                    // per-tensor scale coefficient must travel with them, since
                    // cst.scaleCoef belongs to this projection.
                    MetalBackend::setTensor(mPrefillDualPeer->getWeight().get(), encoder, 7);
                    MetalBackend::setTensor(mPrefillDualPeer->getBias().get(), encoder, 8);
                    MetalBackend::setTensor(mPrefillDualPeer->getDequantScale().get(), encoder, 9);
                    [encoder setBuffer:mPrefillDualCoef offset:0 atIndex:10];
                } else if (mPrefillSiluOn) {
                    MetalBackend::setTensor(mPrefillSiluGate, encoder, 7);
                }
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
