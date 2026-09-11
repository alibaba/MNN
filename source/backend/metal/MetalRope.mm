//
//  MetalRope.mm
//  MNN
//
//  Fused RoPE (Rotary Positional Embedding) kernel for Metal backend.
//
//  Inputs:  x, cos, sin
//  Output:  same shape as x
//
//  For rotary dimension R (must be even), split x[..., 0:R] in half and
//  leave x[..., R:D] unchanged.
//  Then compute
//    q0 = even * cos[i] - odd * sin[i]
//    q1 = odd  * cos[i + ropeHalfD] + even * sin[i + ropeHalfD]
//  and concatenate [q0, q1] along the last dimension.
//

#define MNN_UNUSED(x)
#import "MNNMetalContext.h"
#import "backend/metal/MetalBackend.hpp"
#import "MetalExecution.hpp"
#import "MetalLayerNorm.hpp"
#import "MetalEnv.hpp"
#import "core/TensorUtils.hpp"
#import "core/Macro.h"
#include "MNN_generated.h"
#include <cstring>
#include <vector>

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {

static std::shared_ptr<MetalLayerNorm::Resource> makeRopeNormResource(Backend* backend, const LayerNorm* layerNorm) {
    if (nullptr == layerNorm || nullptr == layerNorm->gamma()) {
        return nullptr;
    }
    int gammaSize = layerNorm->gamma()->size();
    if (gammaSize <= 0) {
        return nullptr;
    }
    auto res = std::make_shared<MetalLayerNorm::Resource>();
    res->mGroup = layerNorm->group();
    res->mEps = layerNorm->epsilon();
    res->mAxisSize = layerNorm->axis() == nullptr ? 1 : layerNorm->axis()->size();
    res->mHasGammaBeta = true;
    res->mRMSNorm = layerNorm->useRMSNorm();
    res->mGammaSize = gammaSize;
    res->mGammaBuffer.reset(Tensor::createDevice<uint8_t>({gammaSize * (int)sizeof(float)}));
    if (!backend->onAcquireBuffer(res->mGammaBuffer.get(), Backend::STATIC)) {
        MNN_ERROR("MetalRope: alloc q/k norm gamma buffer error.\n");
        return nullptr;
    }
    auto gammaPtr = MetalBackend::getBuffer(res->mGammaBuffer.get());
    ::memcpy((uint8_t*)gammaPtr.first.contents + gammaPtr.second, layerNorm->gamma()->data(),
             gammaSize * sizeof(float));
    return res;
}

static bool validRopeC4Input(const Tensor* q, const Tensor* k, int numHead, int kvNumHead, int headDim) {
    if (q == nullptr || k == nullptr || numHead <= 0 || kvNumHead <= 0 || headDim <= 0) {
        return false;
    }
    if (TensorUtils::getDescribe(q)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 ||
        TensorUtils::getDescribe(k)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
        return false;
    }
    if (q->dimensions() != 4 || k->dimensions() != 4 || q->length(0) != k->length(0) || q->length(2) != 1 ||
        q->length(3) != 1 || k->length(2) != 1 || k->length(3) != 1) {
        return false;
    }
    return q->length(1) == numHead * headDim && k->length(1) == kvNumHead * headDim;
}

struct RopeParam {
    int outerSize;
    int workDim;
    int ropeHalfD;
    int D;
    int numHead;
    int kvnumHead;
    int fullHead;
    float qEps;
    float kEps;
};

// Metal kernel source. ftype is float / half selected by MNN_METAL_FLOAT16_STORAGE.
static const char* gMetalRopeKernelSource = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
#ifdef MNN_METAL_FLOAT16_STORAGE
typedef half ftype;
typedef half4 ftype4;
#else
typedef float ftype;
typedef float4 ftype4;
#endif

struct RopeParam {
    int outerSize;
    int workDim;
    int ropeHalfD;
    int D;
    int numHead;
    int kvnumHead;
    int fullHead;
    float qEps;
    float kEps;
};

static inline int c4Offset(int token, int channel, int seqLen) {
    return (channel / 4) * seqLen * 4 + token * 4 + (channel % 4);
}

static inline ftype loadC4(const device ftype* tensor, int token, int base, int offset, int seqLen) {
    if (seqLen == 1) {
        return tensor[base + offset];
    }
    return tensor[c4Offset(token, base + offset, seqLen)];
}

#if defined(Q_NORM) || defined(K_NORM)
kernel void rope_kernel(
                        const device ftype* q           [[ buffer(0) ]],
                        const device ftype* k           [[ buffer(1) ]],
                        const device ftype* cos         [[ buffer(2) ]],
                        const device ftype* sin         [[ buffer(3) ]],
                        device ftype* qo                 [[ buffer(4) ]],
                        device ftype* ko                 [[ buffer(5) ]],
                        constant RopeParam& p           [[ buffer(6) ]],
#ifdef Q_NORM
                        const device float* qGamma      [[ buffer(7) ]],
#endif
#ifdef K_NORM
                        const device float* kGamma      [[ buffer(8) ]],
#endif
#ifdef USE_SG
                        uint3 gid                      [[ threadgroup_position_in_grid]],
                        uint tiisg                     [[ thread_index_in_simdgroup]],
                        uint sgitg                     [[ simdgroup_index_in_threadgroup ]]
#else
                        uint3 gid                      [[ thread_position_in_grid]]
#endif
) {
#ifdef USE_SG
    uint actual_z = gid.z * 2 + sgitg;
    if (gid.y >= (uint)p.outerSize || actual_z >= p.fullHead) {
        return;
    }
    int step = 32;
    int start = tiisg;
#else
    uint actual_z = gid.z;
    if (gid.x >= 1 || gid.y >= (uint)p.outerSize || actual_z >= p.fullHead) {
        return;
    }
    int step = 1;
    int start = 0;
#endif

    bool isQ = true;
    const device ftype* xTensor = q;
    int xBase = actual_z * p.D;
    int xSeq = p.outerSize;
    device ftype* yTensor = qo;
    int yBase = gid.y * p.numHead * p.D + actual_z * p.D;
    if (actual_z >= p.numHead) {
        xTensor = k;
        xBase = (actual_z - p.numHead) * p.D;
        yTensor = ko;
        yBase = gid.y * p.kvnumHead * p.D + (actual_z - p.numHead) * p.D;
        isQ = false;
    }

    // actual_z is uniform across the simdgroup, so isQ is too: the gamma table and
    // the eps are one uniform choice per simdgroup rather than a per-lane branch.
    const device float* gamma = nullptr;
    float eps = 0.0f;
#ifdef Q_NORM
    if (isQ) {
        gamma = qGamma;
        eps = p.qEps;
    }
#endif
#ifdef K_NORM
    if (!isQ) {
        gamma = kGamma;
        eps = p.kEps;
    }
#endif

#ifdef X_CACHE_N
    // The lane's slice of the head, loaded once. The rotate pass needs index i and
    // i + ropeHalfD; the host only defines X_CACHE_N when both D and ropeHalfD are
    // multiples of the simdgroup step, so both land in this lane's slice and every
    // trip count below is a compile-time constant -- which is what keeps xv in
    // registers instead of spilling to thread-local memory.
    ftype xv[X_CACHE_N];
    for (int s = 0; s < X_CACHE_N; ++s) {
        xv[s] = loadC4(xTensor, gid.y, xBase, start + s * step, xSeq);
    }

    float square_sum = 0.0f;
    for (int s = 0; s < X_CACHE_N; ++s) {
        float val = xv[s];
        square_sum += val * val;
    }
#ifdef USE_SG
    square_sum = simd_sum(square_sum);
#endif
    float var = (nullptr == gamma) ? 1.0f : 1.0f / sqrt(square_sum / ROPE_D + eps);

    const int halfSlots = ROPE_HALF_D / 32;
    for (int s = 0; s < halfSlots; ++s) {
        int i = start + s * step;
        ftype evenVal = xv[s];
        ftype oddVal  = xv[s + halfSlots];
        if (nullptr != gamma) {
            evenVal = evenVal * var * gamma[i];
            oddVal  = oddVal * var * gamma[i + ROPE_HALF_D];
        }

        int cosIndex = gid.y * (2 * ROPE_HALF_D) + i;
        ftype cEven = cos[cosIndex];
        ftype cOdd  = cos[cosIndex + ROPE_HALF_D];
        ftype sEven = sin[cosIndex];
        ftype sOdd  = sin[cosIndex + ROPE_HALF_D];

        yTensor[yBase + i] = evenVal * cEven - oddVal * sEven;
        yTensor[yBase + i + ROPE_HALF_D] = oddVal * cOdd + evenVal * sOdd;
    }
    for (int s = 2 * halfSlots; s < X_CACHE_N; ++s) {
        int i = start + s * step;
        ftype value = xv[s];
        if (nullptr != gamma) {
            value = value * var * gamma[i];
        }
        yTensor[yBase + i] = value;
    }
#else
    float square_sum = 0.0f;
    if (nullptr != gamma) {
        for (int i = start; i < p.D; i += step) {
            float val = loadC4(xTensor, gid.y, xBase, i, xSeq);
            square_sum += val * val;
        }
    }
#ifdef USE_SG
    square_sum = simd_sum(square_sum);
#endif
    float var = (nullptr == gamma) ? 1.0f : 1.0f / sqrt(square_sum / p.D + eps);

    for (int i = start; i < p.ropeHalfD; i += step) {
        ftype evenVal = loadC4(xTensor, gid.y, xBase, i, xSeq);
        ftype oddVal  = loadC4(xTensor, gid.y, xBase, i + p.ropeHalfD, xSeq);
        if (nullptr != gamma) {
            evenVal = evenVal * var * gamma[i];
            oddVal  = oddVal * var * gamma[i + p.ropeHalfD];
        }

        int cosIndex = gid.y * (2 * p.ropeHalfD) + i;
        ftype cEven = cos[cosIndex];
        ftype cOdd  = cos[cosIndex + p.ropeHalfD];
        ftype sEven = sin[cosIndex];
        ftype sOdd  = sin[cosIndex + p.ropeHalfD];

        yTensor[yBase + i] = evenVal * cEven - oddVal * sEven;
        yTensor[yBase + i + p.ropeHalfD] = oddVal * cOdd + evenVal * sOdd;
    }
    for (int i = 2 * p.ropeHalfD + start; i < p.D; i += step) {
        ftype value = loadC4(xTensor, gid.y, xBase, i, xSeq);
        if (nullptr != gamma) {
            value = value * var * gamma[i];
        }
        yTensor[yBase + i] = value;
    }
#endif
}

#ifdef ROPE_TILE
static inline ftype4 ropeNormQuad(const threadgroup ftype4* row, int qi, float var,
                                  const device float4* gamma) {
    ftype4 v = row[qi];
    if (nullptr != gamma) {
        v = ftype4(float4(v) * var * gamma[qi]);
    }
    return v;
}

// Token-tiled variant of the kernel above. The C4 input keeps only 4 channels
// (8 B in fp16) contiguous per token, so with one lane per channel-slice every
// simd load degenerates into 32 separate 8 B transactions. Tokens *are* adjacent
// within a plane, so phase 1 maps lanes to ROPE_TILE_T consecutive tokens -- one
// 64 B run per plane group -- and stages the tile in threadgroup memory. Phase 2
// switches back to one lane per output quad, both because the rotate needs the
// (i, i + ropeHalfD) pair in one lane and because the output layout is dense
// [token][head][dim], i.e. already coalesced that way.
kernel void rope_kernel_tile(
                        const device ftype* q           [[ buffer(0) ]],
                        const device ftype* k           [[ buffer(1) ]],
                        const device ftype* cos         [[ buffer(2) ]],
                        const device ftype* sin         [[ buffer(3) ]],
                        device ftype* qo                [[ buffer(4) ]],
                        device ftype* ko                [[ buffer(5) ]],
                        constant RopeParam& p           [[ buffer(6) ]],
#ifdef Q_NORM
                        const device float* qGamma      [[ buffer(7) ]],
#endif
#ifdef K_NORM
                        const device float* kGamma      [[ buffer(8) ]],
#endif
                        uint3 gid                       [[ threadgroup_position_in_grid ]],
                        uint tiisg                      [[ thread_index_in_simdgroup ]],
                        uint sgitg                      [[ simdgroup_index_in_threadgroup ]]) {
    threadgroup ftype4 sm[ROPE_NSG * ROPE_TILE_T * ROPE_PLANES];

    uint actual_z = gid.z * ROPE_NSG + sgitg;
    if (actual_z >= (uint)p.fullHead) {
        return;
    }
    int tok0 = (int)gid.y * ROPE_TILE_T;

    const device ftype4* xTensor = (const device ftype4*)q;
    device ftype4* yTensor = (device ftype4*)qo;
    int planeBase = (int)actual_z * ROPE_PLANES;
    int headStride = p.numHead * ROPE_PLANES;
    int yHead = (int)actual_z * ROPE_PLANES;
    const device float4* gamma = nullptr;
    float eps = 0.0f;
#ifdef Q_NORM
    gamma = (const device float4*)qGamma;
    eps = p.qEps;
#endif
    if (actual_z >= (uint)p.numHead) {
        xTensor = (const device ftype4*)k;
        yTensor = (device ftype4*)ko;
        planeBase = ((int)actual_z - p.numHead) * ROPE_PLANES;
        headStride = p.kvnumHead * ROPE_PLANES;
        yHead = ((int)actual_z - p.numHead) * ROPE_PLANES;
        gamma = nullptr;
        eps = 0.0f;
#ifdef K_NORM
        gamma = (const device float4*)kGamma;
        eps = p.kEps;
#endif
    }

    const int smBase = (int)sgitg * (ROPE_TILE_T * ROPE_PLANES);
    {
        int tl = (int)tiisg % ROPE_TILE_T;
        int pg = (int)tiisg / ROPE_TILE_T;
        int tok = tok0 + tl;
        bool valid = tok < p.outerSize;
        // Reading column 0 for the out-of-range lanes keeps the load uniform and
        // in bounds; the value is dropped and phase 2 never visits that token.
        int col = valid ? tok : 0;
        for (int it = 0; it < ROPE_LOAD_ITERS; ++it) {
            int plane = pg + it * ROPE_PLANE_GROUPS;
            ftype4 v = xTensor[(planeBase + plane) * p.outerSize + col];
            sm[smBase + tl * ROPE_PLANES + plane] = valid ? v : ftype4(0);
        }
    }
    // Each simdgroup only ever reads back its own slice of sm, so simdgroup scope
    // is enough -- which is also what makes the early return above safe.
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const device ftype4* cos4 = (const device ftype4*)cos;
    const device ftype4* sin4 = (const device ftype4*)sin;
    for (int t = 0; t < ROPE_TILE_T; ++t) {
        int tok = tok0 + t;
        if (tok >= p.outerSize) {
            break;
        }
        const threadgroup ftype4* row = sm + smBase + t * ROPE_PLANES;
        float ss = 0.0f;
        for (int c = 0; c < ROPE_QUADS_PER_LANE; ++c) {
            int qi = (int)tiisg + c * 32;
            if (qi < ROPE_PLANES) {
                float4 f = float4(row[qi]);
                ss += dot(f, f);
            }
        }
        ss = simd_sum(ss);
        float var = (nullptr == gamma) ? 1.0f : 1.0f / sqrt(ss / ROPE_D + eps);
        int yBase = tok * headStride + yHead;
        int cosBase = tok * (ROPE_HALF_D / 2);
        for (int c = 0; c < ROPE_QUADS_PER_LANE; ++c) {
            int qi = (int)tiisg + c * 32;
            if (qi >= ROPE_PLANES) {
                break;
            }
            int ch = qi * 4;
            ftype4 out;
            if (ch < 2 * ROPE_HALF_D) {
                // out[c] = xn[c] * cos[c] +- xn[c -+ ropeHalfD] * sin[c]; the quads
                // never straddle the half boundary because ropeHalfD % 4 == 0.
                int partner = (ch < ROPE_HALF_D) ? (qi + ROPE_HALF_QUADS) : (qi - ROPE_HALF_QUADS);
                ftype4 selfN = ropeNormQuad(row, qi, var, gamma);
                ftype4 partN = ropeNormQuad(row, partner, var, gamma);
                ftype4 cv = cos4[cosBase + qi];
                ftype4 sv = sin4[cosBase + qi];
                out = (ch < ROPE_HALF_D) ? (selfN * cv - partN * sv) : (selfN * cv + partN * sv);
            } else {
                out = ropeNormQuad(row, qi, var, gamma);
            }
            yTensor[yBase + qi] = out;
        }
    }
}
#endif // ROPE_TILE
#else
kernel void rope_kernel(
                        const device ftype* q           [[ buffer(0) ]],
                        const device ftype* k           [[ buffer(1) ]],
                        const device ftype* cos         [[ buffer(2) ]],
                        const device ftype* sin         [[ buffer(3) ]],
                        device ftype* qo                 [[ buffer(4) ]],
                        device ftype* ko                 [[ buffer(5) ]],
                        constant RopeParam& p           [[ buffer(6) ]],
                        uint3 gid                      [[ thread_position_in_grid]]) {
    if (gid.x >= (uint)p.workDim || gid.y >= (uint)p.outerSize || gid.z >= p.fullHead) {
        return;
    }
    const device ftype* xTensor = q;
    int xBase = gid.z * p.D;
    int xSeq = p.outerSize;
    device ftype* yTensor = qo;
    int yBase = gid.y * p.numHead * p.D + gid.z * p.D;
    if (gid.z >= p.numHead) {
        xTensor = k;
        xBase = (gid.z - p.numHead) * p.D;
        yTensor = ko;
        yBase = gid.y * p.kvnumHead * p.D + (gid.z - p.numHead) * p.D;
    }
    if (gid.x < (uint)p.ropeHalfD) {
        ftype evenVal = loadC4(xTensor, gid.y, xBase, gid.x, xSeq);
        ftype oddVal  = loadC4(xTensor, gid.y, xBase, gid.x + p.ropeHalfD, xSeq);
        int cosIndex = gid.y * (2 * p.ropeHalfD) + gid.x;
        ftype cEven = cos[cosIndex];
        ftype cOdd  = cos[cosIndex + p.ropeHalfD];
        ftype sEven = sin[cosIndex];
        ftype sOdd  = sin[cosIndex + p.ropeHalfD];

        ftype q0 = evenVal * cEven - oddVal * sEven;
        ftype q1 = oddVal  * cOdd  + evenVal * sOdd;

        yTensor[yBase + gid.x] = q0;
        yTensor[yBase + gid.x + p.ropeHalfD] = q1;
    }
    int tail = 2 * p.ropeHalfD + gid.x;
    if (tail < p.D) {
        yTensor[yBase + tail] = loadC4(xTensor, gid.y, xBase, tail, xSeq);
    }
}
#endif
)metal";

// Token-tiled RoPE geometry. Tokens are the contiguous axis of the C4 input, so
// ROPE_TILE_T tokens x 4 channels is one ROPE_TILE_T * 8 B run in fp16; 2
// simdgroups keeps the 64-thread threadgroup of the scalar path. 8 tokens (one
// 64 B run) measures best -- 16 is neutral and 32 loses to the occupancy cost of
// its 16 KB of staging, so what pays is cutting the request *count*, not widening
// the run.
static constexpr int kRopeTileTDefault = 8;
static constexpr int kRopeTileNsg = 2;
// Below this the kernel is launch-bound and the two arms tie, so decode and tiny
// prefill stay on the scalar path. Above it the tile wins throughout, and wins
// most while the input is still cache-resident: what it removes is L1/L2 request
// count -- one per lane in the scalar path, since C4 keeps only 4 channels
// contiguous -- not DRAM bytes, which both arms move identically.
static constexpr int kRopeTileMinSeq = 64;

class MetalRopeExecution : public MetalExecution {
public:
    explicit MetalRopeExecution(Backend *backend, int ropeCutHeadDim, std::shared_ptr<MetalLayerNorm::Resource> qNorm,
                                std::shared_ptr<MetalLayerNorm::Resource> kNorm, int numHead, int kvNumHead,
                                int headDim)
        : MetalExecution(backend),
          mRopeCutHeadDim(ropeCutHeadDim),
          mNumHead(numHead),
          mKvNumHead(kvNumHead),
          mHeadDim(headDim),
          mQNorm(qNorm),
          mKNorm(kNorm) {
        auto mtbn = static_cast<MetalBackend *>(backend);
        auto context = (__bridge MNNMetalContext *)mtbn->context();
        mParam = [context newDeviceBuffer:sizeof(RopeParam) access:CPUWriteOnly];
        auto rt = static_cast<MetalRuntime*>(mtbn->getRuntime());
        std::vector<std::string> keys = {"rope_kernel"};
        MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
        NSMutableDictionary *macros = [NSMutableDictionary dictionary];
        if (mtbn->useFp16InsteadFp32()) {
            macros[@"MNN_METAL_FLOAT16_STORAGE"] = @"1";
            keys.emplace_back("fp16");
        }
        if (mQNorm) {
            macros[@"Q_NORM"] = @"1";
            keys.emplace_back("q_norm");
        }
        if (mKNorm) {
            macros[@"K_NORM"] = @"1";
            keys.emplace_back("k_norm");
        }
        if ((mQNorm || mKNorm) && rt->supportSimdGroupReduce()) {
            macros[@"USE_SG"] = @"1";
            keys.emplace_back("sg");
            mUseSG = true;
        } else {
            mUseSG = false;
        }
        // Same derivation as onResize; headDim and ropeCutHeadDim are ctor arguments,
        // so the rotary geometry is already fixed here and can be baked in.
        int ropeDim = mRopeCutHeadDim;
        if (ropeDim <= 0 || ropeDim > mHeadDim) {
            ropeDim = mHeadDim;
        }
        ropeDim = (ropeDim / 2) * 2;
        int ropeHalfD = ropeDim / 2;
        const int kStep = 32;
        // 8 entries caps the array at 16 B per lane in fp16, i.e. headDim <= 256.
        if (mUseSG && !MetalEnv::get().ropeXCacheDisabled && ropeHalfD > 0 &&
            mHeadDim % kStep == 0 && ropeHalfD % kStep == 0 && mHeadDim / kStep <= 8) {
            macros[@"X_CACHE_N"] = [NSString stringWithFormat:@"%d", mHeadDim / kStep];
            macros[@"ROPE_D"] = [NSString stringWithFormat:@"%d", mHeadDim];
            macros[@"ROPE_HALF_D"] = [NSString stringWithFormat:@"%d", ropeHalfD];
            keys.emplace_back("xcache_" + std::to_string(mHeadDim) + "_" + std::to_string(ropeHalfD));
        }
        option.preprocessorMacros = macros;
        auto pipeline = rt->findPipeline(keys);
        if (nil == pipeline) {
            pipeline = mtbn->makeComputePipelineWithSourceOption(gMetalRopeKernelSource, "rope_kernel", option);
            rt->insertPipeline(keys, pipeline);
        }
        mPipeline = pipeline;
        if (nil == mPipeline) {
            MNN_ERROR("MetalRope: failed to compile rope_kernel.\n");
        }
        mActivePipeline = mPipeline;
        // Token-tiled variant. Both pipelines are built here and onResize picks by
        // sequence length, because the two arms tie in the launch-bound range.
        // planes % ROPE_PLANE_GROUPS == 0 is what lets the 32 lanes split cleanly
        // into whole token runs.
        int planes = mHeadDim / 4;
        // Tokens per simdgroup: how wide a contiguous run each plane group reads,
        // and equally how much threadgroup memory the tile costs, so the override
        // trades request count against occupancy in both directions.
        int tileT = kRopeTileTDefault;
        {
            int e = MetalEnv::get().ropeTileT;
            if (e > 0 && e <= 32 && (32 % e) == 0 && planes % (32 / e) == 0) {
                tileT = e;
            }
        }
        const int planeGroups = 32 / tileT;
        if (mUseSG && MetalEnv::get().ropeTile >= 0 && ropeHalfD > 0 && mHeadDim % 4 == 0 &&
            ropeHalfD % 4 == 0 && planes % planeGroups == 0 && planes <= 128) {
            mTileT = tileT;
            macros[@"ROPE_TILE"] = @"1";
            macros[@"ROPE_D"] = [NSString stringWithFormat:@"%d", mHeadDim];
            macros[@"ROPE_HALF_D"] = [NSString stringWithFormat:@"%d", ropeHalfD];
            macros[@"ROPE_PLANES"] = [NSString stringWithFormat:@"%d", planes];
            macros[@"ROPE_HALF_QUADS"] = [NSString stringWithFormat:@"%d", ropeHalfD / 4];
            macros[@"ROPE_TILE_T"] = [NSString stringWithFormat:@"%d", tileT];
            macros[@"ROPE_NSG"] = [NSString stringWithFormat:@"%d", kRopeTileNsg];
            macros[@"ROPE_PLANE_GROUPS"] = [NSString stringWithFormat:@"%d", planeGroups];
            macros[@"ROPE_LOAD_ITERS"] = [NSString stringWithFormat:@"%d", planes / planeGroups];
            macros[@"ROPE_QUADS_PER_LANE"] = [NSString stringWithFormat:@"%d", UP_DIV(planes, 32)];
            option.preprocessorMacros = macros;
            auto tileKeys = keys;
            tileKeys.emplace_back("tile_" + std::to_string(mHeadDim) + "_" + std::to_string(ropeHalfD) + "_" +
                                  std::to_string(tileT));
            auto tilePipeline = rt->findPipeline(tileKeys);
            if (nil == tilePipeline) {
                tilePipeline = mtbn->makeComputePipelineWithSourceOption(gMetalRopeKernelSource, "rope_kernel_tile",
                                                                        option);
                rt->insertPipeline(tileKeys, tilePipeline);
            }
            mPipelineTile = tilePipeline;
        }
    }

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        if (inputs.size() != 4 || outputs.size() != 2) {
            MNN_ERROR("MetalRope: expected 4 inputs and 2 outputs, got %zu inputs and %zu outputs.\n", inputs.size(),
                      outputs.size());
            return INVALID_VALUE;
        }
        auto q       = inputs[0];
        auto k       = inputs[1];
        if (!validRopeC4Input(q, k, mNumHead, mKvNumHead, mHeadDim)) {
            MNN_ERROR("MetalRope: invalid C4 input, numHead=%d, kvNumHead=%d, headDim=%d.\n", mNumHead, mKvNumHead,
                      mHeadDim);
            return NOT_SUPPORT;
        }
        int headDim = mHeadDim;
        int batch = 1;
        int seqLen = q->length(0);
        int numHead = mNumHead;
        int kvnumHead = mKvNumHead;

        RopeParam* p = (RopeParam*)(mParam.contents);
        p->outerSize  = static_cast<int>(batch * seqLen);
        int ropeDim = mRopeCutHeadDim;
        if (ropeDim <= 0 || ropeDim > headDim) {
            ropeDim = headDim;
        }
        ropeDim = (ropeDim / 2) * 2;
        p->ropeHalfD  = ropeDim / 2;
        p->workDim    = ALIMAX(p->ropeHalfD, headDim - ropeDim);
        p->D          = headDim;
        p->numHead    = numHead;
        p->kvnumHead  = kvnumHead;
        p->fullHead  = kvnumHead + numHead;
        p->qEps       = mQNorm ? mQNorm->mEps : 0.0f;
        p->kEps       = mKNorm ? mKNorm->mEps : 0.0f;
        auto mtbn = static_cast<MetalBackend *>(backend());
        auto context = (__bridge MNNMetalContext *)mtbn->context();
        if (mQNorm || mKNorm) {
            if (mUseSG) {
                int tileEnv = MetalEnv::get().ropeTile;
                bool useTile = (nil != mPipelineTile) &&
                               (1 == tileEnv || (0 == tileEnv && p->outerSize >= kRopeTileMinSeq));
                if (useTile) {
                    mActivePipeline = mPipelineTile;
                    mThreads = std::make_pair(MTLSizeMake(1, UP_DIV(p->outerSize, mTileT),
                                                         (NSUInteger)UP_DIV(numHead + kvnumHead, kRopeTileNsg)),
                                              MTLSizeMake(32 * kRopeTileNsg, 1, 1));
                } else {
                    mActivePipeline = mPipeline;
                    mThreads = std::make_pair(MTLSizeMake(1, p->outerSize, (NSUInteger)(numHead + kvnumHead + 1) / 2), MTLSizeMake(64, 1, 1));
                }
            } else {
                mActivePipeline = mPipeline;
                mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(1, p->outerSize, (NSUInteger)(numHead + kvnumHead))];
            }
        } else {
            mActivePipeline = mPipeline;
            mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake((NSUInteger)p->workDim, p->outerSize, (NSUInteger)(numHead + kvnumHead))];
        }
        return NO_ERROR;
    }

    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override {
        if (nil == mActivePipeline) {
            return;
        }
        
        auto backend = static_cast<MetalBackend *>(this->backend());

        [encoder setComputePipelineState:mActivePipeline];
        MetalBackend::setTensor(inputs[0], encoder, 0);
        MetalBackend::setTensor(inputs[1], encoder, 1);
        MetalBackend::setTensor(inputs[2], encoder, 2);
        MetalBackend::setTensor(inputs[3], encoder, 3);
        MetalBackend::setTensor(outputs[0], encoder, 4);
        MetalBackend::setTensor(outputs[1], encoder, 5);
        [encoder setBuffer:mParam offset:0 atIndex:6];
        if (mQNorm && mQNorm->mGammaBuffer) {
            MetalBackend::setTensor(mQNorm->mGammaBuffer.get(), encoder, 7);
        }
        if (mKNorm && mKNorm->mGammaBuffer) {
            MetalBackend::setTensor(mKNorm->mGammaBuffer.get(), encoder, 8);
        }
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
    }
    
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (nullptr == dst) {
            return true;
        }
        auto rope = new MetalRopeExecution(bn, mRopeCutHeadDim, mQNorm, mKNorm, mNumHead, mKvNumHead, mHeadDim);
        *dst = rope;
        MNN_METAL_PROFILE_REGISTER_CLONE(bn, op, *dst);
        return true;
    }

private:
    int mRopeCutHeadDim = 0;
    int mNumHead = 0;
    int mKvNumHead = 0;
    int mHeadDim = 0;
    int mTileT = 0;
    bool mUseSG = false;
    std::shared_ptr<MetalLayerNorm::Resource> mQNorm;
    std::shared_ptr<MetalLayerNorm::Resource> mKNorm;
    id<MTLBuffer> mParam = nil;
    id<MTLComputePipelineState> mPipeline = nil;
    id<MTLComputePipelineState> mPipelineTile = nil;
    id<MTLComputePipelineState> mActivePipeline = nil;
    std::pair<MTLSize, MTLSize> mThreads;
};

class MetalRoPECreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        int ropeCutHeadDim = 0;
        std::shared_ptr<MetalLayerNorm::Resource> qNorm;
        std::shared_ptr<MetalLayerNorm::Resource> kNorm;
        int numHead = 0;
        int kvNumHead = 0;
        int headDim = 0;
        auto param = op == nullptr ? nullptr : op->main_as_RoPEParam();
        if (param != nullptr) {
            ropeCutHeadDim = param->rope_cut_head_dim();
            numHead = param->num_head();
            kvNumHead = param->kv_num_head();
            headDim = param->head_dim();
            qNorm = makeRopeNormResource(backend, param->q_norm());
            kNorm = makeRopeNormResource(backend, param->k_norm());
        }
        return new MetalRopeExecution(backend, ropeCutHeadDim, qNorm, kNorm, numHead, kvNumHead, headDim);
    }
};
REGISTER_METAL_OP_CREATOR(MetalRoPECreator, OpType_RoPE);

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
#endif // MNN_METAL_ENABLED
