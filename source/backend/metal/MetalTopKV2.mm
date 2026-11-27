//
//  MetalTopKV2.mm
//  MNN
//
//  Created by MNN on 2026/03/22.
//

#import "core/Macro.h"
#import "MetalBackend.hpp"
#import "MetalCast.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/TensorUtils.hpp"
#include "MNN_generated.h"

#if MNN_METAL_ENABLED
namespace MNN {

static const int kTopKLocalK = 16;

static const char* gTopKV2K1Template = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

#define SIMD_GROUP_WIDTH 32

struct TopKParam {
    int4 size; // rowSize, k, numRows, pad
};

inline bool better(T aValue, int aIndex, T bValue, int bIndex) {
    if (bIndex < 0) {
        return true;
    }
    if (aIndex < 0) {
        return false;
    }
#ifdef SORT_DESC
    if (aValue > bValue) {
        return true;
    }
    if (aValue < bValue) {
        return false;
    }
#else
    if (aValue < bValue) {
        return true;
    }
    if (aValue > bValue) {
        return false;
    }
#endif
    return aIndex < bIndex;
}

kernel void topkv2(device T* outValue [[buffer(0)]],
                  device int* outIndex [[buffer(1)]],
                  const device T* inValue [[buffer(2)]],
                  constant TopKParam& p [[buffer(3)]],
#ifdef SIMD_GROUP_REDUCE
                  uint3 tgp [[threadgroup_position_in_grid]],
                  uint  tiisg [[thread_index_in_simdgroup]],
                  uint  sgitg [[simdgroup_index_in_threadgroup]]
#else
                  uint  tid [[thread_index_in_threadgroup]],
                  uint3 tgp [[threadgroup_position_in_grid]]
#endif
                  ) {
    const uint row = tgp.x;
    const int rowSize = p.size.x;
    const int numRows = p.size.z;
    if ((int)row >= numRows) {
        return;
    }

#ifdef IS_INT
    const T initWorst = (T)(2147483647);
    const T initBestWorst = (T)(-2147483648);
#else
#ifdef USE_FP16
    const T initWorst = (T)(65504.0h);
    const T initBestWorst = (T)(-65504.0h);
#else
    const T initWorst = (T)(FLT_MAX);
    const T initBestWorst = (T)(-FLT_MAX);
#endif
#endif

    const device T* rowIn = inValue + row * (uint)rowSize;

    T bestVal;
    int bestIdx = -1;
#ifdef SORT_DESC
    bestVal = initBestWorst;
#else
    bestVal = initWorst;
#endif

#ifdef SIMD_GROUP_REDUCE
    const uint tid = tiisg + sgitg * SIMD_GROUP_WIDTH;
#endif

    for (int i = (int)tid; i < rowSize; i += (int)THREAD_NUMBER) {
        const T val = rowIn[i];
        if (better(val, i, bestVal, bestIdx)) {
            bestVal = val;
            bestIdx = i;
        }
    }

#ifdef SIMD_GROUP_REDUCE
    // Intra-simdgroup reduce plus a cross-SG merge (needs threadgroup_barrier)
#ifdef SORT_DESC
    T sgBestVal = simd_max(bestVal);
#else
    T sgBestVal = simd_min(bestVal);
#endif
    const int INF_IDX = 2147483647;
    int candidate = (bestIdx >= 0 && bestVal == sgBestVal) ? bestIdx : INF_IDX;
    int sgBestIdx = simd_min(candidate);

    threadgroup T sharedBestVal[THREAD_NUMBER / SIMD_GROUP_WIDTH];
    threadgroup int sharedBestIdx[THREAD_NUMBER / SIMD_GROUP_WIDTH];
    if (tiisg == 0) {
        sharedBestVal[sgitg] = sgBestVal;
        sharedBestIdx[sgitg] = sgBestIdx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0 && sgitg == 0) {
        const uint NUM_SG = THREAD_NUMBER / SIMD_GROUP_WIDTH;
        T finalVal;
        int finalIdx = -1;
#ifdef SORT_DESC
        finalVal = initBestWorst;
#else
        finalVal = initWorst;
#endif
        for (uint i = 0; i < NUM_SG; ++i) {
            T v = sharedBestVal[i];
            int idx = sharedBestIdx[i];
            if (better(v, idx, finalVal, finalIdx)) {
                finalVal = v;
                finalIdx = idx;
            }
        }
        device T* rowOut = outValue + row * (uint)1;
        device int* rowIdx = outIndex + row * (uint)1;
        rowOut[0] = finalVal;
        rowIdx[0] = finalIdx;
    }
#else
    // Fallback: threadgroup tree reduction
    threadgroup T sharedBestVal[THREAD_NUMBER];
    threadgroup int sharedBestIdx[THREAD_NUMBER];
    sharedBestVal[tid] = bestVal;
    sharedBestIdx[tid] = bestIdx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = THREAD_NUMBER / 2; s > 0; s >>= 1) {
        if (tid < s) {
            T val1 = sharedBestVal[tid];
            int idx1 = sharedBestIdx[tid];
            T val2 = sharedBestVal[tid + s];
            int idx2 = sharedBestIdx[tid + s];
            if (!better(val1, idx1, val2, idx2)) {
                sharedBestVal[tid] = val2;
                sharedBestIdx[tid] = idx2;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        device T* rowOut = outValue + row * (uint)1;
        device int* rowIdx = outIndex + row * (uint)1;
        rowOut[0] = sharedBestVal[0];
        rowIdx[0] = sharedBestIdx[0];
    }
#endif
}
)metal";


static const char* gTopKV2K32Template = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

#define LOCAL_K 16

struct TopKParam {
    int4 size; // rowSize, k, numRows, pad
};

inline bool better(T aValue, int aIndex, T bValue, int bIndex) {
    if (bIndex < 0) {
        return true;
    }
    if (aIndex < 0) {
        return false;
    }
#ifdef SORT_DESC
    if (aValue > bValue) {
        return true;
    }
    if (aValue < bValue) {
        return false;
    }
#else
    if (aValue < bValue) {
        return true;
    }
    if (aValue > bValue) {
        return false;
    }
#endif
    return aIndex < bIndex;
}

kernel void topkv2(device T* outValue [[buffer(0)]],
                  device int* outIndex [[buffer(1)]],
                  const device T* inValue [[buffer(2)]],
                  constant TopKParam& p [[buffer(3)]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint3 tgp [[threadgroup_position_in_grid]]) {
    const uint row = tgp.x;
    const int rowSize = p.size.x;
    const int k = p.size.y;
    const int numRows = p.size.z;
    if ((int)row >= numRows) {
        return;
    }

#ifdef IS_INT
    const T initWorst = (T)(2147483647);
    const T initBestWorst = (T)(-2147483648);
#else
#ifdef USE_FP16
    const T initWorst = (T)(65504.0h);
    const T initBestWorst = (T)(-65504.0h);
#else
    const T initWorst = (T)(FLT_MAX);
    const T initBestWorst = (T)(-FLT_MAX);
#endif
#endif

    const device T* rowIn = inValue + row * (uint)rowSize;

    thread T localValue[LOCAL_K];
    thread int localIndex[LOCAL_K];
#ifdef SORT_DESC
    for (uint i = 0; i < LOCAL_K; ++i) {
        localValue[i] = initBestWorst;
        localIndex[i] = -1;
    }
#else
    for (uint i = 0; i < LOCAL_K; ++i) {
        localValue[i] = initWorst;
        localIndex[i] = -1;
    }
#endif

    for (int i = (int)tid; i < rowSize; i += (int)THREAD_NUMBER) {
        const T value = rowIn[i];
        if (!better(value, i, localValue[k - 1], localIndex[k - 1])) {
            continue;
        }

        uint insertPos = k;
        for (uint j = 0; j < k; ++j) {
            if (better(value, i, localValue[j], localIndex[j])) {
                insertPos = j;
                break;
            }
        }
        if (insertPos >= k) {
            continue;
        }
        for (uint j = k - 1; j > 0; --j) {
            if (j == insertPos) break;
            localValue[j] = localValue[j - 1];
            localIndex[j] = localIndex[j - 1];
        }
        localValue[insertPos] = value;
        localIndex[insertPos] = i;
    }

    threadgroup T sharedValue[THREAD_NUMBER][LOCAL_K];
    threadgroup int sharedIndex[THREAD_NUMBER][LOCAL_K];
    for (uint i = 0; i < k; ++i) {
        sharedValue[tid][i] = localValue[i];
        sharedIndex[tid][i] = localIndex[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = THREAD_NUMBER / 2; s > 0; s >>= 1) {
        if (tid < s) {
            T aVals[LOCAL_K];
            int aIdxs[LOCAL_K];
            T bVals[LOCAL_K];
            int bIdxs[LOCAL_K];
            for (uint i = 0; i < k; ++i) {
                aVals[i] = sharedValue[tid][i];
                aIdxs[i] = sharedIndex[tid][i];
                bVals[i] = sharedValue[tid + s][i];
                bIdxs[i] = sharedIndex[tid + s][i];
            }

            uint ai = 0, bi = 0;
            for (uint oi = 0; oi < k; ++oi) {
                if (better(aVals[ai], aIdxs[ai], bVals[bi], bIdxs[bi])) {
                    sharedValue[tid][oi] = aVals[ai];
                    sharedIndex[tid][oi] = aIdxs[ai];
                    ai++;
                } else {
                    sharedValue[tid][oi] = bVals[bi];
                    sharedIndex[tid][oi] = bIdxs[bi];
                    bi++;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        device T* rowOut = outValue + row * (uint)k;
        device int* rowIdx = outIndex + row * (uint)k;
        const int realK = min(k, rowSize);
        for (int i = 0; i < realK; ++i) {
            rowOut[i] = sharedValue[0][i];
            rowIdx[i] = sharedIndex[0][i];
        }
    }
}
)metal";


static const char* gTopKV2FastTemplate = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

struct FastTopKParam {
    int rowSize;
    int k;
    int numBlocks;
    int blockSize;
    int vecAligned;
    int pad0;
    int pad1;
    int pad2;
};

inline T topk_init_worst() {
#ifdef IS_INT
#ifdef SORT_DESC
    return (T)(-2147483648);
#else
    return (T)(2147483647);
#endif
#else
#ifdef SORT_DESC
    return (T)(-INFINITY);
#else
    return (T)(INFINITY);
#endif
#endif
}

inline bool topk_better(T aValue, int aIndex, T bValue, int bIndex) {
    if (bIndex < 0) return true;
    if (aIndex < 0) return false;
#ifdef SORT_DESC
    if (aValue > bValue) return true;
    if (aValue < bValue) return false;
#else
    if (aValue < bValue) return true;
    if (aValue > bValue) return false;
#endif
    return aIndex < bIndex;
}

inline void topk_sg_reduce(T bestVal, int bestIdx, uint tiisg, uint sgitg,
                           threadgroup T* shVal, threadgroup int* shIdx) {
    const T initWorst = topk_init_worst();
    const T vReduce = (bestIdx >= 0) ? bestVal : initWorst;
#ifdef SORT_DESC
    const T sgBest = simd_max(vReduce);
#else
    const T sgBest = simd_min(vReduce);
#endif
    const int INF_IDX = 2147483647;
    const int cand = (bestIdx >= 0 && bestVal == sgBest) ? bestIdx : INF_IDX;
    const int sgIdx = simd_min(cand);
    if (tiisg == 0) {
        shVal[sgitg] = sgBest;
        shIdx[sgitg] = (sgIdx == INF_IDX) ? -1 : sgIdx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void topk_pick_tg_winner(const threadgroup T* shVal, const threadgroup int* shIdx,
                                thread T& winVal, thread int& winIdx) {
    winVal = topk_init_worst();
    winIdx = -1;
    for (int s = 0; s < 8; ++s) {
        const int idx = shIdx[s];
        if (idx >= 0) {
            const T v = shVal[s];
            if (topk_better(v, idx, winVal, winIdx)) {
                winVal = v;
                winIdx = idx;
            }
        }
    }
}

kernel void topk1_stage1(device T* partialVal [[buffer(0)]],
                         device int* partialIdx [[buffer(1)]],
                         const device T* inValue [[buffer(2)]],
                         constant FastTopKParam& p [[buffer(3)]],
                         uint3 tgp [[threadgroup_position_in_grid]],
                         uint tiisg [[thread_index_in_simdgroup]],
                         uint sgitg [[simdgroup_index_in_threadgroup]]) {
    const int blockId = int(tgp.x);
    const int row = int(tgp.y);
    const int tid = int(sgitg * 32 + tiisg);
    const int start = blockId * p.blockSize;
    const int end = min(start + p.blockSize, p.rowSize);
    const device T* rowIn = inValue + (long)row * p.rowSize;

    T bestVal = topk_init_worst();
    int bestIdx = -1;
    if (p.vecAligned != 0) {
        const int vecEnd = end & ~3;
        for (int i = start + tid * 4; i < vecEnd; i += 256 * 4) {
            T4 v = *(const device T4*)(rowIn + i);
            if (topk_better(v[0], i + 0, bestVal, bestIdx)) { bestVal = v[0]; bestIdx = i + 0; }
            if (topk_better(v[1], i + 1, bestVal, bestIdx)) { bestVal = v[1]; bestIdx = i + 1; }
            if (topk_better(v[2], i + 2, bestVal, bestIdx)) { bestVal = v[2]; bestIdx = i + 2; }
            if (topk_better(v[3], i + 3, bestVal, bestIdx)) { bestVal = v[3]; bestIdx = i + 3; }
        }
        const int tailIdx = vecEnd + tid;
        if (tailIdx >= start && tailIdx < end) {
            T v = rowIn[tailIdx];
            if (topk_better(v, tailIdx, bestVal, bestIdx)) { bestVal = v; bestIdx = tailIdx; }
        }
    } else {
        for (int i = start + tid; i < end; i += 256) {
            T v = rowIn[i];
            if (topk_better(v, i, bestVal, bestIdx)) { bestVal = v; bestIdx = i; }
        }
    }

    threadgroup T shVal[8];
    threadgroup int shIdx[8];
    topk_sg_reduce(bestVal, bestIdx, tiisg, sgitg, shVal, shIdx);
    if (tid == 0) {
        T winVal;
        int winIdx;
        topk_pick_tg_winner(shVal, shIdx, winVal, winIdx);
        const int outPos = row * p.numBlocks + blockId;
        partialVal[outPos] = winVal;
        partialIdx[outPos] = winIdx;
    }
}

kernel void topk1_stage2(device T* outValue [[buffer(0)]],
                         device int* outIndex [[buffer(1)]],
                         const device T* partialVal [[buffer(2)]],
                         const device int* partialIdx [[buffer(3)]],
                         constant FastTopKParam& p [[buffer(4)]],
                         uint3 tgp [[threadgroup_position_in_grid]],
                         uint tiisg [[thread_index_in_simdgroup]],
                         uint sgitg [[simdgroup_index_in_threadgroup]]) {
    const int row = int(tgp.x);
    const int tid = int(sgitg * 32 + tiisg);
    const int base = row * p.numBlocks;
    T bestVal = topk_init_worst();
    int bestIdx = -1;
    for (int i = tid; i < p.numBlocks; i += 256) {
        const int idx = partialIdx[base + i];
        if (idx >= 0) {
            const T v = partialVal[base + i];
            if (topk_better(v, idx, bestVal, bestIdx)) { bestVal = v; bestIdx = idx; }
        }
    }
    threadgroup T shVal[8];
    threadgroup int shIdx[8];
    topk_sg_reduce(bestVal, bestIdx, tiisg, sgitg, shVal, shIdx);
    if (tid == 0) {
        T winVal;
        int winIdx;
        topk_pick_tg_winner(shVal, shIdx, winVal, winIdx);
        outValue[row] = winVal;
        outIndex[row] = winIdx;
    }
}

kernel void topk_peel_stage1(device T* partialVal [[buffer(0)]],
                             device int* partialIdx [[buffer(1)]],
                             const device T* inValue [[buffer(2)]],
                             constant FastTopKParam& p [[buffer(3)]],
                             uint3 tgp [[threadgroup_position_in_grid]],
                             uint tiisg [[thread_index_in_simdgroup]],
                             uint sgitg [[simdgroup_index_in_threadgroup]]) {
    const int blockId = int(tgp.x);
    const int row = int(tgp.y);
    const int tid = int(sgitg * 32 + tiisg);
    const int start = blockId * p.blockSize;
    const int end = min(start + p.blockSize, p.rowSize);
    const device T* rowIn = inValue + (long)row * p.rowSize;

    thread T vals[16];
    uint activeMask = 0u;
    for (int j = 0; j < 16; ++j) {
        const int idx = start + j * 256 + tid;
        if (idx < end) {
            vals[j] = rowIn[idx];
            activeMask |= (1u << j);
        }
    }

    threadgroup T shVal[8];
    threadgroup int shIdx[8];
    const int outBase = (row * p.numBlocks + blockId) * p.k;
    for (int step = 0; step < p.k; ++step) {
        T bestVal = topk_init_worst();
        int bestIdx = -1;
        int bestSlot = -1;
        uint m = activeMask;
        while (m != 0u) {
            const uint j = ctz(m);
            m &= m - 1u;
            const int idx = start + int(j) * 256 + tid;
            const T v = vals[j];
            if (topk_better(v, idx, bestVal, bestIdx)) {
                bestVal = v;
                bestIdx = idx;
                bestSlot = int(j);
            }
        }
        topk_sg_reduce(bestVal, bestIdx, tiisg, sgitg, shVal, shIdx);
        T winVal;
        int winIdx;
        topk_pick_tg_winner(shVal, shIdx, winVal, winIdx);
        if (bestSlot >= 0 && bestIdx == winIdx) {
            activeMask &= ~(1u << bestSlot);
        }
        if (tid == 0) {
            partialVal[outBase + step] = winVal;
            partialIdx[outBase + step] = winIdx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void topk_peel_stage2(device T* outValue [[buffer(0)]],
                             device int* outIndex [[buffer(1)]],
                             const device T* partialVal [[buffer(2)]],
                             const device int* partialIdx [[buffer(3)]],
                             constant FastTopKParam& p [[buffer(4)]],
                             uint3 tgp [[threadgroup_position_in_grid]],
                             uint tiisg [[thread_index_in_simdgroup]],
                             uint sgitg [[simdgroup_index_in_threadgroup]]) {
    const int row = int(tgp.x);
    const int tid = int(sgitg * 32 + tiisg);
    const int totalCand = p.numBlocks * p.k;
    const device T* pVal = partialVal + (long)row * totalCand;
    const device int* pIdx = partialIdx + (long)row * totalCand;

    thread T vals[STAGE2_ELEMS];
    thread int idxs[STAGE2_ELEMS];
    uint activeMask = 0u;
    for (int j = 0; j < STAGE2_ELEMS; ++j) {
        const int c = j * 256 + tid;
        if (c < totalCand) {
            const int idx = pIdx[c];
            if (idx >= 0) {
                vals[j] = pVal[c];
                idxs[j] = idx;
                activeMask |= (1u << j);
            }
        }
    }

    threadgroup T shVal[8];
    threadgroup int shIdx[8];
    const int outBase = row * p.k;
    for (int step = 0; step < p.k; ++step) {
        T bestVal = topk_init_worst();
        int bestIdx = -1;
        int bestSlot = -1;
        uint m = activeMask;
        while (m != 0u) {
            const uint j = ctz(m);
            m &= m - 1u;
            const T v = vals[j];
            const int idx = idxs[j];
            if (topk_better(v, idx, bestVal, bestIdx)) {
                bestVal = v;
                bestIdx = idx;
                bestSlot = int(j);
            }
        }
        topk_sg_reduce(bestVal, bestIdx, tiisg, sgitg, shVal, shIdx);
        T winVal;
        int winIdx;
        topk_pick_tg_winner(shVal, shIdx, winVal, winIdx);
        if (bestSlot >= 0 && bestIdx == winIdx) {
            activeMask &= ~(1u << bestSlot);
        }
        if (tid == 0) {
            outValue[outBase + step] = winVal;
            outIndex[outBase + step] = winIdx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
)metal";

class MetalTopKV2 : public MetalExecution {
private:
    id<MTLBuffer> mParam = nil;
    id<MTLBuffer> mFastParam = nil;
    id<MTLBuffer> mPartialVal = nil;
    id<MTLBuffer> mPartialIdx = nil;
    id<MTLComputePipelineState> mPipeline = nil;
    id<MTLComputePipelineState> mFastStage1 = nil;
    id<MTLComputePipelineState> mFastStage2 = nil;
    int mGroupNumber = 0;
    int mTopK = 0;
    bool mLargest;
    int mLocalThreadNumber = 0;
    bool mUseFast = false;
    int mFastRowSize = 0;
    int mFastNumRows = 0;
    int mFastNumBlocks = 0;
    int mFastStage2Elems = 0;
    int mRecordedFast = -1;
    int mRecordedRowSize = 0;
    int mRecordedNumRows = 0;
    int mRecordedNumBlocks = 0;
    int mRecordedTopK = 0;
    int mRecordedGroupNumber = 0;

    bool ensureFastPipelines(MetalBackend* mtbn, const Tensor* input, int k, int stage2Elems) {
        const bool useFp16 = mtbn->useFp16InsteadFp32();
        NSString* T = MetalCast::getScalarType(input->getType(), useFp16);
        NSString* T4 = [T stringByAppendingString:@"4"];
        const std::string tStr([T UTF8String]);
        const std::string mode = (k == 1) ? "k1" : ("peel_" + std::to_string(stage2Elems));
        std::vector<std::string> k1 = {"topkv2_fast_s1", tStr, mLargest ? "desc" : "asc", mode};
        std::vector<std::string> k2 = {"topkv2_fast_s2", tStr, mLargest ? "desc" : "asc", mode};
        auto rt = mtbn->runtime();
        auto p1 = rt->findPipeline(k1);
        auto p2 = rt->findPipeline(k2);
        if (p1 == nil || p2 == nil) {
            auto dic = [NSMutableDictionary dictionaryWithCapacity:6];
            [dic setValue:T forKey:@"T"];
            [dic setValue:T4 forKey:@"T4"];
            [dic setValue:[NSString stringWithFormat:@"%d", stage2Elems] forKey:@"STAGE2_ELEMS"];
            if (mLargest) {
                [dic setValue:@"1" forKey:@"SORT_DESC"];
            }
            if (input->getType().code == halide_type_int && input->getType().bits == 32) {
                [dic setValue:@"1" forKey:@"IS_INT"];
            } else if (useFp16) {
                [dic setValue:@"1" forKey:@"USE_FP16"];
            }
            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            opts.preprocessorMacros = dic;
            const char* entry1 = (k == 1) ? "topk1_stage1" : "topk_peel_stage1";
            const char* entry2 = (k == 1) ? "topk1_stage2" : "topk_peel_stage2";
            if (p1 == nil) {
                p1 = mtbn->makeComputePipelineWithSourceOption(gTopKV2FastTemplate, entry1, opts);
                if (p1 != nil) {
                    rt->insertPipeline(k1, p1);
                }
            }
            if (p2 == nil) {
                p2 = mtbn->makeComputePipelineWithSourceOption(gTopKV2FastTemplate, entry2, opts);
                if (p2 != nil) {
                    rt->insertPipeline(k2, p2);
                }
            }
        }
        if (p1 == nil || p2 == nil || p1.maxTotalThreadsPerThreadgroup < 256 ||
            p2.maxTotalThreadsPerThreadgroup < 256) {
            return false;
        }
        mFastStage1 = p1;
        mFastStage2 = p2;
        return true;
    }

public:
    MetalTopKV2(Backend* backend, bool largest) : MetalExecution(backend), mLargest(largest) {
        auto mtbn = static_cast<MetalBackend*>(backend);
        mParam = mtbn->getConstBuffer(sizeof(int) * 4);
        if (nil == mParam) {
            mValid = false;
        }
    }
    virtual ~ MetalTopKV2() {
        auto mtbn = static_cast<MetalBackend*>(backend());
        mtbn->returnConstBuffer(mParam);
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto input = inputs[0];
        auto output = outputs[0];
        const int rowSize = input->length(input->dimensions() - 1);
        if (rowSize <= 0) {
            mGroupNumber = 0;
            mUseFast = false;
            return NO_ERROR;
        }
        auto mtbn = static_cast<MetalBackend*>(backend());
        const int numRows = input->elementSize() / rowSize;
        const int k = output->length(output->dimensions() - 1);
        if (k <= 0 || k > rowSize) {
            return NOT_SUPPORT;
        }

        const auto dtype = input->getType();
        const bool typeOk = (dtype.code == halide_type_float) ||
                            (dtype.code == halide_type_int && dtype.bits == 32);
        mUseFast = false;
        if (typeOk && mtbn->runtime()->supportSimdGroupReduce() &&
            rowSize >= 4096 && rowSize <= 524288 && numRows >= 1 && numRows <= 16 && k <= 64) {
            int numBlocks = 1;
            int blockSize = rowSize;
            int stage2Elems = 16;
            if (k == 1) {
                numBlocks = std::min(256, UP_DIV(rowSize, 1024));
                blockSize = ((UP_DIV(rowSize, numBlocks) + 3) / 4) * 4;
                numBlocks = UP_DIV(rowSize, blockSize);
            } else {
                numBlocks = UP_DIV(rowSize, 4096);
                if (numBlocks < 32) {
                    numBlocks = std::min(32, UP_DIV(rowSize, 256));
                }
                if (numBlocks > 128) {
                    numBlocks = 128;
                }
                blockSize = UP_DIV(rowSize, numBlocks);
                stage2Elems = (numBlocks * k <= 4096) ? 16 : 32;
            }
            if ((k == 1 || (blockSize <= 4096 && numBlocks * k <= 256 * stage2Elems)) &&
                ensureFastPipelines(mtbn, input, k, stage2Elems)) {
                auto ctx = (__bridge MNNMetalContext*)mtbn->context();
                const size_t elemBytes = (dtype.code == halide_type_float && mtbn->useFp16InsteadFp32()) ? 2 : 4;
                const size_t partialCount = (size_t)numRows * numBlocks * k;
                const size_t valBytes = partialCount * elemBytes;
                const size_t idxBytes = partialCount * sizeof(int);
                if (mFastParam == nil) {
                    mFastParam = [ctx newDeviceBuffer:sizeof(int) * 8 access:CPUWriteOnly];
                }
                if (mPartialVal == nil || mPartialVal.length < valBytes) {
                    mPartialVal = [ctx newDeviceBuffer:valBytes access:CPUTransparent];
                }
                if (mPartialIdx == nil || mPartialIdx.length < idxBytes) {
                    mPartialIdx = [ctx newDeviceBuffer:idxBytes access:CPUTransparent];
                }
                if (mFastParam != nil && mPartialVal != nil && mPartialIdx != nil) {
                    const bool offsetAligned = (TensorUtils::getDescribeOrigin(input)->offset & 15) == 0;
                    const int vecAligned = (offsetAligned && (numRows == 1 || (rowSize & 3) == 0)) ? 1 : 0;
                    auto p = reinterpret_cast<int*>([mFastParam contents]);
                    p[0] = rowSize;
                    p[1] = k;
                    p[2] = numBlocks;
                    p[3] = blockSize;
                    p[4] = vecAligned;
                    p[5] = p[6] = p[7] = 0;
                    mFastRowSize = rowSize;
                    mFastNumRows = numRows;
                    mFastNumBlocks = numBlocks;
                    mFastStage2Elems = stage2Elems;
                    mTopK = k;
                    mGroupNumber = numRows;
                    mUseFast = true;
                    return NO_ERROR;
                }
            }
        }

        int kTopKThreadNumber = static_cast<MetalRuntime*>(mtbn->getRuntime())->maxThreadSize();
        if (k > 1) {
            kTopKThreadNumber = 32768 / kTopKLocalK / (2 * sizeof(float));
        }

        const int kTopKCandidateNumber = kTopKThreadNumber * kTopKLocalK;

        if (k > kTopKLocalK) {
            return NOT_SUPPORT;
        }
        if (k > kTopKCandidateNumber) {
            MNN_ERROR("Metal TopK don't support k=%dlarger than %d\n", k, kTopKCandidateNumber);
            return NOT_SUPPORT;
        }

        const bool useFp16 = mtbn->useFp16InsteadFp32();
        bool largest = mLargest;
        NSString* T = MetalCast::getScalarType(inputs[0]->getType(), useFp16);

        std::vector<std::string> keys = {
            "topkv2",
            std::string([T UTF8String]),
            largest ? "largest" : "smallest",
        };
        mLocalThreadNumber = kTopKThreadNumber;

        const char* sourceTemplate = nullptr;
        if (k == 1) {
            keys.push_back("k1");
            sourceTemplate = gTopKV2K1Template;
        } else if (k <= kTopKLocalK) {
            keys.push_back("smallk");
            sourceTemplate = gTopKV2K32Template;
        }

        auto pipeline = mtbn->runtime()->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions* compileOptions = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:T forKey:@"T"];
            if (largest) {
                [dic setValue:@"1" forKey:@"SORT_DESC"];
            }
            if (inputs[0]->getType().code == halide_type_int && inputs[0]->getType().bits == 32) {
                [dic setValue:@"1" forKey:@"IS_INT"];
            }
            if (useFp16 && inputs[0]->getType().code != halide_type_int) {
                [dic setValue:@"1" forKey:@"USE_FP16"];
            }
            // Keep THREAD_NUMBER in sync with host-side kTopKThreadNumber
            [dic setValue:[NSString stringWithFormat:@"%d", kTopKThreadNumber] forKey:@"THREAD_NUMBER"];
            // Enable SIMD group reduction for K=1 when supported
            if (k == 1 && ((MetalRuntime*)mtbn->runtime())->supportSimdGroupReduce()) {
                [dic setValue:@"1" forKey:@"SIMD_GROUP_REDUCE"];
            }
            compileOptions.preprocessorMacros = dic;

            pipeline = mtbn->makeComputePipelineWithSourceOption(sourceTemplate, "topkv2", compileOptions);
            mtbn->runtime()->insertPipeline(keys, pipeline);
        }
        if (nil == pipeline) {
            MNN_ERROR("Create TopKV2 pipeline error\n");
            return NOT_SUPPORT;
        }
        mPipeline = pipeline;
        auto p = (int*)mParam.contents;
        p[0] = rowSize;
        p[1] = k;
        p[2] = numRows;
        p[3] = 0;

        mGroupNumber = numRows;
        mTopK = k;
        return NO_ERROR;
    }

    bool onReplayUpdate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if ((mUseFast ? 1 : 0) != mRecordedFast) {
            return false;
        }
        if (mUseFast) {
            return mFastRowSize == mRecordedRowSize && mFastNumRows == mRecordedNumRows &&
                   mFastNumBlocks == mRecordedNumBlocks && mTopK == mRecordedTopK;
        }
        return mGroupNumber == mRecordedGroupNumber && mTopK == mRecordedTopK;
    }

    void onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, id<MTLComputeCommandEncoder> encoder) override {
        mRecordedFast = mUseFast ? 1 : 0;
        mRecordedRowSize = mFastRowSize;
        mRecordedNumRows = mFastNumRows;
        mRecordedNumBlocks = mFastNumBlocks;
        mRecordedTopK = mTopK;
        mRecordedGroupNumber = mGroupNumber;
        if (mGroupNumber <= 0) {
            return;
        }
        if (mUseFast) {
            [encoder setComputePipelineState:mFastStage1];
            [encoder setBuffer:mPartialVal offset:0 atIndex:0];
            [encoder setBuffer:mPartialIdx offset:0 atIndex:1];
            MetalBackend::setTensor(inputs[0], encoder, 2);
            [encoder setBuffer:mFastParam offset:0 atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(mFastNumBlocks, mFastNumRows, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [encoder setComputePipelineState:mFastStage2];
            MetalBackend::setTensor(outputs[0], encoder, 0);
            MetalBackend::setTensor(outputs[1], encoder, 1);
            [encoder setBuffer:mPartialVal offset:0 atIndex:2];
            [encoder setBuffer:mPartialIdx offset:0 atIndex:3];
            [encoder setBuffer:mFastParam offset:0 atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(mFastNumRows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            return;
        }
        [encoder setComputePipelineState:mPipeline];
        MetalBackend::setTensor(outputs[0], encoder, 0);
        MetalBackend::setTensor(outputs[1], encoder, 1);
        MetalBackend::setTensor(inputs[0], encoder, 2);
        [encoder setBuffer:mParam offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(mGroupNumber, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(mLocalThreadNumber, 1, 1)];
    }

};

static const char* gMoeRouteTemplate = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

struct TopKParam {
    int4 size;
};

kernel void moe_route(device T* outProbs [[buffer(0)]],
                      device int* outIndex [[buffer(1)]],
                      const device T* inLogits [[buffer(2)]],
                      const device T* inBias [[buffer(3)]],
                      constant TopKParam& p [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
    const int row = (int)gid;
    const int rowSize = p.size.x;
    const int k = p.size.y;
    if (row >= p.size.z) {
        return;
    }
    const device T* x = inLogits + row * rowSize;
    device T* probs = outProbs + row * rowSize;
    T bestVal[LOCAL_K];
    int bestIdx[LOCAL_K];
    for (int j = 0; j < k; ++j) {
#ifdef USE_FP16
        bestVal[j] = (T)(-65504.0h);
#else
        bestVal[j] = (T)(-FLT_MAX);
#endif
        bestIdx[j] = -1;
    }
    for (int i = 0; i < rowSize; ++i) {
        const T pr = (T)(1.0f / (1.0f + exp(-(float)x[i])));
        probs[i] = pr;
        const T s = pr + inBias[i];
        if (bestIdx[k - 1] >= 0 && s <= bestVal[k - 1]) {
            continue;
        }
        int pos = k - 1;
        while (pos > 0 && (bestIdx[pos - 1] < 0 || s > bestVal[pos - 1])) {
            bestVal[pos] = bestVal[pos - 1];
            bestIdx[pos] = bestIdx[pos - 1];
            --pos;
        }
        bestVal[pos] = s;
        bestIdx[pos] = i;
    }
    device int* ids = outIndex + row * k;
    for (int j = 0; j < k; ++j) {
        ids[j] = bestIdx[j];
    }
}
)metal";

class MetalMoeRoute : public MetalExecution {
public:
    static Execution* create(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, Backend* backend) {
        if (inputs.size() != 4 || inputs[0]->getType().code != halide_type_float) {
            return nullptr;
        }
        const int rowSize = inputs[0]->length(inputs[0]->dimensions() - 1);
        const int k = outputs[1]->length(outputs[1]->dimensions() - 1);
        if (rowSize <= 0 || rowSize > 256 || k <= 0 || k > kTopKLocalK || k > rowSize) {
            return nullptr;
        }
        return new MetalMoeRoute(backend);
    }
    MetalMoeRoute(Backend* backend) : MetalExecution(backend) {
        auto mtbn = static_cast<MetalBackend*>(backend);
        mParam = mtbn->getConstBuffer(sizeof(int) * 4);
        if (nil == mParam) {
            mValid = false;
        }
    }
    virtual ~MetalMoeRoute() {
        auto mtbn = static_cast<MetalBackend*>(backend());
        mtbn->returnConstBuffer(mParam);
    }
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto input = inputs[0];
        const int rowSize = input->length(input->dimensions() - 1);
        const int numRows = input->elementSize() / rowSize;
        const int k = outputs[1]->length(outputs[1]->dimensions() - 1);
        auto mtbn = static_cast<MetalBackend*>(backend());
        const bool useFp16 = mtbn->useFp16InsteadFp32();
        NSString* T = MetalCast::getScalarType(input->getType(), useFp16);
        std::vector<std::string> keys = {
            "moe_route",
            std::string([T UTF8String]),
            std::to_string(k),
        };
        auto pipeline = mtbn->runtime()->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions* compileOptions = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:T forKey:@"T"];
            [dic setValue:[NSString stringWithFormat:@"%d", k] forKey:@"LOCAL_K"];
            if (useFp16) {
                [dic setValue:@"1" forKey:@"USE_FP16"];
            }
            compileOptions.preprocessorMacros = dic;
            pipeline = mtbn->makeComputePipelineWithSourceOption(gMoeRouteTemplate, "moe_route", compileOptions);
            mtbn->runtime()->insertPipeline(keys, pipeline);
        }
        if (nil == pipeline) {
            MNN_ERROR("Create MoE route pipeline error\n");
            return NOT_SUPPORT;
        }
        mPipeline = pipeline;
        auto p = (int*)mParam.contents;
        p[0] = rowSize;
        p[1] = k;
        p[2] = numRows;
        p[3] = 0;
        mRows = numRows;
        return NO_ERROR;
    }
    void onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, id<MTLComputeCommandEncoder> encoder) override {
        if (mRows <= 0) {
            return;
        }
        [encoder setComputePipelineState:mPipeline];
        MetalBackend::setTensor(outputs[0], encoder, 0);
        MetalBackend::setTensor(outputs[1], encoder, 1);
        MetalBackend::setTensor(inputs[0], encoder, 2);
        MetalBackend::setTensor(inputs[3], encoder, 3);
        [encoder setBuffer:mParam offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(mRows, 64), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
private:
    id<MTLBuffer> mParam = nil;
    id<MTLComputePipelineState> mPipeline = nil;
    int mRows = 0;
};

class MetalTopKV2Creator : public MetalBackend::Creator {
public:
    Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend, const std::vector<Tensor*>& outputs) const override {
        if (inputs.size() < 2 || outputs.size() != 2) {
            return nullptr;
        }
        if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            return nullptr;
        }
        bool largest = true;
        auto param = op->main_as_TopKV2();
        if (nullptr != param) {
            largest = param->largest();
        }
        if (nullptr != param && param->routeSigmoid()) {
            return MetalMoeRoute::create(inputs, outputs, backend);
        }
        auto output = outputs[0];
        const int rowSize = inputs[0]->length(inputs[0]->dimensions() - 1);
        const int numRows = rowSize > 0 ? (inputs[0]->elementSize() / rowSize) : 0;
        const int k = output->length(output->dimensions() - 1);
        const bool fastRange = (rowSize >= 4096 && rowSize <= 524288 && numRows >= 1 && numRows <= 16 && k <= 64);
        if (k > kTopKLocalK && !fastRange) {
            return nullptr;
        }
        return new MetalTopKV2(backend, largest);
    }
};

REGISTER_METAL_OP_CREATOR(MetalTopKV2Creator, OpType_TopKV2);

} // namespace MNN
#endif
