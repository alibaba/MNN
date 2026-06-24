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
    // SIMD group 内归约 + 跨 SG 合并（需要 threadgroup_barrier）
#ifdef SORT_DESC
    T sgBestVal = simd_max(bestVal);
#else
    T sgBestVal = simd_min(bestVal);
#endif
    const int INF_IDX = 2147483647;
    int candidate = (bestIdx >= 0 && bestVal == sgBestVal) ? bestIdx : INF_IDX;
    int sgBestIdx = simd_min(candidate);

    // 写入每个 simdgroup 的结果
    threadgroup T sharedBestVal[THREAD_NUMBER / SIMD_GROUP_WIDTH];
    threadgroup int sharedBestIdx[THREAD_NUMBER / SIMD_GROUP_WIDTH];
    if (tiisg == 0) {
        sharedBestVal[sgitg] = sgBestVal;
        sharedBestIdx[sgitg] = sgBestIdx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 跨 simdgroup 合并
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


class MetalTopKV2 : public MetalExecution {
private:
    id<MTLBuffer> mParam = nil;
    id<MTLComputePipelineState> mPipeline = nil;
    int mGroupNumber = 0;
    int mTopK = 0;
    bool mLargest;
    int mLocalThreadNumber = 0;
public:
    MetalTopKV2(Backend* backend, bool largest) : MetalExecution(backend), mLargest(largest) {
        auto mtbn = static_cast<MetalBackend*>(backend);
        auto context = (__bridge MNNMetalContext*)mtbn->context();
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
            return NO_ERROR;
        }
        auto mtbn = static_cast<MetalBackend*>(backend());
        int kTopKThreadNumber = static_cast<MetalRuntime*>(mtbn->getRuntime())->maxThreadSize();
        const int numRows = input->elementSize() / rowSize;
        const int k = output->length(output->dimensions() - 1);
        if (k > 1) {
            kTopKThreadNumber = 32768 / kTopKLocalK / (2 * sizeof(float));
        }

        const int kTopKCandidateNumber = kTopKThreadNumber * kTopKLocalK;

        if (k <= 0 || k > rowSize) {
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

    void onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, id<MTLComputeCommandEncoder> encoder) override {
        if (mGroupNumber <= 0) {
            return;
        }
        auto mtbn = static_cast<MetalBackend*>(backend());
        [encoder setComputePipelineState:mPipeline];
        MetalBackend::setTensor(outputs[0], encoder, 0);
        MetalBackend::setTensor(outputs[1], encoder, 1);
        MetalBackend::setTensor(inputs[0], encoder, 2);
        [encoder setBuffer:mParam offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(mGroupNumber, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(mLocalThreadNumber, 1, 1)];
    }

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
        auto output = outputs[0];
        const int k = output->length(output->dimensions() - 1);
        if (k > kTopKLocalK) {
            return nullptr;
        }
        return new MetalTopKV2(backend, largest);
    }
};

REGISTER_METAL_OP_CREATOR(MetalTopKV2Creator, OpType_TopKV2);

} // namespace MNN
#endif
