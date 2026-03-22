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

static const int kTopKThreadNumber = 128;
static const int kTopKLocalK = 8;
static const int kTopKCandidateNumber = kTopKThreadNumber * kTopKLocalK;

static const char* gTopKV2Template = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

#define THREAD_NUMBER 128
#define LOCAL_K 8
#define CANDIDATE_NUMBER (THREAD_NUMBER * LOCAL_K)

struct TopKParam {
    int4 size; // rowSize, k, numRows, pad
};

inline bool afterAsc(T aValue, int aIndex, T bValue, int bIndex) {
    if (aValue > bValue) {
        return true;
    }
    if (aValue < bValue) {
        return false;
    }
    // tie-break: larger index comes after
    return aIndex > bIndex;
}

inline bool better(T aValue, int aIndex, T bValue, int bIndex) {
    // b is invalid => always better
    if (bIndex < 0) {
        return true;
    }
    // a is invalid => never better
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
    // tie-break: smaller index wins
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
    const T initWorst = (T)(FLT_MAX);
    const T initBestWorst = (T)(-FLT_MAX);
#endif

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

    const device T* rowIn = inValue + row * (uint)rowSize;

    for (int i = (int)tid; i < rowSize; i += (int)THREAD_NUMBER) {
        const T value = rowIn[i];
        if (!better(value, i, localValue[LOCAL_K - 1], localIndex[LOCAL_K - 1])) {
            continue;
        }

        uint insertPos = LOCAL_K;
        for (uint j = 0; j < LOCAL_K; ++j) {
            if (better(value, i, localValue[j], localIndex[j])) {
                insertPos = j;
                break;
            }
        }
        if (insertPos >= LOCAL_K) {
            continue;
        }
        for (uint j = LOCAL_K - 1; j > insertPos; --j) {
            localValue[j] = localValue[j - 1];
            localIndex[j] = localIndex[j - 1];
        }
        localValue[insertPos] = value;
        localIndex[insertPos] = i;
    }

    threadgroup T sharedValue[CANDIDATE_NUMBER];
    threadgroup int sharedIndex[CANDIDATE_NUMBER];
    const uint base = tid * LOCAL_K;
    for (uint i = 0; i < LOCAL_K; ++i) {
        sharedValue[base + i] = localValue[i];
        sharedIndex[base + i] = localIndex[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint size = 2; size <= CANDIDATE_NUMBER; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            for (uint idx = tid; idx < CANDIDATE_NUMBER; idx += THREAD_NUMBER) {
                const uint ixj = idx ^ stride;
                if (ixj <= idx) {
                    continue;
                }
                bool up = ((idx & size) == 0);
#ifdef SORT_DESC
                up = !up;
#endif

                const bool after = afterAsc(sharedValue[idx], sharedIndex[idx], sharedValue[ixj], sharedIndex[ixj]);
                if (up == after) {
                    const T tValue = sharedValue[idx];
                    sharedValue[idx] = sharedValue[ixj];
                    sharedValue[ixj] = tValue;
                    const int tIndex = sharedIndex[idx];
                    sharedIndex[idx] = sharedIndex[ixj];
                    sharedIndex[ixj] = tIndex;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid == 0) {
        device T* rowOut = outValue + row * (uint)k;
        device int* rowIdx = outIndex + row * (uint)k;
        const int realK = min(k, rowSize);
        for (int i = 0; i < realK; ++i) {
            rowOut[i] = sharedValue[i];
            rowIdx[i] = sharedIndex[i];
        }
    }
}
)metal";

class MetalTopKV2 : public MetalExecution {
public:
    MetalTopKV2(id<MTLComputePipelineState> pipeline, Backend* backend) : MetalExecution(backend), mPipeline(pipeline) {
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
        const int numRows = input->elementSize() / rowSize;
        const int k = output->length(output->dimensions() - 1);

        auto p = (int*)mParam.contents;
        p[0] = rowSize;
        p[1] = k;
        p[2] = numRows;
        p[3] = 0;

        mGroupNumber = numRows;
        return NO_ERROR;
    }

    void onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, id<MTLComputeCommandEncoder> encoder) override {
        if (mGroupNumber <= 0) {
            return;
        }
        [encoder setComputePipelineState:mPipeline];
        MetalBackend::setTensor(outputs[0], encoder, 0);
        MetalBackend::setTensor(outputs[1], encoder, 1);
        MetalBackend::setTensor(inputs[0], encoder, 2);
        [encoder setBuffer:mParam offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(mGroupNumber, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kTopKThreadNumber, 1, 1)];
    }

private:
    id<MTLBuffer> mParam = nil;
    id<MTLComputePipelineState> mPipeline = nil;
    int mGroupNumber = 0;
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

        const int rowSize = inputs[0]->length(inputs[0]->dimensions() - 1);
        const int k = outputs[0]->length(outputs[0]->dimensions() - 1);
        if (k <= 0 || k > rowSize) {
            return nullptr;
        }
        // Limit by threadgroup candidate capacity: THREAD_NUMBER * LOCAL_K
        if (k > kTopKCandidateNumber) {
            return nullptr;
        }

        bool largest = true;
        auto param = op->main_as_TopKV2();
        if (nullptr != param) {
            largest = param->largest();
        }

        auto mtbn = static_cast<MetalBackend*>(backend);
        const bool useFp16 = mtbn->useFp16InsteadFp32();
        NSString* T = MetalCast::getScalarType(inputs[0]->getType(), useFp16);

        std::vector<std::string> keys = {
            "topkv2",
            std::string([T UTF8String]),
            largest ? "largest" : "smallest",
        };

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
            compileOptions.preprocessorMacros = dic;

            pipeline = mtbn->makeComputePipelineWithSourceOption(gTopKV2Template, "topkv2", compileOptions);
            mtbn->runtime()->insertPipeline(keys, pipeline);
        }
        if (nil == pipeline) {
            MNN_ERROR("Create TopKV2 pipeline error\n");
            return nullptr;
        }
        return new MetalTopKV2(pipeline, backend);
    }
};

REGISTER_METAL_OP_CREATOR(MetalTopKV2Creator, OpType_TopKV2);

} // namespace MNN
#endif
