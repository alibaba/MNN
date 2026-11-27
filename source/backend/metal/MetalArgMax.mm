//
//  MetalArgMax.mm
//  MNN
//
//  Created by MNN on 2023/12/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "core/Macro.h"
#import "MetalCast.hpp"
#import "MetalBackend.hpp"
#import "MNNMetalContext.h"
#include "MNN_generated.h"

#if MNN_METAL_ENABLED
namespace MNN {
static const char* gArgMaxTemplate = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct constBuffer
{
    int4 size;
};
struct sourceBuffer
{
    T data[1];
};
struct destbuffer
{
    int data[1];
};

kernel void main0(device destbuffer& uOutput [[buffer(0)]], const device sourceBuffer& uInput [[buffer(1)]], constant constBuffer& uConst [[buffer(2)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]])
{
    threadgroup T local_buffer[256];
    threadgroup int local_index[256];
    int index = int(gl_GlobalInvocationID.x) / uConst.size.w;
    int lidIndex = int(gl_LocalInvocationID.x);
    int lidx = lidIndex / uConst.size.w;
    int lid = lidIndex % uConst.size.w;
    int x = index % uConst.size.x;
    int y = index / uConst.size.x;
    int W = uConst.size.x;
    int H = uConst.size.y;
    int C = uConst.size.z;
    bool _68 = y < uConst.size.z;
    bool _75;
    if (_68)
    {
        _75 = lid < uConst.size.y;
    }
    else
    {
        _75 = _68;
    }
    if (_75)
    {
        int offset = ((y * H) * W) + x;
        T maxValue = uInput.data[offset + (lid * W)];
        int maxIndex = lid;
        int _107 = lid + uConst.size.w;
        for (int i = _107; i < uConst.size.y; i += uConst.size.w)
        {
            T value = uInput.data[offset + (i * W)];
#ifdef ARGMIN
            if (value < maxValue)
            {
                maxValue = value;
                maxIndex = i;
            }
#else
            if (value > maxValue)
            {
                maxValue = value;
                maxIndex = i;
            }
#endif
        }
        local_buffer[lid + (lidx * uConst.size.w)] = maxValue;
        local_index[lid + (lidx * uConst.size.w)] = maxIndex;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((y < uConst.size.z) && (lid == 0))
    {
        T maxValue_1 = local_buffer[lidx * uConst.size.w];
        int maxIndex_1 = local_index[lidx * uConst.size.w];
        int t = 1;
        for (;;)
        {
            bool _195 = t < uConst.size.w;
            bool _202;
            if (_195)
            {
                _202 = t < uConst.size.y;
            }
            else
            {
                _202 = _195;
            }
            if (_202)
            {
                T next = local_buffer[t + (lidx * uConst.size.w)];
#ifdef ARGMIN
                if (next < maxValue_1)
                {
                    maxValue_1 = next;
                    maxIndex_1 = local_index[t + (lidx * uConst.size.w)];
                }
#else
                if (next > maxValue_1)
                {
                    maxValue_1 = next;
                    maxIndex_1 = local_index[t + (lidx * uConst.size.w)];
                }
#endif
                t++;
                continue;
            }
            else
            {
                break;
            }
        }
        uOutput.data[index] = maxIndex_1;
    }
}
)metal";

// Two-pass multi-threadgroup simdgroup ArgMax/ArgMin for wide contiguous rows
// (e.g. LLM vocab logits). Preserves exact first-index tie-breaking.
static const char* gArgMaxFastTemplate = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

struct FastArgMaxParam {
    int mid;
    int numBlocks;
    int blockSize;
    int vecAligned;
};

inline T argmax_init_worst() {
#ifdef IS_INT
#ifdef ARGMIN
    return (T)(2147483647);
#else
    return (T)(-2147483648);
#endif
#elif defined(USE_FP16)
#ifdef ARGMIN
    return (T)(INFINITY);
#else
    return (T)(-INFINITY);
#endif
#else
#ifdef ARGMIN
    return (T)(INFINITY);
#else
    return (T)(-INFINITY);
#endif
#endif
}

inline bool argmax_better(T aVal, int aIdx, T bVal, int bIdx) {
    if (bIdx < 0) return true;
    if (aIdx < 0) return false;
#ifdef ARGMIN
    if (aVal < bVal) return true;
    if (aVal > bVal) return false;
#else
    if (aVal > bVal) return true;
    if (aVal < bVal) return false;
#endif
    return aIdx < bIdx;
}

inline void argmax_reduce_tg(T bestVal, int bestIdx, uint tiisg, uint sgitg,
                             threadgroup T* shVal, threadgroup int* shIdx,
                             thread T& outVal, thread int& outIdx) {
    const T initWorst = argmax_init_worst();
    const T vReduce = (bestIdx >= 0) ? bestVal : initWorst;
#ifdef ARGMIN
    const T sgBest = simd_min(vReduce);
#else
    const T sgBest = simd_max(vReduce);
#endif
    const int INF_IDX = 2147483647;
    const int cand = (bestIdx >= 0 && bestVal == sgBest) ? bestIdx : INF_IDX;
    const int sgIdx = simd_min(cand);
    if (tiisg == 0) {
        shVal[sgitg] = sgBest;
        shIdx[sgitg] = (sgIdx == INF_IDX) ? -1 : sgIdx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        const bool valid = (tiisg < 8) && (shIdx[tiisg] >= 0);
        const T lv = valid ? shVal[tiisg] : initWorst;
        const int li = valid ? shIdx[tiisg] : -1;
#ifdef ARGMIN
        const T fv = simd_min(lv);
#else
        const T fv = simd_max(lv);
#endif
        const int fcand = (li >= 0 && lv == fv) ? li : INF_IDX;
        const int fi = simd_min(fcand);
        outVal = fv;
        outIdx = (fi == INF_IDX) ? -1 : fi;
    }
}

kernel void argmax_stage1(device T* partialVal [[buffer(0)]],
                          device int* partialIdx [[buffer(1)]],
                          const device T* inData [[buffer(2)]],
                          constant FastArgMaxParam& p [[buffer(3)]],
                          uint3 tgp [[threadgroup_position_in_grid]],
                          uint tiisg [[thread_index_in_simdgroup]],
                          uint sgitg [[simdgroup_index_in_threadgroup]]) {
    const int blockId = int(tgp.x);
    const int row = int(tgp.y);
    const int tid = int(sgitg * 32 + tiisg);
    const int start = blockId * p.blockSize;
    const int end = min(start + p.blockSize, p.mid);
    const device T* rowIn = inData + (long)row * p.mid;

    T bestVal = argmax_init_worst();
    int bestIdx = -1;

    if (p.vecAligned != 0) {
        const int vecEnd = end & ~3;
        for (int i = start + tid * 4; i < vecEnd; i += 256 * 4) {
            T4 v = *(const device T4*)(rowIn + i);
            if (argmax_better(v[0], i + 0, bestVal, bestIdx)) { bestVal = v[0]; bestIdx = i + 0; }
            if (argmax_better(v[1], i + 1, bestVal, bestIdx)) { bestVal = v[1]; bestIdx = i + 1; }
            if (argmax_better(v[2], i + 2, bestVal, bestIdx)) { bestVal = v[2]; bestIdx = i + 2; }
            if (argmax_better(v[3], i + 3, bestVal, bestIdx)) { bestVal = v[3]; bestIdx = i + 3; }
        }
        const int tailIdx = vecEnd + tid;
        if (tailIdx >= start && tailIdx < end) {
            T v = rowIn[tailIdx];
            if (argmax_better(v, tailIdx, bestVal, bestIdx)) { bestVal = v; bestIdx = tailIdx; }
        }
    } else {
        for (int i = start + tid; i < end; i += 256) {
            T v = rowIn[i];
            if (argmax_better(v, i, bestVal, bestIdx)) { bestVal = v; bestIdx = i; }
        }
    }

    threadgroup T shVal[8];
    threadgroup int shIdx[8];
    T outVal;
    int outIdx;
    argmax_reduce_tg(bestVal, bestIdx, tiisg, sgitg, shVal, shIdx, outVal, outIdx);
    if (sgitg == 0 && tiisg == 0) {
        const int outPos = row * p.numBlocks + blockId;
        partialVal[outPos] = outVal;
        partialIdx[outPos] = outIdx;
    }
}

kernel void argmax_stage2(device int* outIndex [[buffer(0)]],
                          const device T* partialVal [[buffer(1)]],
                          const device int* partialIdx [[buffer(2)]],
                          constant FastArgMaxParam& p [[buffer(3)]],
                          uint3 tgp [[threadgroup_position_in_grid]],
                          uint tiisg [[thread_index_in_simdgroup]],
                          uint sgitg [[simdgroup_index_in_threadgroup]]) {
    const int row = int(tgp.x);
    const int tid = int(sgitg * 32 + tiisg);
    const int base = row * p.numBlocks;
    T bestVal = argmax_init_worst();
    int bestIdx = -1;
    for (int i = tid; i < p.numBlocks; i += 256) {
        const int idx = partialIdx[base + i];
        if (idx >= 0) {
            const T v = partialVal[base + i];
            if (argmax_better(v, idx, bestVal, bestIdx)) {
                bestVal = v;
                bestIdx = idx;
            }
        }
    }
    threadgroup T shVal[8];
    threadgroup int shIdx[8];
    T outVal;
    int outIdx;
    argmax_reduce_tg(bestVal, bestIdx, tiisg, sgitg, shVal, shIdx, outVal, outIdx);
    if (sgitg == 0 && tiisg == 0) {
        outIndex[row] = outIdx;
    }
}
)metal";

class MetalArgMax : public MetalExecution {
private:
    int mAxis;
    bool mIsArgMin;
    int mGroupNumber = 0;
    bool mUseFast = false;
    int mFastMid = 0;
    int mFastOutside = 0;
    int mFastNumBlocks = 0;
    int mFastVecAligned = 0;
    int mRecordedFast = -1;
    int mRecordedMid = 0;
    int mRecordedOutside = 0;
    int mRecordedNumBlocks = 0;
    int mRecordedGroupNumber = 0;
    id<MTLBuffer> mParam = nil;
    id<MTLBuffer> mFastParam = nil;
    id<MTLBuffer> mPartialVal = nil;
    id<MTLBuffer> mPartialIdx = nil;
    id<MTLComputePipelineState> mPipeline = nil;
    id<MTLComputePipelineState> mFastStage1 = nil;
    id<MTLComputePipelineState> mFastStage2 = nil;

    bool ensureFastPipelines(MetalBackend* mtbn, const Tensor* input) {
        if (mFastStage1 != nil && mFastStage2 != nil) {
            return true;
        }
        const bool useFp16 = mtbn->useFp16InsteadFp32();
        NSString* type = MetalCast::getScalarType(input->getType(), useFp16);
        NSString* type4 = [type stringByAppendingString:@"4"];
        const std::string tStr([type UTF8String]);
        std::vector<std::string> k1 = {"argmax_fast_s1", tStr};
        std::vector<std::string> k2 = {"argmax_fast_s2", tStr};
        if (mIsArgMin) {
            k1.emplace_back("argmin");
            k2.emplace_back("argmin");
        }
        auto rt = mtbn->runtime();
        auto p1 = rt->findPipeline(k1);
        auto p2 = rt->findPipeline(k2);
        if (p1 == nil || p2 == nil) {
            auto dic = [NSMutableDictionary dictionaryWithCapacity:5];
            [dic setValue:type forKey:@"T"];
            [dic setValue:type4 forKey:@"T4"];
            if (mIsArgMin) {
                [dic setValue:@"1" forKey:@"ARGMIN"];
            }
            if (input->getType().code == halide_type_int && input->getType().bits == 32) {
                [dic setValue:@"1" forKey:@"IS_INT"];
            } else if (useFp16) {
                [dic setValue:@"1" forKey:@"USE_FP16"];
            }
            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            opts.preprocessorMacros = dic;
            if (p1 == nil) {
                p1 = mtbn->makeComputePipelineWithSourceOption(gArgMaxFastTemplate, "argmax_stage1", opts);
                if (p1 != nil) {
                    rt->insertPipeline(k1, p1);
                }
            }
            if (p2 == nil) {
                p2 = mtbn->makeComputePipelineWithSourceOption(gArgMaxFastTemplate, "argmax_stage2", opts);
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
    MetalArgMax(id<MTLComputePipelineState> pipeline, int axis, bool isArgMin, Backend *bn)
        : MetalExecution(bn), mAxis(axis), mIsArgMin(isArgMin), mPipeline(pipeline) {
        auto mtbn = static_cast<MetalBackend *>(bn);
        auto context = (__bridge MNNMetalContext *)mtbn->context();
        mParam = [context newDeviceBuffer:sizeof(int) * 4 access:CPUWriteOnly];
    }

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto input  = inputs[0];
        auto mtbn = static_cast<MetalBackend*>(backend());
        auto axis = mAxis;
        if (axis < 0) {
            axis = input->dimensions() + axis;
        }
        int inside = 1;
        int outside = 1;
        int mid = input->length(axis);
        for (int i = 0; i < axis; ++i) {
            outside *= input->length(i);
        }
        for (int i = axis + 1; i < input->dimensions(); ++i) {
            inside *= input->length(i);
        }

        const auto dtype = input->getType();
        const bool typeOk = (dtype.code == halide_type_float) ||
                            (dtype.code == halide_type_int && dtype.bits == 32);
        mUseFast = false;
        if (inside == 1 && mid >= 4096 && outside >= 1 && outside <= 16 && typeOk &&
            mtbn->runtime()->supportSimdGroupReduce() && ensureFastPipelines(mtbn, input)) {
            int numBlocks = std::min(256, UP_DIV(mid, 1024));
            if (numBlocks < 1) {
                numBlocks = 1;
            }
            int blockSize = UP_DIV(mid, numBlocks);
            // Round blockSize up to multiple of 4 so vector loads stay aligned per block.
            blockSize = ((blockSize + 3) / 4) * 4;
            numBlocks = UP_DIV(mid, blockSize);
            const size_t elemBytes = (dtype.code == halide_type_float && mtbn->useFp16InsteadFp32()) ? 2 : 4;
            const size_t valBytes = (size_t)outside * numBlocks * elemBytes;
            const size_t idxBytes = (size_t)outside * numBlocks * sizeof(int);
            auto ctx = (__bridge MNNMetalContext*)mtbn->context();
            if (mFastParam == nil) {
                mFastParam = [ctx newDeviceBuffer:sizeof(int) * 4 access:CPUWriteOnly];
            }
            if (mPartialVal == nil || mPartialVal.length < valBytes) {
                mPartialVal = [ctx newDeviceBuffer:valBytes access:CPUTransparent];
            }
            if (mPartialIdx == nil || mPartialIdx.length < idxBytes) {
                mPartialIdx = [ctx newDeviceBuffer:idxBytes access:CPUTransparent];
            }
            if (mFastParam != nil && mPartialVal != nil && mPartialIdx != nil) {
                const bool offsetAligned = (TensorUtils::getDescribeOrigin(input)->offset & 15) == 0;
                const int vecAligned = (offsetAligned && (outside == 1 || (mid & 3) == 0)) ? 1 : 0;
                auto p = reinterpret_cast<int*>([mFastParam contents]);
                p[0] = mid;
                p[1] = numBlocks;
                p[2] = blockSize;
                p[3] = vecAligned;
                mFastMid = mid;
                mFastOutside = outside;
                mFastNumBlocks = numBlocks;
                mFastVecAligned = vecAligned;
                mUseFast = true;
                return NO_ERROR;
            }
        }

        auto total = outside * inside;
        int outsideParallel = 1;
        int reduceAxis = 1;
        if (total >= 256) {
            reduceAxis = 1;
            outsideParallel = 256;
        } else if (total < 16) {
            reduceAxis = 256;
            outsideParallel = 1;
        } else {
            reduceAxis = 16;
            outsideParallel = 16;
        }

        {
            auto Argmax = reinterpret_cast<int*>([mParam contents]);
            Argmax[0] = inside;
            Argmax[1] = mid;
            Argmax[2] = outside;
            Argmax[3] = reduceAxis;
        }
        mGroupNumber = UP_DIV(total, outsideParallel);
        return NO_ERROR;
    }

    virtual bool onReplayUpdate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if ((mUseFast ? 1 : 0) != mRecordedFast) {
            return false;
        }
        if (mUseFast) {
            return mFastMid == mRecordedMid && mFastOutside == mRecordedOutside &&
                   mFastNumBlocks == mRecordedNumBlocks;
        }
        return mGroupNumber == mRecordedGroupNumber;
    }

    virtual void onEncode(const std::vector<Tensor *>& inputs, const std::vector<Tensor *>& outputs,
                          id<MTLComputeCommandEncoder> encoder) override {
        mRecordedFast = mUseFast ? 1 : 0;
        mRecordedMid = mFastMid;
        mRecordedOutside = mFastOutside;
        mRecordedNumBlocks = mFastNumBlocks;
        mRecordedGroupNumber = mGroupNumber;
        if (mUseFast) {
            [encoder setComputePipelineState:mFastStage1];
            [encoder setBuffer:mPartialVal offset:0 atIndex:0];
            [encoder setBuffer:mPartialIdx offset:0 atIndex:1];
            MetalBackend::setTensor(inputs[0], encoder, 2);
            [encoder setBuffer:mFastParam offset:0 atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(mFastNumBlocks, mFastOutside, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [encoder setComputePipelineState:mFastStage2];
            MetalBackend::setTensor(outputs[0], encoder, 0);
            [encoder setBuffer:mPartialVal offset:0 atIndex:1];
            [encoder setBuffer:mPartialIdx offset:0 atIndex:2];
            [encoder setBuffer:mFastParam offset:0 atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(mFastOutside, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            return;
        }
        [encoder setComputePipelineState:mPipeline];
        MetalBackend::setTensor(outputs[0], encoder, 0);
        MetalBackend::setTensor(inputs[0], encoder, 1);
        [encoder setBuffer:mParam offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(mGroupNumber, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
};

class MetalArgMaxCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *bn, const std::vector<Tensor *> &outputs) const {
        if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            // Don't support legency version
            return nullptr;
        }
        auto axis = op->main_as_ArgMax()->axis();
        auto mtbn = static_cast<MetalBackend *>(bn);
        auto type = MetalCast::getScalarType(inputs[0]->getType(), mtbn->useFp16InsteadFp32());
        const bool isArgMin = (op->type() == OpType_ArgMin);
        std::vector<std::string> keys {
            "argmax",
            std::string([type UTF8String])
        };
        if (isArgMin) {
            keys.emplace_back("argmin");
        }
        auto pipeline = mtbn->runtime()->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
            if (!isArgMin) {
                compileOptions.preprocessorMacros = @{
                    @"T" : type,
                };
            } else {
                compileOptions.preprocessorMacros = @{
                    @"T" : type,
                    @"ARGMIN": @"1"
                };
            }
            pipeline = mtbn->makeComputePipelineWithSourceOption(gArgMaxTemplate, "main0", compileOptions);
            mtbn->runtime()->insertPipeline(keys, pipeline);
        }
        if (nil == pipeline) {
            MNN_ERROR("Create ArgMax pipeline error\n");
            return nullptr;
        }
        return new MetalArgMax(pipeline, axis, isArgMin, bn);
    }
};
REGISTER_METAL_OP_CREATOR(MetalArgMaxCreator, OpType_ArgMax);
REGISTER_METAL_OP_CREATOR(MetalArgMaxCreator, OpType_ArgMin);
};
#endif
