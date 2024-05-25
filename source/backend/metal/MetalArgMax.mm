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

class MetalArgMax : public MetalExecution {
private:
    int mAxis;
    int mGroupNumber;
    id<MTLBuffer> mParam;
    id<MTLComputePipelineState> mPipeline;
public:
    MetalArgMax(id<MTLComputePipelineState> pipeline, int axis, Backend *bn) : MetalExecution(bn) {
        mPipeline = pipeline;
        mAxis = axis;
        auto mtbn = static_cast<MetalBackend *>(bn);
        auto context = (__bridge MNNMetalContext *)mtbn->context();
        mParam = [context newDeviceBuffer:sizeof(int) * 4 access:CPUWriteOnly];
    };
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto input  = inputs[0];
        auto output = outputs[0];

        auto inputFormat = TensorUtils::getDescribe(input)->dimensionFormat;
        auto axis = mAxis;
        if (axis < 0) {
            axis = input->dimensions() + axis;
        }
        int inside = 1;
        int outside = 1;
        int mid = input->length(axis);
        for (int i=0; i<axis; ++i) {
            outside *= input->length(i);
        }
        for (int i=axis+1; i<input->dimensions(); ++i) {
            inside *= input->length(i);
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
        
        // gpu param
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
    virtual void onEncode(const std::vector<Tensor *>& inputs, const std::vector<Tensor *>& outputs,
                          id<MTLComputeCommandEncoder> encoder) override {
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
        std::vector<std::string> keys {
            "argmax",
            std::string([type UTF8String])
        };
        if (op->type() == OpType_ArgMin) {
            keys.emplace_back("argmin");
        }
        auto pipeline = mtbn->runtime()->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
            if (op->type() != OpType_ArgMin) {
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
        return new MetalArgMax(pipeline, axis, bn);
    }
};
REGISTER_METAL_OP_CREATOR(MetalArgMaxCreator, OpType_ArgMax);
REGISTER_METAL_OP_CREATOR(MetalArgMaxCreator, OpType_ArgMin);
};
#endif
