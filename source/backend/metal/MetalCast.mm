//
//  MetalCast.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalCast.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {
static const char* gCastTemplate =
        R"glsl(
    #include <metal_stdlib>
    using namespace metal;
    kernel void main0(const device T0 *in [[buffer(0)]],
                                device T1 *out      [[buffer(1)]],
                                device uint4& s   [[buffer(2)]],
                                uint3 gid               [[thread_position_in_grid]]) {
        if (gid.x < (uint)s.x) {
            int off = gid.x;
            T0 x = in[off];
            T1 y;
            y.x = x.x;
            y.y = x.y;
            y.z = x.z;
            y.w = x.w;
            TRANSOFRM;
            out[off] = y;
        }
    }
    )glsl";

MetalCast::MetalCast(Backend *backend, id<MTLComputePipelineState> pipeline)
    : MetalExecution(backend) {
    auto mtbn = static_cast<MetalBackend *>(backend);
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mPipeline = pipeline;
    mConstBuffer = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
}
ErrorCode MetalCast::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    auto input = inputs[0];
    auto element = input->elementSize();
    auto sizeDiv4 = UP_DIV(element, 4);
    ((int *)mConstBuffer.contents)[0] = sizeDiv4;
    mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(sizeDiv4, 1, 1)];
    return NO_ERROR;
}

void MetalCast::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    [encoder setComputePipelineState:mPipeline];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input)->extra.offset atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:1];
    [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
}
static DataType _mapDataType(DataType src) {
    if (DataType_DT_BOOL == src) {
        return DataType_DT_INT32;
    }
    if (DataType_DT_INT64 == src) {
        return DataType_DT_INT32;
    }
    if (DataType_DT_DOUBLE == src) {
        return DataType_DT_FLOAT;
    }
    return src;
}
NSString* MetalCast::getScalarType(const halide_type_t& srcType, bool useFp16) {
    NSString* T0 = nil;
    switch (srcType.code) {
        case halide_type_float:
            if (useFp16) {
                T0 = @"half";
            } else {
                T0 = @"float";
            }
            break;
        case halide_type_int:
        {
            if (srcType.bits == 32) {
                T0 = @"int";
            } else if (srcType.bits == 8) {
                T0 = @"char";
            } else {
                MNN_ERROR("Don't support ScalarType src : %d\n", srcType.code);
                return nullptr;
            }
            break;
        }
        case halide_type_uint:
        {
            if (srcType.bits == 32) {
                T0 = @"uint";
            } else if (srcType.bits == 8) {
                T0 = @"uchar";
            } else {
                MNN_ERROR("Don't support ScalarType src : %d\n", srcType.code);
                return nil;
            }
            break;
        }
        default:
            MNN_ERROR("Don't support ScalarType src : %d\n", srcType.code);
            return nil;
    }
    return T0;
}
NSString* MetalCast::getVecType(const halide_type_t& srcType, bool useFp16) {
    NSString* T0 = nil;
    switch (srcType.code) {
        case halide_type_float:
            if (useFp16) {
                T0 = @"half4";
            } else {
                T0 = @"float4";
            }
            break;
        case halide_type_int:
        {
            if (srcType.bits == 32) {
                T0 = @"int4";
            } else if (srcType.bits == 8) {
                T0 = @"char4";
            } else {
                MNN_ERROR("Don't support cast src : %d\n", srcType.code);
                return nullptr;
            }
            break;
        }
        case halide_type_uint:
        {
            if (srcType.bits == 32) {
                T0 = @"uint4";
            } else if (srcType.bits == 8) {
                T0 = @"uchar4";
            } else {
                MNN_ERROR("Don't support cast src : %d\n", srcType.code);
                return nil;
            }
            break;
        }
        default:
            MNN_ERROR("Don't support cast src : %d\n", srcType.code);
            return nil;
    }
    return T0;
}
class MetalCastCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        auto mtbn = static_cast<MetalBackend *>(backend);
        NSString* T0 = nil;
        NSString* T1 = nil;
        NSString* TRANSOFRM = @"";
        auto dstT = op->main_as_CastParam()->dstT();
        if (dstT == DataType_DT_BOOL) {
            TRANSOFRM = @"y=select(int4(0),int4(1),y>0);";
        }
        auto dstType = _mapDataType(dstT);
        bool useFp16 = mtbn->useFp16InsteadFp32();
        switch (dstType) {
            case DataType_DT_FLOAT:
                if (useFp16) {
                    T1 = @"half4";
                } else {
                    T1 = @"float4";
                }
                break;
            case DataType_DT_INT8:
                T1 = @"char4";
                break;
            case DataType_DT_UINT8:
                T1 = @"uchar4";
                break;
            case DataType_DT_INT32:
                T1 = @"int4";
                break;
            default:
                MNN_ERROR("Don't support cast dst : %d\n", dstType);
                return nullptr;
                break;
        }
        auto srcType = inputs[0]->getType();
        T0 = MetalCast::getVecType(srcType, useFp16);
        std::vector<std::string> keys = {
            std::string([T0 UTF8String]),
            std::string([T1 UTF8String]),
            std::string([TRANSOFRM UTF8String]),
            "cast"
        };
        auto pipeline = mtbn->runtime()->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
            compileOptions.preprocessorMacros = @{
                @"T0" : T0,
                @"T1" : T1,
                @"TRANSOFRM" : TRANSOFRM
            };
            pipeline = mtbn->makeComputePipelineWithSourceOption(gCastTemplate, "main0", compileOptions);
            mtbn->runtime()->insertPipeline(keys, pipeline);
        }
        if (nil == pipeline) {
            MNN_ERROR("Create Cast execution error for metal\n");
            return nullptr;
        }
        return new MetalCast(backend, pipeline);
    }
};
REGISTER_METAL_OP_CREATOR(MetalCastCreator, OpType_Cast);


static const char* gSelectTemplate = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
kernel void main0(device T* uOutput [[buffer(0)]], const device int* uSelect [[buffer(1)]], const device T* uInput0 [[buffer(2)]], const device T* uInput1 [[buffer(3)]], constant int4& uStride [[buffer(4)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    int i = int(gl_GlobalInvocationID.x);
    if (i < uStride.w)
    {
        if (uSelect[uStride.x*i] > 0) {
            uOutput[i] = uInput0[uStride.y*i];
        } else {
            uOutput[i] = uInput1[uStride.z*i];
        }
    }
}
)metal";


class MetalSelect : public MetalCast {
public:
    MetalSelect(Backend *backend, id<MTLComputePipelineState> pipeline) : MetalCast(backend, pipeline) {
        // Do nothing
    }
    virtual ~MetalSelect() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto backend = static_cast<MetalBackend *>(this->backend());
        auto context = (__bridge MNNMetalContext *)backend->context();
        auto inSize0 = inputs[0]->elementSize();
        auto inSize1 = inputs[1]->elementSize();
        auto inSize2 = inputs[2]->elementSize();
        auto outSize = outputs[0]->elementSize();

        auto param = reinterpret_cast<int*>(mConstBuffer.contents);
        param[0] = inSize0 > 1 ? 1 : 0;
        param[1] = inSize1 > 1 ? 1 : 0;
        param[2] = inSize2 > 1 ? 1 : 0;
        param[3] = outSize;
        mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(outSize, 1, 1)];
        return NO_ERROR;
    }
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override {
        auto backend = static_cast<MetalBackend *>(this->backend());
        auto context = (__bridge MNNMetalContext *)backend->context();
        [encoder setComputePipelineState:mPipeline];
        MetalBackend::setTensor(outputs[0], encoder, 0);
        MetalBackend::setTensor(inputs[0], encoder, 1);
        MetalBackend::setTensor(inputs[1], encoder, 2);
        MetalBackend::setTensor(inputs[2], encoder, 3);
        [encoder setBuffer:mConstBuffer offset:0 atIndex:4];
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
    }
};

class MetalSelectCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        auto mtbn = static_cast<MetalBackend *>(backend);
        NSString* T = MetalCast::getScalarType(outputs[0]->getType(), mtbn->useFp16InsteadFp32());
        std::vector<std::string> keys = {
            std::string([T UTF8String]),
            "select"
        };
        auto pipeline = mtbn->runtime()->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
            compileOptions.preprocessorMacros = @{
                @"T" : T,
            };
            pipeline = mtbn->makeComputePipelineWithSourceOption(gSelectTemplate, "main0", compileOptions);
            mtbn->runtime()->insertPipeline(keys, pipeline);
        }
        if (nil == pipeline) {
            MNN_ERROR("Create Select execution error for metal\n");
            return nullptr;
        }
        return new MetalSelect(backend, pipeline);
    }
};
REGISTER_METAL_OP_CREATOR(MetalSelectCreator, OpType_Select);


class MetalRange : public MetalSelect {
public:
    MetalRange(Backend *backend, id<MTLComputePipelineState> pipeline) : MetalSelect(backend, pipeline) {
        // Do nothing
    }
    virtual ~MetalRange() = default;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override {
        auto backend = static_cast<MetalBackend *>(this->backend());
        auto context = (__bridge MNNMetalContext *)backend->context();
        [encoder setComputePipelineState:mPipeline];
        MetalBackend::setTensor(outputs[0], encoder, 0);
        MetalBackend::setTensor(inputs[0], encoder, 1);
        MetalBackend::setTensor(inputs[2], encoder, 2);
        [encoder setBuffer:mConstBuffer offset:0 atIndex:3];
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
    }

};


static const char* gRangeTemplate = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
kernel void main0(device T* uOutput [[buffer(0)]], const device T* uStart [[buffer(1)]], const device T* uDelta [[buffer(2)]], constant int4& uSize [[buffer(3)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    int i = int(gl_GlobalInvocationID.x);
    if(i < uSize.w) {
        uOutput[i] = (T)(i) * uDelta[0] + uStart[0];
    }
}
)metal";
class MetalRangeCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        auto mtbn = static_cast<MetalBackend *>(backend);
        NSString* T = MetalCast::getScalarType(outputs[0]->getType(), mtbn->useFp16InsteadFp32());
        std::vector<std::string> keys = {
            std::string([T UTF8String]),
            "range"
        };
        auto pipeline = mtbn->runtime()->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
            compileOptions.preprocessorMacros = @{
                @"T" : T,
            };
            pipeline = mtbn->makeComputePipelineWithSourceOption(gRangeTemplate, "main0", compileOptions);
            mtbn->runtime()->insertPipeline(keys, pipeline);
        }
        if (nil == pipeline) {
            MNN_ERROR("Create Select execution error for metal\n");
            return nullptr;
        }
        return new MetalRange(backend, pipeline);
    }
};
REGISTER_METAL_OP_CREATOR(MetalRangeCreator, OpType_Range);

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
