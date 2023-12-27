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
class MetalCastCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        auto mtbn = static_cast<MetalBackend *>(backend);
        MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
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
                    return nullptr;
                }
                break;
            }
            default:
                MNN_ERROR("Don't support cast src : %d\n", srcType.code);
                return nullptr;
        }

        compileOptions.preprocessorMacros = @{
            @"T0" : T0,
            @"T1" : T1,
            @"TRANSOFRM" : TRANSOFRM
        };
        auto pipeline = mtbn->makeComputePipelineWithSourceOption(gCastTemplate, "main0", compileOptions);
        if (nil == pipeline) {
            MNN_ERROR("Create Cast execution error for metal\n");
            return nullptr;
        }
        return new MetalCast(backend, pipeline);
    }
};
REGISTER_METAL_OP_CREATOR(MetalCastCreator, OpType_Cast);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
