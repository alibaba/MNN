//
//  MetalUnary.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalCast.hpp"
#import "core/Macro.h"
#import "MNNMetalContext.h"
#import "MetalBackend.hpp"
#import "MetalUnary.hpp"

#if MNN_METAL_ENABLED
namespace MNN {
static const char* gBinaryTemplate = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct unary_shape {
    int width;
    int height;
    int size;
};

static inline float4 MNNEXP(float4 tmp) {
    tmp = clamp(tmp, (float4)-87.0, (float4)87.0);
    return exp(tmp);
}

static inline float4 MNNTANH(float4 value) {
    float4 tmp = MNNEXP((float4)(2.0)*value);
    return (tmp-(float4)1.0)/(tmp+(float4)1.0);
}
static inline float4 neg(float4 value) { return -value; }
static inline float4 square(float4 value) { return value * value; }
static inline float4 expm1(float4 value) {return MNNEXP(value) - 1;}
static inline float4 reciprocal(float4 value) {return 1.0/(value);}
static inline float4 sigmoid(float4 value) {return 1.f / (1.f + MNNEXP(-value));}
static inline float4 log1p(float4 value) {return log(1.f + value);}
static inline float4 hardswish(float4 value) {
    return (float4)(1.0/6.0) * (value * min(max(value+(float4)3, 0), (float4)6));
}
static inline float4 gelu(float4 value) {
    float4 temp = (float4)0.044715 * value * value * value;
    temp = (float4)0.79788458 * (temp + value);
    temp = clamp(temp, (float4)-5.0, (float4)5.0);
    float4 result = ((float4)1.0 + MNNTANH(temp)) * value * (float4)0.5;
    return result;
}

kernel void main0(const device T *in [[buffer(0)]], \
                            device T *out      [[buffer(1)]], \
                            device unary_shape& s   [[buffer(2)]], \
                            uint3 gid               [[thread_position_in_grid]]) { \
    if (gid.x < (uint)s.width) { \
        int off = gid.z * s.size + gid.y * s.width + gid.x; \
        out[off] = (T)(FUNC((float4)(in[off]))); \
    } \
}
)metal";

static NSString *kernelForType(UnaryOpOperation type) {
#define op_case(type, imp)        \
    case UnaryOpOperation_##type: \
        return @#imp
    switch (type) {
        op_case(ABS, abs);
        op_case(NEG, neg);
        op_case(FLOOR, floor);
        op_case(CEIL, ceil);
        op_case(ROUND, round);
        op_case(SQUARE, square);
        op_case(SQRT, sqrt);
        op_case(RSQRT, rsqrt);
        op_case(EXP, MNNEXP);
        op_case(EXPM1, expm1);
        op_case(LOG, log);
        op_case(SIN, sin);
        op_case(COS, cos);
        op_case(TAN, tan);
        op_case(TANH, MNNTANH);
        op_case(SIGMOID, sigmoid);
        op_case(ASIN, asin);
        op_case(ACOS, acos);
        op_case(ATAN, atan);
        op_case(SIGN, sign);
        op_case(RECIPROCAL, reciprocal);
        op_case(LOG1P, log1p);
        op_case(ACOSH, acosh);
        op_case(COSH, cosh);
        op_case(SINH, sinh);
        op_case(ASINH, asinh);
        op_case(ATANH, atanh);
        op_case(HARDSWISH, hardswish);
        op_case(GELU, gelu);
        op_case(GELU_STANDARD, gelu);
        default:
            FUNC_PRINT_ALL(EnumNameUnaryOpOperation(type), s);
            return nil;
    }
}

MetalUnary::MetalUnary(Backend *backend, id<MTLComputePipelineState> pipeline) : MetalExecution(backend) {
    auto mtbn = static_cast<MetalBackend *>(backend);
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mConstBuffer                 = [context newDeviceBuffer:3 * sizeof(int) access:CPUWriteOnly];
    mPipeline = pipeline;
}
ErrorCode MetalUnary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    auto input = inputs[0];
    auto element = input->elementSize();
    auto sizeDiv4 = UP_DIV(element, 4);
    ((int *)mConstBuffer.contents)[0] = sizeDiv4;
    mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(sizeDiv4, 1, 1)];
    return NO_ERROR;
}

void MetalUnary::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto input = inputs[0], output = outputs[0];
    [encoder setComputePipelineState:mPipeline];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input)->extra.offset atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:1];
    [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
}

class MetalUnaryCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        UnaryOpOperation optype;
        if (op->type() == OpType_TanH) {
            optype = UnaryOpOperation_TANH;
        } else if (op->type() == OpType_Sigmoid) {
            optype = UnaryOpOperation_SIGMOID;
        } else {
            optype = op->main_as_UnaryOp()->opType();
        }
        if (UnaryOpOperation_ERF == optype || UnaryOpOperation_ERFC == optype || UnaryOpOperation_ERFINV == optype) {
            return nullptr;
        }
        auto kernel = kernelForType(optype);
        if (nil == kernel) {
            return nullptr;
        }
        auto mtbn = static_cast<MetalBackend *>(backend);
        NSString* T = MetalCast::getVecType(outputs[0]->getType(), mtbn->useFp16InsteadFp32());
        std::vector<std::string> keys = {
            std::string([T UTF8String]),
            std::string([kernel UTF8String]),
            "unary"
        };
        auto pipeline = mtbn->runtime()->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
            compileOptions.preprocessorMacros = @{
                @"T" : T,
                @"FUNC" : kernel,
            };
            pipeline = mtbn->makeComputePipelineWithSourceOption(gBinaryTemplate, "main0", compileOptions);
            mtbn->runtime()->insertPipeline(keys, pipeline);
        }
        if (nil == pipeline) {
            MNN_ERROR("Make Unary shader error\n");
            return nullptr;
        }
        return new MetalUnary(backend, pipeline);
    }
};
REGISTER_METAL_OP_CREATOR(MetalUnaryCreator, OpType_UnaryOp);
REGISTER_METAL_OP_CREATOR(MetalUnaryCreator, OpType_TanH);
REGISTER_METAL_OP_CREATOR(MetalUnaryCreator, OpType_Sigmoid);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
