//
//  MetalBinary.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalBinary.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalBinary::MetalBinary(Backend *backend, id<MTLComputePipelineState> pipeline, id<MTLComputePipelineState> plane) : MetalExecution(backend) {
    auto mtbn = static_cast<MetalBackend *>(backend);
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mConstBuffer             = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    mPipeline = pipeline;
    mPlanePipeline = plane;
}
ErrorCode MetalBinary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input0 = inputs[0], input1 = inputs[1], output = outputs[0];
    const int input0_data_count = TensorUtils::getRawSize(input0);
    const int input1_data_count = TensorUtils::getRawSize(input1);
    mUsePlane = false;

    int outdatacount = output->elementSize();
    ((int *)mConstBuffer.contents)[0] = input0_data_count == 1 ? 0 : 1;
    ((int *)mConstBuffer.contents)[1] = input1_data_count == 1 ? 0 : 1;
    ((int *)mConstBuffer.contents)[2] = outdatacount;
    mUsePlane = input0_data_count != 1 && input1_data_count != 1;
    mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(outdatacount, 1, 1)];
    return NO_ERROR;
}

void MetalBinary::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto input0 = inputs[0], input1 = inputs[1], output = outputs[0];
    if (mUsePlane) {
        [encoder setComputePipelineState:mPlanePipeline];
    } else {
        [encoder setComputePipelineState:mPipeline];
    }
    MetalBackend::setTensor(input0, encoder, 0);
    MetalBackend::setTensor(input1, encoder, 1);
    MetalBackend::setTensor(output, encoder, 2);
    [encoder setBuffer:mConstBuffer offset:0 atIndex:3];
    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
}

#define CHECK(t, i) if (originOp == t) return i;
NSString* MetalBinary::convert(int originOp, bool inputFloat) {
    if (BinaryOpOperation_MOD == originOp) {
        if (inputFloat) {
            return @"fmod((float)V0,(float)V1)";
        }
        /**
         auto res = x % y;
         if ((res < 0 && y > 0) || (res > 0 && y < 0)) {
             res += y;
         }
         */
        return @"select(V0%V1,(V0%V1)+V1,(V0%V1<0&&V1>0)||(V0%V1>0&&V1<0))";
    }
    CHECK(BinaryOpOperation_ADD, @"V0+V1");
    CHECK(BinaryOpOperation_ATAN2, @"atan2(V0,V1)");
    CHECK(BinaryOpOperation_SUB, @"V0-V1");
    CHECK(BinaryOpOperation_MUL, @"V0*V1");
    CHECK(BinaryOpOperation_FLOORMOD, @"V0-floor((float)V0/(float)V1)*V1");
    CHECK(BinaryOpOperation_FLOORDIV, @"floor((float)V0/(float)V1)");
    CHECK(BinaryOpOperation_MINIMUM, @"min(V0,V1)");
    CHECK(BinaryOpOperation_MAXIMUM, @"max(V0,V1)");
    CHECK(BinaryOpOperation_DIV, @"V1==0?0:V0/V1");
    CHECK(BinaryOpOperation_REALDIV, @"V1==0?0:V0/V1");
    CHECK(BinaryOpOperation_POW, @"pow(V0,V1)");
    CHECK(BinaryOpOperation_SquaredDifference, @"(V0-V1)*(V0-V1)");
    CHECK(BinaryOpOperation_EQUAL, @"(V0==V1)?1:0");
    CHECK(BinaryOpOperation_LESS, @"(V0<V1)?1:0");
    CHECK(BinaryOpOperation_LESS_EQUAL, @"(V0<=V1)?1:0");
    CHECK(BinaryOpOperation_GREATER, @"(V0>V1)?1:0");
    CHECK(BinaryOpOperation_GREATER_EQUAL, @"(V0>=V1)?1:0");
    CHECK(BinaryOpOperation_NOTEQUAL, @"(V0!=V1)?1:0");
    CHECK(BinaryOpOperation_LOGICALOR, @"V0||V1");
    CHECK(BinaryOpOperation_BITWISE_AND, @"V0&V1");
    CHECK(BinaryOpOperation_BITWISE_OR, @"V0|V1");
    CHECK(BinaryOpOperation_BITWISE_XOR, @"V0^V1");
    return nil;
}

static const char* gBinaryTemplate = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
kernel void binary(const device T0 *in0 [[buffer(0)]],
    const device T1 *in1 [[buffer(1)]], device T2 *out [[buffer(2)]], constant int4& s [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= s.z) return;
#ifdef SAME_SIZE
    auto V0 = in0[int(gid)];
    auto V1 = in1[int(gid)];
#else
    auto V0 = in0[s.x * int(gid)];
    auto V1 = in1[s.y * int(gid)];
#endif
    auto val = CUSTOM;
#ifdef RELU
    val = (val < (T2)0 ? (T2)0 : val);
#endif
    out[int(gid)] = val;
}
)metal";

class MetalBinaryCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        auto binaryop = op->main_as_BinaryOp();
        auto mtbn = static_cast<MetalBackend *>(backend);
        NSString* T2 = MetalCast::getScalarType(outputs[0]->getType(), mtbn->useFp16InsteadFp32());
        NSString* T0 = MetalCast::getScalarType(inputs[0]->getType(), mtbn->useFp16InsteadFp32());
        NSString* T1 = MetalCast::getScalarType(inputs[1]->getType(), mtbn->useFp16InsteadFp32());
        MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
        auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
        std::vector<std::string> keys = {
            std::string([T0 UTF8String]),
            std::string([T1 UTF8String]),
            std::string([T2 UTF8String]),
            std::to_string(binaryop->opType()),
            "binary"
        };
        [dic setValue:T0 forKey:@"T0"];
        [dic setValue:T1 forKey:@"T1"];
        [dic setValue:T2 forKey:@"T2"];
        if (binaryop->activationType() == 1) {
            keys.emplace_back("RELU");
            [dic setValue:@"1" forKey:@"RELU"];
        }
        auto pipeline = mtbn->runtime()->findPipeline(keys);
        auto plane = pipeline;
        if (nil == pipeline) {
            NSString* custom = MetalBinary::convert(binaryop->opType(), inputs[0]->getType().code == halide_type_float);
            if (nil == custom) {
                MNN_ERROR("Metal Don't support binary - %d \n", binaryop->opType());
                return nullptr;
            }
            [dic setValue:custom forKey:@"CUSTOM"];
            compileOptions.preprocessorMacros = dic;
            pipeline = mtbn->makeComputePipelineWithSourceOption(gBinaryTemplate, "binary", compileOptions);
            mtbn->runtime()->insertPipeline(keys, pipeline);
            keys.emplace_back("PLANE");
            [dic setValue:@"1" forKey:@"SAME_SIZE"];
            compileOptions.preprocessorMacros = dic;
            plane = mtbn->makeComputePipelineWithSourceOption(gBinaryTemplate, "binary", compileOptions);
            mtbn->runtime()->insertPipeline(keys, plane);
        } else {
            keys.emplace_back("PLANE");
            plane = mtbn->runtime()->findPipeline(keys);
        }
        if (nil == pipeline) {
            MNN_ERROR("Make Binary shader error\n");
            return nullptr;
        }
        return new MetalBinary(backend, pipeline, plane);
    }
};
REGISTER_METAL_OP_CREATOR(MetalBinaryCreator, OpType_BinaryOp);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
