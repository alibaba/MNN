//
//  MetalUnary.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalUnary.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalUnary::MetalUnary(Backend *backend, UnaryOpOperation optype) : Execution(backend), mOpType(optype) {
    // nothing to do
}

static NSString *kernelForType(UnaryOpOperation type) {
#define op_case(type, imp)        \
    case UnaryOpOperation_##type: \
        return @"unary_" #imp "_x4"
    switch (type) {
        op_case(ABS, abs);
        op_case(NEG, neg);
        op_case(FLOOR, floor);
        op_case(CEIL, ceil);
        op_case(ROUND, round);
        op_case(SQUARE, square);
        op_case(SQRT, sqrt);
        op_case(RSQRT, rsqrt);
        op_case(EXP, exp);
        op_case(EXPM1, expm1);
        op_case(LOG, log);
        op_case(SIN, sin);
        op_case(COS, cos);
        op_case(TAN, tan);
        op_case(TANH, tanh);
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
        default:
            FUNC_PRINT_ALL(EnumNameUnaryOpOperation(type), s);
            return nil;
    }
}

ErrorCode MetalUnary::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();

    // prepare
    auto input = inputs[0], output = outputs[0];
    auto element = input->elementSize();
    auto sizeDiv4 = UP_DIV(element, 4);
    // create shape
    auto shape                 = [context newDeviceBuffer:3 * sizeof(int) access:CPUWriteOnly];
    ((int *)shape.contents)[0] = sizeDiv4;

    auto encoder   = [context encoder];
    auto bandwidth = [context load:kernelForType(mOpType) encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:shape offset:0 atIndex:2];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) sizeDiv4, (NSUInteger)1, (NSUInteger)1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalUnaryCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        if (op->type() == OpType_TanH) {
            return new MetalUnary(backend, UnaryOpOperation_TANH);
        }
        if (op->type() == OpType_Sigmoid) {
            return new MetalUnary(backend, UnaryOpOperation_SIGMOID);
        }
        auto optype = op->main_as_UnaryOp()->opType();
        if (UnaryOpOperation_ERF == optype || UnaryOpOperation_ERFC == optype || UnaryOpOperation_ERFINV == optype) {
            return nullptr;
        }
        return new MetalUnary(backend, optype);
    }
};
REGISTER_METAL_OP_CREATOR(MetalUnaryCreator, OpType_UnaryOp);
REGISTER_METAL_OP_CREATOR(MetalUnaryCreator, OpType_TanH);
REGISTER_METAL_OP_CREATOR(MetalUnaryCreator, OpType_Sigmoid);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
