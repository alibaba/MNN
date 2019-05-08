//
//  MetalUnary.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalUnary.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalUnary::MetalUnary(Backend *backend, UnaryOpOperation optype) : Execution(backend), mOpType(optype) {
    // nothing to do
}

static NSString *kernelForType(UnaryOpOperation type, BOOL simd) {
#define op_case(type, imp)        \
    case UnaryOpOperation_##type: \
        return simd ? @"unary_" #imp "_x4" : @"unary_" #imp "_x1"
    switch (type) {
        op_case(ABS, abs);
        op_case(NEG, neg);
        //            op_case(FLOOR, floor);
        op_case(CEIL, ceil);
        op_case(SQUARE, square);
        op_case(SQRT, sqrt);
        op_case(RSQRT, rsqrt);
        op_case(EXP, exp);
        //            op_case(LOG, log);
        //            op_case(SIN, sin);
        //            op_case(COS, cos);
        //            op_case(TAN, tan);
        //            op_case(ASIN, asin);
        //            op_case(ACOS, acos);
        //            op_case(ATAN, atan);
        //            op_case(RECIPROCAL, reciprocal);
        default:
            return nil;
    }
}

ErrorCode MetalUnary::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();

    // prepare
    auto input = inputs[0], output = outputs[0];
    auto tf = output->getDimensionType() == Tensor::TENSORFLOW;
    int w   = output->width();
    int h   = output->height();
    int c   = output->channel();
    int b   = output->batch();
    if (input->buffer().dimensions == 1) { //支持标量处理
        w = w == 0 ? 1 : w;
        h = h == 0 ? 1 : h;
        c = c == 0 ? 1 : c;
        b = b == 0 ? 1 : b;
    }
    int z = tf ? c * b : UP_DIV(c, 4) * b;

    // create shape
    auto shape                 = [context newDeviceBuffer:3 * sizeof(int) access:CPUWriteOnly];
    ((int *)shape.contents)[0] = w;
    ((int *)shape.contents)[1] = h;
    ((int *)shape.contents)[2] = w * h;

    auto encoder   = [context encoder];
    auto bandwidth = [context load:kernelForType(mOpType, !tf) encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:shape offset:0 atIndex:2];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) w, (NSUInteger)h, (NSUInteger)z } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalUnaryCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto optype = op->main_as_UnaryOp()->opType();
        return new MetalUnary(backend, optype);
    }
};
REGISTER_METAL_OP_CREATOR(MetalUnaryCreator, OpType_UnaryOp);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
