//
//  MetalDequantize.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalDequantize.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalDequantize::MetalDequantize(Backend *backend, const Dequantize *dq)
    : Execution(backend), mModeFormat(dq->modelFormat()), mType(dq->type()), mMode(dq->mode()) {
    // nothing to do
}

ErrorCode MetalDequantize::onMinCombined(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto &input0 = inputs[0], &input1 = inputs[1], &input2 = inputs[2], &output0 = outputs[0];
    auto encoder   = [context encoder];
    auto bandwidth = (MetalBandwidth){};
    switch (mType) {
        case DataType_DT_QUINT8:
            bandwidth = [context load:@"dequantize_min_combined_uint8" encoder:encoder];
            break;
        case DataType_DT_QUINT16:
            bandwidth = [context load:@"dequantize_min_combined_uint16" encoder:encoder];
            break;
        case DataType_DT_QINT8:
            bandwidth = [context load:@"dequantize_min_combined_int8" encoder:encoder];
            break;
        case DataType_DT_QINT16:
            bandwidth = [context load:@"dequantize_min_combined_int16" encoder:encoder];
            break;
        case DataType_DT_QINT32:
            bandwidth = [context load:@"dequantize_min_combined_int32" encoder:encoder];
            break;
        default:
            MNN_ASSERT(false); // unsupported type
            return NOT_SUPPORT;
    }
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input0->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input1->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input2->deviceId() offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output0->deviceId() offset:0 atIndex:3];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) output0->elementSize(), 1, 1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

ErrorCode MetalDequantize::onMinFirst(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto &input0 = inputs[0], &input1 = inputs[1], &input2 = inputs[2], &output0 = outputs[0];
    auto encoder   = [context encoder];
    auto bandwidth = (MetalBandwidth){};
    switch (mType) {
        case DataType_DT_QUINT8:
            bandwidth = [context load:@"dequantize_min_first_uint8" encoder:encoder];
            break;
        case DataType_DT_QUINT16:
            bandwidth = [context load:@"dequantize_min_first_uint16" encoder:encoder];
            break;
        case DataType_DT_QINT8:
            bandwidth = [context load:@"dequantize_min_first_int8" encoder:encoder];
            break;
        case DataType_DT_QINT16:
            bandwidth = [context load:@"dequantize_min_first_int16" encoder:encoder];
            break;
        case DataType_DT_QINT32:
            bandwidth = [context load:@"dequantize_min_first_int32" encoder:encoder];
            break;
        default:
            MNN_ASSERT(false); // unsupported type
            return NOT_SUPPORT;
    }
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input0->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input1->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input2->deviceId() offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output0->deviceId() offset:0 atIndex:3];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) output0->elementSize(), 1, 1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

ErrorCode MetalDequantize::onScaled(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto &input0 = inputs[0], &input1 = inputs[1], &input2 = inputs[2], &output0 = outputs[0];
    auto encoder   = [context encoder];
    auto bandwidth = (MetalBandwidth){};
    switch (mType) {
        case DataType_DT_QUINT8:
            bandwidth = [context load:@"dequantize_scaled_uint8" encoder:encoder];
            break;
        case DataType_DT_QUINT16:
            bandwidth = [context load:@"dequantize_scaled_uint16" encoder:encoder];
            break;
        case DataType_DT_QINT8:
            bandwidth = [context load:@"dequantize_scaled_int8" encoder:encoder];
            break;
        case DataType_DT_QINT16:
            bandwidth = [context load:@"dequantize_scaled_int16" encoder:encoder];
            break;
        case DataType_DT_QINT32:
            bandwidth = [context load:@"dequantize_scaled_int32" encoder:encoder];
            break;
        default:
            MNN_ASSERT(false); // unsupported type
            return NOT_SUPPORT;
    }
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input0->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input1->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input2->deviceId() offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output0->deviceId() offset:0 atIndex:3];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) output0->elementSize(), 1, 1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

ErrorCode MetalDequantize::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    switch (mMode) {
        case QuantizeMode_MIN_COMBINED:
            return onMinCombined(inputs, outputs);
        case QuantizeMode_MIN_FIRST:
            return onMinFirst(inputs, outputs);
        case QuantizeMode_SCALED:
            return onScaled(inputs, outputs);
        default:
            return NOT_SUPPORT;
    }
}

class MetalDequantizeCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalDequantize(backend, op->main_as_Dequantize());
    }
};
REGISTER_METAL_OP_CREATOR(MetalDequantizeCreator, OpType_Dequantize);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
