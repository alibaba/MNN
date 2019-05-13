//
//  MetalSpatialProduct.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalSpatialProduct.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalSpatialProduct::MetalSpatialProduct(Backend *backend) : Execution(backend) {
    // nothing to do
}

ErrorCode MetalSpatialProduct::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], weight = inputs[1], output = outputs[0];
    int w = output->width(), h = output->height(), z = UP_DIV(output->channel(), 4), b = output->batch();

    auto shape                 = [context newDeviceBuffer:2 * sizeof(int) access:CPUWriteOnly];
    ((int *)shape.contents)[0] = w * h;
    ((int *)shape.contents)[1] = z * b;
    
    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"spartial_product" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)weight->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:2];
    [encoder setBuffer:shape offset:0 atIndex:3];
    [context dispatchEncoder:encoder threads:{ (NSUInteger)w * h, (NSUInteger)z * b, 1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalSpatialProductCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalSpatialProduct(backend);
    }
};
REGISTER_METAL_OP_CREATOR(MetalSpatialProductCreator, OpType_SpatialProduct);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
