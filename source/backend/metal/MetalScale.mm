//
//  MetalScale.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalScale.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalScale::MetalScale(Backend *backend, const Scale *scale) : Execution(backend) {
    auto context  = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    auto channel4 = UP_DIV(scale->channels(), 4) * 4;
    mScale = [context newDeviceBuffer:channel4 * sizeof(float) bytes:scale->scaleData()->data() access:CPUWriteOnly];
    mBias  = scale->biasData()
                ? [context newDeviceBuffer:channel4 * sizeof(float) bytes:scale->biasData()->data() access:CPUWriteOnly]
                : [context newDeviceBuffer:channel4 * sizeof(float) access:CPUTransparent];
}

ErrorCode MetalScale::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];

    // shape
    auto tf = input->getDimensionType() == Tensor::TENSORFLOW;
    int w   = tf ? output->tfWidth() : output->width();
    int h   = tf ? output->tfHeight() : output->height();
    int c   = tf ? output->tfChannel() : output->channel();
    int z   = UP_DIV(c, 4);

    auto shape                 = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    ((int *)shape.contents)[0] = w * h;
    
    // tensorflow
    if (tf) {
        ((int *)shape.contents)[1] = c;
        ((int *)shape.contents)[2] = output->tfBatch();

        auto encoder   = [context encoder];
        auto bandwidth = [context load:@"scale_tf" encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
        [encoder setBuffer:shape offset:0 atIndex:2];
        [encoder setBuffer:mScale offset:0 atIndex:3];
        [encoder setBuffer:mBias offset:0 atIndex:4];
        [context dispatchEncoder:encoder
                         threads:{ (NSUInteger) c, (NSUInteger)w * h * output->tfBatch(), 1 }
                       bandwidth:bandwidth];
        [encoder endEncoding];
        MNN_PRINT_ENCODER(context, encoder);
    }
    // caffe
    else {
        ((int *)shape.contents)[1] = z;
        ((int *)shape.contents)[2] = output->batch();

        auto encoder   = [context encoder];
        auto bandwidth = [context load:@"scale_ca" encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
        [encoder setBuffer:shape offset:0 atIndex:2];
        [encoder setBuffer:mScale offset:0 atIndex:3];
        [encoder setBuffer:mBias offset:0 atIndex:4];
        [context dispatchEncoder:encoder
                         threads:{ (NSUInteger)w * h, (NSUInteger)z * output->batch(), 1 }
                       bandwidth:bandwidth];
        [encoder endEncoding];
        MNN_PRINT_ENCODER(context, encoder);
    }
    return NO_ERROR;
}

class MetalScaleCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalScale(backend, op->main_as_Scale());
    }
};
static MetalCreatorRegister<MetalScaleCreator> __ec(OpType_Scale);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
