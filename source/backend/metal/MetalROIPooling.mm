//
//  MetalROIPooling.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalROIPooling.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalROIPooling::MetalROIPooling(Backend *backend, float spatialScale)
    : MetalExecution(backend), mSpatialScale(spatialScale) {
    // nothing to do
}
ErrorCode MetalROIPooling::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], roi = inputs[1], output = outputs[0];
    int iw = input->width(), ih = input->height();
    int ow = output->width(), oh = output->height(), oz = UP_DIV(output->channel(), 4), ob = output->batch();

    auto shape                   = [context newDeviceBuffer:7 * sizeof(int) + sizeof(float) access:CPUWriteOnly];
    ((int *)shape.contents)[0]   = iw;
    ((int *)shape.contents)[1]   = ih;
    ((int *)shape.contents)[2]   = iw * ih;
    ((int *)shape.contents)[3]   = ow;
    ((int *)shape.contents)[4]   = oh;
    ((int *)shape.contents)[5]   = ow * oh;
    ((int *)shape.contents)[6]   = oz;
    ((float *)shape.contents)[7] = mSpatialScale;
    mShape = shape;
    return NO_ERROR;
}

void MetalROIPooling::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], roi = inputs[1], output = outputs[0];
    int iw = input->width(), ih = input->height();
    int ow = output->width(), oh = output->height(), oz = UP_DIV(output->channel(), 4), ob = output->batch();

    auto bandwidth = [context load:@"ROI_pooling" encoder:encoder fp16:backend->useFp16InsteadFp32()];

    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input)->extra.offset atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)roi->deviceId())->getBuffer() offset:TensorUtils::getDescribe(roi)->extra.offset atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:2];
    [encoder setBuffer:mShape offset:0 atIndex:3];
}

class MetalROIPoolingCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        return new MetalROIPooling(backend, op->main_as_RoiParameters()->spatialScale());
    }
};
REGISTER_METAL_OP_CREATOR(MetalROIPoolingCreator, OpType_ROIPooling);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
