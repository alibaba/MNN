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
    auto mtbn = static_cast<MetalBackend *>(backend);
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mShape = [context newDeviceBuffer:8 * sizeof(int) + sizeof(float) access:CPUWriteOnly];
    mPipeline = [context pipelineWithName:@"ROI_pooling" fp16:mtbn->useFp16InsteadFp32()];
}
ErrorCode MetalROIPooling::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], roi = inputs[1], output = outputs[0];
    int iw = input->width(), ih = input->height();
    int ow = output->width(), oh = output->height(), oz = UP_DIV(output->channel(), 4), ob = output->batch();

    ((int *)mShape.contents)[0]   = iw;
    ((int *)mShape.contents)[1]   = ih;
    ((int *)mShape.contents)[2]   = iw * ih;
    ((int *)mShape.contents)[3]   = input->batch();
    ((int *)mShape.contents)[4]   = ow;
    ((int *)mShape.contents)[5]   = oh;
    ((int *)mShape.contents)[6]   = ow * oh;
    ((int *)mShape.contents)[7]   = ob;
    ((float *)mShape.contents)[8] = mSpatialScale;
    return NO_ERROR;
}

void MetalROIPooling::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], roi = inputs[1], output = outputs[0];
    int iw = input->width(), ih = input->height();
    int ow = output->width(), oh = output->height(), oz = UP_DIV(output->channel(), 4), ob = output->batch();
    [encoder setComputePipelineState:mPipeline];
    MetalBackend::setTensor(input, encoder, 0);
    MetalBackend::setTensor(roi, encoder, 1);
    MetalBackend::setTensor(output, encoder, 2);
    [encoder setBuffer:mShape offset:0 atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(ow, 16), UP_DIV(oh, 16), ob * oz) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
}

class MetalROIPoolingCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        auto roi = inputs[1];
        if (TensorUtils::getDescribe(roi)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            // Don't support old roipooling
            return nullptr;
        }
        return new MetalROIPooling(backend, op->main_as_RoiParameters()->spatialScale());
    }
};
REGISTER_METAL_OP_CREATOR(MetalROIPoolingCreator, OpType_ROIPooling);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
