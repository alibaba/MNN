//
//  MetalSoftmax.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MNNMetalContext.h"
#import "backend/metal/MetalSoftmax.hpp"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"
#import "core/TensorUtils.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalSoftmax::MetalSoftmax(Backend *backend, int32_t axis) : MetalExecution(backend), mAxis(axis) {
    auto mtbn = static_cast<MetalBackend *>(backend);
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mShapeBuffer               = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
}

ErrorCode MetalSoftmax::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    auto dimensions    = input->buffer().dimensions;
    auto realAxis      = mAxis < 0 ? dimensions + mAxis : mAxis;
    auto c4 = TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4; // even dims != 4
    auto reorder       = realAxis == 1 && c4;
    // shape
    auto inside = 1;
    auto flat = input->length(realAxis);
    auto axis = flat;
    auto outside = 1;
    
    for (int i = 0; i < realAxis; i++) {
        auto length = input->length(i);
        if (1 == i && c4) {
            length = UP_DIV(length, 4);
        }
        outside *= length;
    }
    for (int i = realAxis + 1; i < input->dimensions(); i++) {
        auto length = input->length(i);
        if (1 == i && c4) {
            length = UP_DIV(length, 4);
        }
        inside *= length;
    }
    if (reorder) {
        axis = UP_DIV(axis, 4);
    }
    ((int *)mShapeBuffer.contents)[0] = inside;
    ((int *)mShapeBuffer.contents)[1] = axis;
    ((int *)mShapeBuffer.contents)[2] = outside;
    ((int *)mShapeBuffer.contents)[3] = flat;
    
    // encode
    auto plane     = !(TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4);
    auto kernel = plane ? @"softmax_plane" : reorder ? @"softmax_on_reorder" : @"softmax_off_reorder";
    mPipeline = [context pipelineWithName:kernel fp16:backend->useFp16InsteadFp32()];
    
    mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake((NSUInteger)inside, (NSUInteger)outside, 1)];
    return NO_ERROR;
}

void MetalSoftmax::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto input = inputs[0], output = outputs[0];
    [encoder setComputePipelineState:mPipeline];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input)->extra.offset atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:1];
    [encoder setBuffer:mShapeBuffer offset:0 atIndex:2];

    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
    MNN_PRINT_ENCODER(context, encoder);
}

class MetalSoftmaxCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *> &outputs) const {
        auto softmax = op->main_as_Axis();
        return new MetalSoftmax(backend, softmax->axis());
    }
};
REGISTER_METAL_OP_CREATOR(MetalSoftmaxCreator, OpType_Softmax);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
