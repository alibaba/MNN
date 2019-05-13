//
//  MetalReshape.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalReshape.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"
#import "TensorUtils.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalReshape::MetalReshape(Backend *backend, MNN_DATA_FORMAT dimType) : Execution(backend), mDimType(dimType) {
    // nothing to do
}

ErrorCode MetalReshape::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0], output = outputs[0];
    if (input->getDimensionType() != Tensor::TENSORFLOW) {
        auto fmt = mDimType == MNN_DATA_FORMAT_NHWC ? MNN_DATA_FORMAT_NHWC : MNN_DATA_FORMAT_NCHW;
        mMiddle.reset(new Tensor);
        mCarbon.reset(new Tensor);

        if (mDimType == MNN_DATA_FORMAT_NHWC && input->dimensions() == 4) {
            mMiddle->buffer().dim[0].extent = input->buffer().dim[0].extent;
            mMiddle->buffer().dim[1].extent = input->buffer().dim[3].extent;
            mMiddle->buffer().dim[2].extent = input->buffer().dim[2].extent;
            mMiddle->buffer().dim[3].extent = input->buffer().dim[1].extent;
            TensorUtils::setLinearLayout(mMiddle.get());
        } else {
            TensorUtils::copyShape(input, mMiddle.get());
        }
        if (mDimType == MNN_DATA_FORMAT_NHWC && output->dimensions() == 4) {
            mCarbon->buffer().dim[0].extent = output->buffer().dim[0].extent;
            mCarbon->buffer().dim[1].extent = output->buffer().dim[3].extent;
            mCarbon->buffer().dim[2].extent = output->buffer().dim[2].extent;
            mCarbon->buffer().dim[3].extent = output->buffer().dim[1].extent;
            TensorUtils::setLinearLayout(mCarbon.get());
        } else {
            TensorUtils::copyShape(output, mCarbon.get());
        }

        TensorUtils::getDescribe(mMiddle.get())->dimensionFormat = fmt;
        TensorUtils::getDescribe(mCarbon.get())->dimensionFormat = fmt;
        mMiddle->buffer().dim[1].flags                           = 0;
        mCarbon->buffer().dim[1].flags                           = 0;

        // acquire buffer space
        auto backend = static_cast<MetalBackend *>(this->backend());
        backend->onAcquireBuffer(mMiddle.get(), Backend::DYNAMIC);
        mCarbon->buffer().device = mMiddle->buffer().device; // share device

        // release temp buffer space
        backend->onReleaseBuffer(mMiddle.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

ErrorCode MetalReshape::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];

    // tensorflow
    if (input->getDimensionType() == Tensor::TENSORFLOW) {
        auto encoder   = [context encoder];
        auto bandwidth = [context load:@"copy_float" encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
        [context dispatchEncoder:encoder threads:{ (NSUInteger) output->elementSize(), 1, 1 } bandwidth:bandwidth];
        [encoder endEncoding];
        MNN_PRINT_ENCODER(context, encoder);
    }
    // caffe
    else {
        id encoder = nil;
        backend->onCopyBuffer(input, mMiddle.get(), encoder);
        backend->onCopyBuffer(mCarbon.get(), output, encoder);
    }
    return NO_ERROR;
}

class MetalReshapeCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalReshape(backend, op->main_as_Reshape()->dimType());
    }
};
REGISTER_METAL_OP_CREATOR(MetalReshapeCreator, OpType_Reshape);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
