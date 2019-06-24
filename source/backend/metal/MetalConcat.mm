//
//  MetalConcat.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalConcat.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"
#import "TensorUtils.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalConcat::MetalConcat(Backend *bn, int axis) : Execution(bn), mAxis(axis) {
    // nothing to do
}

ErrorCode MetalConcat::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto output  = outputs[0];
    auto tf      = output->getDimensionType() == Tensor::TENSORFLOW;
    mBlits       = [context newDeviceBuffer:5 * inputs.size() * sizeof(int) access:CPUReadWrite];

    // enable fast mode if concat on slice
    mFastMode = !tf && mAxis == 1;
    for (int i = 0; i < inputs.size(); ++i) {
        if (inputs[i]->channel() % 4 != 0) {
            mFastMode = false;
            break;
        }
    }

    // set up blits
    auto blits = (int *)mBlits.contents;
    for (int i = 0; i < inputs.size(); i++, blits += 5) {
        auto input = inputs[i];
        if (mFastMode) {
            blits[0] = input->width() * input->height();
            blits[1] = UP_DIV(input->channel(), 4);
            blits[2] = input->batch();
            blits[3] = blits[0] * blits[1];
            blits[4] = blits[0] * UP_DIV(output->channel(), 4);
        } else {
            auto axis = mAxis > 0 ? mAxis : input->buffer().dimensions + mAxis;
            blits[0]  = 1;
            for (int j = axis + 1; j < input->buffer().dimensions; j++) {
                blits[0] *= input->length(j);
            }
            blits[1] = input->length(axis);
            blits[2] = 1;
            for (int j = 0; j < axis; j++) {
                blits[2] *= input->length(j);
            }
            blits[3] = blits[0] * blits[1];
            blits[4] = blits[0] * output->length(axis);
        }
    }

    // acquire space for temp input/output & set up blits
    if (!tf && !mFastMode) {
        // reset temp output
        mTempOutput.reset(new Tensor);
        TensorUtils::copyShape(output, mTempOutput.get());
        TensorUtils::getDescribe(mTempOutput.get())->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        mTempOutput->buffer().dim[1].flags                           = 0; // force NCHW
        TensorUtils::setLinearLayout(mTempOutput.get());
        backend->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);

        // reset temp inputs
        mTempInputs.clear();
        for (int i = 0; i < inputs.size(); i++, blits += 5) {
            std::shared_ptr<Tensor> tempInput(new Tensor);
            TensorUtils::copyShape(inputs[i], tempInput.get());
            TensorUtils::getDescribe(tempInput.get())->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            tempInput->buffer().dim[1].flags                           = 0; // force NCHW
            TensorUtils::setLinearLayout(tempInput.get());
            mTempInputs.push_back(tempInput);
            backend->onAcquireBuffer(tempInput.get(), Backend::DYNAMIC);
        }

        // make space reusable
        backend->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
        for (auto &tempInput : mTempInputs) {
            backend->onReleaseBuffer(tempInput.get(), Backend::DYNAMIC);
        }
    }
    return NO_ERROR;
}

ErrorCode MetalConcat::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto output  = outputs[0];
    auto tf      = output->getDimensionType() == Tensor::TENSORFLOW;
    auto encoder = [context encoder];
    auto start   = 0;

    if (tf) {
        auto bandwidth           = [context load:@"concat_x1" encoder:encoder];
        bandwidth.zAxisProtected = YES;
        for (int i = 0; i < inputs.size(); i++) {
            auto input = inputs[i];
            auto blits = (int *)mBlits.contents + i * 5;
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)(input->buffer().device) offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)(output->buffer().device) offset:start atIndex:1];
            [encoder setBuffer:mBlits offset:i * 5 * sizeof(int) atIndex:2];
            [context dispatchEncoder:encoder
                             threads:{ (NSUInteger) blits[0], (NSUInteger)blits[1], (NSUInteger)blits[2] }
                           bandwidth:bandwidth];
            start += blits[3] * sizeof(metal_float);
        }
    } else if (mFastMode) {
        auto bandwidth           = [context load:@"concat_x4" encoder:encoder];
        bandwidth.zAxisProtected = YES;
        for (int i = 0; i < inputs.size(); i++) {
            auto input = inputs[i];
            auto blits = (int *)mBlits.contents + i * 5;
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)(input->buffer().device) offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)(output->buffer().device) offset:start atIndex:1];
            [encoder setBuffer:mBlits offset:i * 5 * sizeof(int) atIndex:2];
            [context dispatchEncoder:encoder
                             threads:{ (NSUInteger) blits[0], (NSUInteger)blits[1], (NSUInteger)blits[2] }
                           bandwidth:bandwidth];
            start += blits[3] * 4 * sizeof(metal_float);
        }
    } else {
        for (int i = 0; i < inputs.size(); i++) {
            // convert forth
            backend->onCopyBuffer(inputs[i], mTempInputs[i].get(), encoder);

            // blit
            auto blits               = (int *)mBlits.contents + i * 5;
            auto bandwidth           = [context load:@"concat_x1" encoder:encoder];
            bandwidth.zAxisProtected = YES;
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)(mTempInputs[i]->buffer().device) offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)(mTempOutput->buffer().device) offset:start atIndex:1];
            [encoder setBuffer:mBlits offset:i * 5 * sizeof(int) atIndex:2];
            [context dispatchEncoder:encoder
                             threads:{ (NSUInteger) blits[0], (NSUInteger)blits[1], (NSUInteger)blits[2] }
                           bandwidth:bandwidth];
            start += blits[3] * sizeof(metal_float);
        }

        // convert back
        backend->onCopyBuffer(mTempOutput.get(), output, encoder);
    }
    [encoder endEncoding];

    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalConcatCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto concat = op->main_as_Axis();
        auto axis = concat->axis();
        if (axis < 0) {
            axis = axis + inputs[0]->dimensions();
        }
        return new MetalConcat(backend, axis);
    }
};
REGISTER_METAL_OP_CREATOR(MetalConcatCreator, OpType_Concat);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
