//
//  MetalPooling.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalPooling.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalPooling::MetalPooling(Backend *backend, const Pool *pooling)
    : Execution(backend),
      mGlobal(pooling->isGlobal()),
      mPoolType(pooling->type()),
      mKernelX(pooling->kernelX()),
      mKernelY(pooling->kernelY()),
      mStrideX(pooling->strideX()),
      mStrideY(pooling->strideY()),
      mPadX(pooling->padX()),
      mPadY(pooling->padY()) {
    // nothing to do
}

ErrorCode MetalPooling::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];

    int strideWidth  = mStrideX;
    int strideHeight = mStrideY;
    int padWidth     = mPadX;
    int padHeight    = mPadY;
    int kernelWidth  = (int)MIN(mKernelX, input->width());
    int kernelHeight = (int)MIN(mKernelY, input->height());
    if (mGlobal) {
        kernelWidth  = (int)input->width();
        kernelHeight = (int)input->height();
        strideWidth  = (int)input->width();
        strideHeight = (int)input->height();
        padWidth     = 0;
        padHeight    = 0;
    }

    mConstBuffer                       = [context newDeviceBuffer:11 * sizeof(int) access:CPUWriteOnly];
    ((int *)mConstBuffer.contents)[0]  = input->width();
    ((int *)mConstBuffer.contents)[1]  = input->height();
    ((int *)mConstBuffer.contents)[2]  = output->width();
    ((int *)mConstBuffer.contents)[3]  = output->height();
    ((int *)mConstBuffer.contents)[4]  = UP_DIV(output->channel(), 4) * output->batch();
    ((int *)mConstBuffer.contents)[5]  = kernelWidth;
    ((int *)mConstBuffer.contents)[6]  = kernelHeight;
    ((int *)mConstBuffer.contents)[7]  = strideWidth;
    ((int *)mConstBuffer.contents)[8]  = strideHeight;
    ((int *)mConstBuffer.contents)[9]  = padWidth;
    ((int *)mConstBuffer.contents)[10] = padHeight;

    return NO_ERROR;
}

ErrorCode MetalPooling::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    auto ow = output->width(), oh = output->height(), slice = UP_DIV(output->channel(), 4) * output->batch();

    auto encoder   = [context encoder];
    auto bandwidth = [context load:(mPoolType == PoolType_MAXPOOL) ? @"pooling_max" : @"pooling_avg" encoder:encoder];
    bandwidth.zAxisProtected = YES;
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
    [context dispatchEncoder:encoder
                     threads:{ (NSUInteger) ow, (NSUInteger)oh, (NSUInteger)slice }
                   bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalPoolingCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalPooling(backend, op->main_as_Pool());
    }
};
REGISTER_METAL_OP_CREATOR(MetalPoolingCreator, OpType_Pooling);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
