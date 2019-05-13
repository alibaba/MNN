//
//  MetalQuantizedMaxPool.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalQuantizedMaxPool.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalQuantizedMaxPool::MetalQuantizedMaxPool(Backend *backend, const QuantizedMaxPool *pool)
    : Execution(backend),
      mKernelX(pool->kernelX()),
      mKernelY(pool->kernelY()),
      mStrideX(pool->strideX()),
      mStrideY(pool->strideY()),
      mPadType(pool->padType()),
      mPadX(pool->padX()),
      mPadY(pool->padY()),
      mActivationMin(pool->outputActivationMin()),
      mActivationMax(pool->outputActivationMax()) {
    // nothing to do
}

ErrorCode MetalQuantizedMaxPool::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];

    int strideWidth  = mStrideX;
    int strideHeight = mStrideY;
    int padWidth     = mPadX;
    int padHeight    = mPadY;
    int kernelWidth  = (int)MIN(mKernelX, input->width());
    int kernelHeight = (int)MIN(mKernelY, input->height());
    if (mPadType == PoolPadType_VALID) {
        padWidth  = 0;
        padHeight = 0;
    }

    mConstBuffer                       = [context newDeviceBuffer:14 * sizeof(int) access:CPUWriteOnly];
    ((int *)mConstBuffer.contents)[0]  = input->batch();
    ((int *)mConstBuffer.contents)[1]  = input->height();
    ((int *)mConstBuffer.contents)[2]  = input->width();
    ((int *)mConstBuffer.contents)[3]  = output->height();
    ((int *)mConstBuffer.contents)[4]  = output->width();
    ((int *)mConstBuffer.contents)[5]  = input->channel();
    ((int *)mConstBuffer.contents)[6]  = kernelWidth;
    ((int *)mConstBuffer.contents)[7]  = kernelHeight;
    ((int *)mConstBuffer.contents)[8]  = strideWidth;
    ((int *)mConstBuffer.contents)[9]  = strideHeight;
    ((int *)mConstBuffer.contents)[10] = padWidth;
    ((int *)mConstBuffer.contents)[11] = padHeight;
    ((int *)mConstBuffer.contents)[12] = mActivationMin;
    ((int *)mConstBuffer.contents)[13] = mActivationMax;
    return NO_ERROR;
}

ErrorCode MetalQuantizedMaxPool::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    auto b = output->batch(), h = output->height(), w = output->width(), c = output->channel();

    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"quantized_max_pool" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) c, (NSUInteger)w, (NSUInteger)b *h } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalQuantizedMaxPoolCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalQuantizedMaxPool(backend, op->main_as_QuantizedMaxPool());
    }
};
REGISTER_METAL_OP_CREATOR(MetalQuantizedMaxPoolCreator, OpType_QuantizedMaxPool);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
