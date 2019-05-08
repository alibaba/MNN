//
//  MetalQuantizedAvgPool.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalQuantizedAvgPool.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalQuantizedAvgPool::MetalQuantizedAvgPool(Backend *backend, const QuantizedAvgPool *pool)
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

ErrorCode MetalQuantizedAvgPool::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];

    int strideWidth  = mStrideX;
    int strideHeight = mStrideY;
    int padWidth     = mPadX;
    int padHeight    = mPadY;
    int kernelWidth  = (int)MIN(mKernelX, input->width());
    int kernelHeight = (int)MIN(mKernelY, input->height());
    if (mPadType == PoolPadType_SAME) {
        padWidth  = MAX(0, ((output->width() - 1) * mStrideX + mKernelX - input->width()) / 2);
        padHeight = MAX(0, ((output->height() - 1) * mStrideY + mKernelY - input->height()) / 2);
    }

    mConstBuffer  = [context newDeviceBuffer:14 * sizeof(int) access:CPUWriteOnly];
    auto contents = (int *)mConstBuffer.contents;
    contents[0]   = input->batch();
    contents[1]   = input->height();
    contents[2]   = input->width();
    contents[3]   = output->height();
    contents[4]   = output->width();
    contents[5]   = input->channel();
    contents[6]   = kernelWidth;
    contents[7]   = kernelHeight;
    contents[8]   = strideWidth;
    contents[9]   = strideHeight;
    contents[10]  = padWidth;
    contents[11]  = padHeight;
    contents[12]  = mActivationMin;
    contents[13]  = mActivationMax;
    return NO_ERROR;
}

ErrorCode MetalQuantizedAvgPool::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    auto b = output->batch(), h = output->height(), w = output->width(), c = output->channel();

    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"quantized_avg_pool" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) c, (NSUInteger)w, (NSUInteger)b *h } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalQuantizedAvgPoolCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalQuantizedAvgPool(backend, op->main_as_QuantizedAvgPool());
    }
};
REGISTER_METAL_OP_CREATOR(MetalQuantizedAvgPoolCreator, OpType_QuantizedAvgPool);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
