//
//  MetalConvolution1x1.mm
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalConvolution1x1.hpp"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED

namespace MNN {
bool MetalConvolution1x1::isValid(const Convolution2D *conv, const Tensor *input) {
    auto common = conv->common();
    auto kx = common->kernelX(), ky = common->kernelY();
    auto dx = common->dilateX(), dy = common->dilateY();
    auto sx = common->strideX(), sy = common->strideY();
    auto px = common->padX(), py = common->padY();
    return kx == 1 && ky == 1 && dx == 1 && dy == 1 && px == 0 && py == 0 && sx == 1 && sy == 1;
}

MetalConvolution1x1::MetalConvolution1x1(Backend *backend, const MNN::Op *op) : MetalConvolutionCommon(backend, op) {
    loadWeight(op->main_as_Convolution2D());
}

ErrorCode MetalConvolution1x1::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MetalConvolutionCommon::onResize(inputs, outputs);

    // prepare
    auto input = inputs[0], output = outputs[0];
    auto is = input->width() * input->height(), iz = UP_DIV(input->channel(), 4), igz = iz / mGroups;
    auto os = output->width() * output->height(), oz = UP_DIV(output->channel(), 4), ogz = oz / mGroups;
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();

    // create const buffer
    int constants[] = {is, igz, iz, os, ogz, oz, output->batch(), mActivationType};
    mConstBuffer.reset(sizeof(constants));
    ::memcpy(mConstBuffer.buffer().contents, constants, sizeof(constants));
    auto w = output->width(), h = output->height(), z = UP_DIV(output->channel(), 4), b = output->batch();;
    if (mGroups == 1 && (w * h >= 32)) {
        mPipeline = [context pipelineWithName:@"conv1x1_g1z4"];
        mThreads   = [context computeBestGroupAndLocal:mPipeline threads:{(NSUInteger)UP_DIV(w * h, 4), (NSUInteger)z, (NSUInteger)b}];
    } else {
        mPipeline = [context pipelineWithName:@"conv1x1"];
        mThreads   = [context computeBestGroupAndLocal:mPipeline threads:{(NSUInteger)w * h, (NSUInteger)z, (NSUInteger)b}];
    }
    return NO_ERROR;
}

ErrorCode MetalConvolution1x1::onFloat(const Tensor *input, const Tensor *output) {
    auto backend = static_cast<MetalBackend *>(this->backend());

    auto encoder    = backend->encoder();
    [encoder setComputePipelineState:mPipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:mConstBuffer.buffer() offset:0 atIndex:2];
    [encoder setBuffer:mWeight offset:0 atIndex:3];
    [encoder setBuffer:mBias offset:0 atIndex:4];
    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
    return NO_ERROR;
}
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
