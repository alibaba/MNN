//
//  MetalConvolution1x1.mm
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalConvolution1x1.hpp"
#import "Macro.h"
#import "MetalBackend.hpp"

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
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    auto is = input->width() * input->height(), iz = UP_DIV(input->channel(), 4), igz = iz / mGroups;
    auto os = output->width() * output->height(), oz = UP_DIV(output->channel(), 4), ogz = oz / mGroups;

    // create const buffer
    int constants[] = {is, igz, iz, os, ogz, oz, output->batch(), mActivationType};
    mConstBuffer    = [context newDeviceBuffer:sizeof(constants) bytes:constants access:CPUWriteOnly];
    return NO_ERROR;
}

ErrorCode MetalConvolution1x1::onQuantized(const Tensor *input, const Tensor *output) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto w = output->width(), h = output->height(), z = UP_DIV(output->channel(), 4), b = output->batch();
    
    auto encoder    = [context encoder];
    auto bandwidth  = (MetalBandwidth){};
    MTLSize threads = {};
    if (mGroups == 1 && (w * h * b >= 32 ? z >= 16 : z >= 128)) {
        bandwidth = [context load:@"qntconv1x1_g1z4" encoder:encoder];
        threads   = {(NSUInteger)w * h, (NSUInteger)UP_DIV(z, 4), (NSUInteger)b};
    } else {
        bandwidth = [context load:@"qntconv1x1" encoder:encoder];
        threads   = {(NSUInteger)w * h, (NSUInteger)z, (NSUInteger)b};
    }
    bandwidth.zAxisProtected = YES;
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
    [encoder setBuffer:mWeight offset:0 atIndex:3];
    [encoder setBuffer:mBias offset:0 atIndex:4];
    [encoder setBuffer:mAlpha offset:0 atIndex:5];
    [context dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

ErrorCode MetalConvolution1x1::onFloat(const Tensor *input, const Tensor *output) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto w = output->width(), h = output->height(), z = UP_DIV(output->channel(), 4), b = output->batch();;
    
    auto encoder    = [context encoder];
    auto bandwidth  = (MetalBandwidth){};
    MTLSize threads = {};
    if (mGroups == 1 && (w * h * b >= 32 ? z >= 16 : z >= 128)) {
        bandwidth = [context load:@"conv1x1_g1z4" encoder:encoder];
        threads   = {(NSUInteger)w * h, (NSUInteger)UP_DIV(z, 4), (NSUInteger)b};
    } else {
        bandwidth = [context load:@"conv1x1" encoder:encoder];
        threads   = {(NSUInteger)w * h, (NSUInteger)z, (NSUInteger)b};
    }
    bandwidth.zAxisProtected = YES;
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
    [encoder setBuffer:mWeight offset:0 atIndex:3];
    [encoder setBuffer:mBias offset:0 atIndex:4];
    [context dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
