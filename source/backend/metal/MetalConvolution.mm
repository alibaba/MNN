//
//  MetalConvolution.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalConvolution.hpp"
#import "Macro.h"
#import "MetalBackend.hpp"
#import "MetalConvolution1x1.hpp"
#import "MetalConvolutionGEMM.hpp"
#import "MetalConvolutionWinograd.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalConvolution::MetalConvolution(Backend *backend, const MNN::Op *op) : MetalConvolutionCommon(backend, op) {
    loadWeight(op->main_as_Convolution2D());
}

// definitely less than max threadgroup memory to ensure that it won't take too long in one step.
#define kMaxGemmStepMemory (8 * 1024)

bool MetalConvolution::isThreadgroupLocalPreferred(const Tensor *input, const Tensor *output) {
    if (output->width() * output->height() > 256) {
        return false;
    }

    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    int igz      = UP_DIV(input->channel(), 4) / mGroups;
    int ogz      = UP_DIV(output->channel(), 4) / mGroups;

    int unit          = mQnt ? sizeof(int8_t) : sizeof(metal_float);
    int sliceMemory   = 4 * mKernelY * mKernelX * 4 * unit;
    int maxMemory     = sliceMemory > kMaxGemmStepMemory ? (int)context.maxThreadgroupMemoryLength : kMaxGemmStepMemory;
    int maxStepSlices = maxMemory / sliceMemory;
    int steps         = UP_DIV(igz, maxStepSlices);

    static int kGemmUnroll = 4;
    return ogz * ogz * kGemmUnroll / steps / steps >= output->width() * output->height();
}

ErrorCode MetalConvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MetalConvolutionCommon::onResize(inputs, outputs);

    // prepare
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    auto iw = input->width(), ih = input->height(), igz = UP_DIV(input->channel(), 4) / mGroups;
    auto ow = output->width(), oh = output->height(), ogz = UP_DIV(output->channel(), 4) / mGroups;

    // pad mode support
    int padX = mPadX, padY = mPadY;
    if (mPadMode == PadMode_SAME) {
        int kernelWidthSize = (mKernelX - 1) * mDilateX + 1;
        int kernelHeightSize = (mKernelY - 1) * mDilateY + 1;
        int pw = (ow - 1) * mStrideX + kernelWidthSize - iw;
        int ph = (oh - 1) * mStrideY + kernelHeightSize - ih;
        padX   = pw / 2;
        padY   = ph / 2;
    }

    // update threadgroup memory if needed
    int stepSlices  = igz;
    mLocalPreferred = isThreadgroupLocalPreferred(input, output);
    if (mLocalPreferred) {
        int unit        = mQnt ? sizeof(int8_t) : sizeof(metal_float);
        int sliceMemory = 4 * mKernelY * mKernelX * 4 * unit;
        int maxMemory = sliceMemory > kMaxGemmStepMemory ? (int)context.maxThreadgroupMemoryLength : kMaxGemmStepMemory;
        int maxStepSlices  = maxMemory / sliceMemory;
        int steps          = UP_DIV(igz, maxStepSlices);
        stepSlices         = UP_DIV(igz, steps);
        mThreadgroupMemory = stepSlices * sliceMemory;
    }

    // create const buffer
    int constants[] = {iw,
                       ih,
                       iw * ih,
                       igz,
                       ow,
                       oh,
                       ow * oh,
                       ogz,
                       stepSlices,

                       mKernelX,
                       mKernelY,
                       mKernelX * mKernelY,
                       mStrideX,
                       mStrideY,
                       padX,
                       padY,
                       mDilateX,
                       mDilateY,
                       mActivationType};
    mConstBuffer = [context newDeviceBuffer:sizeof(constants) bytes:constants access:CPUWriteOnly];
    return NO_ERROR;
}

ErrorCode MetalConvolution::onQuantized(const Tensor *input, const Tensor *output) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto iw = input->width(), ih = input->height(), iz = UP_DIV(input->channel(), 4);
    auto ow = output->width(), oh = output->height(), oz = UP_DIV(output->channel(), 4), ogz = oz / mGroups;
    auto unit = sizeof(int8_t);
    auto ib = iw * ih * iz * 4 * unit, ig = ib / mGroups;
    auto ob = ow * oh * oz * 4 * sizeof(metal_float), og = ob / mGroups;

    auto encoder    = [context encoder];
    auto bandwidth  = (MetalBandwidth){};
    MTLSize threads = {};
    if (mLocalPreferred) {
        bandwidth = [context load:@"qntconv_local" encoder:encoder];
        threads   = {(NSUInteger)UP_DIV(ow, 4), (NSUInteger)oh, (NSUInteger)ogz};
    } else if (ow * oh >= 32 ? ogz >= 16 : ogz >= 128) {
        bandwidth = [context load:@"qntconv_z4" encoder:encoder];
        threads   = {(NSUInteger)ow, (NSUInteger)oh, (NSUInteger)UP_DIV(ogz, 4)};
    } else {
        bandwidth = [context load:@"qntconv" encoder:encoder];
        threads   = {(NSUInteger)ow, (NSUInteger)oh, (NSUInteger)ogz};
    }

    for (int b = 0; b < input->batch(); b++) {
        for (int g = 0; g < mGroups; g++) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:b * ib + g * ig atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:b * ob + g * og atIndex:1];
            [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
            [encoder setBuffer:mWeight offset:g * mWeight.length / mGroups atIndex:3];
            [encoder setBuffer:mBias offset:g * mBias.length / mGroups atIndex:4];
            [encoder setBuffer:mAlpha offset:g * mAlpha.length / mGroups atIndex:5];
            if (mLocalPreferred) {
                [encoder setThreadgroupMemoryLength:mThreadgroupMemory atIndex:0];
                [context dispatchEncoder:encoder
                                 threads:threads
                         threadsPerGroup:{ 1, 1, (NSUInteger)ogz }
                               bandwidth:bandwidth];
            } else {
                [context dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
            }
        }
    }
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

ErrorCode MetalConvolution::onFloat(const Tensor *input, const Tensor *output) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto iw = input->width(), ih = input->height(), iz = UP_DIV(input->channel(), 4);
    auto ow = output->width(), oh = output->height(), oz = UP_DIV(output->channel(), 4), ogz = oz / mGroups;
    auto unit = sizeof(metal_float);
    auto ib = iw * ih * iz * 4 * unit, ig = ib / mGroups;
    auto ob = ow * oh * oz * 4 * sizeof(metal_float), og = ob / mGroups;

    auto encoder    = [context encoder];
    auto bandwidth  = (MetalBandwidth){};
    MTLSize threads = {};
    if (mLocalPreferred) {
        bandwidth = [context load:@"conv_local" encoder:encoder];
        threads   = {(NSUInteger)UP_DIV(ow, 4), (NSUInteger)oh, (NSUInteger)ogz};
    } else if (ow * oh >= 32 ? ogz >= 16 : ogz >= 128) {
        bandwidth = [context load:@"conv_z4" encoder:encoder];
        threads   = {(NSUInteger)ow, (NSUInteger)oh, (NSUInteger)UP_DIV(ogz, 4)};
    } else {
        bandwidth = [context load:@"conv" encoder:encoder];
        threads   = {(NSUInteger)ow, (NSUInteger)oh, (NSUInteger)ogz};
    }

    for (int b = 0; b < input->batch(); b++) {
        for (int g = 0; g < mGroups; g++) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:b * ib + g * ig atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:b * ob + g * og atIndex:1];
            [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
            [encoder setBuffer:mWeight offset:g * mWeight.length / mGroups atIndex:3];
            [encoder setBuffer:mBias offset:g * mBias.length / mGroups atIndex:4];
            if (mLocalPreferred) {
                [encoder setThreadgroupMemoryLength:mThreadgroupMemory atIndex:0];
                [context dispatchEncoder:encoder
                                 threads:threads
                         threadsPerGroup:{ 1, 1, (NSUInteger)ogz }
                               bandwidth:bandwidth];
            } else {
                [context dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
            }
        }
    }
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalConvolutionCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        if (op->type() == OpType_Convolution) {
            auto conv  = op->main_as_Convolution2D();
            auto input = inputs[0];
            if (MetalConvolution1x1::isValid(conv, input)) {
                return new MetalConvolution1x1(backend, op);
            }
            if (MetalConvolutionWinograd::isValid(conv, input)) {
                return new MetalConvolutionWinograd(backend, input, op);
            }
            if (MetalConvolutionGEMM::isValid(conv, input)) {
                return new MetalConvolutionGEMM(backend, input, op);
            }
        }
        return new MetalConvolution(backend, op);
    }
};
REGISTER_METAL_OP_CREATOR(MetalConvolutionCreator, OpType_Convolution);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
