//
//  MetalConvolution.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalConvolution.hpp"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"
#import "backend/metal/MetalConvolution1x1.hpp"
#import "backend/metal/MetalConvolutionGEMM.hpp"
#import "backend/metal/MetalConvolutionWinograd.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalConvolution::MetalConvolution(Backend *backend, const MNN::Op *op) : MetalConvolutionCommon(backend, op) {
    mOp = op;
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

    int unit          = sizeof(metal_float);
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
    auto pads = ConvolutionCommon::convolutionPad(input, output, mOp->main_as_Convolution2D()->common());
    auto padX = pads.first;
    auto padY = pads.second;

    // update threadgroup memory if needed
    int stepSlices  = igz;
    mLocalPreferred = isThreadgroupLocalPreferred(input, output);
    if (mLocalPreferred) {
        int unit        = sizeof(metal_float);
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
    mConstBuffer.reset(sizeof(constants));
    ::memcpy(mConstBuffer.buffer().contents, constants, sizeof(constants));
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

    auto encoder    = backend->encoder();
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
            [encoder setBuffer:mConstBuffer.buffer() offset:0 atIndex:2];
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
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalConvolutionCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto param = op->main_as_Convolution2D();
        if (param->quanParameter() != nullptr) {
            if (param->quanParameter()->has_scaleInt()) {
                return nullptr;
            }
        }
        if (op->type() == OpType_Convolution) {
            auto conv  = op->main_as_Convolution2D();
            auto input = inputs[0];
            if (MetalConvolutionWinograd::isValid(conv, input)) {
                return new MetalConvolutionWinograd(backend, input, op);
            }
            if (MetalConvolutionGEMM::isValid(conv, input)) {
                return new MetalConvolutionGEMM(backend, input, op);
            }
            if (MetalConvolution1x1::isValid(conv, input)) {
                return new MetalConvolution1x1(backend, op);
            }
        }
        return new MetalConvolution(backend, op);
    }
};
REGISTER_METAL_OP_CREATOR(MetalConvolutionCreator, OpType_Convolution);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
