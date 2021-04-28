//
//  MetalConvolutionDepthwise.mm
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalConvolutionDepthwise.hpp"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED

namespace MNN {
MetalConvolutionDepthwise::MetalConvolutionDepthwise(Backend *backend, const MNN::Op *op)
    : MetalConvolutionCommon(backend, op) {
    loadWeight(op->main_as_Convolution2D());
}

ErrorCode MetalConvolutionDepthwise::onResize(const std::vector<Tensor *> &inputs,
                                              const std::vector<Tensor *> &outputs) {
    MetalConvolutionCommon::onResize(inputs, outputs);

    // prepare
    auto input = inputs[0], output = outputs[0];
    auto iw = input->width(), ih = input->height(), iz = UP_DIV(input->channel(), 4);
    auto ow = output->width(), oh = output->height(), ob = output->batch();

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

    // create const buffer
    int constants[] = {iw,
                       ih,
                       iw * ih,
                       ow,
                       oh,
                       ow * oh,
                       iz,
                       ob,

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
    
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    mPipeline = [context pipelineWithName:@"conv_depthwise"];
    
    auto w = output->width();
    auto h = output->height();
    auto z = UP_DIV(output->channel(), 4);
    auto b = output->batch();
            
    NSUInteger gid_x = w;
    NSUInteger gid_y = h;
    NSUInteger gid_z = z*b;
            
    NSArray *arr = [NSArray arrayWithObjects:(__bridge id<MTLBuffer>)(void *)input->deviceId(),
                    (__bridge id<MTLBuffer>)((void *)output->deviceId()),
                    mConstBuffer.buffer(), mWeight, mBias, nil];

    mThreads = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr];

    return NO_ERROR;
}

ErrorCode MetalConvolutionDepthwise::onFloat(const Tensor *input, const Tensor *output) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    if(backend->isCommandEncoderSet()) {
        return NO_ERROR;
    }
    
    auto func = [=](){
        auto encoder    = backend->encoder();
        [encoder setComputePipelineState:mPipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
        [encoder setBuffer:mConstBuffer.buffer() offset:0 atIndex:2];
        [encoder setBuffer:mWeight offset:0 atIndex:3];
        [encoder setBuffer:mBias offset:0 atIndex:4];
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
            
        auto context = (__bridge MNNMetalContext *)backend->context();
        if(context.isCommitEachShader) {
            backend->flushEncoder();
            [context commit_net];
        }
    };
    func();
    backend->addOpEncoder(func);

    return NO_ERROR;
}

template <typename FType, typename TType>
static id<MTLBuffer> weightInBlock(MNNMetalContext *context, int group, int kh, int kw, const FType *src) {
    auto buffer = [context newDeviceBuffer:UP_DIV(group, 4) * 4 * kw * kh * sizeof(TType) access:CPUWriteOnly];
    auto dst    = (TType *)buffer.contents;
    for (int g = 0; g < group; g++) {
        auto z = g / 4, r = g % 4;
        auto z_dst = dst + z * kh * kw * 4 + r;
        for (int h = 0; h < kh; h++) {
            for (int w = 0; w < kw; w++) {
                // to   [g/4][h][w][4]
                // from [g][h][w]
                // dst[(z * kh * kw + h * kw + w) * 4 + r] =
                // src[ g * kh * kw + h * kw + w];
                z_dst[(h * kw + w) * 4] = *src++;
            }
        }
    }
    return buffer;
}

id<MTLBuffer> MetalConvolutionDepthwise::weightForFloat(int group, int oc, int ic, int kh, int kw, const float *src) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    return weightInBlock<float, metal_float>(context, group, kh, kw, src);
}

class MetalConvolutionDepthwiseCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        if (inputs.size() > 1) {
            return nullptr;
        }
        return new MetalConvolutionDepthwise(backend, op);
    }
};
REGISTER_METAL_OP_CREATOR(MetalConvolutionDepthwiseCreator, OpType_ConvolutionDepthwise);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
