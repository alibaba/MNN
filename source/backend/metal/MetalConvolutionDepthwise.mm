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
    auto backend = static_cast<MetalBackend *>(this->backend());

    // prepare
    auto input = inputs[0], output = outputs[0];
    auto iw = input->width();
    auto ih = input->height();
    auto ic_4 = UP_DIV(input->channel(), 4);
    auto ow = output->width();
    auto oh = output->height();
    auto ob = output->batch();
    auto oc_4 = UP_DIV(output->channel(), 4);

    auto pads = ConvolutionCommon::convolutionPad(input, output, mOp->main_as_Convolution2D()->common());
    auto padX = pads.first;
    auto padY = pads.second;

    // create const buffer
    int constants[] = {iw,
                       ih,
                       iw * ih,
                       ow,
                       oh,
                       ow * oh,
                       ic_4,
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
    mConstBuffer = backend->getConstBuffer(sizeof(constants));

    ::memcpy(mConstBuffer.contents, constants, sizeof(constants));
    
    auto context = (__bridge MNNMetalContext *)backend->context();
    mPipeline = [context pipelineWithName:@"conv_depthwise"];
            
    NSUInteger gid_x = ow;
    NSUInteger gid_y = oh;
    NSUInteger gid_z = oc_4*ob;
            
    NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                    (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                    mConstBuffer, mWeight, mBias, nil];

    std::string name = "conv_depthwise";
    MetalRuntime *rt = (MetalRuntime *)backend->runtime();
    auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name];
    mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
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
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset: TensorUtils::getDescribe(input)->extra.offset
atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset: TensorUtils::getDescribe(output)->extra.offset
atIndex:1];
        [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
        [encoder setBuffer:mWeight offset:0 atIndex:3];
        [encoder setBuffer:mBias offset:0 atIndex:4];
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
            
        auto context = (__bridge MNNMetalContext *)backend->context();
        if(backend->isCmdBufferCommit()) {
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
