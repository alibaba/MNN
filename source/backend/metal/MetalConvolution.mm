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
#import "backend/metal/MetalConvolutionWinograd.hpp"
#include <string>

#if MNN_METAL_ENABLED
namespace MNN {

MetalConvolution::MetalConvolution(Backend *backend, const MNN::Op *op) : MetalConvolutionCommon(backend, op, nullptr) {
    loadWeight(op->main_as_Convolution2D());
}
MetalConvolution::MetalConvolution(Backend *backend, const MNN::Op *op, std::shared_ptr<MNN::Tensor> weight, std::shared_ptr<MNN::Tensor> bias) : MetalConvolutionCommon(backend, op, bias) {
    mWeight = weight;
}
bool MetalConvolution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new MetalConvolution(bn, op, mWeight, mBias);
    return true;
}
ErrorCode MetalConvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    // prepare
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto mtbn = backend;
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0];
    auto output = outputs[0];
    auto iw   = input->width();
    auto ih   = input->height();
    auto ic_4 = UP_DIV(input->channel(), 4);
    auto ow   = output->width();
    auto oh   = output->height();
    auto oc_4 = UP_DIV(output->channel(), 4);
    auto ob   = output->batch();
    
    auto pads = ConvolutionCommon::convolutionPad(input, output, mOp->main_as_Convolution2D()->common());
    auto padX = pads.first;
    auto padY = pads.second;
    int stepSlices  = ic_4;

    // create const buffer
    int constants[] = {iw,
                       ih,
                       iw * ih,
                       ic_4,
                       ow,
                       oh,
                       ow * oh,
                       oc_4,
                       ob,
                       oc_4 * ob,
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
    mConstBuffer = backend->getConstBuffer(sizeof(constants));
    ::memcpy(mConstBuffer.contents, constants, sizeof(constants));
    
    mParam = "_ic" + std::to_string(ic_4) + "oc" + std::to_string(oc_4) +
             "k" + std::to_string(mKernelX) + "x" + std::to_string(mKernelY) +
             "s" + std::to_string(mStrideX) + "x" + std::to_string(mStrideY) +
             "d" + std::to_string(mDilateX) + "x" + std::to_string(mDilateY);

    MetalRuntime* rt = (MetalRuntime *)backend->runtime();
    bool isS1D1 = (mStrideX==1 && mStrideY==1 && mDilateX==1 && mDilateY==1);
    bool isS1D1P0 = isS1D1 && (padX==0 && padY==0 && mKernelX>1 && mKernelX%2==1);
    bool is3x3s1Conv = (mKernelX==3 && mKernelY==3 && mStrideX==1 && mStrideY==1 && padX==1 && padY==1 && mDilateX==1 && mDilateY==1);

    // printf("isS1D1P0: %d, c:%d %d, K:%d %d, s:%d %d, p:%d %d, iwh:%d %d, owh:%d %d\n", isS1D1P0, ic_4, oc_4, mKernelX, mKernelY, mStrideX, mStrideY, padX, padY, iw, ih, ow, oh);

    if(rt->getTuneLevel() == Never) {
        int packW = 1;
        int packC = 2;
        NSString* kernelName = @"conv_z2";
        if(isS1D1P0) {
            packW = 2;
            packC = 1;
            kernelName = @"conv_s1d1p0_w2";
        }
        NSUInteger gid_x = UP_DIV(ow, packW);
        NSUInteger gid_y = oh;
        NSUInteger gid_z = UP_DIV(oc_4, packC) * ob;

        mPipeline = [context pipelineWithName:kernelName fp16:backend->useFp16InsteadFp32()];
        NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                        (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                        mConstBuffer, ((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId())->getBuffer(), ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(), nil];
        const Tensor* weight = mWeight.get();
        const Tensor* bias = mBias.get();
        int buffer_offset[] = {TensorUtils::getDescribe(input)->extra.offset, TensorUtils::getDescribe(output)->extra.offset, 0, TensorUtils::getDescribe(weight)->extra.offset, TensorUtils::getDescribe(bias)->extra.offset};
        std::string name = [kernelName UTF8String] + mParam;
        auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name offsets:buffer_offset queue:backend->queue()];
        mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
    } else {
        const int total_kernel = 5;
        NSString* shaderName[total_kernel] = {@"conv",  @"conv_z4", @"conv_z2", @"conv_s1d1p0_w2", @"conv_s1d1p0_w4"};
        int itemW[total_kernel] = {1, 1, 1, 2, 4};
        int itemH[total_kernel] = {1, 1, 1, 1, 1};
        int itemC[total_kernel] = {1, 4, 2, 1, 1};
        
        int actual_kernel = 3;
        if(isS1D1P0) {
            actual_kernel = 4;
            if(mKernelX == 3) {
                actual_kernel = 5;
            }
        } else if(is3x3s1Conv) {
            actual_kernel = 4;
            shaderName[3] = @"convk3s1d1p1_w2z4";
            itemW[3] = 2;
            itemH[3] = 1;
            itemC[3] = 4;
        } else {
            actual_kernel = 3;
        }

        std::pair<NSUInteger, int> min_cost(INT_MAX, 0);//(min_time, min_index)
        
        NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                        (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                        mConstBuffer, (((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId()))->getBuffer(), ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(), nil];
        const Tensor* weight = mWeight.get();
        const Tensor* bias = mBias.get();
        int buffer_offset[] = {
            TensorUtils::getDescribe(input)->extra.offset,
            TensorUtils::getDescribe(output)->extra.offset,
            0,
            TensorUtils::getDescribe(weight)->extra.offset,
            TensorUtils::getDescribe(bias)->extra.offset
        };

        for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
            id<MTLComputePipelineState> pipeline = [context pipelineWithName:shaderName[knl_idx] fp16:mtbn->useFp16InsteadFp32()];
            NSUInteger gid_x = UP_DIV(ow, itemW[knl_idx]);
            NSUInteger gid_y = UP_DIV(oh, itemH[knl_idx]);
            NSUInteger gid_z = UP_DIV(oc_4, itemC[knl_idx]) * ob;

            std::string name = [shaderName[knl_idx] UTF8String] + mParam;
            auto ret = [context getGridAndThreadgroup:pipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name offsets: buffer_offset queue:backend->queue()];
            
            if(min_cost.first > std::get<2>(ret)) {
                min_cost.first = std::get<2>(ret);
                min_cost.second = knl_idx;
                mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
            }
            // printf("conv idx:%d %s, global:%d %d %d, local:%d %d %d, min_cost: %d -> %d\n", knl_idx, name.c_str(), (int)std::get<0>(ret).width, (int)std::get<0>(ret).height, (int)std::get<0>(ret).depth, (int)std::get<1>(ret).width, (int)std::get<1>(ret).height, (int)std::get<1>(ret).depth, std::get<2>(ret), (int)min_cost.first);
        }
        // printf("conv idx:%d, min_cost:%d\n", (int)min_cost.second, (int)min_cost.first);
        // std::string tmp = [shaderName[min_cost.second] UTF8String];
        // printf("!!~ %s\n", tmp.c_str());
        mPipeline = [context pipelineWithName:shaderName[min_cost.second] fp16:mtbn->useFp16InsteadFp32()];
    }
    return NO_ERROR;
}

void MetalConvolution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    [encoder setComputePipelineState:mPipeline];
    MetalBackend::setTensor(input, encoder, 0);
    MetalBackend::setTensor(output, encoder, 1);
    [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
    MetalBackend::setTensor(mWeight.get(), encoder, 3);
    MetalBackend::setTensor(mBias.get(), encoder, 4);

    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
}

class MetalConvolutionCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        auto param = op->main_as_Convolution2D();
        if (param->quanParameter() != nullptr) {
            if (param->quanParameter()->has_scaleInt()) {
                return nullptr;
            }
        }
        if (inputs.size() > 1) {
            return nullptr;
        }
        auto conv  = op->main_as_Convolution2D();
        if (conv->common()->group() > 1) {
            return nullptr;
        }
        if (op->type() == OpType_Convolution) {
            auto input = inputs[0];
            if (MetalConvolutionWinograd::isValid(conv, inputs[0], outputs[0])) {
                return new MetalConvolutionWinograd(backend, op);
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
