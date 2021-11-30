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
    auto input = inputs[0];
    auto output = outputs[0];
    
    auto is  = input->width() * input->height();
    auto ic_4  = UP_DIV(input->channel(), 4);
    auto ow  = output->width();
    auto oh  = output->height();
    auto os  = ow * oh;
    auto ob  = output->batch();
    auto oc  = output->channel();
    auto oc_4  = UP_DIV(output->channel(), 4);
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();

    // create const buffer
    int constants[] = {is, ic_4, ow, oh, os, oc_4, ob, mActivationType};
    mConstBuffer = backend->getConstBuffer(sizeof(constants));
    ::memcpy(mConstBuffer.contents, constants, sizeof(constants));
    
    MetalRuntime* rt = (MetalRuntime *)backend->runtime();

    if(rt->getTuneLevel() == Never) {
        if (ow * oh >= 128) {
            NSUInteger gid_x = UP_DIV(ow * oh, 8);
            NSUInteger gid_y = oc_4;
            NSUInteger gid_z = ob;

            mPipeline = [context pipelineWithName:@"conv1x1_g1z8"];

            NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                            (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                            mConstBuffer, mWeight, mBias, nil];
            
            std::string name = "conv1x1_g1z8";
            MetalRuntime *rt = (MetalRuntime *)backend->runtime();
            auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name];
            mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
        } else {
            NSUInteger gid_x = UP_DIV(ow * oh, 4);
            NSUInteger gid_y = oc_4;
            NSUInteger gid_z = ob;
            
            mPipeline = [context pipelineWithName:@"conv1x1_g1z4"];
            
            NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                            (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                            mConstBuffer, mWeight, mBias, nil];

            std::string name = "conv1x1_g1z4";
            MetalRuntime *rt = (MetalRuntime *)backend->runtime();
            auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name];
            mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
            //printf("conv1x1_z4, %d %d %d %d\n", ow, oh, oc_4, ic_4);
        }
    } else {
        // {@"conv1x1_w4h2", @"conv1x1_w2h2", @"conv1x1_w2c2", @"conv1x1_w1h1" };
        const int total_kernel = 5;
        NSString* shaderName[total_kernel] = {@"conv1x1_w4h2", @"conv1x1_w2h2", @"conv1x1_w2c2", @"conv1x1_w2h2c2", @"conv1x1_w4h4"};
        int itemW[total_kernel] = {4, 2, 2, 2, 4};
        int itemH[total_kernel] = {2, 2, 1, 2, 4};
        int itemC[total_kernel] = {4, 4, 8, 8, 4};
     
        int actual_kernel = 5;
        std::pair<NSUInteger, int> min_cost(INT_MAX, 0);//(min_time, min_index)
        
        NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                        (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                        mConstBuffer, mWeight, mBias, nil];
        
        for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
            id<MTLComputePipelineState> pipeline = [context pipelineWithName:shaderName[knl_idx]];
            NSUInteger gid_x = UP_DIV(ow, itemW[knl_idx]);
            NSUInteger gid_y = UP_DIV(oh, itemH[knl_idx]);
            NSUInteger gid_z = ob * UP_DIV(oc, itemC[knl_idx]);
            
            std::string name = [shaderName[knl_idx] UTF8String];
            auto ret = [context getGridAndThreadgroup:pipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name];
            
            if(min_cost.first > std::get<2>(ret)) {
                min_cost.first = std::get<2>(ret);
                min_cost.second = knl_idx;
                mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
            }
            //printf("conv1x1 idx:%d, global:%d %d %d, local:%d %d %d, min_cost:%d\n", knl_idx, (int)retTune.second.first.width, (int)retTune.second.first.height, (int)retTune.second.first.depth, (int)retTune.second.second.width, (int)retTune.second.second.height, (int)retTune.second.second.depth, (int)retTune.first);
        }
        //printf("conv1x1 idx:%d, min_cost:%d\n", (int)min_cost.second, (int)min_cost.first);
        mPipeline = [context pipelineWithName:shaderName[min_cost.second]];
    }

    return NO_ERROR;
}

ErrorCode MetalConvolution1x1::onFloat(const Tensor *input, const Tensor *output) {
    auto backend = static_cast<MetalBackend *>(this->backend());

    if(backend->isCommandEncoderSet()) {
        return NO_ERROR;
    }
    
    auto func = [=](){
        auto encoder    = backend->encoder();
        [encoder setComputePipelineState:mPipeline];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input)->extra.offset atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:1];
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
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
