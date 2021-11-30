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
    int ic_4      = UP_DIV(input->channel(), 4);
    int oc_4      = UP_DIV(output->channel(), 4);

    int unit          = sizeof(metal_float);
    int sliceMemory   = 4 * mKernelY * mKernelX * 4 * unit;
    int maxMemory     = sliceMemory > kMaxGemmStepMemory ? (int)context.maxThreadgroupMemoryLength : kMaxGemmStepMemory;
    int maxStepSlices = maxMemory / sliceMemory;
    int steps         = UP_DIV(ic_4, maxStepSlices);

    static int kGemmUnroll = 4;
    return oc_4 * oc_4 * kGemmUnroll / steps / steps >= output->width() * output->height();
}

ErrorCode MetalConvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MetalConvolutionCommon::onResize(inputs, outputs);

    // prepare
    auto backend = static_cast<MetalBackend *>(this->backend());
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
    
    // update threadgroup memory if needed
    mLocalPreferred = isThreadgroupLocalPreferred(input, output);
    mLocalPreferred = false;//not used temporarily
    
    if (mLocalPreferred) {
        int unit        = sizeof(metal_float);
        int sliceMemory = 4 * mKernelY * mKernelX * 4 * unit;
        int maxMemory = sliceMemory > kMaxGemmStepMemory ? (int)context.maxThreadgroupMemoryLength : kMaxGemmStepMemory;
        int maxStepSlices  = maxMemory / sliceMemory;
        int steps          = UP_DIV(ic_4, maxStepSlices);
        stepSlices         = UP_DIV(ic_4, steps);
        mThreadgroupMemory = stepSlices * sliceMemory;
    }
    
    MetalRuntime* rt = (MetalRuntime *)backend->runtime();
    bool isMuchComputer = (ow * oh >= 32 ? oc_4 >= 4 : oc_4 >= 128);
    bool is3x3s1Conv = (mKernelX==3 && mKernelY==3 && mStrideX==1 && mStrideY==1 && padX==1 && padY==1 && mDilateX==1 && mDilateY==1);
    
    if(isMuchComputer && is3x3s1Conv) {
        mPipeline = [context pipelineWithName:@"convk3s1d1p1_w2z4"];
        
        NSUInteger gid_x = UP_DIV(ow, 2);
        NSUInteger gid_y = oh;
        NSUInteger gid_z = UP_DIV(oc_4, 4) * ob;
            
        NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                        (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                        mConstBuffer, mWeight, mBias, nil];
        
        std::string name = "convk3s1d1p1_w2z4";
        MetalRuntime *rt = (MetalRuntime *)backend->runtime();
        auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name];
        mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));

        //printf("conv3x3_w2z4, cost:%d\n", (int)std::get<2>(ret));
    } else {
        if(rt->getTuneLevel() == Never) {
            int packC = 1;
            NSString* kernelName = @"conv";
            if(isMuchComputer) {
                packC = 4;
                kernelName = @"conv_z4";
            }
            NSUInteger gid_x = ow;
            NSUInteger gid_y = oh;
            NSUInteger gid_z = UP_DIV(oc_4, packC) * ob;

            mPipeline = [context pipelineWithName:kernelName];
            NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                            (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                            mConstBuffer, mWeight, mBias, nil];
            
            std::string name = [kernelName UTF8String];
            auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name];
            mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
        } else {
            // {"conv_2d_c4h1w4", "conv_2d_c4h1w2", "conv_2d_c4h1w1", "conv_2d_c8h1w1", };
            const int total_kernel = 2;
            NSString* shaderName[total_kernel] = {@"conv", @"conv_z4"};
            int itemW[total_kernel] = {1, 1};
            int itemH[total_kernel] = {1, 1};
            int itemC[total_kernel] = {1, 4};
            
            int actual_kernel = 2;
            std::pair<NSUInteger, int> min_cost(INT_MAX, 0);//(min_time, min_index)
            
            NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                            (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                            mConstBuffer, mWeight, mBias, nil];

            for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
                id<MTLComputePipelineState> pipeline = [context pipelineWithName:shaderName[knl_idx]];
                NSUInteger gid_x = UP_DIV(ow, itemW[knl_idx]);
                NSUInteger gid_y = UP_DIV(oh, itemH[knl_idx]);
                NSUInteger gid_z = UP_DIV(oc_4, itemC[knl_idx]) * ob;

                std::string name = [shaderName[knl_idx] UTF8String];
                auto ret = [context getGridAndThreadgroup:pipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name];
                
                if(min_cost.first > std::get<2>(ret)) {
                    min_cost.first = std::get<2>(ret);
                    min_cost.second = knl_idx;
                    mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
                }
                //printf("conv1x1 idx:%d, global:%d %d %d, local:%d %d %d, min_cost:%d\n", knl_idx, (int)retTune.second.first.width, (int)retTune.second.first.height, (int)retTune.second.first.depth, (int)retTune.second.second.width, (int)retTune.second.second.height, (int)retTune.second.second.depth, (int)retTune.first);
            }
            //printf("conv idx:%d, min_cost:%d\n", (int)min_cost.second, (int)min_cost.first);

            mPipeline = [context pipelineWithName:shaderName[min_cost.second]];
        }
    }
    return NO_ERROR;
}

ErrorCode MetalConvolution::onFloat(const Tensor *input, const Tensor *output) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();

    if(backend->isCommandEncoderSet()) {
        return NO_ERROR;
    }
    
    auto func = [=](){
        auto oc_4 = UP_DIV(output->channel(), 4);
        auto encoder    = backend->encoder();
        
        auto bandwidth  = (MetalBandwidth){mPipeline.threadExecutionWidth, mPipeline.maxTotalThreadsPerThreadgroup, NO};

        [encoder setComputePipelineState:mPipeline];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input)->extra.offset atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:1];
        [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
        [encoder setBuffer:mWeight offset:0 atIndex:3];
        [encoder setBuffer:mBias offset:0 atIndex:4];
        if (mLocalPreferred) {
            [encoder setThreadgroupMemoryLength:mThreadgroupMemory atIndex:0];
            //[encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
            [context dispatchEncoder:encoder threads:_mThreads.first threadsPerGroup:{ 1, 1, (NSUInteger)oc_4 } bandwidth:_mThreads.second];
        } else {
            [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
        }
        
        //need to commit
        if(backend->isCmdBufferCommit()) {
            backend->flushEncoder();
            [context commit_net];
        }
    };
    
    func();
    backend->addOpEncoder(func);

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
        if (inputs.size() > 1) {
            return nullptr;
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
