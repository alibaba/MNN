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
    int stepSlices  = igz;

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
    
    // update threadgroup memory if needed
    mLocalPreferred = isThreadgroupLocalPreferred(input, output);
    mLocalPreferred = false;//not used temporarily
    
    if (mLocalPreferred) {
        int unit        = sizeof(metal_float);
        int sliceMemory = 4 * mKernelY * mKernelX * 4 * unit;
        int maxMemory = sliceMemory > kMaxGemmStepMemory ? (int)context.maxThreadgroupMemoryLength : kMaxGemmStepMemory;
        int maxStepSlices  = maxMemory / sliceMemory;
        int steps          = UP_DIV(igz, maxStepSlices);
        stepSlices         = UP_DIV(igz, steps);
        mThreadgroupMemory = stepSlices * sliceMemory;
    }
    
    if (mLocalPreferred) {
        mPipeline = [context pipelineWithName:@"conv_local"];
        MTLSize global    = {(NSUInteger)UP_DIV(ow, 4), (NSUInteger)oh, (NSUInteger)ogz};

        MetalBandwidth bandwidth = {mPipeline.threadExecutionWidth, mPipeline.maxTotalThreadsPerThreadgroup, NO};
        _mThreads  = std::make_pair(global, bandwidth);
    } else if (ow * oh >= 32 ? ogz >= 4 : ogz >= 128){
        if(mKernelX==3 && mKernelY==3 && mStrideX==1 && mStrideY==1 && padX==1 && padY==1 && mDilateX==1 && mDilateY==1) {

            mPipeline = [context pipelineWithName:@"convk3s1d1p1_w2z4"];
            
            NSUInteger gid_x = UP_DIV(ow, 2);
            NSUInteger gid_y = oh;
            NSUInteger gid_z = UP_DIV(ogz, 4);
                
            NSArray *arr = [NSArray arrayWithObjects:(__bridge id<MTLBuffer>)(void *)input->deviceId(),
                            (__bridge id<MTLBuffer>)((void *)output->deviceId()),
                            mConstBuffer.buffer(), mWeight, mBias, nil];

            mThreads = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr];
            //printf("conv_w2z4, %d %d %d %d\n", ow, oh, ogz, igz);
        } else {
            mPipeline = [context pipelineWithName:@"conv_z4"];
            
            NSUInteger gid_x = ow;
            NSUInteger gid_y = oh;
            NSUInteger gid_z = UP_DIV(ogz, 4);
            
            NSArray *arr = [NSArray arrayWithObjects:(__bridge id<MTLBuffer>)(void *)input->deviceId(),
                            (__bridge id<MTLBuffer>)((void *)output->deviceId()),
                            mConstBuffer.buffer(), mWeight, mBias, nil];
            
            mThreads = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr];
            //printf("conv_z4, %d %d %d %d\n", ow, oh, ogz, igz);
        }
    } else {
        mPipeline = [context pipelineWithName:@"conv"];
        
        NSUInteger gid_x = ow;
        NSUInteger gid_y = oh;
        NSUInteger gid_z = ogz;
            
        NSArray *arr = [NSArray arrayWithObjects:(__bridge id<MTLBuffer>)(void *)input->deviceId(),
                        (__bridge id<MTLBuffer>)((void *)output->deviceId()),
                        mConstBuffer.buffer(), mWeight, mBias, nil];

        mThreads = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr];
        //printf("conv_z1, %d %d %d %d\n", ow, oh, ogz, igz);
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

        auto iw = input->width(), ih = input->height(), iz = UP_DIV(input->channel(), 4);
        auto ow = output->width(), oh = output->height(), oz = UP_DIV(output->channel(), 4), ogz = oz / mGroups;
        auto unit = sizeof(metal_float);
        auto ib = iw * ih * iz * 4 * unit, ig = ib / mGroups;
        auto ob = ow * oh * oz * 4 * sizeof(metal_float), og = ob / mGroups;

        auto encoder    = backend->encoder();
        
        auto bandwidth  = (MetalBandwidth){mPipeline.threadExecutionWidth, mPipeline.maxTotalThreadsPerThreadgroup, NO};

        for (int b = 0; b < input->batch(); b++) {
            for (int g = 0; g < mGroups; g++) {
                [encoder setComputePipelineState:mPipeline];
                [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:b * ib + g * ig atIndex:0];
                [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:b * ob + g * og atIndex:1];
                [encoder setBuffer:mConstBuffer.buffer() offset:0 atIndex:2];
                [encoder setBuffer:mWeight offset:g * mWeight.length / mGroups atIndex:3];
                [encoder setBuffer:mBias offset:g * mBias.length / mGroups atIndex:4];
                if (mLocalPreferred) {
                    [encoder setThreadgroupMemoryLength:mThreadgroupMemory atIndex:0];
                    //[encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
                    [context dispatchEncoder:encoder threads:_mThreads.first threadsPerGroup:{ 1, 1, (NSUInteger)ogz } bandwidth:_mThreads.second];
                } else {
                    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
                }
                
                //need to commit
                if(context.isCommitEachShader) {
                    backend->flushEncoder();
                    [context commit_net];
                }
            }
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
