//
//  MetalLayerNorm.mm
//  MNN
//
//  Created by MNN on 2022/06/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalLayerNorm.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalLayerNorm::MetalLayerNorm(Backend *backend, const LayerNorm *layernorm)
    : Execution(backend), mGroup(layernorm->group()),
        mEps(layernorm->epsilon()) {
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();

    int axis_size = layernorm->axis()->size();
    mAxis.resize(axis_size);
    for (int i = 0; i < axis_size; ++i) {
        mAxis[i] = layernorm->axis()->Get(i);
    }

    if (layernorm->gamma() && layernorm->beta()) {
        has_gamma_beta_ = true;
        int gamma_size = layernorm->gamma()->size();
        const float* gamma_data = layernorm->gamma()->data();
        mGammaBuffer =
            [context newDeviceBuffer:gamma_size * sizeof(float) access:CPUWriteOnly];

        memcpy(mGammaBuffer.contents, (const void *)gamma_data, gamma_size * sizeof(float));
        
        if (layernorm->beta()->size() != gamma_size) {
            MNN_ERROR("Size of gamma and beta are not match in MetalLayerNorm.\n");
        }

        const float* beta_data = layernorm->beta()->data();
        mBetaBuffer =
            [context newDeviceBuffer:gamma_size * sizeof(float) access:CPUWriteOnly];
        memcpy(mBetaBuffer.contents, (const void *)beta_data, gamma_size * sizeof(float));
    }
}

ErrorCode MetalLayerNorm::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();

    auto input = inputs[0], output = outputs[0];
    
    mOutside = 1;
    mInside = 1;
    int rank = input->dimensions();
    if (mGroup > 1) {
        mOutside = input->length(0) * mGroup;
        for (int i = 1; i < rank; i++) {
            mInside *= input->length(i);
        }
        mInside /= mGroup;
        return NO_ERROR;
    }
    std::vector<int> axis(mAxis.size());
    for (int i = 0; i < mAxis.size(); ++i) {
        if (mAxis[i] < 0) {
            mAxis[i] += rank;
        }
    }
    std::sort(mAxis.begin(), mAxis.end());

    for (int i = 0; i < rank - axis.size(); ++i) {
        mOutside *= input->length(i);
    }
    for (int i = rank - axis.size(); i < rank; ++i) {
        mInside *= input->length(i);
    }
    
    mShapeBuffer = [context newDeviceBuffer:3 * sizeof(int) + sizeof(float) access:CPUWriteOnly];
    ((int *)mShapeBuffer.contents)[0]   = mInside;
    ((int *)mShapeBuffer.contents)[1]   = mOutside;
    ((float *)mShapeBuffer.contents)[2] = mEps;
    ((int *)mShapeBuffer.contents)[3]   = (int)has_gamma_beta_;

    
    bool parallel = (mInside > 32) && ((mInside & 3) == 0);
    mPipeline = [context pipelineWithName:parallel ? @"layernorm_x4" : @"layernorm_x1"];
    
    auto inside = parallel ? mInside/4 : mInside;
    mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake((NSUInteger)inside, (NSUInteger)mOutside, 1)];
    return NO_ERROR;
}

ErrorCode MetalLayerNorm::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    
    if(backend->isCommandEncoderSet()) {
        return NO_ERROR;
    }
    
    auto func = [=](){
        auto input = inputs[0], output = outputs[0];

        auto encoder   = backend->encoder();
        [encoder setComputePipelineState:mPipeline];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input)->extra.offset atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:1];
        [encoder setBuffer:mShapeBuffer offset:0 atIndex:2];
        [encoder setBuffer:mGammaBuffer offset:0 atIndex:3];
        [encoder setBuffer:mBetaBuffer offset:0 atIndex:4];

        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
        MNN_PRINT_ENCODER(context, encoder);

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

class MetalLayerNormCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *> &outputs) const {
        return new MetalLayerNorm(backend, op->main_as_LayerNorm());
    }
};
REGISTER_METAL_OP_CREATOR(MetalLayerNormCreator, OpType_LayerNorm);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
