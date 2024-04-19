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
    : MetalExecution(backend), mGroup(layernorm->group()),
        mEps(layernorm->epsilon()) {
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();

    int axis_size = 0;
    if (nullptr != layernorm->axis()) {
        axis_size = layernorm->axis()->size();
    }
    mAxisSize = axis_size;

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
    mShapeBuffer = [context newDeviceBuffer:3 * sizeof(int) + sizeof(float) access:CPUWriteOnly];
    RMSNorm = layernorm->useRMSNorm();
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
    } else {
        for (int i = 0; i < rank - mAxisSize; ++i) {
            mOutside *= input->length(i);
        }
        for (int i = rank - mAxisSize; i < rank; ++i) {
            mInside *= input->length(i);
        }
    }
    
    ((int *)mShapeBuffer.contents)[0]   = mInside;
    ((int *)mShapeBuffer.contents)[1]   = mOutside;
    ((float *)mShapeBuffer.contents)[2] = mEps;
    ((int *)mShapeBuffer.contents)[3]   = (int)has_gamma_beta_;

    
    bool parallel = (mInside > 32) && ((mInside & 3) == 0);
    if(RMSNorm){
        mPipeline = [context pipelineWithName:parallel ? @"layernorm_x4_rms" : @"layernorm_x1_rms" fp16:backend->useFp16InsteadFp32()];
    }else{
        mPipeline = [context pipelineWithName:parallel ? @"layernorm_x4" : @"layernorm_x1" fp16:backend->useFp16InsteadFp32()];
    }
    
    auto inside = parallel ? mInside/4 : mInside;
    mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake((NSUInteger)inside, (NSUInteger)mOutside, 1)];
    return NO_ERROR;
}

void MetalLayerNorm::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    [encoder setComputePipelineState:mPipeline];
    MetalBackend::setTensor(input, encoder, 0);
    MetalBackend::setTensor(output, encoder, 1);
    [encoder setBuffer:mShapeBuffer offset:0 atIndex:2];
    if (!has_gamma_beta_) {
        // Set fake buffer to avoid validate
        MetalBackend::setTensor(input, encoder, 3);
        MetalBackend::setTensor(input, encoder, 4);
    } else {
        [encoder setBuffer:mGammaBuffer offset:0 atIndex:3];
        [encoder setBuffer:mBetaBuffer offset:0 atIndex:4];
    }

    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
    MNN_PRINT_ENCODER(context, encoder);
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
