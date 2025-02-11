//
//  MetalScale.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalScale.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalScale::MetalScale(Backend *backend, const Scale *scale) : MetalExecution(backend) {
    auto mtbn = static_cast<MetalBackend *>(backend);
    auto bufferAlloc = mtbn->getStaticBufferPool();
    auto context  = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    auto channel4 = UP_DIV(scale->channels(), 4) * 4;
    mBiasOffset = channel4 / 4;
    mScaleBias = bufferAlloc->alloc(2 * channel4 * sizeof(float));
    if (mScaleBias.first == nullptr) {
        mValid = false;
        return;
    }
    auto scalePtr = MetalBackend::getMemPtr(mScaleBias);
    ::memset(scalePtr, 0, 2 * channel4 * sizeof(float));
    ::memcpy(scalePtr, scale->scaleData()->data(), scale->channels() * sizeof(float));
    auto biasPtr = scalePtr + channel4 * sizeof(float);
    if (nullptr != scale->biasData()) {
        ::memcpy(biasPtr, scale->biasData()->data(), scale->channels() * sizeof(float));
    }
    mConst = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    mPipeline = [context pipelineWithName:@"scale_ca" fp16:mtbn->useFp16InsteadFp32()];
}
MetalScale::~MetalScale() {
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto bufferAlloc = mtbn->getStaticBufferPool();
    if (nullptr != mScaleBias.first) {
        bufferAlloc->free(mScaleBias);
    }
}

ErrorCode MetalScale::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto output = outputs[0];

    // shape
    int w   = output->width();
    int h   = output->height();
    int c   = output->channel();
    int z   = UP_DIV(c, 4);
    ((int *)mConst.contents)[0] = w*h;
    ((int *)mConst.contents)[1] = z;
    ((int *)mConst.contents)[2] = output->batch();
    ((int *)mConst.contents)[3] = mBiasOffset;
    mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(w*h, z * outputs[0]->batch(), 1)];
    return NO_ERROR;
}

void MetalScale::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto input = inputs[0], output = outputs[0];
    [encoder setComputePipelineState:mPipeline];
    MetalBackend::setTensor(input, encoder, 0);
    MetalBackend::setTensor(output, encoder, 1);
    [encoder setBuffer:mConst offset:0 atIndex:2];
    MetalBackend::setMem(mScaleBias, encoder, 3);
    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];

}

class MetalScaleCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        return new MetalScale(backend, op->main_as_Scale());
    }
};
REGISTER_METAL_OP_CREATOR(MetalScaleCreator, OpType_Scale);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
