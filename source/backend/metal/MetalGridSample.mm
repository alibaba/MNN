//
//  MetalGridSample.mm
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalGridSample.hpp"
#import "backend/metal/MNNMetalContext.h"

#if MNN_METAL_ENABLED
namespace MNN {

MetalGridSample::MetalGridSample(Backend *backend, const GridSample *gridSample)
        : Execution(backend) {
    mMode = gridSample->mode();
    mPaddingMode = gridSample->paddingMode();
    mAlignCorners = gridSample->alignCorners();

    auto metal_backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)metal_backend->context();
    mParams = [context newDeviceBuffer:9*sizeof(int) access:CPUWriteOnly];
}

ErrorCode MetalGridSample::onResize(const std::vector<Tensor *> &inputs,
                                    const std::vector<Tensor *> &outputs) {
    auto inputTensor = inputs[0];
    auto outputTensor = outputs[0];

    ((int *)mParams.contents)[0] = inputTensor->batch();//inputTensor->buffer().dim[0].extent; // batches
    ((int *)mParams.contents)[1] = inputTensor->channel();//->buffer().dim[1].extent; // channels
    ((int *)mParams.contents)[2] = inputTensor->height();//buffer().dim[2].extent; // inH
    ((int *)mParams.contents)[3] = inputTensor->width();//buffer().dim[3].extent; // inW
    ((int *)mParams.contents)[4] = outputTensor->height();//->buffer().dim[2].extent; // outH
    ((int *)mParams.contents)[5] = outputTensor->width();//->buffer().dim[3].extent; // outW
    ((int *)mParams.contents)[6] = mMode;
    ((int *)mParams.contents)[7] = mPaddingMode;
    ((int *)mParams.contents)[8] = mAlignCorners;

    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    mPipeline = [context pipelineWithName:@"grid_sample"];

    int batches = ((int *)mParams.contents)[0];
    int channels = ((int *)mParams.contents)[1];
    int outH = ((int *)mParams.contents)[4];
    int outW = ((int *)mParams.contents)[5];
    mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(outW, outH, batches)];

    //printf("re:%d %d %d, %d %d %d, %d %d\n", mThreads.first.width, mThreads.first.height, mThreads.first.depth, mThreads.second.width, mThreads.second.height, mThreads.second.depth, ((int *)mParams.contents)[3], ((int *)mParams.contents)[2]);
    return NO_ERROR;
}

ErrorCode MetalGridSample::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());

    if(backend->isCommandEncoderSet()) {
        return NO_ERROR;
    }
    
    auto func = [=](){
        auto encoder = backend->encoder();
        [encoder setComputePipelineState:mPipeline];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)inputs[0]->deviceId())->getBuffer() offset:TensorUtils::getDescribe(inputs[0])->extra.offset atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)inputs[1]->deviceId())->getBuffer() offset:TensorUtils::getDescribe(inputs[1])->extra.offset atIndex:1];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)outputs[0]->deviceId())->getBuffer() offset:TensorUtils::getDescribe(outputs[0])->extra.offset atIndex:2];
        [encoder setBuffer:mParams offset:0 atIndex:3];
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

class MetalGridSampleCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op,
                                Backend *backend, const std::vector<Tensor *>& outputs) const override {
        return new MetalGridSample(backend, op->main_as_GridSample());
    }
};

REGISTER_METAL_OP_CREATOR(MetalGridSampleCreator, OpType_GridSample);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
