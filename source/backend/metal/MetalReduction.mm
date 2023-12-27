//
//  MetalReduction.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalReduction.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"
#import "core/TensorUtils.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalReduction::MetalReduction(Backend *backend, const ReductionParam *p, halide_type_t type) : MetalExecution(backend) {
    auto integer = type.code == halide_type_int;
    NSString *kernel;
    switch (p->operation()) {
        case ReductionType_SUM:
            kernel = integer ? @"reduce_sum_s" : @"reduce_sum_f";
            break;
        case ReductionType_ASUM:
        case ReductionType_SUMSQ:
            MNN_ASSERT(false); // both un-supported
            break;
        case ReductionType_MEAN:
            kernel = integer ? @"reduce_mean_s" : @"reduce_mean_f";
            break;
        case ReductionType_MAXIMUM:
            kernel = integer ? @"reduce_max_s" : @"reduce_max_f";
            break;
        case ReductionType_MINIMUM:
            kernel = integer ? @"reduce_min_s" : @"reduce_min_f";
            break;
        case ReductionType_PROD:
            kernel = integer ? @"reduce_prod_s" : @"reduce_prod_f";
            break;
        default:
            break;
    }
    // The reduce after geometry compute has only one axis
    mAxis = p->dim()->data()[0];
    auto mkbn = static_cast<MetalBackend *>(backend);
    auto context = (__bridge MNNMetalContext *)mkbn->context();
    mConst = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    mPipeline = [context pipelineWithName:kernel  fp16:mkbn->useFp16InsteadFp32()];
}

ErrorCode MetalReduction::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    int outsideSize = 1, axisSize = 1, insideSize = 1;
    for (int i = 0; i < mAxis; i++) {
        outsideSize *= inputs[0]->length(i);
    }
    axisSize = inputs[0]->length(mAxis);
    for (int i = mAxis + 1; i < inputs[0]->dimensions(); i++) {
        insideSize *= inputs[0]->length(i);
    }
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    ((int *)mConst.contents)[0] = outsideSize;
    ((int *)mConst.contents)[1] = axisSize;
    ((int *)mConst.contents)[2] = insideSize;
    ((int *)mConst.contents)[3] = axisSize * insideSize;
    mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(outsideSize, insideSize, 1)];
    return NO_ERROR;
}

void MetalReduction::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto &input = inputs[0], &output = outputs[0];
    [encoder setComputePipelineState:mPipeline];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input)->extra.offset atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:1];
    [encoder setBuffer:mConst offset:0 atIndex:2];
    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
}

class MetalReductionCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        auto param = op->main_as_ReductionParam();
        switch (param->operation()) {
            case ReductionType_ALL:
            case ReductionType_ANY:
            case ReductionType_ASUM:
            case ReductionType_SUMSQ:
                return nullptr;
            default:
                break;
        };

        return new MetalReduction(backend, op->main_as_ReductionParam(), inputs[0]->getType());
    }
};
REGISTER_METAL_OP_CREATOR(MetalReductionCreator, OpType_Reduction);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
