//
//  MetalReduction.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalReduction.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "Macro.h"
#import "MetalBackend.hpp"
#import "TensorUtils.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalReduction::MetalReduction(Backend *backend, const ReductionParam *p) : Execution(backend) {
    auto integer = p->dType() == DataType_DT_INT32;
    switch (p->operation()) {
        case ReductionType_SUM:
            mKernel = integer ? @"reduce_sum_s" : @"reduce_sum_f";
            break;
        case ReductionType_ASUM:
        case ReductionType_SUMSQ:
            MNN_ASSERT(false); // both un-supported
            break;
        case ReductionType_MEAN:
            mKernel = integer ? @"reduce_mean_s" : @"reduce_mean_f";
            break;
        case ReductionType_MAXIMUM:
            mKernel = integer ? @"reduce_max_s" : @"reduce_max_f";
            break;
        case ReductionType_MINIMUM:
            mKernel = integer ? @"reduce_min_s" : @"reduce_min_f";
            break;
        case ReductionType_PROD:
            mKernel = integer ? @"reduce_prod_s" : @"reduce_prod_f";
            break;
        default:
            break;
    }
    for (int i = 0; i < p->dim()->size(); i++) {
        mDims.push_back(p->dim()->data()[i]);
    }
}

ErrorCode MetalReduction::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mDims.size() <= 1)
        return NO_ERROR;

    auto backend = static_cast<MetalBackend *>(this->backend());
    auto &input  = inputs[0];
    mMiddles.clear();
    for (int i = 0; i < mDims.size() - 1; i++) {
        auto middle = new Tensor(input->buffer().dimensions);
        TensorUtils::copyShape(input, middle);
        for (int j = 0; j <= i; j++) {
            middle->buffer().dim[mDims[j]].extent = 1;
        }
        backend->onAcquireBuffer(middle, Backend::DYNAMIC);
        mMiddles.push_back(std::shared_ptr<Tensor>(middle));
    }
    for (auto &t : mMiddles) {
        backend->onReleaseBuffer(t.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

static void encode(MNNMetalContext *context, id<MTLComputeCommandEncoder> encoder, MetalBandwidth bandwidth,
                   const Tensor *input, const Tensor *output, int axis) {
    auto ib         = input->buffer();
    int outsideSize = 1, axisSize = 1, insideSize = 1;
    if (axis >= 0) {
        for (int i = 0; i < axis; i++)
            outsideSize *= ib.dim[i].extent;
        axisSize = ib.dim[axis].extent;
        for (int i = axis + 1; i < ib.dimensions; i++)
            insideSize *= ib.dim[i].extent;
    } else {
        axisSize = input->elementSize();
    }

    auto shape                 = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    ((int *)shape.contents)[0] = outsideSize;
    ((int *)shape.contents)[1] = axisSize;
    ((int *)shape.contents)[2] = insideSize;
    ((int *)shape.contents)[3] = axisSize * insideSize;

    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:shape offset:0 atIndex:2];
    [context dispatchEncoder:encoder
                     threads:{ (NSUInteger) outsideSize, (NSUInteger)insideSize, 1 }
                   bandwidth:bandwidth];
}

ErrorCode MetalReduction::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto &input = inputs[0], &output = outputs[0];
    auto encoder   = [context encoder];
    auto bandwidth = [context load:mKernel encoder:encoder];

    if (mDims.empty()) {
        encode(context, encoder, bandwidth, input, output, -1);
    } else if (mDims.size() == 1) {
        encode(context, encoder, bandwidth, input, output, mDims[0]);
    } else {
        encode(context, encoder, bandwidth, input, mMiddles[0].get(), mDims[0]);
        for (int i = 1; i < mMiddles.size(); i++) {
            encode(context, encoder, bandwidth, mMiddles[i - 1].get(), mMiddles[i].get(), mDims[i]);
        }
        encode(context, encoder, bandwidth, mMiddles.back().get(), output, mDims.back());
    }

    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalReductionCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalReduction(backend, op->main_as_ReductionParam());
    }
};
REGISTER_METAL_OP_CREATOR(MetalReductionCreator, OpType_Reduction);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
