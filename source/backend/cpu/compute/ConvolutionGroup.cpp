//
//  ConvolutionGroup.cpp
//  MNN
//
//  Created by MNN on 2018/08/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvolutionGroup.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
ConvolutionGroup::ConvolutionGroup(Backend *b, const std::vector<std::shared_ptr<Execution>> &subConvolution)
    : MNN::Execution(b) {
    mSubConvolution = subConvolution;
    MNN_ASSERT(subConvolution.size() > 1);

    mInputRaw.reset(new Tensor(4));
    mInputUnit.reset(new Tensor(4, Tensor::CAFFE_C4));
    mOutputRaw.reset(new Tensor(4));
    mOutputUnit.reset(new Tensor(4, Tensor::CAFFE_C4));

    mInputUnitWrap.push_back(mInputUnit.get());
    mOutputUnitWrap.push_back(mOutputUnit.get());
}

ErrorCode ConvolutionGroup::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto ib = inputs[0]->buffer();
    auto ob = outputs[0]->buffer();
    ::memcpy(mInputRaw->buffer().dim, ib.dim, ib.dimensions * sizeof(halide_dimension_t));
    mInputRaw->buffer().dimensions    = ib.dimensions;
    mInputRaw->buffer().dim[0].extent = 1;

    ::memcpy(mInputUnit->buffer().dim, ib.dim, ib.dimensions * sizeof(halide_dimension_t));
    mInputUnit->buffer().dimensions    = ib.dimensions;
    mInputUnit->buffer().dim[1].extent = ib.dim[1].extent / mSubConvolution.size();
    mInputUnit->buffer().dim[0].extent = 1;
    TensorUtils::getDescribe(mInputUnit.get())->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
    TensorUtils::setLinearLayout(mInputUnit.get());

    ::memcpy(mOutputRaw->buffer().dim, ob.dim, ob.dimensions * sizeof(halide_dimension_t));
    mOutputRaw->buffer().dimensions    = ob.dimensions;
    mOutputRaw->buffer().dim[0].extent = 1;

    ::memcpy(mOutputUnit->buffer().dim, ob.dim, ob.dimensions * sizeof(halide_dimension_t));
    mOutputUnit->buffer().dimensions    = ob.dimensions;
    mOutputUnit->buffer().dim[1].extent = ob.dim[1].extent / mSubConvolution.size();
    mOutputUnit->buffer().dim[0].extent = 1;
    TensorUtils::getDescribe(mOutputUnit.get())->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
    TensorUtils::setLinearLayout(mOutputUnit.get());

    backend()->onAcquireBuffer(mOutputUnit.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mInputUnit.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mInputRaw.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mOutputRaw.get(), Backend::DYNAMIC);

    for (auto &iter : mSubConvolution) {
        iter->onResize(mInputUnitWrap, mOutputUnitWrap);
    }

    backend()->onReleaseBuffer(mOutputUnit.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mInputUnit.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mInputRaw.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mOutputRaw.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode ConvolutionGroup::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input           = inputs[0];
    auto output          = outputs[0];
    int batch            = input->buffer().dim[0].extent;
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto inputBatchSize  = input->width() * input->height() * UP_DIV(input->channel(), core->pack) * core->pack;
    auto outputBatchSize = output->width() * output->height() * UP_DIV(output->channel(), core->pack) * core->pack;

    for (int b = 0; b < batch; ++b) {
        auto srcOrigin = input->host<uint8_t>() + b * inputBatchSize * core->bytes;
        auto dstOrigin = output->host<uint8_t>() + b * outputBatchSize * core->bytes;

        core->MNNUnpackCUnit(mInputRaw->host<float>(), (float*)srcOrigin, input->width() * input->height(), input->channel());
        int inputGroupSize   = input->width() * input->height() * input->channel() / mSubConvolution.size();
        int outputGroupSize  = output->width() * output->height() * output->channel() / mSubConvolution.size();
        int subInputChannel  = input->channel() / mSubConvolution.size();
        int subOutputChannel = output->channel() / mSubConvolution.size();
        for (int group = 0; group < mSubConvolution.size(); ++group) {
            core->MNNPackCUnit(mInputUnit->host<float>(), (const float*)(mInputRaw->host<uint8_t>() + group * inputGroupSize * core->bytes),
                      input->width() * input->height(), subInputChannel);
            mSubConvolution[group]->onExecute(mInputUnitWrap, mOutputUnitWrap);
            core->MNNUnpackCUnit((float*)(mOutputRaw->host<uint8_t>() + group * outputGroupSize * core->bytes), mOutputUnit->host<float>(),
                        output->width() * output->height(), subOutputChannel);
        }
        core->MNNPackCUnit((float*)dstOrigin, mOutputRaw->host<float>(), output->width() * output->height(), output->channel());
    }
    return NO_ERROR;
}
} // namespace MNN
