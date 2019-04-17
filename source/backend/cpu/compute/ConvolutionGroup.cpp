//
//  ConvolutionGroup.cpp
//  MNN
//
//  Created by MNN on 2018/08/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvolutionGroup.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"

namespace MNN {
ConvolutionGroup::ConvolutionGroup(Backend *b, const std::vector<std::shared_ptr<Execution>> &subConvolution)
    : MNN::Execution(b) {
    mSubConvolution = subConvolution;
    auto group      = subConvolution.size();
    MNN_ASSERT(group > 1);

    mInputRaw.reset(new Tensor(4));
    mInputUnit.reset(new Tensor(4));
    mOutputRaw.reset(new Tensor(4));
    mOutputUnit.reset(new Tensor(4));

    mInputUnitWrap.push_back(mInputUnit.get());
    mOutputUnitWrap.push_back(mOutputUnit.get());
}

ErrorCode ConvolutionGroup::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto ib = inputs[0]->buffer();
    auto ob = outputs[0]->buffer();
    ::memcpy(mInputRaw->buffer().dim, ib.dim, ib.dimensions * sizeof(halide_dimension_t));
    mInputRaw->buffer().dimensions    = ib.dimensions;
    mInputRaw->buffer().dim[1].flags  = 0;
    mInputRaw->buffer().dim[0].extent = 1;

    ::memcpy(mInputUnit->buffer().dim, ib.dim, ib.dimensions * sizeof(halide_dimension_t));
    mInputUnit->buffer().dimensions    = ib.dimensions;
    mInputUnit->buffer().dim[1].flags  = Tensor::REORDER_4;
    mInputUnit->buffer().dim[1].extent = ib.dim[1].extent / mSubConvolution.size();
    mInputUnit->buffer().dim[0].extent = 1;

    ::memcpy(mOutputRaw->buffer().dim, ob.dim, ob.dimensions * sizeof(halide_dimension_t));
    mOutputRaw->buffer().dimensions    = ob.dimensions;
    mOutputRaw->buffer().dim[1].flags  = 0;
    mOutputRaw->buffer().dim[0].extent = 1;

    ::memcpy(mOutputUnit->buffer().dim, ob.dim, ob.dimensions * sizeof(halide_dimension_t));
    mOutputUnit->buffer().dimensions    = ob.dimensions;
    mOutputUnit->buffer().dim[1].flags  = Tensor::REORDER_4;
    mOutputUnit->buffer().dim[1].extent = ob.dim[1].extent / mSubConvolution.size();
    mOutputUnit->buffer().dim[0].extent = 1;

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
    auto inputBatchSize  = input->width() * input->height() * ALIGN_UP4(input->channel());
    auto outputBatchSize = output->width() * output->height() * ALIGN_UP4(output->channel());

    for (int b = 0; b < batch; ++b) {
        auto srcOrigin = input->host<float>() + b * inputBatchSize;
        auto dstOrigin = output->host<float>() + b * outputBatchSize;

        MNNUnpackC4(mInputRaw->host<float>(), srcOrigin, input->width() * input->height(), input->channel());
        int inputGroupSize   = input->width() * input->height() * input->channel() / mSubConvolution.size();
        int outputGroupSize  = output->width() * output->height() * output->channel() / mSubConvolution.size();
        int subInputChannel  = input->channel() / mSubConvolution.size();
        int subOutputChannel = output->channel() / mSubConvolution.size();
        for (int group = 0; group < mSubConvolution.size(); ++group) {
            MNNPackC4(mInputUnit->host<float>(), mInputRaw->host<float>() + group * inputGroupSize,
                      input->width() * input->height(), subInputChannel);
            mSubConvolution[group]->onExecute(mInputUnitWrap, mOutputUnitWrap);
            MNNUnpackC4(mOutputRaw->host<float>() + group * outputGroupSize, mOutputUnit->host<float>(),
                        output->width() * output->height(), subOutputChannel);
        }
        MNNPackC4(dstOrigin, mOutputRaw->host<float>(), output->width() * output->height(), output->channel());
    }

    return NO_ERROR;
}
} // namespace MNN
