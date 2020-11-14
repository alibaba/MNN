//
//  CPUInnerProduct.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUInnerProduct.hpp"
#include "core/AutoStorage.h"
#include "backend/cpu/CPUConvolution.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "core/Macro.h"

namespace MNN {

class CPUInnerProductExecutor : public Execution {
public:
    CPUInnerProductExecutor(Backend *bn, const MNN::Op *op) : Execution(bn) {
        auto parameter  = op->main_as_InnerProduct();
        int outputCount = parameter->outputCount();
        int srcCount    = parameter->weight()->size() / outputCount;
        mWeight.reset(CPUConvolution::reorderWeightSize(srcCount, outputCount, 1, 4, 4));
        if (mWeight.get() == nullptr) {
            mValid = false;
            return;
        }
        mWeight.clear();
        AutoStorage<float> cache(mWeight.size());
        CPUConvolution::reorderWeight(mWeight.get(), parameter->weight()->data(), srcCount, outputCount, 1, cache.get());
        mBias.reset(ALIGN_UP4(outputCount));
        mBias.clear();
        ::memcpy(mBias.get(), parameter->bias()->data(), parameter->bias()->size() * sizeof(float));
        mInputPad.reset(new Tensor(2));
        mOutputPad.reset(new Tensor(2));
    }
    virtual ~CPUInnerProductExecutor() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto input                         = inputs[0];
        auto output                        = outputs[0];
        mOutputPad->buffer().dim[1].extent = ALIGN_UP4(output->buffer().dim[1].extent);
        mOutputPad->buffer().dim[0].extent = output->buffer().dim[0].extent;
        mInputPad->buffer().dim[1].extent  = ALIGN_UP4(input->buffer().dim[1].extent);
        mInputPad->buffer().dim[0].extent  = input->buffer().dim[0].extent;

        backend()->onAcquireBuffer(mOutputPad.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mInputPad.get(), Backend::DYNAMIC);

        backend()->onReleaseBuffer(mOutputPad.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mInputPad.get(), Backend::DYNAMIC);
        return NO_ERROR;
    }

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto input  = inputs[0];
        auto output = outputs[0];

        auto originSource = input->host<float>();
        int srcDepthQuad  = mInputPad->buffer().dim[1].extent / 4;
        int dstDepthQuad  = mOutputPad->buffer().dim[1].extent / 4;
        auto width        = mInputPad->buffer().dim[0].extent;

        auto source = mInputPad->host<float>();
        MNNPackC4(source, originSource, width, input->buffer().dim[1].extent);
        auto dest = mOutputPad->host<float>();
        MNNGemmFloatCommon_4(dest, source, mWeight.get(), srcDepthQuad, 4 * width, dstDepthQuad, width, 0);

        MNNAddBias(dest, mBias.get(), width, dstDepthQuad);
        auto originDest = output->host<float>();
        MNNUnpackC4(originDest, dest, width, output->buffer().dim[1].extent);

        return NO_ERROR;
    }

private:
    AutoStorage<float> mWeight;
    AutoStorage<float> mBias;

    std::unique_ptr<Tensor> mInputPad;
    std::unique_ptr<Tensor> mOutputPad;
};

Execution *CPUInnerProductCreator::onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                            const MNN::Op *op, Backend *backend) const {
    return new CPUInnerProductExecutor(backend, op);
}

REGISTER_CPU_OP_CREATOR(CPUInnerProductCreator, OpType_InnerProduct);

} // namespace MNN
