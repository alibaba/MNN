//
//  CPUReverseSequence.cpp
//  MNN
//
//  Created by MNN on 2019/08/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUReverseSequence.hpp"
namespace MNN {
ErrorCode CPUReverseSequence::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (inputs[1]->length(0) != inputs[0]->length(mBatchDim)) {
        return NOT_SUPPORT;
    }

    int mid0 = mSeqDim;
    int mid1 = mBatchDim;
    if (mid0 > mid1) {
        auto temp = mid1;
        mid1      = mid0;
        mid0      = temp;
    }
    mInsideStride = inputs[0]->stride(mid1);
    mOutsideSize  = 1;
    for (int i = 0; i < mid0; ++i) {
        mOutsideSize *= inputs[0]->length(i);
    }
    mOutSideStride = inputs[0]->stride(mid0);
    mMidSize       = 1;
    for (int i = mid0 + 1; i < mid1; ++i) {
        mMidSize *= inputs[0]->length(i);
    }
    mMidStride = inputs[0]->stride(mid1);
    return NO_ERROR;
}
ErrorCode CPUReverseSequence::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input   = inputs[0];
    auto reverse = inputs[1];
    auto output  = outputs[0];
    ::memcpy(output->host<int32_t>(), input->host<int32_t>(), input->size());
    auto batchSize = input->length(mBatchDim);

    for (int n = 0; n < batchSize; ++n) {
        auto q = reverse->host<int32_t>()[n];
        if (q > input->length(mSeqDim)) {
            return INPUT_DATA_ERROR;
        }
        const auto inputBatch = input->host<int32_t>() + n * input->stride(mBatchDim);
        auto outputBatch      = output->host<int32_t>() + n * output->stride(mBatchDim);
        for (int p = 0; p < q; ++p) {
            // Map (p, n, ...) -> (q-p-1, n, ...)
            const auto inputP = inputBatch + (q - p - 1) * input->stride(mSeqDim);
            auto outputP      = outputBatch + p * output->stride(mSeqDim);
            for (int o = 0; o < mOutsideSize; ++o) {
                const auto srcO = inputP + o * mOutSideStride;
                auto dstO       = outputP + o * mOutSideStride;

                for (int m = 0; m < mMidSize; ++m) {
                    const auto srcM = srcO + m * mMidStride;
                    auto dstM       = dstO + m * mMidStride;

                    ::memcpy(dstM, srcM, mInsideStride * sizeof(int32_t));
                }
            }
        }
    }
    return NO_ERROR;
}
class CPUReverseSequenceCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (nullptr == op->main_as_ReverseSequenceParam()) {
            MNN_ERROR("Dont's has Parameters for OpType_ReverseSequence\n");
            return nullptr;
        }
        auto seqDim = op->main_as_ReverseSequenceParam()->seqDim();
        if (seqDim < 0) {
            seqDim += inputs[0]->dimensions();
        }
        auto batchDim = op->main_as_ReverseSequenceParam()->batchDim();
        if (batchDim < 0) {
            batchDim += inputs[0]->dimensions();
        }
        if (seqDim == batchDim) {
            MNN_ERROR("seq and batch dim can't be the same\n");
            return nullptr;
        }
        if (inputs[0]->getType().bits != 32) {
            MNN_ERROR("Don't support %d bit's ReverseSequence\n", inputs[0]->getType().bits);
            return nullptr;
        }
        return new CPUReverseSequence(backend, seqDim, batchDim);
    }
};

REGISTER_CPU_OP_CREATOR(CPUReverseSequenceCreator, OpType_ReverseSequence);
} // namespace MNN
