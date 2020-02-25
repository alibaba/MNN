//
//  CPUPermute.cpp
//  MNN
//
//  Created by MNN on 2018/07/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUPermute.hpp"
#include "backend/cpu/CPUTranspose.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

CPUPermute::CPUPermute(Backend *b, const MNN::Op *op) : MNN::Execution(b) {
    auto shape = op->main_as_Permute()->dims();
    for (int i = 0; i < shape->size(); ++i) {
        mDims.push_back(shape->data()[i]);
    }
}

ErrorCode CPUPermute::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}

ErrorCode CPUPermute::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size());
    MNN_ASSERT(1 == outputs.size());

    auto input  = inputs[0];
    auto output = outputs[0];

    MNN_ASSERT(output->dimensions() == input->dimensions());
    MNN_ASSERT(2 <= output->dimensions() && output->dimensions() <= 5); // 2 <= tensor dim <= 5

    auto originInput  = input->host<float>();
    auto originOutput = output->host<float>();

    {
        bool noChange = true;
        for (int i = 0; i < (int)mDims.size(); ++i) {
            if (mDims[i] != i) {
                noChange = false;
                break;
            }
        }
        // mDims[i] == i, no change at all.
        if (noChange) {
            ::memcpy(originOutput, originInput, inputs[0]->size());
            return NO_ERROR;
        }
    }
    const int outputChannel = output->length(1);

    int strides[5][4];  // map from change of output index to change of input index on N, C4, H and W

    for (int i = 0; i < 5; ++i) {
        if (i >= input->dimensions()) {
            strides[i][0] = strides[i][1] = strides[i][2] = strides[i][3] = 0;
            continue;
        }
        int dim = mDims[i];
        int temp = input->stride(dim);
        if (dim >= 1) {
            temp *= 4;
        }
        if (dim == 1) {
            strides[i][0] = strides[i][1] = strides[i][2] = 1;
            strides[i][3]                                 = temp - 3;
        } else {
            strides[i][0] = strides[i][1] = strides[i][2] = strides[i][3] = temp;
        }
    }
    const int ocTotalStride = strides[1][0] + strides[1][1] + strides[1][2] + strides[1][3];
    // compute prefix sum of output 0 dim stride to avoid frequent assignment of variables in the deepest loops
    for (int i = 1; i < 4; ++i) {
        strides[1][i] += strides[1][i - 1];
    }
#define PTR_DEFINE(i) \
const int outputLength##i = ALIMAX(output->length(i - 1), 1); \

#define PTR_BEGIN(var, i) \
for (int var = 0; var < outputLength##i; ++var) { \
const int inputIndex##i = inputIndex;

#define PTR_END(var, i) \
inputIndex = inputIndex##i + strides[i - 1][var % 4]; \
}

    PTR_DEFINE(3)
    PTR_DEFINE(4)
    PTR_DEFINE(5)
    for (int ob = 0, outputIndex = 0, inputIndex = 0; ob < output->length(0); ++ob) {
        const int inputIndex1 = inputIndex;
        for (int oz = 0; oz <= outputChannel - 4; oz += 4) {
            const int inputIndex2 = inputIndex;
            PTR_BEGIN(od, 3)
            PTR_BEGIN(oy, 4)
            PTR_BEGIN(ox, 5)
            originOutput[outputIndex++] = originInput[inputIndex];
            originOutput[outputIndex++] = originInput[inputIndex + strides[1][0]];
            originOutput[outputIndex++] = originInput[inputIndex + strides[1][1]];
            originOutput[outputIndex++] = originInput[inputIndex + strides[1][2]];
            PTR_END(ox, 5)
            PTR_END(oy, 4)
            PTR_END(od, 3)
            inputIndex = inputIndex2 + ocTotalStride;
        }
        if (outputChannel % 4 != 0) {
            PTR_BEGIN(od, 3)
            PTR_BEGIN(oy, 4)
            PTR_BEGIN(ox, 5)
            originOutput[outputIndex++] = originInput[inputIndex];
            for (int oz = 0; oz < outputChannel % 4 - 1; ++oz) {
                originOutput[outputIndex++] = originInput[inputIndex + strides[1][oz]];
            }
            for (int oz = outputChannel % 4; oz < 4; ++oz) {
                originOutput[outputIndex++] = 0.0f;
            }
            PTR_END(ox, 5)
            PTR_END(oy, 4)
            PTR_END(od, 3)
        }
        inputIndex = inputIndex1 + strides[0][ob % 4];
    }
#undef PTR_DEFINE
#undef PTR_BEGIN
#undef PTR_END

    return NO_ERROR;
}

class CPUWrapPermute : public Execution {
public:
    CPUWrapPermute(Backend *bn, const MNN::Op *op) : Execution(bn) {
        auto shape = op->main_as_Permute()->dims();
        mDims.reset(Tensor::create<int>({(int)shape->size()}));
        if (nullptr == mDims->host<int>()) {
            mValid = false;
            return;
        }
        ::memcpy(mDims->host<int>(), shape->data(), mDims->size());
        mTranspose.reset(new CPUTranspose(bn, DataType_DT_FLOAT));
    }
    virtual ~CPUWrapPermute() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto format = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        if (MNN_DATA_FORMAT_NC4HW4 == format) {
            mTempSource.reset(Tensor::createDevice<float>(inputs[0]->shape(), Tensor::CAFFE));
            mTempDest.reset(Tensor::createDevice<float>(outputs[0]->shape(), Tensor::CAFFE));

            bool valid = backend()->onAcquireBuffer(mTempSource.get(), Backend::DYNAMIC);
            valid      = valid && backend()->onAcquireBuffer(mTempDest.get(), Backend::DYNAMIC);
            if (!valid) {
                return OUT_OF_MEMORY;
            }

            backend()->onReleaseBuffer(mTempSource.get(), Backend::DYNAMIC);
            backend()->onReleaseBuffer(mTempDest.get(), Backend::DYNAMIC);

            mWrapInputs  = {mTempSource.get(), mDims.get()};
            mWrapOutputs = {mTempDest.get()};
            mNeedCopy    = true;
        } else {
            mWrapOutputs = outputs;
            mWrapInputs  = {inputs[0], mDims.get()};
            mNeedCopy    = false;
        }
        return mTranspose->onResize(mWrapInputs, mWrapOutputs);
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        if (mNeedCopy) {
            backend()->onCopyBuffer(inputs[0], mTempSource.get());
        }
        auto code = mTranspose->onExecute(mWrapInputs, mWrapOutputs);
        if (NO_ERROR != code) {
            return code;
        }
        if (mNeedCopy) {
            backend()->onCopyBuffer(mTempDest.get(), outputs[0]);
        }
        return NO_ERROR;
    }

private:
    std::shared_ptr<Tensor> mDims;
    std::shared_ptr<Tensor> mTempSource;
    std::shared_ptr<Tensor> mTempDest;
    std::shared_ptr<Execution> mTranspose;
    std::vector<Tensor *> mWrapInputs;
    std::vector<Tensor *> mWrapOutputs;
    bool mNeedCopy = false;
};

class CPUPermuteCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (op->main_as_Permute()->dims()->size() > 5 ||
            TensorUtils::getDescribe(inputs[0])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
            return new CPUWrapPermute(backend, op);
        }
        return new CPUPermute(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUPermuteCreator, OpType_Permute);
} // namespace MNN
