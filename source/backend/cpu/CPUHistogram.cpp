//
//  CPUHistogram.cpp
//  MNN
//
//  Created by MNN on 2022/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUHistogram.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include <cmath>

namespace MNN {

CPUHistogram::CPUHistogram(Backend *backend, const Op* op): Execution(backend) {
    auto param = op->main_as_ArgMax();
    mChannel = param->axis();
    mBinNum = param->outMaxVal();
    mMin = param->softmaxThreshold();
    mMax = param->topK();
    mAlpha = static_cast<float>(mBinNum) / (mMax - mMin);
    mBeta = mAlpha * mMin;
}

template <typename T>
ErrorCode CPUHistogram::histogram(Tensor* input, Tensor* output) {
    auto iptr = input->host<T>() + mChannel;
    auto optr = output->host<float>();
    memset(optr, 0, mBinNum * sizeof(float));
    for (int i = 0; i < mSize; i++) {
        T val = iptr[i * mStride];
        if (val >= mMin && val <= mMax) {
            const int bin = (int)(val * mAlpha - mBeta);
            optr[std::min(bin, mBinNum -1)]++;
        }
    }
    return NO_ERROR;
}

template <>
ErrorCode CPUHistogram::histogram<uint8_t>(Tensor* input, Tensor* output) {
    auto iptr = input->host<uint8_t>() + mChannel;
    auto optr = output->host<float>();
    int hist_map[256] = { 0 };
    // add hist_ptr to avoid iOS compile error: cannot refer to declaration with an array type inside block
    int* hist_ptr = hist_map;
//    auto numberThread = ((CPUBackend*)backend())->threadNumber();
    // TODO: Support multi thread
    int numberThread = 1;
    int sizeDivide = mSize / numberThread;
    MNN_CONCURRENCY_BEGIN(tId, numberThread) {
        int number = sizeDivide;
        if (tId == numberThread - 1) {
            number = mSize - tId * sizeDivide;
        }
        auto src = iptr + tId * sizeDivide * mStride;
        for (int i = 0; i < number; i++) {
            hist_ptr[src[i * mStride]]++;
        }
    }
    MNN_CONCURRENCY_END();
    memset(optr, 0, mBinNum * sizeof(float));
    for (int i = std::max(mMin, 0); i <= std::min(mMax, 255); i++) {
        int bin = std::min((int)(i * mAlpha - mBeta), mBinNum -1);
        optr[bin] = hist_map[i];
    }
    return NO_ERROR;
}

ErrorCode CPUHistogram::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0], output = outputs[0];
    /*
     1. mAlpha, mBeta
        binIdx = (val - mMin) / (mMax - mMin) * mBinNum
        mAlpha = mBinNum / (mMax - mMin)
        mBeta  = mAlpha * mMin
            -> binIdx = val * mAlpha - mBeta

     2. mChannel, mSize, mStride
        mChannel < 0  : compute all element of input
        mChannel >= 0 : last dim as channel, and other dim is plane; compute the mChannel plane of input;
     */
    if (mChannel < 0) {
        mSize = input->elementSize();
        mStride = 1;
        mChannel = 0;
    } else {
        mSize = 1;
        int lastDim = input->dimensions() - 1;
        for (int i = 0; i < lastDim; i++) {
            mSize *= input->length(i);
        }
        mStride = input->length(lastDim);
        MNN_ASSERT(mChannel <= mStride);
        mChannel = std::min(mChannel, mStride);
    }
    if (input->getType() == halide_type_of<float>()) {
        return histogram<float>(input, output);
    }
    if (input->getType() == halide_type_of<int>()) {
        return histogram<int>(input, output);
    }
    if (input->getType() == halide_type_of<uint8_t>()) {
        return histogram<uint8_t>(input, output);
    }
    return NOT_SUPPORT;
}

class CPUHistogramCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPUHistogram(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUHistogramCreator, OpType_Histogram);
} // namespace MNN
