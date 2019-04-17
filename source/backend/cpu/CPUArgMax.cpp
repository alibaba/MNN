//
//  CPUArgMax.cpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUArgMax.hpp"
#include <float.h>
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"

namespace MNN {

CPUArgMax::CPUArgMax(Backend *backend, int topk, int outMaxVal, int softmaxThreshold)
    : Execution(backend), mTopk(topk), mOutMaxVal(outMaxVal), mSoftmaxThreshold(softmaxThreshold) {
    // nothing to do
}

ErrorCode CPUArgMax::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // acquire buffer space
    auto &input                      = inputs[0]->buffer();
    mInputBuffer.buffer().dimensions = input.dimensions;
    memcpy(mInputBuffer.buffer().dim, input.dim, sizeof(halide_dimension_t) * input.dimensions);
    backend()->onAcquireBuffer(&mInputBuffer, Backend::DYNAMIC);

    auto &output                = outputs[0]->buffer();
    mBuffer.buffer().dimensions = output.dimensions;
    memcpy(mBuffer.buffer().dim, output.dim, sizeof(halide_dimension_t) * output.dimensions);
    backend()->onAcquireBuffer(&mBuffer, Backend::DYNAMIC);

    // release temp buffer space
    backend()->onReleaseBuffer(&mBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mInputBuffer, Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUArgMax::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input            = inputs[0];
    auto output           = outputs[0];
    const int inputWidth  = std::max(1, input->width());
    const int inputHeight = std::max(1, input->height());
    MNNUnpackC4(mInputBuffer.host<float>(), input->host<float>(), inputWidth * inputHeight, input->channel());

    // get params
    int num = 0, dim = 0, keyExtent = 0;
    int iw = input->width(), ow = output->width();
    int ih = input->height(), oh = output->height();
    int ic = input->channel(), oc = output->channel();
    if (iw > 1) {
        num       = ic * ih;
        dim       = iw;
        keyExtent = ow;
    } else if (ih > 1) { // iw = ow = 1
        num       = ic;
        dim       = ih;
        keyExtent = oh;
    } else { // iw = ow = 1, ih = oh = 1;
        num       = 1;
        dim       = ic;
        keyExtent = oc;
    }

    // threshold
    float softmaxThreshold = -FLT_MAX;
    if (mSoftmaxThreshold) {
        softmaxThreshold = 1.0f / dim;
    }

    float *srcOrigin = mInputBuffer.host<float>(); // used as NCHW input
    float *dstOrigin = mBuffer.host<float>();
    for (int i = 0; i < num; ++i) {
        float *iptr = srcOrigin + i * dim;
        float *optr = dstOrigin + i * keyExtent;

        using sortElementT = std::tuple<int, float>;
#define element_index(ele) (std::get<0>(ele))
#define element_value(ele) (std::get<1>(ele))

        // apply threshold
        std::vector<sortElementT> vec;
        vec.reserve(dim);
        for (int j = 0; j < dim; ++j) {
            float val = iptr[j];
            if (val >= softmaxThreshold) {
                vec.emplace_back(std::make_tuple(j, val));
            }
        }
        size_t sortDim = vec.size();

        // sort
        auto comp = [](const sortElementT &a, const sortElementT &b) -> int {
            float va = element_value(a);
            float vb = element_value(b);
            return va > vb;
        };
        int realTopK = std::min(mTopk, (int)sortDim);

        std::partial_sort(vec.begin(), vec.begin() + realTopK, vec.end(), comp);

        // copy index
        for (int j = 0; j < mTopk; ++j) {
            if (j < sortDim) {
                optr[j] = element_index(vec[j]);
            } else {
                optr[j] = 0.f;
            }
        }

        // copy max value
        if (mOutMaxVal) {
            for (int j = 0; j < mTopk; ++j) {
                if (j < sortDim) {
                    optr[mTopk + j] = element_value(vec[j]);
                } else {
                    optr[mTopk + j] = 0.f;
                }
            }
        }
    }

    // upload buffer to output
    const int outputWidth  = std::max(1, output->width());
    const int outputHeight = std::max(1, output->height());
    MNNPackC4(output->host<float>(), mBuffer.host<float>(), outputWidth * outputHeight, output->channel());

    return NO_ERROR;
}

class CPUArgMaxCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto argMax = op->main_as_ArgMax();
        return new CPUArgMax(backend, argMax->topK(), argMax->outMaxVal(), argMax->softmaxThreshold());
    }
};
REGISTER_CPU_OP_CREATOR(CPUArgMaxCreator, OpType_ArgMax);
} // namespace MNN
