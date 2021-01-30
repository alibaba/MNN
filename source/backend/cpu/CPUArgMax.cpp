//
//  CPUArgMax.cpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUArgMax.hpp"
#include <float.h>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/TensorUtils.hpp"
#include <vector>

namespace MNN {

CPUArgMax::CPUArgMax(Backend *backend, ArgMinOrMax mode, int topk, int outMaxVal, int softmaxThreshold, int axis)
    : Execution(backend), mTopk(topk), mOutMaxVal(outMaxVal), mSoftmaxThreshold(softmaxThreshold), mAxis(axis), mMode(mode) {
    // nothing to do
}

ErrorCode CPUArgMax::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // acquire buffer space
    auto input                = inputs[0];
    auto output               = outputs[0];
    auto inputDimensionFromat = TensorUtils::getDescribe(input)->dimensionFormat;

    mFromNHWC = inputDimensionFromat != MNN_DATA_FORMAT_NC4HW4;

    if (!mFromNHWC) {
        // if the input format is NC4HW4, convert to be NCHW from NC4HW4 firstly
        TensorUtils::copyShape(input, &mInputBuffer);
        TensorUtils::copyShape(output, &mOutputBuffer);

        backend()->onAcquireBuffer(&mInputBuffer, Backend::DYNAMIC);
        backend()->onAcquireBuffer(&mOutputBuffer, Backend::DYNAMIC);

        // release temp buffer space
        backend()->onReleaseBuffer(&mInputBuffer, Backend::DYNAMIC);
        backend()->onReleaseBuffer(&mOutputBuffer, Backend::DYNAMIC);
    }

    // compute params
    mNum       = 1;
    mDim       = 1;
    mKeyExtent = 1;

    if(mAxis < 0){
        mAxis = mAxis + input->dimensions();
    }

    if (mFromNHWC) {
        const int dimensions = input->dimensions();
        for (int i = 0; i < mAxis; ++i) {
            mNum = mNum * input->length(i);
        }
        mDim = input->length(mAxis);
        for (int i = mAxis + 1; i < dimensions; ++i) {
            mKeyExtent = mKeyExtent * input->length(i);
        }
    } else {
        if (mAxis == 0) {
            // Legacy code
            // really legacy
            int iw = input->width(), ow = output->width();
            int ih = input->height(), oh = output->height();
            int ic = input->channel(), oc = output->channel();
            if (iw > 1) {
                mNum       = ic * ih;
                mDim       = iw;
                mKeyExtent = ow;
            } else if (ih > 1) { // iw = ow = 1
                mNum       = ic;
                mDim       = ih;
                mKeyExtent = oh;
            } else { // iw = ow = 1, ih = oh = 1;
                mNum       = 1;
                mDim       = ic;
                mKeyExtent = oc;
            }
        // in caffe, axis may not exist, we set it to 10000 to indicate this situation
        // see file: tools/converter/source/caffe/ArgMax.cpp
        } else if (mAxis != 10000) {
            const int dimensions = input->dimensions();
            for (int i = 0; i < mAxis; ++i) {
                mNum = mNum * input->length(i);
            }
            mDim = input->length(mAxis);
            for (int i = mAxis + 1; i < dimensions; ++i) {
                mKeyExtent = mKeyExtent * input->length(i);
            }
        } else {
            MNN_PRINT("error in argmax, not implemented error.");
            MNN_ASSERT(false);
        }
    }

    return NO_ERROR;
}

ErrorCode CPUArgMax::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    using sortElementT = std::tuple<int, float>;
#define element_index(ele) (std::get<0>(ele))
#define element_value(ele) (std::get<1>(ele))
    auto comp = [](const sortElementT &a, const sortElementT &b) -> int {
        float va = element_value(a);
        float vb = element_value(b);
        return va > vb;
    };

    if (mFromNHWC) {
        if (mMode == ARGMAX) {
            auto srcOrigin = input->host<float>();
            auto dstOrigin = output->host<int>();
            for (int i = 0; i < mNum; ++i) {
                auto iptr = srcOrigin + i * mDim * mKeyExtent;
                auto optr = dstOrigin + i * mKeyExtent;

                for(int k = 0; k < mKeyExtent; ++k){
                    int index      = 0;
                    float maxValue = -FLT_MAX;
                    for (int j = 0; j < mDim; ++j) {
                        auto val = iptr[k + j * mKeyExtent];
                        if (val > maxValue) {
                            maxValue = val;
                            index    = j;
                        }
                    }
                    optr[k] = index;
                }
            }
        } else {
            auto srcOrigin = input->host<float>();
            auto dstOrigin = output->host<int>();
            for (int i = 0; i < mNum; ++i) {
                auto iptr = srcOrigin + i * mDim * mKeyExtent;
                auto optr = dstOrigin + i * mKeyExtent;

                for(int k = 0; k < mKeyExtent; ++k){
                    int index      = 0;
                    float minValue = FLT_MAX;
                    for (int j = 0; j < mDim; ++j) {
                        auto val = iptr[k + j * mKeyExtent];
                        if (val < minValue) {
                            minValue = val;
                            index    = j;
                        }
                    }
                    optr[k] = index;
                }
            }
        }

    } else {
        MNN_ASSERT(mMode == ARGMAX); // caffe does not have argmin layer
        // Legacy code for CAFFE
        backend()->onCopyBuffer(input, &mInputBuffer);

        // threshold
        float softmaxThreshold = -FLT_MAX;
        if (mSoftmaxThreshold) {
            softmaxThreshold = 1.0f / mDim;
        }

        float *srcOrigin = mInputBuffer.host<float>(); // used as NCHW input
        if (mAxis == 0) {
            // really legacy
            float *dstOrigin = mOutputBuffer.host<float>();
            for (int i = 0; i < mNum; ++i) {
                float *iptr = srcOrigin + i * mDim;
                float *optr = dstOrigin + i * mKeyExtent;

                // apply threshold
                std::vector<sortElementT> vec;
                vec.reserve(mDim);
                for (int j = 0; j < mDim; ++j) {
                    float val = iptr[j];
                    if (val >= softmaxThreshold) {
                        vec.emplace_back(std::make_tuple(j, val));
                    }
                }
                size_t sortDim = vec.size();

                // sort

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
            backend()->onCopyBuffer(&mOutputBuffer, output);
        } else {
            float *dstOrigin = output->host<float>();
            int outMaxValNum = mOutMaxVal + 1;
            for (int i = 0; i < mNum; ++i) {
                float *iptr = srcOrigin + i * mDim * mKeyExtent;
                float *optr = dstOrigin + i * mKeyExtent * mTopk * outMaxValNum;

                for (int k = 0; k < mKeyExtent; ++k) {
                    // apply threshold
                    std::vector<sortElementT> vec;
                    vec.reserve(mDim);
                    for (int j = 0; j < mDim; ++j) {
                        float val = iptr[k + j * mKeyExtent];
                        if (val >= softmaxThreshold) {
                            vec.emplace_back(std::make_tuple(j, val));
                        }
                    }
                    size_t sortDim = vec.size();

                    // sort

                    int realTopK = std::min(mTopk, (int) sortDim);

                    std::partial_sort(vec.begin(), vec.begin() + realTopK, vec.end(), comp);

                    // copy index
                    for (int j = 0; j < mTopk; ++j) {
                        if (j < sortDim) {
                            optr[k * outMaxValNum * mTopk + j] = element_index(vec[j]);
                        } else {
                            optr[k * outMaxValNum * mTopk + j] = 0.f;
                        }
                    }

                    // copy max value
                    if (mOutMaxVal) {
                        for (int j = 0; j < mTopk; ++j) {
                            if (j < sortDim) {
                                optr[k * outMaxValNum * mTopk + mTopk + j] = element_value(vec[j]);
                            } else {
                                optr[k * outMaxValNum * mTopk + mTopk + j] = 0.f;
                            }
                        }
                    }
                }
            }
        }
    }

    return NO_ERROR;
}

class CPUArgMaxCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto argMax = op->main_as_ArgMax();
        if (op->type() == OpType_ArgMin) {
            return new CPUArgMax(backend, CPUArgMax::ArgMinOrMax::ARGMIN,
                    argMax->topK(), argMax->outMaxVal(), argMax->softmaxThreshold(), argMax->axis());
        } else {
            return new CPUArgMax(backend, CPUArgMax::ArgMinOrMax::ARGMAX,
                    argMax->topK(), argMax->outMaxVal(), argMax->softmaxThreshold(), argMax->axis());
        }
    }
};
REGISTER_CPU_OP_CREATOR(CPUArgMaxCreator, OpType_ArgMax);
REGISTER_CPU_OP_CREATOR(CPUArgMaxCreator, OpType_ArgMin);
} // namespace MNN
