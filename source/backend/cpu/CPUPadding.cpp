//
//  CPUPadding.cpp
//  MNN
//
//  Created by MNN on 2019/6/24.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#include "CPUPadding.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"
#include <string.h>
#include "CPUTensorConvert.hpp"
namespace MNN {
void CPUPadding::execute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input   = inputs[0];
    auto output  = outputs[0];
    auto padding = inputs[1]->host<int32_t>();
    ::memset(output->host<char>(), 0, output->size());
    auto outputData = output->host<char>();
    auto inputData = input->host<char>();
#define MAX_DIM 6
    MNN_ASSERT(output->dimensions() <= MAX_DIM);
    int dims[MAX_DIM];
    int oStride[MAX_DIM];
    int iStride[MAX_DIM];
    int pad[MAX_DIM];
    auto bytes = input->getType().bytes();
    for (int i=0; i<MAX_DIM; ++i) {
        pad[i] = 0;
        dims[i] = 1;
        oStride[i] = 0;
        iStride[i] = 0;
    }
    int offset = MAX_DIM - input->dimensions();
    for (int i=0; i<input->dimensions(); ++i) {
        pad[offset+i] = padding[2*i];
        dims[offset+i] = input->length(i);
        oStride[offset+i] = output->stride(i) * bytes;
        iStride[offset+i] = input->stride(i) * bytes;
    }
    for (int w = 0; w < dims[0]; ++w) {
        auto ow  = outputData + (w+pad[0]) * oStride[0];
        auto sw = inputData + w * iStride[0];
#define PTR(x, y, i)                      \
auto o##x  = o##y + (x+pad[i]) * oStride[i];    \
auto s##x = s##y + x * iStride[i]; \

        for (int v = 0; v < dims[1]; ++v) {
            PTR(v, w, 1);
            for (int u = 0; u < dims[2]; ++u) {
                PTR(u, v, 2);
                for (int z = 0; z < dims[3]; ++z) {
                    PTR(z, u, 3);
                    for (int y = 0; y < dims[4]; ++y) {
                        PTR(y, z, 4);
                        ::memcpy(oy+pad[5]*oStride[5], sy, iStride[4]);
                    }
                }
            }
        }
    }
#undef MAX_DIM
#undef PTR
}

ErrorCode CPUPaddingPacked::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto padding    = inputs[1];
    auto paddingPtr = padding->host<int32_t>();
    if (paddingPtr[2] != 0 || paddingPtr[3] != 0) {
        mNeedConvert = true;
    }
    if (!mNeedConvert) {
        return NO_ERROR;
    }
    mTempOutput.reset(Tensor::createDevice<float>(outputs[0]->shape(), Tensor::CAFFE));
    mTempInput.reset(Tensor::createDevice<float>(inputs[0]->shape(), Tensor::CAFFE));
    bool res = backend()->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);
    res = res && backend()->onAcquireBuffer(mTempInput.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    mTempInputs = {mTempInput.get(), inputs[1]};
    mTempOutputs = {mTempOutput.get()};
    backend()->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempInput.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode CPUPaddingPacked::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    if (mNeedConvert) {
        CPUTensorConverter::convert(input, mTempInput.get());
        CPUPadding::execute(mTempInputs, mTempOutputs);
        CPUTensorConverter::convert(mTempOutput.get(), output);
        return NO_ERROR;
    }
    auto iw     = input->width();
    auto ih     = input->height();
    auto ic     = input->channel();
    auto ib     = input->batch();

    auto ow      = output->width();
    auto oh      = output->height();
    auto icC4    = UP_DIV(ic, 4);
    auto padding = inputs[1]->host<int32_t>();
    ::memset(output->host<float>(), 0, output->size());
    for (int n = 0; n < ib; ++n) {
        auto inputN  = input->host<float>() + input->stride(0) * n;
        auto outputN = output->host<float>() + output->stride(0) * (padding[2 * 0] + n);
        for (int c = 0; c < icC4; ++c) {
            auto inputC  = inputN + c * iw * ih * 4;
            auto outputC = outputN + c * ow * oh * 4;

            for (int h = 0; h < ih; ++h) {
                auto inputH  = inputC + h * iw * 4;
                auto outputH = outputC + (h + padding[2 * 2]) * ow * 4;

                ::memcpy(outputH + padding[2 * 3] * 4, inputH, iw * 4 * sizeof(float));
            }
        }
    }

    return NO_ERROR;
}
class CPUPaddingCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        if (TensorUtils::getDescribe(inputs[0])->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
            return new CPUPadding(backend);
        }
        if (inputs[0]->dimensions() != 4) {
            MNN_ERROR("Currently padding only support 4 dimension for NC4HW4\n");
            return nullptr;
        }
        if (inputs[0]->buffer().type.bits != 32) {
            MNN_ERROR("Currently padding NC4HW4 only support 32 bit padding\n");
            return nullptr;
        }
        return new CPUPaddingPacked(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUPaddingCreator, OpType_Padding);
}; // namespace MNN
