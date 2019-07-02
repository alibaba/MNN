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
namespace MNN {
ErrorCode CPUPadding::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input   = inputs[0];
    auto output  = outputs[0];
    auto padding = inputs[1]->host<int32_t>();
    ::memset(output->host<char>(), 0, output->size());
    auto bytes = input->getType().bytes();
    auto unit  = input->length(3) * bytes;
    for (int b = 0; b < input->length(0); ++b) {
        auto outputB = output->host<char>() + output->stride(0) * (b + padding[2 * 0]) * bytes;
        auto inputB  = input->host<char>() + input->stride(0) * b * bytes;
        for (int h = 0; h < input->length(1); ++h) {
            auto outputH = outputB + output->stride(1) * (h + padding[2 * 1]) * bytes;
            auto inputH  = inputB + input->stride(1) * h * bytes;
            for (int w = 0; w < input->length(2); ++w) {
                auto outputW = outputH + output->stride(2) * (w + padding[2 * 2]) * bytes;
                auto inputW  = inputH + input->stride(2) * w * bytes;
                ::memcpy(outputW + padding[3 * 2] * bytes, inputW, unit);
            }
        }
    }
    return NO_ERROR;
}

ErrorCode CPUPaddingPacked::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
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
        if (inputs[0]->dimensions() != 4) {
            MNN_ERROR("Currently padding only support NHWC or NC4HW4\n");
            return nullptr;
        }
        auto padding    = inputs[1];
        auto paddingPtr = padding->host<int32_t>();
        if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NHWC) {
            return new CPUPadding(backend);
        }
        if (paddingPtr[2] != 0 || paddingPtr[3] != 0) {
            MNN_ERROR("Currently padding NC4HW4 don't support channel padding\n");
            return nullptr;
        }
        if (inputs[0]->buffer().type.code != halide_type_float) {
            MNN_ERROR("Currently padding NC4HW4 only support float padding\n");
            return nullptr;
        }
        return new CPUPaddingPacked(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUPaddingCreator, OpType_Padding);
}; // namespace MNN
