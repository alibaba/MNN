//
//  CPUSoftmaxGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUSoftmaxGrad.hpp"
#include "CommonOptFunction.h"
#include "ConvOpt.h"
#include "Macro.h"
#include "TensorUtils.hpp"
#include "Vec4.hpp"
using namespace MNN::Math;
namespace MNN {
ErrorCode CPUSoftmaxGrad::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(1 == mAxis);
    auto softmax        = inputs[0];
    auto gradSoftmax    = inputs[1];
    auto gradX          = outputs[0];
    auto gradXPtr       = gradX->host<float>();
    auto softmaxPtr     = softmax->host<float>();
    auto gradSoftmaxPtr = gradSoftmax->host<float>();
    auto batch          = softmax->length(0);
    if (TensorUtils::getDescribe(gradX)->dimensionFormat == MNN_DATA_FORMAT_NHWC) {
        // NHWC
        auto channel = softmax->length(1);
        MNN_ASSERT(channel > 0);
        for (int i = 0; i < batch; ++i) {
            auto s0 = softmaxPtr + i * channel;
            auto s1 = gradSoftmaxPtr + i * channel;

            auto dst   = gradXPtr + i * channel;
            float sumV = 0.0f;
            for (int j = 0; j < channel; ++j) {
                sumV = sumV + s1[j] * s0[j];
            }
            for (int j = 0; j < channel; ++j) {
                dst[j] = s0[j] * (s1[j] - sumV);
            }
        }
        return NO_ERROR;
    }
    auto channel       = softmax->channel();
    auto channelC4     = channel / 4;
    auto channelAlign  = ALIGN_UP4(channel);
    auto channelRemain = channelC4 * 4;

    for (int i = 0; i < batch; ++i) {
        auto s0 = softmaxPtr + i * channelAlign;
        auto s1 = gradSoftmaxPtr + i * channelAlign;

        auto dst = gradXPtr + i * channelAlign;
        ::memset(dst, 0, channelAlign * sizeof(float));
        Vec4 sumV(0.0f);
        for (int j = 0; j < channelC4; ++j) {
            sumV = sumV + Vec4::load(s1 + 4 * j) * Vec4::load(s0 + 4 * j);
        }
        float sum = sumV[0] + sumV[1] + sumV[2] + sumV[3];
        for (int j = channelRemain; j < channel; ++j) {
            sum += s1[j] * s0[j];
        }
        sumV = Vec4(sum);
        for (int j = 0; j < channelC4; ++j) {
            Vec4::save(dst + 4 * j, Vec4::load(s0 + 4 * j) * (Vec4::load(s1 + 4 * j) - sumV));
        }
        for (int j = channelRemain; j < channel; ++j) {
            dst[j] = s0[j] * (s1[j] - sum);
        }
    }
    return NO_ERROR;
}
class CPUSoftmaxGradCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto axis = op->main_as_Axis()->axis();
        if (axis < 0) {
            axis = inputs[0]->dimensions() - 1;
        }
        return new CPUSoftmaxGrad(axis, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUSoftmaxGradCreator, OpType_SoftmaxGrad);

} // namespace MNN
