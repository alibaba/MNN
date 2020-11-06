//
//  CPUCosineSimilarity.cpp
//  MNN
//
//  Created by MNN on 2019/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUCosineSimilarity.hpp"
#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "math/Vec.hpp"

using Vec4 = MNN::Math::Vec<float, 4>;

namespace MNN {

ErrorCode CPUCosineSimilarity::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto x1     = inputs[0];
    auto x2     = inputs[1];
    auto output = outputs[0];

    const int batch         = x1->batch();
    const int batchStride   = x1->stride(0);
    const int channel       = x1->channel();
    const int channleStride = x1->stride(1);
    const float eps         = 1e-8f;
    const auto x1DataPtr    = x1->host<float>();
    const auto x2DataPtr    = x2->host<float>();
    auto outputDataPtr      = output->host<float>();

    // the layout of input tensor is nchw
    for (int i = 0; i < batch; ++i) {
        const auto x1DataBatchPtr = x1DataPtr + i * batchStride;
        const auto x2DataBatchPtr = x2DataPtr + i * batchStride;
        auto outputDataBathPtr    = outputDataPtr + i * channleStride;

        int j = 0;
        for (; j < channleStride; j += 4) {
            const auto x1ChannelPtr = x1DataBatchPtr + j;
            const auto x2ChannelPtr = x2DataBatchPtr + j;

            Vec4 innerProduct(.0f);
            Vec4 x1Square(.0f);
            Vec4 x2Square(.0f);
            for (int c = 0; c < channel; ++c) {
                Vec4 x1Data = Vec4::load(x1ChannelPtr + c * channleStride);
                Vec4 x2Data = Vec4::load(x2ChannelPtr + c * channleStride);
                auto x1Xx2        = x1Data * x2Data;
                innerProduct      = innerProduct + x1Xx2;
                x1Square          = x1Square + x1Data * x1Data;
                x2Square          = x2Square + x2Data * x2Data;
            }
            for (int k = 0; k < 4; ++k) {
                outputDataBathPtr[j + k] = innerProduct[k] / sqrt(x1Square[k] * x2Square[k] + eps);
            }
        }
        for (; j < channleStride; ++j) {
            const auto x1ChannelPtr = x1DataBatchPtr + j;
            const auto x2ChannelPtr = x2DataBatchPtr + j;

            float innerProduct = .0f;
            float x1Square     = .0f;
            float x2Square     = .0f;
            for (int c = 0; c < channel; ++c) {
                float x1Data = x1ChannelPtr[c * channleStride];
                float x2Data = x2ChannelPtr[c * channleStride];
                innerProduct += x1Data * x2Data;
                x1Square += x1Data * x1Data;
                x2Square += x2Data * x2Data;
            }
            outputDataBathPtr[j] = innerProduct / sqrt(x1Square * x2Square + eps);
        }
    }

    return NO_ERROR;
}

class CPUCosineSimilarityCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPUCosineSimilarity(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUCosineSimilarityCreator, OpType_CosineSimilarity);

} // namespace MNN
