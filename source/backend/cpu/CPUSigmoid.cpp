//
//  CPUSigmoid.cpp
//  MNN
//
//  Created by MNN on 2018/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUSigmoid.hpp"
#include <math.h>
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"

namespace MNN {
ErrorCode CPUSigmoid::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(1 == inputs.size());
    MNN_ASSERT(1 == outputs.size());
    auto inputData  = inputs[0]->host<float>();
    auto outputData = outputs[0]->host<float>();

    const int dataSize = outputs[0]->elementSize();
    int countC8        = dataSize / 8;
    if (countC8 > 0) {
        // Align to eight so asm is easier to write
        static float parameters[] = {
            (float)log(2.0f), 1.0f / (float)log(2.0f), 1.0f, 1.0f, 0.5f, 1.0f / 6.0f, 1.0f / 24.0f, 1.0f / 120.0f};
        MNNExpC8(outputData, inputData, parameters, countC8);
        int cc8 = countC8 * 8;
        for (int i = 0; i < cc8; ++i) {
            outputData[i] = 1.0f / (1.0f + outputData[i]);
        }
    }
    int remain = countC8 * 8;
    auto param = log(2.0f);
    for (int i = remain; i < dataSize; i++) {
        /*Origin Function*/
        // outputData[i] = 1.0f/(1.0f+exp(-inputData[i]));

        /*Approciate Function*/
        auto x         = -inputData[i];
        int div        = (x / param);
        auto xReamin   = x - div * param;
        div            = std::min(div, 24);
        div            = std::max(div, -24);
        float expBasic = 1.0;
        if (div < 0) {
            expBasic = 1.0f / (1 << (-div));
        } else {
            expBasic = (float)(1 << div);
        }
        auto t         = xReamin;
        auto expRemain = ((((1.0f / 120 * t + 1.0f / 24) * t + 1.0f / 6) * t + 0.5f) * t + 1.0f) * t + 1.0f;
        outputData[i]  = 1.0f / (1.0f + expBasic * expRemain);
    }

    return NO_ERROR;
}

class CPUSigmoidCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPUSigmoid(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUSigmoidCreator, OpType_Sigmoid);
} // namespace MNN
