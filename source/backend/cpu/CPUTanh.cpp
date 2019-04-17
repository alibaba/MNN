//
//  CPUTanh.cpp
//  MNN
//
//  Created by MNN on 2018/08/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUTanh.hpp"
#include <math.h>
#include "CPUBackend.hpp"
#include "Macro.h"

namespace MNN {

// Lambert's series with 7 divisions
// reference from
// https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
inline float tanhf_poly(float value) {
    if (value > 5.0) {
        return 1.0;
    } else if (value <= -5.0) {
        return -1.0;
    } else {
        float x2 = value * value;
        float a  = value * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
        float b  = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
        return a / b;
    }
}

ErrorCode CPUTanh::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size());
    MNN_ASSERT(1 == outputs.size());
    auto inputData  = inputs[0]->host<float>();
    auto outputData = outputs[0]->host<float>();

    const int dataSize = outputs[0]->elementSize();

    for (int i = 0; i < dataSize; i++) {
        // outputData[i] = 1 - 2 / (expf(2 * inputData[i]) + 1);
        outputData[i] = tanhf_poly(inputData[i]);
    }

    return NO_ERROR;
}

class CPUTanhCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUTanh(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUTanhCreator, OpType_TanH);
} // namespace MNN
