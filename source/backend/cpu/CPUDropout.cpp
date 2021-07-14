//
//  CPURelu.cpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUDropout.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "CPUBackend.hpp"
#include <string.h>
#include <random>
#include <iostream>
namespace MNN {
ErrorCode CPUDropout::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto& ib = inputs[0]->buffer();
    auto& ratiob = inputs[1]->buffer();
    auto& ob = outputs[0]->buffer();
    auto& maskb = outputs[1]->buffer();
    const float* srcO = (const float*)ib.host;
    const float* ratioO = (const float*)ratiob.host;
    float* dstO       = (float*)ob.host;
    float* maskO       = (float*)maskb.host;
    mDropRatio = ratioO[0];
    auto eltSize = outputs[0]->elementSize();
    float scale  = 1. / (1. - mDropRatio);
    std::mt19937 mGenerator;
    mGenerator.seed(std::random_device()());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for(int i=0;i<eltSize;i++){
	dstO[i] = dis(mGenerator);
	dstO[i] = dstO[i] < mDropRatio ? 0.0f : scale;
        dstO[i] = srcO[i]*dstO[i];
    }
    return NO_ERROR;
}



class CPUDropoutCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        auto ratio = 0.5f;
        if (nullptr != op->main()) {
            ratio = op->main_as_Dropout()->ratio();
        }
        return new CPUDropout(backend, ratio);
    }
};


REGISTER_CPU_OP_CREATOR(CPUDropoutCreator, OpType_Dropout);
} // namespace MNN
