//
//  CPULayerNorm.cpp
//  MNN
//
//  Created by MNN on 2020/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cmath>
#include "backend/cpu/CPUDynamicQuant.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include "core/Execution.hpp"
#include "core/Concurrency.h"
#include "core/OpCommonUtils.hpp"
#include "MNN_generated.h"
namespace MNN {


CPUDynamicQuant::CPUDynamicQuant(const MNN::Op* op, Backend* backend) : Execution(backend) {
    
}

ErrorCode CPUDynamicQuant::onResize(const std::vector<Tensor*> &inputs,
                                 const std::vector<Tensor*> &outputs) {
    return NO_ERROR;
}

ErrorCode CPUDynamicQuant::onExecute(const std::vector<Tensor*> &inputs,
                                  const std::vector<Tensor*> &outputs) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto int8core = static_cast<CPUBackend*>(backend())->int8Functions();
    float *inputPtr = inputs[0]->host<float>();
    int8_t *outputPtr = outputs[0]->host<int8_t>();
    int size = static_cast<CPUBackend*>(backend())->getTensorSize(inputs[0]);
    float quantScale = 0.f, dequantScale = 0.f, zeroPoint = 0.f;
    float maxVal = 0.f, minVal = 0.f;
    core->MNNCountMaxMinValue(inputPtr, &minVal, &maxVal, size);
    // Compute scale and zero
    float range = maxVal - minVal;
    MNN_ASSERT(range != 0);
    quantScale = 255.0f / range;
    dequantScale = range / 255.0f;
    zeroPoint = std::min(255.f, std::max(roundf(-(minVal * 255.f) / range), 0.f)) - 128.0f;
    int pack = core->pack;
    std::vector<float> qsVec(pack, quantScale);
    int sizeDiv = UP_DIV(size, pack);
    int8core->MNNFloat2Int8(inputPtr, outputPtr, sizeDiv, qsVec.data(), -128, 127, (ssize_t)zeroPoint);
    float* scale = outputs[1]->host<float>();
    float* zeros = outputs[2]->host<float>();
    *scale = dequantScale;
    *zeros = zeroPoint;
    
    return NO_ERROR;
}



CPUDynamicQuant::~CPUDynamicQuant() {

}

class CPUDynamicQuantCreator : public CPUBackend::Creator {
public:
    Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* backend) const override {
        return new CPUDynamicQuant(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDynamicQuantCreator, OpType_DynamicQuant);

}  // namespace MNN
