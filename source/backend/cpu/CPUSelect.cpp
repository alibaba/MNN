//
//  CPUSelect.cpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#include "backend/cpu/CPUSelect.hpp"
#include "core/TensorUtils.hpp"
#include "compute/CommonOptFunction.h"
namespace MNN {

template<typename T>
void selectMain(const int* select, const T* i1, const T* i2, T* out, size_t outSize, int inOff0, int inOff1, int inOff2) {
    for (int i = 0; i < outSize; i++) {
        if (*select) {
            *out = *i1;
        } else {
            *out = *i2;
        }
        out++;
        select+=inOff0;
        i1+=inOff1;
        i2+=inOff2;
    }
}


ErrorCode CPUSelect::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inSize0 = static_cast<CPUBackend*>(backend())->getTensorSize(inputs[0]);
    auto inSize1 = static_cast<CPUBackend*>(backend())->getTensorSize(inputs[1]);
    auto inSize2 = static_cast<CPUBackend*>(backend())->getTensorSize(inputs[2]);
    auto outSize = static_cast<CPUBackend*>(backend())->getTensorSize(outputs[0]);
    int inOff0 = inSize0 == 1 ? 0 : 1;
    int inOff1 = inSize1 == 1 ? 0 : 1;
    int inOff2 = inSize2 == 1 ? 0 : 1;
    auto select = inputs[0]->host<int32_t>();
    auto dataBytes = CPUBackend::getBytes(backend(), outputs[0]);
    switch (dataBytes) {
        case 4:
            selectMain(select, inputs[1]->host<int32_t>(), inputs[2]->host<int32_t>(), outputs[0]->host<int32_t>(), outSize, inOff0, inOff1, inOff2);
            break;
        case 2:
            selectMain(select, inputs[1]->host<int16_t>(), inputs[2]->host<int16_t>(), outputs[0]->host<int16_t>(), outSize, inOff0, inOff1, inOff2);
            break;
        case 1:
            selectMain(select, inputs[1]->host<int8_t>(), inputs[2]->host<int8_t>(), outputs[0]->host<int8_t>(), outSize, inOff0, inOff1, inOff2);
            break;
        default:
            break;
    }
    return NO_ERROR;
}

class CPUSelectCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto cpubn = static_cast<CPUBackend*>(backend);
        auto format = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        if (cpubn->functions()->pack != 4 && MNN_DATA_FORMAT_NC4HW4 == format) {
            // For ARM82 backend, int32 is pack4 but float is pack8, don't support this case
            return nullptr;
        }
        return new CPUSelect(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUSelectCreator, OpType_Select);
} // namespace MNN
