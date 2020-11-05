
//
//  CPUGatherND.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/*Ref:
 https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/kernels/internal/reference/reference_ops.h
 */

#include <string.h>
#include "backend/cpu/CPUGatherND.hpp"
#include "core/Concurrency.h"

namespace MNN {
ErrorCode CPUGatherND::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto params = inputs[0];
    auto indice = inputs[1];
    mSliceN = 1;
    mSliceSize = 1;
    for (int i=0; i<indice->dimensions()-1; ++i) {
        mSliceN *=  indice->length(i);
    }
    auto indiceNd = indice->length(indice->dimensions()-1);
    mDimsToCount.resize(indiceNd);
    for (int i=indiceNd; i<params->dimensions(); ++i) {
        mSliceSize *=  params->length(i);
    }
    auto paramSize = params->elementSize();
    for (int i=0; i<indiceNd; ++i) {
        mDimsToCount[i] = paramSize / params->length(i);
        paramSize = mDimsToCount[i];
    }
    mDimsToCount.resize(indiceNd);
    return NO_ERROR;
}

ErrorCode CPUGatherND::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto params = inputs[0];
    auto indice = inputs[1];
    auto indiceNd = indice->length(indice->dimensions()-1);
    auto indiceData = indice->host<int32_t>();
    auto output = outputs[0];
    auto bytes = output->getType().bytes();
    MNN_CONCURRENCY_BEGIN(i, mSliceN) {
        int fromPos = 0;
        for (int j=0; j<indiceNd; ++j) {
            fromPos += mDimsToCount[j] * indiceData[i*indiceNd + j];
        }
        ::memcpy(output->host<uint8_t>() + bytes * i * mSliceSize, params->host<uint8_t>() + bytes * fromPos, bytes * mSliceSize);
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

class CPUGatherNDCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUGatherND(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUGatherNDCreator, OpType_GatherND);

} // namespace MNN
