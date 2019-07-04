//
//  CPUCast.cpp
//  MNN
//
//  Created by MNN on 2018/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUCast.hpp"
#include "Macro.h"

namespace MNN {

template <typename srcT, typename dstT>
class CastDataType : public Execution {
public:
    CastDataType(Backend *b) : Execution(b) {
        // nothing to do
    }
    virtual ~CastDataType() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto input                = inputs[0];
        auto output               = outputs[0];
        auto srcData              = input->host<srcT>();
        auto dstData              = output->host<dstT>();
        const auto inputDataSize  = input->elementSize();
        const auto outputDataSize = output->elementSize();
        MNN_ASSERT(inputDataSize == outputDataSize);
        for (int i = 0; i < inputDataSize; i++) {
            dstData[i] = static_cast<dstT>(srcData[i]);
        }
        return NO_ERROR;
    }
};

class CopyExecution : public Execution {
public:
    CopyExecution(Backend *b) : Execution(b) {
        // nothing to do
    }
    virtual ~CopyExecution() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto input                = inputs[0];
        auto output               = outputs[0];
        auto srcData              = input->host<char>();
        auto dstData              = output->host<char>();
        const auto inputDataSize  = input->size();
        const auto outputDataSize = output->size();
        if (inputDataSize != outputDataSize) {
            return INPUT_DATA_ERROR;
        }
        ::memcpy(dstData, srcData, inputDataSize);
        return NO_ERROR;
    }
};

static DataType _mapDataType(DataType src) {
    if (DataType_DT_BOOL == src) {
        return DataType_DT_INT32;
    }
    if (DataType_DT_INT64 == src) {
        return DataType_DT_INT32;
    }
    if (DataType_DT_DOUBLE == src) {
        return DataType_DT_FLOAT;
    }
    return src;
}
Execution *CPUCastCreator::onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                    const MNN::Op *op, Backend *backend) const {
    auto cast = op->main_as_CastParam();
    auto srcT = _mapDataType(cast->srcT());
    auto dstT = _mapDataType(cast->dstT());
    if (inputs[0]->buffer().type == outputs[0]->buffer().type) {
        return new CopyExecution(backend);
    }
    if (dstT == MNN::DataType_DT_INT32 && srcT == MNN::DataType_DT_FLOAT) {
        return new CastDataType<float, int>(backend);
    }
    if (dstT == MNN::DataType_DT_INT32 && srcT == MNN::DataType_DT_DOUBLE) {
        return new CastDataType<double, int>(backend);
    }
    if (dstT == MNN::DataType_DT_FLOAT && srcT == MNN::DataType_DT_INT32) {
        return new CastDataType<int, float>(backend);
    }
    if (dstT == MNN::DataType_DT_FLOAT && srcT == MNN::DataType_DT_UINT8) {
        return new CastDataType<uint8_t, float>(backend);
    }
    if (dstT == MNN::DataType_DT_INT32 && srcT == MNN::DataType_DT_INT64) {
        return new CastDataType<int64_t, int32_t>(backend);
    }
    MNN_PRINT("Don't support cast form %d to %d\n", cast->srcT(), cast->dstT());
    return nullptr;
}

REGISTER_CPU_OP_CREATOR(CPUCastCreator, OpType_Cast);
} // namespace MNN
