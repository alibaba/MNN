//
//  CPUCast.cpp
//  MNN
//
//  Created by MNN on 2018/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUCast.hpp"
#include "core/Macro.h"

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
class Bit32ToBool : public Execution {
public:
    Bit32ToBool(Backend *b) : Execution(b) {
        // nothing to do
    }
    virtual ~Bit32ToBool() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto input                = inputs[0];
        auto output               = outputs[0];
        auto srcData              = input->host<int>();
        auto dstData              = output->host<int>();
        const auto inputDataSize  = input->elementSize();
        const auto outputDataSize = output->elementSize();
        MNN_ASSERT(inputDataSize == outputDataSize);
        for (int i = 0; i < inputDataSize; i++) {
            int value  = srcData[i] == 0 ? 0 : 1;
            dstData[i] = value;
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
    // cast param srcT is invalid
    // auto srcT = _mapDataType(cast->srcT());
    auto dstT = _mapDataType(cast->dstT());

    const auto &inputDataType = inputs[0]->getType();

    if (inputs[0]->buffer().type == outputs[0]->buffer().type) {
        return new CopyExecution(backend);
    }
    if ((halide_type_of<int32_t>() == inputDataType || halide_type_of<float>() == inputDataType) &&
        cast->dstT() == MNN::DataType_DT_BOOL) {
        return new Bit32ToBool(backend);
    }
    if (dstT == MNN::DataType_DT_INT32 && halide_type_of<float>() == inputDataType) {
        return new CastDataType<float, int>(backend);
    }
    if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<int32_t>() == inputDataType) {
        return new CastDataType<int, float>(backend);
    }
    if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<uint8_t>() == inputDataType) {
        return new CastDataType<uint8_t, float>(backend);
    }
    if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<int8_t>() == inputDataType) {
        return new CastDataType<int8_t, float>(backend);
    }
    if (dstT == MNN::DataType_DT_INT8 && halide_type_of<float>() == inputDataType) {
        return new CastDataType<float, int8_t>(backend);
    }
    if (dstT == MNN::DataType_DT_INT32 && halide_type_of<uint8_t>() == inputDataType) {
        return new CastDataType<uint8_t, int32_t>(backend);
    }
    MNN_PRINT("Don't support cast form %d to %d\n", cast->srcT(), cast->dstT());
    return nullptr;
}

REGISTER_CPU_OP_CREATOR(CPUCastCreator, OpType_Cast);
} // namespace MNN
