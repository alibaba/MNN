//
//  CPUCast.cpp
//  MNN
//
//  Created by MNN on 2018/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUCast.hpp"
#include "core/TensorUtils.hpp"
#include "core/Macro.h"
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include "compute/CommonOptFunction.h"
#include <cmath>

namespace MNN {
ErrorCode CPUCastCreator::cast(const void* inputRaw, void* outputRaw, ConvertType type,
                               int number, float scale, float zero, float min, float max, const CPUBackend* bn) {
    auto pack = bn->functions()->pack;
    int c4Size = number / pack;
    int remain = number % pack;
    if (type == FlOAT_TO_INT8) {
        scale = (scale == 0.f ? 0.f : 1.f / scale);
        std::vector<float> scales(pack, scale);
        bn->int8Functions()->MNNFloat2Int8((float*)(inputRaw), (int8_t*)(outputRaw), c4Size, scales.data(), min, max, zero);
        if (remain > 0) {
            std::vector<float> tempSrc(pack);
            std::vector<int8_t> tempDst(pack);
            ::memcpy(tempSrc.data(), (float*)(inputRaw) + c4Size * pack, remain * sizeof(float));
            bn->int8Functions()->MNNFloat2Int8(tempSrc.data(), tempDst.data(), 1, scales.data(), min, max, zero);
            ::memcpy(static_cast<int8_t*>(outputRaw) + c4Size * pack, tempDst.data(), remain * sizeof(int8_t));
        }
        return NO_ERROR;
    }
    if (type == INT8_TO_FlOAT) {
        std::vector<float> scales(pack, scale);
        bn->int8Functions()->MNNInt8ScaleToFloat((float*)(outputRaw), (int8_t*)(inputRaw), scales.data(), c4Size, zero);
        if (remain > 0) {
            std::vector<float> tempDst(pack);
            std::vector<int8_t> tempSrc(pack);
            ::memcpy(tempSrc.data(), (int8_t*)(inputRaw) + c4Size * pack, remain * sizeof(int8_t));
            bn->int8Functions()->MNNInt8ScaleToFloat(tempDst.data(), tempSrc.data(), scales.data(), 1, zero);
            ::memcpy(static_cast<float*>(outputRaw) + c4Size * pack, tempDst.data(), remain * sizeof(float));
        }
        return NO_ERROR;
    }
    MNN_ERROR("Don't support cast type \n");
    return NOT_SUPPORT;
}

ErrorCode CPUCastCreator::cast(const Tensor* input, const Tensor* output, const CPUBackend* bn, ConvertType type) {
    auto& ib     = input->buffer();
    auto& ob     = output->buffer();
    int totalSize = bn->getTensorSize(input);
    auto quantAttr = TensorUtils::getDescribe(input)->quantAttr;
    if (quantAttr == nullptr) {
        MNN_ERROR("No quant info for Cast\n");
        return INVALID_VALUE;
    }
    auto code = cast(ib.host, ob.host, type, totalSize, quantAttr->scale, quantAttr->zero, quantAttr->min, quantAttr->max, bn);
    if (NO_ERROR != code) {
        MNN_ERROR("Error in CPUCast\n");
        return code;
    }
    return NO_ERROR;
}

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
        MNN_ASSERT(inputDataSize == output->elementSize());
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
        MNN_ASSERT(inputDataSize == output->elementSize());
        for (int i = 0; i < inputDataSize; i++) {
            int value  = srcData[i] == 0 ? 0 : 1;
            dstData[i] = value;
        }
        return NO_ERROR;
    }
};
class BF16ToFP32 : public Execution {
public:
    BF16ToFP32(Backend *b) : Execution(b) {
        // nothing to do
    }
    virtual ~BF16ToFP32() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto input                = inputs[0];
        auto output               = outputs[0];
        auto srcData              = input->host<int16_t>();
        auto dstData              = output->host<int16_t>();
        const auto inputDataSize  = input->elementSize();
        MNN_ASSERT(inputDataSize == output->elementSize());
        for (int i = 0; i < inputDataSize; i++) {
            dstData[i * 2] = 0;
            dstData[i * 2 + 1] = srcData[i];
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

    if (inputDataType.bytes() == 4 && cast->dstT() == MNN::DataType_DT_BOOL) {
        return new Bit32ToBool(backend);
    }
    if (inputs[0]->buffer().type == outputs[0]->buffer().type) {
        return new CopyExecution(backend);
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
    if (dstT == MNN::DataType_DT_FLOAT && halide_type_t(halide_type_bfloat, 16) == inputDataType) {
        return new BF16ToFP32(backend);
    }
    if (dstT == MNN::DataType_DT_INT8 && halide_type_of<float>() == inputDataType) {
        return new CastDataType<float, int8_t>(backend);
    }
    if (dstT == MNN::DataType_DT_INT8 && halide_type_of<int32_t>() == inputDataType) {
        return new CastDataType<int32_t, int8_t>(backend);
    }
    if (dstT == MNN::DataType_DT_UINT8 && halide_type_of<float>() == inputDataType) {
        return new CastDataType<float, uint8_t>(backend);
    }
    if (dstT == MNN::DataType_DT_UINT8 && halide_type_of<int32_t>() == inputDataType) {
        return new CastDataType<int32_t, uint8_t>(backend);
    }
    if (dstT == MNN::DataType_DT_UINT8 && halide_type_of<int8_t>() == inputDataType) {
        return new CastDataType<int8_t, uint8_t>(backend);
    }
    if (dstT == MNN::DataType_DT_INT32 && halide_type_of<uint8_t>() == inputDataType) {
        return new CastDataType<uint8_t, int32_t>(backend);
    }
    if (dstT == MNN::DataType_DT_INT32 && halide_type_of<int8_t>() == inputDataType) {
        return new CastDataType<int8_t, int32_t>(backend);
    }
    MNN_PRINT("Don't support cast form %d, %d to %d\n", inputDataType.code, inputDataType.bits, cast->dstT());
    return nullptr;
}

REGISTER_CPU_OP_CREATOR(CPUCastCreator, OpType_Cast);
} // namespace MNN
