//
//  CPUAsString.cpp
//  MNN
//
//  Created by MNN on 2018/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUAsString.hpp"
#include <string.h>
#include <functional>
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
#define INT_CAPACITY 10
#define FLOAT_CAPACITY 30
inline std::string _int2String(int number) {
    char result[INT_CAPACITY];
    snprintf(result, INT_CAPACITY, "%d", number);
    return std::string(result);
}

class AsStringExecutor : public Execution {
public:
    AsStringExecutor(Backend* backend, const AsString* op) : Execution(backend) {
        mWidth      = op->width();
        mPrecision  = op->precision();
        mScientific = op->scientific();
        if (nullptr != op->fillString()) {
            mFillString = op->fillString()->str();
        }
        mSourceType = op->T();
        mShortest   = op->shortest();
    }
    virtual ~AsStringExecutor() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto inputTensor  = inputs[0];
        auto outputTensor = outputs[0];
        TensorUtils::clearHandleData(outputTensor);
        std::string format = "%";

        if (mWidth >= 0) {
            format = format + mFillString + _int2String(mWidth);
        }

        if (mPrecision >= 0) {
            format = format + "." + _int2String(mPrecision);
        }

        switch (mSourceType) {
            case DataType_DT_INT8:
            case DataType_DT_INT32:
                format = format + "d";
                break;

            case DataType_DT_INT64:
                format = format + "lld";
                break;

            case DataType_DT_FLOAT:
            case DataType_DT_DOUBLE:
            case DataType_DT_COMPLEX64: {
                if (mShortest) {
                    format += "g";
                } else if (mScientific) {
                    format += "e";
                } else {
                    format += "f";
                }
            }
            default:
                break;
        }

        if (mSourceType == DataType_DT_COMPLEX64) {
            format = std::string("(") + format + "," + format + ")";
        }

        auto size       = inputTensor->size() / inputTensor->buffer().type.bytes();
        auto outputData = outputTensor->host<char*>();

        MNN_ASSERT(mSourceType == DataType_DT_FLOAT || mSourceType == DataType_DT_BOOL);
        switch (mSourceType) {
            case DataType_DT_FLOAT: {
                auto data = inputTensor->host<float>();
                for (int i = 0; i < size; i++) {
                    auto tempData = (char*)::malloc(FLOAT_CAPACITY + 1);
                    snprintf(tempData, FLOAT_CAPACITY, format.c_str(), data[i]);
                    tempData[FLOAT_CAPACITY] = 0;
                    outputData[i]            = ::strdup(tempData);
                    ::free(tempData);
                }
                break;
            }
            case DataType_DT_BOOL: {
                auto data = inputTensor->host<int32_t>();
                for (int i = 0; i < size; i++) {
                    if (data[i] > 0) {
                        outputData[i] = ::strdup("true");
                    } else {
                        outputData[i] = ::strdup("false");
                    }
                }
                break;
            }
            default:
                return NOT_SUPPORT;
        }
        return NO_ERROR;
    }

private:
    int mWidth;
    int mPrecision;
    bool mScientific;
    bool mShortest;
    std::string mFillString;
    DataType mSourceType;
};

Execution* CPUAsStringCreator::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                        const MNN::Op* op, Backend* backend) const {
    return new AsStringExecutor(backend, op->main_as_AsString());
}

REGISTER_CPU_OP_CREATOR(CPUAsStringCreator, OpType_AsString);
} // namespace MNN
