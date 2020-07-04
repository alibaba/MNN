//
//  CPUDequantize.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUDequantize.hpp"
#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"

#define UNIT 4
#define TILE 2

extern "C" {
void dequantizeMinFirst(uint8_t* input, float* output, float* rangeScale, float* resultAdd, size_t lengthUnit) {
    for (int i = 0; i < lengthUnit; i++) {
        for (int m = 0; m < TILE; m++) {
            for (int j = 0; j < UNIT; j++) {
                output[i * UNIT * TILE + m * UNIT + j] =
                    (float)input[i * UNIT * TILE + m * UNIT + j] * (*rangeScale) + (*resultAdd);
            }
        }
    }
}
}

namespace MNN {

template <typename T>
CPUDequantize<T>::CPUDequantize(Backend* backend, QuantizeMode mode, const Op* op) :  Execution(backend), mMode(mode) {
    auto param = op->main_as_Dequantize();
    mIsLiteDequantize = param->modelFormat() == ModeFormat_TFLITE;
    mZeroPoint = param->inputQuantizedParam()->zeroPoint();
    mScale = param->inputQuantizedParam()->scale();
    mHalfRange = !std::is_signed<T>::value ? 0.0f
                                           : (static_cast<double>(std::numeric_limits<T>::max()) -
                                              static_cast<double>(std::numeric_limits<T>::min()) + 1) /
                                                 2.0f;
}

template <typename T>
ErrorCode CPUDequantize<T>::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    T* src       = (T*)input->host<T>();
    auto dest    = output->host<float>();
    
    if(mIsLiteDequantize){
        const int elements = input->elementSize();
        
        for(int i = 0; i < elements; ++i){
            dest[i] = mScale * (static_cast<int32_t>(src[i]) - mZeroPoint);
        }
        
        return NO_ERROR;
    }
    
    const float minRange = inputs[1]->host<float>()[0];
    const float maxRange = inputs[2]->host<float>()[0];
    
    int length   = 1;
    for (int i = 0; i < input->buffer().dimensions; i++) {
        length *= input->buffer().dim[i].extent;
    }

    if (mMode == QuantizeMode_MIN_COMBINED) {
        const float scaleFactor = (maxRange - minRange) / (static_cast<double>(std::numeric_limits<T>::max()) -
                                                           static_cast<double>(std::numeric_limits<T>::min()));
        for (int i = 0; i < length; i++) {
            dest[i] = ((static_cast<int>(src[i]) + mHalfRange) * scaleFactor) + minRange;
        }
    } else if (mMode == QuantizeMode_MIN_FIRST) {
        if (std::is_same<T, uint8_t>::value) {
            constexpr int numberOfBits      = sizeof(T) * 8;
            constexpr int64_t numberOfSteps = static_cast<int64_t>(1) << numberOfBits;
            float rangeScale                = (maxRange - minRange) / (numberOfSteps - 1.0);
            float rangeMinRounded = maxRange == minRange ? minRange : round(minRange / rangeScale) * rangeScale;
            float lowestQuantized = static_cast<float>(std::numeric_limits<T>::lowest());
            float resultAdd       = (rangeMinRounded - lowestQuantized * rangeScale);

            int32_t lengthUnit = length / (UNIT * TILE);
            int32_t remain     = length - (lengthUnit * UNIT * TILE);
            dequantizeMinFirst((uint8_t*)src, (float*)dest, &rangeScale, &resultAdd, lengthUnit);
            if (remain > 0) {
                int32_t currentIndex = lengthUnit * UNIT * TILE;
                for (int i = 0; i < remain; i++) {
                    dest[currentIndex + i] = (float)src[i] * rangeScale + resultAdd;
                }
            }
        } else {
            constexpr int numberOfBits      = sizeof(T) * 8;
            constexpr int64_t numberOfSteps = static_cast<int64_t>(1) << numberOfBits;
            float rangeScale                = (maxRange - minRange) / (numberOfSteps - 1.0);
            float rangeMinRounded = maxRange == minRange ? minRange : round(minRange / rangeScale) * rangeScale;
            float lowestQuantized = static_cast<float>(std::numeric_limits<T>::lowest());
            float resultAdd       = (rangeMinRounded - lowestQuantized * rangeScale);
            for (int i = 0; i < length; i++) {
                dest[i] = (float)src[i] * rangeScale + resultAdd;
            }
        }
    } else if (mMode == QuantizeMode_SCALED) {
        const float scaleFactor =
            std::numeric_limits<T>::min() == 0
                ? (maxRange / std::numeric_limits<T>::max())
                : std::max(minRange / std::numeric_limits<T>::min(), maxRange / std::numeric_limits<T>::max());
        for (int i = 0; i < length; i++) {
            dest[i] = static_cast<int>(src[i]) * scaleFactor;
        }
    }
    return NO_ERROR;
}

class CPUDequantizeCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        auto dequantize = op->main_as_Dequantize();
        switch (dequantize->type()) {
            case DataType_DT_QUINT8:
                return new CPUDequantize<uint8_t>(backend, dequantize->mode(), op);
            case DataType_DT_QUINT16:
                return new CPUDequantize<uint16_t>(backend, dequantize->mode(), op);
            case DataType_DT_QINT8:
                return new CPUDequantize<int8_t>(backend, dequantize->mode(), op);
            case DataType_DT_QINT16:
                return new CPUDequantize<int16_t>(backend, dequantize->mode(), op);
            case DataType_DT_QINT32:
                return new CPUDequantize<int32_t>(backend, dequantize->mode(), op);
            default:
                MNN_ASSERT(false); // unsupported type
                return nullptr;
        }
    }
};
REGISTER_CPU_OP_CREATOR(CPUDequantizeCreator, OpType_Dequantize);

} // namespace MNN
