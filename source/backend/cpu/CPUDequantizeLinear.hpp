//
//  CPUDequantizeLinear.hpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDequantizeLinear_hpp
#define CPUDequantizeLinear_hpp

#include "core/AutoStorage.h"
#include "core/Execution.hpp"
#include "compute/Int8FunctionsOpt.h"

namespace MNN {
typedef void(*dequantFunc)(float* dst, const int8_t* source, int inputDim, int inputSize, int size, int UNIT, float* scales, int8_t* zeros, const CoreInt8Functions* core);
class CPUDequantizeLinear : public Execution {
public:
    CPUDequantizeLinear(Backend *b, float* scales, int8_t* zeroPoints, int size = 1, int axis = 0, int inputBits = 8);
    virtual ~CPUDequantizeLinear() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    std::vector<float> mQuantScales;
    std::vector<int8_t> mQuantZeroPoints;
    int mSize = 1;
    int mAxis = 0;
    int mInputBits = 8;
    dequantFunc mFunc;
};

template<typename T>
void dequantizeFunc(float* dst, const int8_t* source, int inputDim, int inputSize, int size, int UNIT, float* scales, int8_t* zeros, const CoreInt8Functions* core) {
#ifdef MNN_USE_SSE
    auto src = (uint8_t*)source;
    int offset = 128;
#else
    auto src = (int8_t*)source;
    int offset = 0;
#endif
//    auto src = (T*)source;
    if (inputDim == 1) {
        for (int i = 0; i < size; ++i) {
            dst[i] = static_cast<float>(src[i] - zeros[i] - offset) * scales[i];
        }
        return;
    }
    int chw = 1;
    if (inputDim > 1) {
        chw = inputSize / (size * sizeof(T));
    }

    if (size == 1) {
        if (sizeof(T) == 1) {
            core->MNNInt8ScaleToFloat(dst, (int8_t*)src, scales, chw / UNIT, zeros[0]);
            int sizeDiv = (int)chw / UNIT;
            for (int k = sizeDiv * UNIT; k < chw; ++k) {
                dst[k] = static_cast<float>(src[k] - zeros[0] - offset) * scales[0];
            }
        } else {
            for (int k = 0; k < chw; ++k) {
                dst[k] = static_cast<float>(src[k] - zeros[0] - offset) * scales[0];
            }
        }
        
    } else {
        for (int i = 0; i < size; ++i) {
            std::vector<float> tmp(4, scales[i]);
            //core->MNNInt8ScaleToFloat(dst, src, tmp.data(), sizeDiv, mQuantZeroPoints[i]);
            for (int k = 0; k < chw; ++k) {
                dst[k] = static_cast<float>(src[k] - zeros[i] - offset) * scales[i];
            }
            src += chw;
            dst += chw;
        }
    }
}
} // namespace MNN

#endif /* CPUDequantizeLinear_hpp */
