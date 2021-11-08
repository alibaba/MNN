//
//  TestUtils.h
//  MNN
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TestUtils_h
#define TestUtils_h

#include <assert.h>
#include <stdio.h>
#include <functional>
#include <string>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>
#include <math.h>
#include <iostream>
#include "core/Backend.hpp"
#include "MNN_generated.h"
/**
 * @brief dispatch payload on all available backends
 * @param payload   test to perform
 */
void dispatch(std::function<void(MNNForwardType)> payload);
/**
 * @brief dispatch payload on given backend
 * @param payload   test to perform
 * @param backend   given backend
 */
void dispatch(std::function<void(MNNForwardType)> payload, MNNForwardType backend);

/**
 @brief check the result with the ground truth
 @param result data
 @param rightData
 @param size
 @param threshold
 */
template <typename T>
bool checkVector(const T* result, const T* rightData, int size, T threshold){
    MNN_ASSERT(result != nullptr);
    MNN_ASSERT(rightData != nullptr);
    MNN_ASSERT(size >= 0);
    for(int i = 0; i < size; ++i){
        if(fabs(result[i] - rightData[i]) > threshold){
            std::cout << "right: " << rightData[i] << ", compute: " << result[i] << std::endl;
            return false;
        }
    }
    return true;
}

template <typename T>
bool checkVectorByRelativeError(const T* result, const T* rightData, int size, float rtol) {
    MNN_ASSERT(result != nullptr);
    MNN_ASSERT(rightData != nullptr);
    MNN_ASSERT(size >= 0);

    float maxValue = 0.0f;
    for(int i = 0; i < size; ++i){
        maxValue = fmax(fabs(rightData[i]), maxValue);
    }
    for(int i = 0; i < size; ++i){
        if (fabs(result[i] - rightData[i]) > maxValue * rtol) {
            std::cout << i << ": right: " << rightData[i] << ", compute: " << result[i] << std::endl;
            return false;
        }
    }
    return true;
}

#ifdef MNN_SUPPORT_BF16
// simulate bf16, prune fp32 tailing precision to bf16 precision
inline float convertFP32Precision(float fp32Value) {
    int32_t* s32Value = (int32_t*)(&fp32Value);
    *s32Value &= 0xffff0000;
    return fp32Value;
}
#else
// simulate fp16
inline float convertFP32Precision(float fp32Value) {
    // todo: convert exp part and fraction part.
    return fp32Value;
}

#endif

inline float keepFP32Precision(float fp32Value) {
    return fp32Value;
}

using ConvertFP32 = float(*)(float fp32Value);
const static ConvertFP32 FP32Converter[MNN::BackendConfig::Precision_Low + 1] = {keepFP32Precision, keepFP32Precision, convertFP32Precision};

#endif /* TestUtils_h */
