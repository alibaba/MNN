//
//  quantizeWeight.cpp
//  MNN
//
//  Created by MNN on 2019/04/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "quantizeWeight.hpp"
#include <math.h>
#include <algorithm>
#include <cmath>
#include "logkit.h"
#include <MNN/MNNDefine.h>

void InitAlpha(const float* weight, const int weightNum, const int kernelNum, float* alpha, const float weightClampValue) {
    const int kernelDim = weightNum / kernelNum;

    for (int i = 0; i < kernelNum; i++) {
        float avg = 0;
        float max = 0;
        float absVal;

        for (int j = 0; j < kernelDim; j++) {
            absVal = std::fabs(weight[i * kernelDim + j]);
            avg += absVal;
            if (absVal > max) {
                max = absVal;
            }
        }
        avg = avg / float(kernelDim);

        if (weightClampValue > 1) {
            alpha[i] = max / (weightClampValue * 1.25);
        }
        else {
            alpha[i] = avg;
        }
    }
}

void UpdateQuantizedWeights(const float* weight, const int weightNum, const int kernelNum, float* alpha,
        const float weightClampValue, int8_t* quantizedWeight) {
    const int kernelDim = weightNum / kernelNum;
    const float eps = 1e-9f;
    float weightQuan;
    CHECK((int)weightClampValue >= 7) << "quantization bits less than 4 not supported yet.";

    for (int i = 0; i < weightNum; i++) {
        weightQuan = weight[i] / (alpha[i / kernelDim]+ eps);
        quantizedWeight[i] = std::min(weightClampValue, std::max(-weightClampValue, std::roundf(weightQuan)));
    }
}

void UpdateAlpha(const float* weight, const int weightNum, const int kernelNum, float* alpha, int8_t* quantizedWeight) {
    const int kernelDim = weightNum / kernelNum;
    const float eps = 1e-9f;

    for (int i = 0; i < kernelNum; i++) {
        const int offset = i * kernelDim;
        float sum1 = 0;
        float sum2 = 0;

        for (int j = 0; j < kernelDim; j++) {
            sum1 += weight[offset + j] * quantizedWeight[offset + j];
            sum2 += quantizedWeight[offset + j] * quantizedWeight[offset + j];
        }
        alpha[i] = sum1 / (sum2+eps);
    }
}

// weight format is [co, ci, kh, kw]
int QuantizeWeightADMM(const float* weight, const int weightNum, int8_t* quantizedWeight, float* alpha,
                            const int kernelNum, const float weightClampValue) {
    // channels: co
    DCHECK((weightNum % kernelNum) == 0) << "weight size error!";
    const int kernelDim     = weightNum / kernelNum; // ci * kh * kw

    InitAlpha(weight, weightNum, kernelNum, alpha, weightClampValue);

    int iter = 0;
    float diffRate = 1;
    float preSum = 0;
    float curSum = 0;
    const int maxIter = 1000;

    for (int i = 0; i < weightNum; i++){
        preSum += std::fabs(weight[i]);
    }
    // update weights quan
    while(iter < maxIter) {
        UpdateQuantizedWeights(weight, weightNum, kernelNum, alpha, weightClampValue, quantizedWeight);
        UpdateAlpha(weight, weightNum, kernelNum, alpha, quantizedWeight);
        iter++;
    }

    for (int i = 0; i < weightNum; i++){
        curSum += std::fabs(quantizedWeight[i]*alpha[i/kernelDim]);
    }
    DLOG(INFO) << "iter: " << iter << " with diff "<< preSum-curSum;
    return 0;
}

// weight format is [co, ci, kh, kw]
int SymmetricQuantizeWeight(const float* weight, const int size, int8_t* quantizedWeight, float* scale,
                            const int channels, float weightClampValue) {
    DCHECK((size % channels) == 0) << "weight size error!";
    const int channelStride     = size / channels;
    const int quantizedMaxValue = weightClampValue;

    for (int c = 0; c < channels; ++c) {
        const auto weightChannelStart    = weight + c * channelStride;
        auto quantizedWeightChannelStart = quantizedWeight + c * channelStride;
        auto minmaxValue                 = std::minmax_element(weightChannelStart, weightChannelStart + channelStride);
        const float dataAbsMax           = std::fmax(std::fabs(*minmaxValue.first), std::fabs(*minmaxValue.second));

        float scaleDataToInt8 = 1.0f;
        if (dataAbsMax == 0) {
            scale[c] = 0.0f;
        } else {
            scale[c]        = dataAbsMax / quantizedMaxValue;
            scaleDataToInt8 = quantizedMaxValue / dataAbsMax;
        }

        for (int i = 0; i < channelStride; ++i) {
            const int32_t quantizedInt8Value = static_cast<int32_t>(roundf(weightChannelStart[i] * scaleDataToInt8));
            quantizedWeightChannelStart[i] =
                std::min(quantizedMaxValue, std::max(-quantizedMaxValue, quantizedInt8Value));
        }
    }

    return 0;
}

int QuantizeConvPerChannel(const float* weight, const int size, const float* bias, int8_t* quantizedWeight,
                           int32_t* quantizedBias, float* scale, const float inputScale, const float outputScale,
                           const int inputChannel, const int outputChannel, std::string method, float weightClampValue, bool mergeChannel) {
    const int icXoc          = inputChannel * outputChannel;
    DCHECK(size % icXoc == 0) << "Input Data Size Error!";

    std::vector<float> quantizedWeightScale(outputChannel);

    float inputScalexWeight = 1.0f;
    if (mergeChannel) {
        if (method == "MAX_ABS"){
            SymmetricQuantizeWeight(weight, size, quantizedWeight, quantizedWeightScale.data(), outputChannel, weightClampValue);
        }
        else if (method == "ADMM") {
            QuantizeWeightADMM(weight, size, quantizedWeight, quantizedWeightScale.data(), outputChannel, weightClampValue);
        }
        inputScalexWeight = inputScale;
    } else {
        const int kernelSize = size / icXoc;
        const int ocStride   = size / outputChannel;

        std::vector<float> weightMultiByInputScale(size);
        for (int oc = 0; oc < outputChannel; ++oc) {
            for (int ic = 0; ic < inputChannel; ++ic) {
                for (int i = 0; i < kernelSize; ++i) {
                    const int index                = oc * ocStride + ic * kernelSize + i;
                    weightMultiByInputScale[index] = inputScale * weight[index];
                }
            }
        }
        if (method == "MAX_ABS"){
            SymmetricQuantizeWeight(weightMultiByInputScale.data(), size, quantizedWeight, quantizedWeightScale.data(), outputChannel, weightClampValue);
        }
        else if (method == "ADMM") {
            QuantizeWeightADMM(weightMultiByInputScale.data(), size, quantizedWeight, quantizedWeightScale.data(), outputChannel, weightClampValue);
        }
    }

    for (int i = 0; i < outputChannel; ++i) {
        if (fabs(outputScale) <= 1e-6) {
            scale[i] = 0.0f;
        } else {
            scale[i] = inputScalexWeight * quantizedWeightScale[i] / outputScale;
        }
    }

    if (bias) {
        for (int i = 0; i < outputChannel; ++i) {
            if (fabs(inputScalexWeight) <= 1e-6 || fabs(quantizedWeightScale[i]) <= 1e-6) {
                quantizedBias[i] = 0;
            } else {
                quantizedBias[i] = static_cast<int32_t>(bias[i] / (inputScalexWeight * quantizedWeightScale[i]));
            }
        }
    }

    return 0;
}

int QuantizeDepthwiseConv(const float* weight, const int size, const float* bias, int8_t* quantizedWeight,
                          int32_t* quantizedBias, float* scale, const float inputScale, const float outputScale,
                          const int inputChannel, const int outputChannel, std::string method, float weightClampValue, bool mergeChannel) {
    DCHECK(inputChannel == outputChannel) << "Input Data Size Error!";

    std::vector<float> quantizedWeightScale(inputChannel);
    if (method == "MAX_ABS") {
        SymmetricQuantizeWeight(weight, size, quantizedWeight, quantizedWeightScale.data(), inputChannel, weightClampValue);
    }
    else if (method == "ADMM") {
        QuantizeWeightADMM(weight, size, quantizedWeight, quantizedWeightScale.data(), inputChannel, weightClampValue);
    }

    for (int c = 0; c < inputChannel; ++c) {
        const int index = c;
        if (fabs(outputScale) <= 1e-6) {
            scale[index] = 0.0f;
        } else {
            scale[index] = inputScale * quantizedWeightScale[c] / outputScale;
        }
    }

    if (bias) {
        for (int i = 0; i < outputChannel; ++i) {
            if (fabs(inputScale) <= 1e-6 || fabs(quantizedWeightScale[i]) <= 1e-6) {
                quantizedBias[i] = 0;
            } else {
                quantizedBias[i] = static_cast<int32_t>(bias[i] / (inputScale * quantizedWeightScale[i]));
            }
        }
    }

    return 0;
}
