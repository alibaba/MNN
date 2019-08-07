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
#include "MNNDefine.h"

bool IsZero(float a) {
    if (std::fabs(a) < 1e-9) {
        return true;
    }
    else {
        return false;
    }
}

void InitAlpha(const float* weight, const int weightNum, const int kernelNum, float* alpha, const int quantizeBits) {
    const int kernelDim = weightNum / kernelNum;
    const int bound = std::pow(2, quantizeBits-1) - 1;
    
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

        if (quantizeBits > 5) {
            alpha[i] = max / (bound * 1.25);
        }
        else {
            alpha[i] = avg;
        }
    }
}

void UpdateQuantizedWeights(const float* weight, const int weightNum, const int kernelNum, float* alpha,
        const int quantizeBits, int8_t* quantizedWeight) {
    const int kernelDim = weightNum / kernelNum;
    const float bound = std::pow(2, quantizeBits - 1) - 1;
    float weightQuan;
    CHECK(quantizeBits > 4) << "quantization bits less than 4 not supported yet.";

    for (int i = 0; i < weightNum; i++) {
        weightQuan = IsZero(alpha[i / kernelDim]) ? 0 : weight[i] / alpha[i / kernelDim];
        quantizedWeight[i] = std::min(bound, std::max(-bound, std::roundf(weightQuan)));
    }
}

void UpdateAlpha(const float* weight, const int weightNum, const int kernelNum, float* alpha, int8_t* quantizedWeight) {
    const int kernelDim = weightNum / kernelNum;

    for (int i = 0; i < kernelNum; i++) {
        const int offset = i * kernelDim;
        float sum1 = 0;
        float sum2 = 0;

        for (int j = 0; j < kernelDim; j++) {
            sum1 += weight[offset + j] * quantizedWeight[offset + j];
            sum2 += quantizedWeight[offset + j] * quantizedWeight[offset + j];
        }
        if (IsZero(sum2)) {
            sum1 = 0;
            sum2 = 1;
        }
        alpha[i] = sum1 / sum2;
    }
}

// weight format is [co, ci, kh, kw]
int QuantizeWeightADMM(const float* weight, const int weightNum, int8_t* quantizedWeight, float* alpha,
                            const int kernelNum) {
    // channels: co
    DCHECK((weightNum % kernelNum) == 0) << "weight size error!";
    const int kernelDim     = weightNum / kernelNum; // ci * kh * kw
    const int quantizeBits = 8;

    InitAlpha(weight, weightNum, kernelNum, alpha, quantizeBits);

    int iter = 0;
    float diffRate = 1;
    float preSum = 0;
    float curSum = 0;
    const int maxIter = 1000;

    for (int i = 0; i < kernelNum; i++){
        preSum += std::fabs(alpha[i]);
    }

    while(iter < maxIter) {

        UpdateQuantizedWeights(weight, weightNum, kernelNum, alpha, quantizeBits, quantizedWeight);

        UpdateAlpha(weight, weightNum, kernelNum, alpha, quantizedWeight);

        for (int i = 0; i < kernelNum; i++) {
            curSum += std::fabs(alpha[i]);
        }
        if (curSum != curSum) {
            DLOG(INFO) << "curSum is nan.";
        }

        diffRate = std::fabs(curSum - preSum) / preSum;
        preSum = curSum;
        iter++;
    }
    DLOG(INFO) << "iter: " << iter;
    DLOG(INFO) << "diffRate: " << diffRate;

    return 0;
}

// weight format is [co, ci, kh, kw]
int SymmetricQuantizeWeight(const float* weight, const int size, int8_t* quantizedWeight, float* scale,
                            const int channels) {
    DCHECK((size % channels) == 0) << "weight size error!";
    const int channelStride     = size / channels;
    const int quantizedMaxValue = 127;

    for (int c = 0; c < channels; ++c) {
        const auto weightChannelStart    = weight + c * channelStride;
        auto quantizedWeightChannelStart = quantizedWeight + c * channelStride;
        auto minmaxValue                 = std::minmax_element(weightChannelStart, weightChannelStart + channelStride);
        const float dataAbsMax           = std::max(std::abs(*minmaxValue.first), std::abs(*minmaxValue.second));

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
                           int32_t* quantizedBias, float* scale, const std::vector<float>& inputScale,
                           const std::vector<float>& outputScale, std::string method, bool mergeChannel) {
    const int inputChannels  = inputScale.size();
    const int outputChannels = outputScale.size();
    const int icXoc          = inputChannels * outputChannels;
    DCHECK(size % icXoc == 0) << "Input Data Size Error!";

    std::vector<float> quantizedWeightScale(outputChannels);

    float inputScalexWeight = 1.0f;
    if (mergeChannel) {
        if (method == "MAX_ABS"){
            SymmetricQuantizeWeight(weight, size, quantizedWeight, quantizedWeightScale.data(), outputChannels);
        }
        else if (method == "ADMM") {
            QuantizeWeightADMM(weight, size, quantizedWeight, quantizedWeightScale.data(), outputChannels);
        }
        inputScalexWeight = inputScale[0];
    } else {
        const int kernelSize = size / icXoc;
        const int ocStride   = size / outputChannels;

        std::vector<float> weightMultiByInputScale(size);
        for (int oc = 0; oc < outputChannels; ++oc) {
            for (int ic = 0; ic < inputChannels; ++ic) {
                for (int i = 0; i < kernelSize; ++i) {
                    const int index                = oc * ocStride + ic * kernelSize + i;
                    weightMultiByInputScale[index] = inputScale[ic] * weight[index];
                }
            }
        }
        if (method == "MAX_ABS"){
            SymmetricQuantizeWeight(weightMultiByInputScale.data(), size, quantizedWeight, quantizedWeightScale.data(), outputChannels);
        }
        else if (method == "ADMM") {
            QuantizeWeightADMM(weightMultiByInputScale.data(), size, quantizedWeight, quantizedWeightScale.data(), outputChannels);
        }
    }

    for (int i = 0; i < outputChannels; ++i) {
        if (outputScale[i] == 0) {
            scale[i] = 0.0f;
        } else {
            scale[i] = inputScalexWeight * quantizedWeightScale[i] / outputScale[0];
        }
    }

    if (bias) {
        for (int i = 0; i < outputChannels; ++i) {
            if (inputScalexWeight == 0 || quantizedWeightScale[i] == 0) {
                quantizedBias[i] = 0;
            } else {
                quantizedBias[i] = static_cast<int32_t>(bias[i] / (inputScalexWeight * quantizedWeightScale[i]));
            }
        }
    }

    return 0;
}

int QuantizeDepthwiseConv(const float* weight, const int size, const float* bias, int8_t* quantizedWeight,
                          int32_t* quantizedBias, float* scale, const std::vector<float>& inputScale,
                          const std::vector<float>& outputScale, std::string method) {
    const int inputChannels  = inputScale.size();
    const int outputChannels = outputScale.size();
    DCHECK(inputChannels == outputChannels) << "Input Data Size Error!";

    std::vector<float> quantizedWeightScale(inputChannels);
    if (method == "MAX_ABS") {
        SymmetricQuantizeWeight(weight, size, quantizedWeight, quantizedWeightScale.data(), inputChannels);
    }
    else if (method == "ADMM") {
        QuantizeWeightADMM(weight, size, quantizedWeight, quantizedWeightScale.data(), inputChannels);
    }

    for (int c = 0; c < inputChannels; ++c) {
        const int index = c;
        if (outputScale[c] == 0) {
            scale[index] = 0.0f;
        } else {
            scale[index] = inputScale[c] * quantizedWeightScale[c] / outputScale[c];
        }
    }

    if (bias) {
        for (int i = 0; i < outputChannels; ++i) {
            if (inputScale[i] == 0 || quantizedWeightScale[i] == 0) {
                quantizedBias[i] = 0;
            } else {
                quantizedBias[i] = static_cast<int32_t>(bias[i] / (inputScale[i] * quantizedWeightScale[i]));
            }
        }
    }

    return 0;
}
