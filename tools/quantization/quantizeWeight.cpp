//
//  quantizeWeight.cpp
//  MNN
//
//  Created by MNN on 2019/04/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "quantizeWeight.hpp"
#include <algorithm>
#include <cmath>
#include <math.h>
#include "logkit.h"

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
            const int32_t quantizedInt8Value =
                static_cast<int32_t>(roundf(weightChannelStart[i] * scaleDataToInt8));
            quantizedWeightChannelStart[i] =
                std::min(quantizedMaxValue, std::max(-quantizedMaxValue, quantizedInt8Value));
        }
    }

    return 0;
}

int QuantizeConvPerChannel(const float* weight, const int size, const float* bias, int8_t* quantizedWeight,
                           int32_t* quantizedBias, float* scale, const std::vector<float>& inputScale,
                           const std::vector<float>& outputScale) {
    const int inputChannels  = inputScale.size();
    const int outputChannels = outputScale.size();
    const int icXoc          = inputChannels * outputChannels;
    DCHECK(size % icXoc == 0) << "Input Data Size Error!";
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
    std::vector<float> quantizedWeightScale(outputChannels);
    SymmetricQuantizeWeight(weightMultiByInputScale.data(), size, quantizedWeight, quantizedWeightScale.data(),
                            outputChannels);

    for (int i = 0; i < outputChannels; ++i) {
        if (outputScale[i] == 0) {
            scale[i] = 0.0f;
        } else {
            scale[i] = quantizedWeightScale[i] / outputScale[i];
        }
    }

    if (bias) {
        for (int i = 0; i < outputChannels; ++i) {
            if (quantizedWeightScale[i] == 0) {
                quantizedBias[i] = 0;
            } else {
                quantizedBias[i] = static_cast<int32_t>(bias[i] / quantizedWeightScale[i]);
            }
        }
    }

    return 0;
}

int QuantizeDepthwiseConv(const float* weight, const int size, const float* bias, int8_t* quantizedWeight,
                          int32_t* quantizedBias, float* scale, const std::vector<float>& inputScale,
                          const std::vector<float>& outputScale) {
    const int inputChannels  = inputScale.size();
    const int outputChannels = outputScale.size();
    DCHECK(inputChannels == outputChannels) << "Input Data Size Error!";

    std::vector<float> quantizedWeightScale(inputChannels);
    SymmetricQuantizeWeight(weight, size, quantizedWeight, quantizedWeightScale.data(), inputChannels);

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
