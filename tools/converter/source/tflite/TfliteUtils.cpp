//
//  TfliteUtils.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <cmath>
#include <memory>

#include "TfliteUtils.hpp"

void CalculateActivationRangeQuantizedImpl(const MNN::FusedActivation activation, const int32_t qmin,
                                           const int32_t qmax, const tfliteQuanParam& outputQuan, int32_t* act_min,
                                           int32_t* act_max) {
    const auto scale        = outputQuan->scale[0];
    const int32_t zeroPoint = static_cast<int32_t>(outputQuan->zero_point[0]);

    auto quantize = [scale, zeroPoint](float f) { return zeroPoint + static_cast<int32_t>(std::round(f / scale)); };

    if (activation == MNN::FusedActivation_kTfLiteActRelu) {
        *act_min = std::max(qmin, quantize(0.0));
        *act_max = qmax;
    } else if (activation == MNN::FusedActivation_kTfLiteActRelu6) {
        *act_min = std::max(qmin, quantize(0.0));
        *act_max = std::min(qmax, quantize(6.0));
    } else if (activation == MNN::FusedActivation_kTfLiteActRelu1) {
        *act_min = std::max(qmin, quantize(-1.0));
        *act_max = std::min(qmax, quantize(1.0));
    } else {
        *act_min = qmin;
        *act_max = qmax;
    }
}

double GetQuantizedConvolutionMultipler(const tfliteQuanParam& inputQuan, const tfliteQuanParam& weightQuan,
                                        const tfliteQuanParam& biasQuan, const tfliteQuanParam& outputQuan) {
    const double inputProductScale = inputQuan->scale[0] * weightQuan->scale[0];
    const double biasScale         = static_cast<double>(biasQuan->scale[0]);
    const double outputScale       = static_cast<double>(outputQuan->scale[0]);

    DCHECK(std::abs(inputProductScale - biasScale) <= (1e-6 * std::min(inputProductScale, biasScale)))
        << "Scale ERROR!";

    DCHECK(inputProductScale >= 0) << "Scale ERROR!";

    return inputProductScale / outputScale;
}

void CalculateActivationRangeUint8(const MNN::FusedActivation activation, const tfliteQuanParam& outputQuan,
                                   int32_t* act_min, int32_t* act_max) {
    const int32_t qmin      = std::numeric_limits<uint8_t>::min();
    const int32_t qmax      = std::numeric_limits<uint8_t>::max();
    const auto scale        = outputQuan->scale[0];
    const int32_t zeroPoint = static_cast<int32_t>(outputQuan->zero_point[0]);

    auto quantize = [scale, zeroPoint](float f) { return zeroPoint + static_cast<int32_t>(std::round(f / scale)); };

    if (activation == MNN::FusedActivation_kTfLiteActRelu) {
        *act_min = std::max(qmin, quantize(0.0));
        *act_max = qmax;
    } else if (activation == MNN::FusedActivation_kTfLiteActRelu6) {
        *act_min = std::max(qmin, quantize(0.0));
        *act_max = std::min(qmax, quantize(6.0));
    } else if (activation == MNN::FusedActivation_kTfLiteActRelu1) {
        *act_min = std::max(qmin, quantize(-1.0));
        *act_max = std::min(qmax, quantize(1.0));
    } else {
        *act_min = qmin;
        *act_max = qmax;
    }
}

void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier, int* shift) {
    if (double_multiplier == 0.) {
        *quantized_multiplier = 0;
        *shift                = 0;
        return;
    }

    const double q = std::frexp(double_multiplier, shift);
    auto q_fixed   = static_cast<int64_t>(std::round(q * (1ll << 31)));

    DCHECK(q_fixed <= (1ll << 31)) << "Quantize Multiplier ERROR!";
    if (q_fixed == (1ll << 31)) {
        q_fixed /= 2;
        ++*shift;
    }
    DCHECK_LE(q_fixed, std::numeric_limits<int32_t>::max()) << "ERROR";
    *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

bool convertDataFormatTflite(const float* src, float* dst, int KH, int KW, int CI, int CO, bool deconv) {
    DCHECK(KH > 0);
    DCHECK(KW > 0);
    DCHECK(CI > 0);
    DCHECK(CO > 0);
    DCHECK(src != nullptr);
    // deconv: CI KH KW CO --> CO CI KH KW
    // conv  : CO KH KW CI --> CO CI KH KW
    for (int oc = 0; oc < CO; ++oc) {
        for (int ic = 0; ic < CI; ++ic) {
            for (int h = 0; h < KH; ++h) {
                for (int w = 0; w < KW; ++w) {
                    dst[(oc * CI + ic) * KH * KW + h * KW + w] = deconv ? src[(ic * KH + h) * KW * CO + w * CO + oc] : src[(oc * KH + h) * KW * CI + w * CI + ic];
                }
            }
        }
    }
    return true;
}

MNN::DataType TfliteDataTypeToMNN(tflite::TensorType type) {
    if (type == tflite::TensorType_FLOAT32) {
        return MNN::DataType_DT_FLOAT;
    }
    if (type == tflite::TensorType_INT8) {
        return MNN::DataType_DT_INT8;
    }
    if (type == tflite::TensorType_UINT8) {
        return MNN::DataType_DT_UINT8;
    }
    if (type == tflite::TensorType_INT32) {
        return MNN::DataType_DT_INT32;
    }
    return MNN::DataType_DT_INVALID;
}

MNN::DataType TfliteDequantDataTypeToMNN(tflite::TensorType type) {
    if (type == tflite::TensorType_FLOAT32) {
        return MNN::DataType_DT_FLOAT;
    }
    if (type == tflite::TensorType_INT8) {
        return MNN::DataType_DT_QINT8;
    }
    if (type == tflite::TensorType_UINT8) {
        return MNN::DataType_DT_QUINT8;
    }
    if (type == tflite::TensorType_INT32) {
        return MNN::DataType_DT_QINT32;
    }
    return MNN::DataType_DT_INVALID;
}
