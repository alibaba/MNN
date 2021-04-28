//
//  TfliteUtils.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TfliteUtils_hpp
#define TfliteUtils_hpp

#include <stdio.h>

#include "MNN_generated.h"
#include "schema_generated.h"
#include "logkit.h"

typedef std::unique_ptr<tflite::QuantizationParametersT> tfliteQuanParam;

void CalculateActivationRangeQuantizedImpl(const MNN::FusedActivation activation, const int32_t qmin,
                                           const int32_t qmax, const tfliteQuanParam& outputQuan, int32_t* act_min,
                                           int32_t* act_max);

double GetQuantizedConvolutionMultipler(const tfliteQuanParam& inputQuan, const tfliteQuanParam& weightQuan,
                                        const tfliteQuanParam& biasQuan, const tfliteQuanParam& outputQuan);

void CalculateActivationRangeUint8(const MNN::FusedActivation activation, const tfliteQuanParam& outputQuan,
                                   int32_t* actMin, int32_t* actMax);

void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier, int* shift);

// weight format converter
// CO KH KW CI --> CO CI KH KW
bool convertDataFormatTflite(const float* src, float* dst, int KH, int KW, int CI, int CO, bool deconv = false);
template <typename T>
bool convertDataFormatTfliteDequant(const T* src, float* dst, int KH, int KW, int CI, int CO,
                                    tflite::QuantizationParametersT* quantParam, bool deconv = false) {
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
                    auto x = deconv ? src[(ic * KH + h) * KW * CO + w * CO + oc] : src[(oc * KH + h) * KW * CI + w * CI + ic];
                    dst[(oc * CI + ic) * KH * KW + h * KW + w] = (x - quantParam->zero_point[0]) * quantParam->scale[0];
                }
            }
        }
    }
    return true;
}

MNN::DataType TfliteDataTypeToMNN(tflite::TensorType type);

MNN::DataType TfliteDequantDataTypeToMNN(tflite::TensorType type);

#endif /* TfliteUtils_hpp */
