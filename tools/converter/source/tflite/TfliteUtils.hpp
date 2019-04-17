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
bool convertDataFormatTflite(const float* src, float* dst, int KH, int KW, int CI, int CO);

#endif /* TfliteUtils_hpp */
