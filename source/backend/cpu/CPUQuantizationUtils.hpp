/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/

// edited from tensorflow - quantization_utils.cc by MNN.


#ifndef QUANTIZATION_HPP
#define QUANTIZATION_HPP

#include <math.h>
#include <stdio.h>
#include <cmath>
#include <limits>
#include "TFQuantizeOp_generated.h"

namespace MNN {

inline int CalculateInputRadius(int inputIntegerBits, int inputLeftShift) {
    const double maxInputRescaled =
        1.0 * ((1 << inputIntegerBits) - 1) * (1ll << (31 - inputIntegerBits)) / (1ll << inputLeftShift);
    return static_cast<int>(std::floor(maxInputRescaled));
}

inline void QuantizeMultiplier(double doubleMultiplier, int32_t* quantizedMultiplier, int* shift) {
    if (doubleMultiplier == 0.) {
        *quantizedMultiplier = 0;
        *shift               = 0;
        return;
    }
    const double q = std::frexp(doubleMultiplier, shift);
    auto qFixed    = static_cast<int64_t>(round(q * (1ll << 31)));
    MNN_ASSERT(qFixed <= (1ll << 31));
    if (qFixed == (1ll << 31)) {
        qFixed /= 2;
        ++*shift;
    }
    MNN_ASSERT(qFixed <= std::numeric_limits<int32_t>::max());
    *quantizedMultiplier = static_cast<int32_t>(qFixed);
}

inline void QuantizeMultiplierGreaterThanOne(double doubleMultiplier, int32_t* quantizedMultiplier, int* leftShift) {
    MNN_ASSERT(doubleMultiplier > 1.);
    QuantizeMultiplier(doubleMultiplier, quantizedMultiplier, leftShift);
    MNN_ASSERT(*leftShift >= 0);
}

inline void PreprocessSoftmaxScaling(double beta, double inputScale, int inputIntegerBits, int32_t* quantizedMultiplier,
                                     int* leftShift) {
    const double inputBetaRealMultiplier =
        std::min(beta * inputScale * (1 << (31 - inputIntegerBits)), (1ll << 31) - 1.0);

    QuantizeMultiplierGreaterThanOne(inputBetaRealMultiplier, quantizedMultiplier, leftShift);
}
inline void CalculateActivationRangeUint8(FusedActivation activation, int outputZeropoint, float inputScale,
                                          int32_t* actMin, int32_t* actMax) {
    const int32_t qmin = std::numeric_limits<uint8_t>::min();
    const int32_t qmax = std::numeric_limits<uint8_t>::max();

    const auto scale     = inputScale;
    const auto zeroPoint = outputZeropoint;

    auto quantize = [scale, zeroPoint](float f) { return zeroPoint + static_cast<int32_t>(round(f / scale)); };

    if (activation == FusedActivation_kTfLiteActRelu) {
        *actMin = std::max(qmin, quantize(0.0));
        *actMax = qmax;
    } else if (activation == FusedActivation_kTfLiteActRelu6) {
        *actMin = std::max(qmin, quantize(0.0));
        *actMax = std::min(qmax, quantize(6.0));
    } else if (activation == FusedActivation_kTfLiteActRelu1) {
        *actMin = std::max(qmin, quantize(-1.0));
        *actMax = std::min(qmax, quantize(1.0));
    } else {
        *actMin = qmin;
        *actMax = qmax;
    }
}

inline void QuantizeMultiplierSmallerThanOne(double doubleMultiplier, int32_t* quantizedMultiplier, int* rightShift) {
    MNN_ASSERT(doubleMultiplier < 1.);
    MNN_ASSERT(doubleMultiplier > 0.);
    int shift;
    QuantizeMultiplier(doubleMultiplier, quantizedMultiplier, &shift);
    MNN_ASSERT(shift <= 0);
    *rightShift = -shift;
}

template <class T>
float FloatForOneQuantizedLevel(float rangeMin, float rangeMax) {
    const int64_t highest                 = static_cast<int64_t>(std::numeric_limits<T>::max());
    const int64_t lowest                  = static_cast<int64_t>(std::numeric_limits<T>::min());
    const float floatForOneQuantizedLevel = (rangeMax - rangeMin) / (highest - lowest);
    return floatForOneQuantizedLevel;
}

template <class T1, class T2, class T3>
void QuantizationRangeForMultiplication(float minA, float maxA, float minB, float maxB, float* minC, float* maxC) {
    const float aFloatForOneQuantLevel = FloatForOneQuantizedLevel<T1>(minA, maxA);
    const float bFloatForOneQuantLevel = FloatForOneQuantizedLevel<T2>(minB, maxB);

    const int64_t cHighest             = static_cast<int64_t>(std::numeric_limits<T3>::max());
    const int64_t cLowest              = static_cast<int64_t>(std::numeric_limits<T3>::min());
    const float cFloatForOneQuantLevel = aFloatForOneQuantLevel * bFloatForOneQuantLevel;

    *minC = cFloatForOneQuantLevel * cLowest;
    *maxC = cFloatForOneQuantLevel * cHighest;
}

template <class T>
int64_t FloatToQuantizedUnclamped(float input, float rangeMin, float rangeMax) {
    const int64_t lowestQuantized = static_cast<double>(std::numeric_limits<T>::min());
    if (rangeMin == rangeMax) {
        return lowestQuantized;
    }
    const int numberOfBits      = sizeof(T) * 8;
    const int64_t numberOfSteps = static_cast<int64_t>(1) << numberOfBits;
    const double rangeAdjust    = (numberOfSteps / (numberOfSteps - 1.0));
    const double range          = ((rangeMax - rangeMin) * rangeAdjust);
    const double rangeScale     = (numberOfSteps / range);
    int64_t quantized           = (round(input * rangeScale) - round(rangeMin * rangeScale));
    quantized += lowestQuantized;
    return quantized;
}

template <class T>
int64_t FloatToQuantizedUnclampedOpt(float input, float rangeMin, float rangeMax) {
    const double rangeScale = (((static_cast<int64_t>(1) << 32) - 1.0) / (rangeMax - rangeMin));
    int64_t quantized       = (round(input * rangeScale) - round(rangeMin * rangeScale));
    quantized += -(static_cast<int64_t>(1) << 31);
    return quantized;
}

template <class T>
T FloatToQuantized(float input, float rangeMin, float rangeMax) {
    if (std::is_same<T, float>::value) {
        // Specialization for float. This is used in reference implementation
        // for float which is useful to compare performance between float
        // and quantized type.
        return input;
    }
    int64_t quantized              = FloatToQuantizedUnclamped<T>(input, rangeMin, rangeMax);
    const int64_t lowestQuantized  = static_cast<int64_t>(std::numeric_limits<T>::min());
    const int64_t highestQuantized = static_cast<int64_t>(std::numeric_limits<T>::max());
    quantized                      = std::max(quantized, lowestQuantized);
    quantized                      = std::min(quantized, highestQuantized);
    return static_cast<T>(static_cast<int32_t>(quantized));
}

template <class T>
float QuantizedToFloat(T input, float rangeMin, float rangeMax) {
    if (std::is_same<T, float>::value) {
        // Specialization for float. This is used in reference implementation
        // for float which is useful to compare performance between float
        // and quantized type.
        return input;
    }
    if (rangeMin == rangeMax) {
        return rangeMin;
    }
    const int numberOfBits        = sizeof(T) * 8;
    const int64_t numberOfSteps   = static_cast<int64_t>(1) << numberOfBits;
    const double rangeAdjust      = (numberOfSteps / (numberOfSteps - 1.0));
    const double range            = ((rangeMax - rangeMin) * rangeAdjust);
    const double rangeScale       = (range / numberOfSteps);
    const int64_t lowestQuantized = static_cast<int64_t>(std::numeric_limits<T>::min());
    const double offsetInput      = static_cast<double>(input) - lowestQuantized;
    // For compatibility with DEQUANTIZE_WITH_EIGEN, we should convert
    // rangeScale to a float, otherwise rangeMinRounded might be slightly
    // different.
    const double rangeMinRounded = round(rangeMin / static_cast<float>(rangeScale)) * static_cast<float>(rangeScale);
    const double result          = rangeMinRounded + (offsetInput * rangeScale);
    return static_cast<float>(result);
}

template <class T>
float QuantizedToFloatOpt(T input, float rangeMin, float rangeMax) {
    if (std::is_same<T, float>::value) {
        // Specialization for float. This is used in reference implementation
        // for float which is useful to compare performance between float
        // and quantized type.
        return input;
    }
    if (rangeMin == rangeMax) {
        return rangeMin;
    }
    const int numberOfBits        = sizeof(int32_t) * 8;
    const int64_t numberOfSteps   = static_cast<int64_t>(1) << numberOfBits;
    const int64_t lowestQuantized = static_cast<int64_t>(1) << (numberOfBits - 1);
    const double rangeScale       = ((rangeMax - rangeMin) / (numberOfSteps - 1.0));
    const double result           = rangeMin + ((input + lowestQuantized) * rangeScale);
    return static_cast<float>(result);
}

template <class T1, class T2>
inline T2 RequantizeInNewRange(T1 input, float minInput, float maxInput, float minNew, float maxNew) {
    const float inputFloat = QuantizedToFloat<T1>(input, minInput, maxInput);
    T2 result              = FloatToQuantized<T2>(inputFloat, minNew, maxNew);
    return result;
}

// Because converting 32-bit accumulated results down to eight bit is a common
// case, we have a specialized code path to handle it as efficiently as
// possible using only fixed-point math for the inner loop.
inline void RequantizeManyInNewRangeReference(const int32_t* input, int64_t count, float minInput, float maxInput,
                                              float minOutput, float maxOutput, uint8_t* output) {
    // Initially we calculate all the constants we need once, before we go into
    // the inner loop.  If this is updated, also update the Eigen version.
    const int fpShift            = 16;
    const float inputRange       = maxInput - minInput;
    const float outputRange      = maxOutput - minOutput;
    const float recipOutputRange = outputRange == 0.0 ? 0.0 : (255.0 / outputRange);
    const float inputRezero      = (minInput + maxInput) / 2.0;
    const int64_t rangeScaleFp =
        outputRange == 0.0 ? 0.0 : static_cast<int64_t>(255.0 * (1 << fpShift) * inputRange / outputRange);
    const int64_t inputOffsetFp = static_cast<int64_t>(inputRezero * recipOutputRange * (1 << fpShift));
    const int64_t outputOffsetFp =
        outputRange == 0.0 ? 0 : static_cast<int64_t>((1 << fpShift) * (minOutput * 255.0) / outputRange);
    const int64_t roundingDelta = 1 << (fpShift - 1);

    // Inside this loop we just do minimal adds, multiplies, and shifts, in a way
    // that could be easily adapted for a SIMD implementation. It should also be
    // possible to perform all the calculations in 32-bit rather than 64, but
    // that's not been implemented yet.
    for (size_t index = 0; index < count; ++index) {
        const int64_t inputValue         = static_cast<int64_t>(input[index]);
        const int64_t fpValue            = ((inputValue * rangeScaleFp) >> 32) + inputOffsetFp;
        const int64_t offsetIntermediate = fpValue - outputOffsetFp;
        const int64_t roundIntermediate  = offsetIntermediate + roundingDelta;
        int64_t quantizedInt64           = roundIntermediate >> fpShift;
        quantizedInt64                   = std::max(quantizedInt64, int64_t(0));
        quantizedInt64                   = std::min(quantizedInt64, int64_t(255));
        output[index]                    = static_cast<uint8_t>(static_cast<int32_t>(quantizedInt64));
    }
}

// Another common case is converting eight bit inputs up to thirty two bits, so
// we have specialized fixed-point code to accelerate that. There is also a NEON
// version for ARM devices below.
inline void RequantizeManyInNewRange8To32BitReference(const uint8_t* input, int64_t count, float minInput,
                                                      float maxInput, float minOutput, float maxOutput,
                                                      int32_t* output) {
    const float code0Float         = QuantizedToFloat<uint8_t>(0, minInput, maxInput);
    const float code1Float         = QuantizedToFloat<uint8_t>(1, minInput, maxInput);
    const int64_t code0Int64       = FloatToQuantizedUnclamped<int32_t>(code0Float, minOutput, maxOutput);
    const int64_t code1Int64       = FloatToQuantizedUnclamped<int32_t>(code1Float, minOutput, maxOutput);
    const int32_t multInt32        = static_cast<int32_t>(code1Int64 - code0Int64);
    const int64_t lowestQuantized  = static_cast<int64_t>(std::numeric_limits<int32_t>::min());
    const int64_t highestQuantized = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    for (int64_t i = 0; i < count; ++i) {
        const int64_t inputValue = static_cast<int64_t>(input[i]);
        int64_t outputValue      = code0Int64 + (inputValue * multInt32);
        outputValue              = std::max(outputValue, lowestQuantized);
        outputValue              = std::min(outputValue, highestQuantized);
        output[i]                = static_cast<int32_t>(outputValue);
    }
}

template <class T1, class T2>
inline void RequantizeManyInNewRange(const T1* input, int64_t count, float minInput, float maxInput, float minOutput,
                                     float maxOutput, T2* output) {
    for (size_t index = 0; index < count; ++index) {
        const float inputFloat = QuantizedToFloat<T1>(input[index], minInput, maxInput);
        output[index]          = FloatToQuantized<T2>(inputFloat, minOutput, maxOutput);
    }
}

//    template <>
//    inline void RequantizeManyInNewRange<int32_t, uint8_t>(
//                                                         const int32_t* input, int64_t count, float minInput, float
//                                                         maxInput, float minOutput, float maxOutput, uint8_t*
//                                                         output) {
//        RequantizeManyInNewRangeReference(input, count, minInput, maxInput,
//                                          minOutput, maxOutput, output);
//    }

template <>
inline void RequantizeManyInNewRange<uint8_t, int32_t>(const uint8_t* input, int64_t count, float minInput,
                                                       float maxInput, float minOutput, float maxOutput,
                                                       int32_t* output) {
    RequantizeManyInNewRange8To32BitReference(input, count, minInput, maxInput, minOutput, maxOutput, output);
}

inline void CalculateUsedRange(Tensor* input, int32_t* usedMinQuantized, int32_t* usedMaxQuantized) {
    int inputDataSize = 1;
    for (int i = 0; i < input->buffer().dimensions; i++) {
        inputDataSize *= input->buffer().dim[i].extent;
    }
    int32_t* inputData = (int32_t*)input->buffer().host;

    usedMinQuantized[0] = inputData[0];
    usedMaxQuantized[0] = inputData[0];
    for (int i = 0; i < inputDataSize; i++) {
        if (inputData[i] < usedMinQuantized[0]) {
            usedMinQuantized[0] = inputData[i];
        }
        if (inputData[i] > usedMaxQuantized[0]) {
            usedMaxQuantized[0] = inputData[i];
        }
    }
}

inline void GetOutputMinAndMaxForQuantizedAdd(float inputMin, float inputMax, float smallerInputMin,
                                              float smallerInputMax, float* outputMin, float* outputMax) {
    // We need to have a good range to add our two arguments together in. This
    // is surprisingly tricky, since it has to satisfy a few different needs:
    //  - Must be symmetrical around zero, so that 0 + 0 = 0.
    //  - Must hold the largest of the argument ranges.
    //  - Should have enough range that the bits of the lowest and highest
    //    arguments overlap if possible without the lower getting truncated.
    //  - Should have some headroom so that there's no overflow.
    //  - Needs to be signed.
    // This leads us to use a scheme where we (assuming the inputs are eight bit
    // and the output is 32-bit) use the bottom 32 - 17 = 15 bits to store the
    // accumulated results. This gives us all the properties we need.
    *outputMax = std::max(inputMax, std::max(-inputMin, std::max(smallerInputMax, -smallerInputMin))) * (1 << 17);
    *outputMin = -(*outputMax);
}
} // namespace MNN

#endif /* CPUQuantizedBiasAdd_hpp */
