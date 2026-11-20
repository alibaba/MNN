#include "HexagonQuantization.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace MNN {
namespace {

uint64_t roundPositiveToNearestEven(double value) {
    const double floorValue = std::floor(value);
    const double remainder = value - floorValue;
    uint64_t rounded = static_cast<uint64_t>(floorValue);
    if (remainder > 0.5 || (remainder == 0.5 && (rounded & 1u) != 0u)) {
        ++rounded;
    }
    return rounded;
}

uint64_t roundUnsignedRightShift(uint64_t value, int64_t shift) {
    if (shift <= 0) {
        return value;
    }
    if (shift > 63) {
        return 0;
    }
    const uint64_t quotient = value >> shift;
    const uint64_t mask = (uint64_t{1} << shift) - 1u;
    const uint64_t remainder = value & mask;
    const uint64_t half = uint64_t{1} << (shift - 1);
    if (remainder > half || (remainder == half && (quotient & 1u) != 0u)) {
        return quotient + 1u;
    }
    return quotient;
}

int64_t applyFixedPointScale(int32_t value, int32_t multiplier, int32_t shift) {
    if (value == 0 || multiplier <= 0) {
        return 0;
    }
    const bool negative = value < 0;
    const uint64_t magnitude =
        negative ? static_cast<uint64_t>(-static_cast<int64_t>(value))
                 : static_cast<uint64_t>(value);
    const uint64_t product = magnitude * static_cast<uint32_t>(multiplier);
    const int64_t binaryShift = static_cast<int64_t>(shift) - 31;

    uint64_t scaledMagnitude = 0;
    if (binaryShift >= 0) {
        if (binaryShift >= 63 ||
            product > (static_cast<uint64_t>(
                           std::numeric_limits<int64_t>::max()) >>
                       binaryShift)) {
            return negative ? std::numeric_limits<int64_t>::min()
                            : std::numeric_limits<int64_t>::max();
        }
        scaledMagnitude = product << binaryShift;
    } else {
        scaledMagnitude = roundUnsignedRightShift(product, -binaryShift);
    }
    if (scaledMagnitude >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return negative ? std::numeric_limits<int64_t>::min()
                        : std::numeric_limits<int64_t>::max();
    }
    const int64_t scaled = static_cast<int64_t>(scaledMagnitude);
    return negative ? -scaled : scaled;
}

int32_t roundToNearestEven(float value) {
    const int32_t base = static_cast<int32_t>(value);
    const float remainder = value - static_cast<float>(base);
    if (remainder > 0.5f || (remainder == 0.5f && (base & 1) != 0)) {
        return base + 1;
    }
    if (remainder < -0.5f || (remainder == -0.5f && (base & 1) != 0)) {
        return base - 1;
    }
    return base;
}

size_t hostNcOffset(int batchIndex, int channel, int areaIndex, int batch,
                    int channels, int area, int hostPack, bool hostNhwc) {
    if (hostNhwc) {
        return (static_cast<size_t>(batchIndex) * area + areaIndex) * channels + channel;
    }
    if (hostPack == 1) {
        return (static_cast<size_t>(batchIndex) * channels + channel) * area +
               areaIndex;
    }
    return ((static_cast<size_t>(channel / hostPack) * batch + batchIndex) *
                area +
            areaIndex) *
               hostPack +
           channel % hostPack;
}

size_t deviceNcOffset(int batchIndex, int channel, int areaIndex, int batch,
                      int area, int devicePack) {
    return ((static_cast<size_t>(channel / devicePack) * batch + batchIndex) *
                area +
            areaIndex) *
               devicePack +
           channel % devicePack;
}

} // namespace

DSPRequantParam makeRequantMultiplier(double realScale, int32_t outputZero,
                                      int32_t clampMin, int32_t clampMax) {
    DSPRequantParam param = {0, 0, outputZero, clampMin, clampMax};
    if (!std::isfinite(realScale) || realScale <= 0.0 ||
        clampMin > clampMax || outputZero < clampMin ||
        outputZero > clampMax) {
        return param;
    }
    int exponent = 0;
    const double fraction = std::frexp(realScale, &exponent);
    uint64_t multiplier = roundPositiveToNearestEven(
        std::ldexp(fraction, 31));
    if (multiplier == (uint64_t{1} << 31)) {
        multiplier >>= 1;
        ++exponent;
    }
    param.multiplier = static_cast<int32_t>(multiplier);
    param.shift = static_cast<int32_t>(exponent);
    return param;
}

int32_t requantizeReference(int32_t value, const DSPRequantParam& param) {
    const int64_t scaled =
        applyFixedPointScale(value, param.multiplier, param.shift);
    const int64_t lower =
        static_cast<int64_t>(param.clampMin) - param.outputZero;
    const int64_t upper =
        static_cast<int64_t>(param.clampMax) - param.outputZero;
    if (scaled < lower) {
        return param.clampMin;
    }
    if (scaled > upper) {
        return param.clampMax;
    }
    return static_cast<int32_t>(scaled + param.outputZero);
}

size_t w8a8Conv1x1PackedWeightBytes(int inputChannels, int outputChannels) {
    if (inputChannels <= 0 || outputChannels <= 0) {
        return 0;
    }
    return static_cast<size_t>((inputChannels + 31) / 32) *
           static_cast<size_t>((outputChannels + 31) / 32) * 1024;
}

bool packW8A8Conv1x1Weight(std::vector<int8_t>& packed,
                           const int8_t* weight, size_t weightCount,
                           int inputChannels, int outputChannels) {
    const size_t packedBytes =
        w8a8Conv1x1PackedWeightBytes(inputChannels, outputChannels);
    const size_t rawBytes =
        static_cast<size_t>(inputChannels) * outputChannels;
    if (packedBytes == 0 || weight == nullptr || weightCount < rawBytes) {
        return false;
    }
    packed.assign(packedBytes, 0);
    const int inputTiles = (inputChannels + 31) / 32;
    for (int outputChannel = 0; outputChannel < outputChannels;
         ++outputChannel) {
        const int outputTile = outputChannel / 32;
        const int outputLane = outputChannel % 32;
        for (int inputChannel = 0; inputChannel < inputChannels;
             ++inputChannel) {
            const int inputTile = inputChannel / 32;
            const int inputLane = inputChannel % 32;
            const int inputPair = inputLane / 2;
            const int pairLane = outputLane * 2 + (inputLane & 1);
            const size_t offset =
                static_cast<size_t>(outputTile * inputTiles + inputTile) *
                    1024 +
                static_cast<size_t>(inputPair / 2) * 128 +
                static_cast<size_t>(pairLane) * 2 + (inputPair & 1);
            packed[offset] =
                weight[static_cast<size_t>(outputChannel) * inputChannels +
                       inputChannel];
        }
    }
    return true;
}

bool buildW8A8Conv1x1PostParams(
    std::vector<int32_t>& biasInt32,
    std::vector<DSPRequantParam>& requant, const float* bias, int biasCount,
    const float* weightScale, int weightScaleCount, int outputChannels,
    float inputScale, float outputScale, bool relu, bool relu6,
    int32_t outputMin, int32_t outputMax) {
    if (outputChannels <= 0 || weightScale == nullptr ||
        weightScaleCount < outputChannels ||
        (bias != nullptr && biasCount < outputChannels) ||
        !std::isfinite(inputScale) || inputScale <= 0.0f ||
        !std::isfinite(outputScale) || outputScale <= 0.0f ||
        outputMin > outputMax) {
        return false;
    }

    int32_t clampMin = outputMin;
    int32_t clampMax = outputMax;
    if (relu || relu6) {
        clampMin = std::max<int32_t>(clampMin, 0);
    }
    if (relu6) {
        const double quantizedSix = std::nearbyint(6.0 / outputScale);
        if (!std::isfinite(quantizedSix)) {
            return false;
        }
        clampMax = std::min<int32_t>(
            clampMax,
            static_cast<int32_t>(std::max<double>(
                std::numeric_limits<int32_t>::min(),
                std::min<double>(std::numeric_limits<int32_t>::max(),
                                 quantizedSix))));
    }
    if (clampMin > clampMax) {
        return false;
    }

    const int paddedChannels = ((outputChannels + 31) / 32) * 32;
    biasInt32.assign(paddedChannels, 0);
    requant.assign(
        paddedChannels,
        makeRequantMultiplier(1.0, 0, clampMin, clampMax));
    for (int outputChannel = 0; outputChannel < outputChannels;
         ++outputChannel) {
        const double scale = weightScale[outputChannel];
        const double accumulatorScale =
            static_cast<double>(inputScale) * scale;
        if (!std::isfinite(scale) || scale <= 0.0 ||
            !std::isfinite(accumulatorScale) || accumulatorScale <= 0.0) {
            return false;
        }
        const double quantizedBias =
            bias == nullptr
                ? 0.0
                : std::nearbyint(
                      static_cast<double>(bias[outputChannel]) /
                      accumulatorScale);
        if (!std::isfinite(quantizedBias) ||
            quantizedBias < std::numeric_limits<int32_t>::min() ||
            quantizedBias > std::numeric_limits<int32_t>::max()) {
            return false;
        }
        biasInt32[outputChannel] = static_cast<int32_t>(quantizedBias);
        requant[outputChannel] = makeRequantMultiplier(
            accumulatorScale / outputScale, 0, clampMin, clampMax);
        if (requant[outputChannel].multiplier <= 0) {
            return false;
        }
    }
    return true;
}

bool packFloatToInt8Nc(const float* src, int8_t* dst, int batch,
                       int channels, int height, int width, int devicePack,
                       int hostPack, float scale, int32_t zero,
                       int32_t clampMin, int32_t clampMax, bool hostNhwc) {
    if (src == nullptr || dst == nullptr || batch <= 0 || channels <= 0 ||
        height <= 0 || width <= 0 || devicePack <= 0 ||
        (hostPack != 1 && hostPack != 4) || (hostNhwc && hostPack != 1) || !std::isfinite(scale) ||
        scale <= 0.0f || zero < clampMin || zero > clampMax ||
        clampMin < -128 || clampMax > 127 || clampMin > clampMax) {
        return false;
    }
    const int area = height * width;
    const size_t deviceElements =
        static_cast<size_t>((channels + devicePack - 1) / devicePack) *
        batch * area * devicePack;
    const size_t hostElements =
        hostPack == 1
            ? static_cast<size_t>(batch) * channels * area
            : static_cast<size_t>((channels + hostPack - 1) / hostPack) *
                  batch * area * hostPack;
    for (size_t i = 0; i < hostElements; ++i) {
        if (!std::isfinite(src[i])) {
            return false;
        }
    }
    std::memset(dst, 0, deviceElements);
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int a = 0; a < area; ++a) {
                const float value =
                    src[hostNcOffset(n, c, a, batch, channels, area,
                                     hostPack, hostNhwc)];
                const float scaled = value / scale;
                int32_t quantized = clampMin;
                if (scaled >= static_cast<float>(clampMax - zero)) {
                    quantized = clampMax;
                } else if (scaled >
                           static_cast<float>(clampMin - zero)) {
                    quantized = std::max(
                        clampMin,
                        std::min(clampMax,
                                 roundToNearestEven(scaled) + zero));
                }
                dst[deviceNcOffset(n, c, a, batch, area, devicePack)] =
                    static_cast<int8_t>(quantized);
            }
        }
    }
    return true;
}

bool unpackInt8NcToFloat(const int8_t* src, float* dst, int batch,
                         int channels, int height, int width, int devicePack,
                         int hostPack, float scale, int32_t zero, bool hostNhwc) {
    if (src == nullptr || dst == nullptr || batch <= 0 || channels <= 0 ||
        height <= 0 || width <= 0 || devicePack <= 0 ||
        (hostPack != 1 && hostPack != 4) || (hostNhwc && hostPack != 1) || !std::isfinite(scale) ||
        scale <= 0.0f) {
        return false;
    }
    const int area = height * width;
    if (hostPack == 4) {
        const size_t hostElements =
            static_cast<size_t>((channels + 3) / 4) * batch * area * 4;
        std::fill(dst, dst + hostElements, 0.0f);
    }
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int a = 0; a < area; ++a) {
                const int32_t value =
                    src[deviceNcOffset(n, c, a, batch, area, devicePack)];
                dst[hostNcOffset(n, c, a, batch, channels, area, hostPack, hostNhwc)] =
                    static_cast<float>(value - zero) * scale;
            }
        }
    }
    return true;
}

} // namespace MNN
