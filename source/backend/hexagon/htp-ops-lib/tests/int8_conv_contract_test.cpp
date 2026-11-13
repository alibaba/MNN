#include "dsp/int8_cnn.h"
#include "../../backend/HexagonQuantization.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

#define CHECK(condition)                                                                            \
    do {                                                                                            \
        if (!(condition)) {                                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);                        \
            return false;                                                                           \
        }                                                                                           \
    } while (0)

size_t packedOffset(int batchIndex, int channel, int areaIndex, int batch,
                    int area, int pack) {
    return ((static_cast<size_t>(channel / pack) * batch +
             batchIndex) *
                area +
            areaIndex) *
               pack +
           channel % pack;
}

bool runConvCase(int batch, int inputChannels, int outputChannels, int area) {
    constexpr int pack = 64;
    const int inputBlocks = (inputChannels + pack - 1) / pack;
    const int outputBlocks = (outputChannels + pack - 1) / pack;
    std::vector<uint8_t> input(
        static_cast<size_t>(batch) * inputBlocks * area * pack, 0);
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < inputChannels; ++c) {
            for (int a = 0; a < area; ++a) {
                input[packedOffset(
                    n, c, a, batch, area, pack)] =
                    static_cast<uint8_t>(static_cast<int8_t>(
                        (n * 13 + c * 7 + a * 5) % 31 - 15));
            }
        }
    }

    std::vector<int8_t> rawWeight(
        static_cast<size_t>(inputChannels) * outputChannels);
    for (int oc = 0; oc < outputChannels; ++oc) {
        for (int ic = 0; ic < inputChannels; ++ic) {
            rawWeight[static_cast<size_t>(oc) * inputChannels + ic] =
                static_cast<int8_t>((oc * 11 + ic * 3) % 17 - 8);
        }
    }
    std::vector<int8_t> packedWeight;
    CHECK(MNN::packW8A8Conv1x1Weight(
        packedWeight, rawWeight.data(), rawWeight.size(),
        inputChannels, outputChannels));

    const int paddedOutputChannels =
        ((outputChannels + 31) / 32) * 32;
    std::vector<int32_t> bias(paddedOutputChannels, 0);
    std::vector<DSPRequantParam> requant(paddedOutputChannels);
    for (int oc = 0; oc < paddedOutputChannels; ++oc) {
        bias[oc] = oc < outputChannels ? (oc % 9) - 4 : 0;
        requant[oc] = MNN::makeRequantMultiplier(
            0.0625 * (1 + oc % 3), 0, -127, 127);
    }

    DSPConv2DInt8Param params = {};
    params.batch = batch;
    params.inputHeight = 1;
    params.inputWidth = area;
    params.inputChannels = inputChannels;
    params.outputHeight = 1;
    params.outputWidth = area;
    params.outputChannels = outputChannels;
    params.kernelY = 1;
    params.kernelX = 1;
    params.strideY = 1;
    params.strideX = 1;
    params.dilateY = 1;
    params.dilateX = 1;
    params.groups = 1;
    params.channelPack = pack;

    const size_t outputBytes =
        static_cast<size_t>(batch) * outputBlocks * area * pack;
    std::vector<uint8_t> expected(outputBytes, 0);
    for (int n = 0; n < batch; ++n) {
        for (int a = 0; a < area; ++a) {
            for (int oc = 0; oc < outputChannels; ++oc) {
                int64_t sum = bias[oc];
                for (int ic = 0; ic < inputChannels; ++ic) {
                    const int32_t inputValue = static_cast<int8_t>(
                        input[packedOffset(n, ic, a, batch, area, pack)]);
                    const int32_t weightValue =
                        rawWeight[static_cast<size_t>(oc) * inputChannels + ic];
                    sum += static_cast<int64_t>(inputValue) * weightValue;
                }
                const int32_t accumulator =
                    sum < std::numeric_limits<int32_t>::min()
                        ? std::numeric_limits<int32_t>::min()
                        : (sum > std::numeric_limits<int32_t>::max()
                               ? std::numeric_limits<int32_t>::max()
                               : static_cast<int32_t>(sum));
                expected[packedOffset(n, oc, a, batch, area, pack)] =
                    static_cast<uint8_t>(static_cast<int8_t>(
                        MNN::requantizeReference(accumulator, requant[oc])));
            }
        }
    }
    std::vector<uint8_t> reference(outputBytes, 0x5a);
    std::vector<uint8_t> actual(outputBytes, 0xa5);
    CHECK(conv1x1_w8a8_reference(
              reference.data(), input.data(),
              reinterpret_cast<const uint8_t*>(packedWeight.data()),
              bias.data(), requant.data(), &params) == AEE_SUCCESS);
    CHECK(htp_ops_conv1x1_w8a8(
              actual.data(), input.data(),
              reinterpret_cast<const uint8_t*>(packedWeight.data()),
              bias.data(), requant.data(), &params) == AEE_SUCCESS);
    CHECK(reference == expected);
    CHECK(actual == expected);
    return true;
}

bool testInvalidContract() {
    uint8_t byte = 0;
    int32_t bias = 0;
    DSPRequantParam requant = MNN::makeRequantMultiplier(0.5);
    DSPConv2DInt8Param params = {};
    params.batch = 1;
    params.inputHeight = 1;
    params.inputWidth = 1;
    params.inputChannels = 1;
    params.outputHeight = 1;
    params.outputWidth = 1;
    params.outputChannels = 1;
    params.kernelY = 1;
    params.kernelX = 1;
    params.strideY = 1;
    params.strideX = 1;
    params.dilateY = 1;
    params.dilateX = 1;
    params.groups = 1;
    params.channelPack = 64;
    params.inputZero = 1;
    CHECK(htp_ops_conv1x1_w8a8(
              &byte, &byte, &byte, &bias, &requant,
              &params) == AEE_EBADPARM);
    return true;
}

} // namespace

int main() {
    if (!runConvCase(1, 35, 37, 6) ||
        !runConvCase(2, 65, 67, 5) ||
        !runConvCase(1, 96, 48, 9) ||
        !testInvalidContract()) {
        return 1;
    }
    std::printf("PASS: Hexagon W8A8 Conv1x1 contract\n");
    return 0;
}
