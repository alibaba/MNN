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

DSPConv2DInt8Param makeConvParams(int inputChannels, int outputChannels,
                                 int area) {
    DSPConv2DInt8Param params = {};
    params.batch = 1;
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
    params.channelPack = 64;
    return params;
}

bool testHmxRouteBoundary() {
    DSPConv2DInt8Param params = makeConvParams(96, 48, 33);
    std::vector<DSPRequantParam> requant(
        64, MNN::makeRequantMultiplier(0.125, 0, -127, 127));
    CHECK(htp_ops_conv1x1_w8a8_hmx_eligible(
              requant.data(), &params) == 1);

    params.inputWidth = 32;
    params.outputWidth = 32;
    CHECK(htp_ops_conv1x1_w8a8_hmx_eligible(
              requant.data(), &params) == 0);

    params.inputWidth = 33;
    params.outputWidth = 33;
    params.channelPack = 32;
    CHECK(htp_ops_conv1x1_w8a8_hmx_eligible(
              requant.data(), &params) == 0);

    params.channelPack = 64;
    requant[7].outputZero = 1;
    CHECK(htp_ops_conv1x1_w8a8_hmx_eligible(
              requant.data(), &params) == 0);
    return true;
}

bool testHmxVtcmLayoutBoundary() {
    constexpr uint32_t oneTileCapacity = 26624;
    DSPConv1x1W8A8HmxLayout layout = {};
    CHECK(htp_ops_conv1x1_w8a8_hmx_layout(
              96, oneTileCapacity, &layout) == AEE_SUCCESS);
    CHECK(layout.inputTiles == 3);
    CHECK(layout.activationOffset == 14336);
    CHECK(layout.activationBytesPerAreaTile == 6144);
    CHECK(layout.rawInputBytesPerAreaTile == 4096);
    CHECK(layout.stagedOutputBytesPerAreaTile == 2048);
    CHECK(layout.maxAreaTilesPerChunk == 1);
    CHECK(layout.vtcmBytes == oneTileCapacity);

    CHECK(htp_ops_conv1x1_w8a8_hmx_layout(
              96, oneTileCapacity - 1, &layout) == AEE_ENOMEMORY);
    CHECK(htp_ops_conv1x1_w8a8_hmx_layout(
              std::numeric_limits<int32_t>::max(),
              std::numeric_limits<uint32_t>::max(),
              &layout) == AEE_EBADPARM);
    return true;
}

bool testHmxActivationAndWeightTails() {
    constexpr int batch = 2;
    constexpr int channels = 35;
    constexpr int area = 35;
    constexpr int pack = 64;
    std::vector<uint8_t> input(
        static_cast<size_t>(batch) * area * pack, 0);
    for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        for (int row = 0; row < area; ++row) {
            for (int channel = 0; channel < channels; ++channel) {
                input[packedOffset(
                    batchIndex, channel, row, batch, area, pack)] =
                    static_cast<uint8_t>(static_cast<int8_t>(
                        (batchIndex * 13 + row * 5 + channel * 3) %
                            31 -
                        15));
            }
        }
    }

    std::vector<uint8_t> activationTile(2048, 0xa5);
    CHECK(htp_ops_pack_hmx_activation_u8_m32k64(
              activationTile.data(), input.data(), 1, batch, 32, 3,
              32, 3, channels, area, pack) == AEE_SUCCESS);
    for (int row = 0; row < 32; ++row) {
        for (int channel = 0; channel < 32; ++channel) {
            const uint8_t expected =
                row < 3 && channel < 3
                    ? static_cast<uint8_t>(
                          static_cast<int8_t>(
                              input[packedOffset(
                                  1, 32 + channel, 32 + row, batch,
                                  area, pack)]) +
                          128)
                    : 0;
            CHECK(activationTile[static_cast<size_t>(row) * 32 +
                                 channel] == expected);
        }
    }

    constexpr int outputChannels = 5;
    std::vector<int8_t> rawWeight(
        static_cast<size_t>(channels) * outputChannels);
    for (int outputChannel = 0; outputChannel < outputChannels;
         ++outputChannel) {
        for (int inputChannel = 0; inputChannel < channels;
             ++inputChannel) {
            rawWeight[static_cast<size_t>(outputChannel) * channels +
                      inputChannel] =
                static_cast<int8_t>(
                    (outputChannel * 11 + inputChannel * 7) % 23 - 11);
        }
    }
    std::vector<int8_t> packedWeight;
    CHECK(MNN::packW8A8Conv1x1Weight(
        packedWeight, rawWeight.data(), rawWeight.size(), channels,
        outputChannels));
    std::vector<uint8_t> hmxWeight(2048, 0xa5);
    CHECK(htp_ops_pack_hmx_weight_s8_k64n32(
              hmxWeight.data(),
              reinterpret_cast<const uint8_t*>(packedWeight.data()),
              2, 0, 0, channels, outputChannels) == AEE_SUCCESS);
    for (int outputChannel = 0; outputChannel < 32; ++outputChannel) {
        for (int inputChannel = 0; inputChannel < 64; ++inputChannel) {
            const size_t offset =
                static_cast<size_t>(inputChannel / 4) * 128 +
                static_cast<size_t>(outputChannel / 4) * 16 +
                static_cast<size_t>(outputChannel & 3) * 4 +
                static_cast<size_t>(inputChannel & 3);
            const int8_t expected =
                outputChannel < outputChannels &&
                        inputChannel < channels
                    ? rawWeight[
                          static_cast<size_t>(outputChannel) * channels +
                          inputChannel]
                    : 0;
            CHECK(static_cast<int8_t>(hmxWeight[offset]) == expected);
        }
    }
    return true;
}

bool testHmxDmaChunkTail() {
    int32_t areaTileBase = -1;
    int32_t areaTilesInChunk = -1;
    CHECK(htp_ops_conv1x1_w8a8_hmx_chunk(
              5, 2, 0, &areaTileBase,
              &areaTilesInChunk) == AEE_SUCCESS);
    CHECK(areaTileBase == 0 && areaTilesInChunk == 2);
    CHECK(htp_ops_conv1x1_w8a8_hmx_chunk(
              5, 2, 1, &areaTileBase,
              &areaTilesInChunk) == AEE_SUCCESS);
    CHECK(areaTileBase == 2 && areaTilesInChunk == 2);
    CHECK(htp_ops_conv1x1_w8a8_hmx_chunk(
              5, 2, 2, &areaTileBase,
              &areaTilesInChunk) == AEE_SUCCESS);
    CHECK(areaTileBase == 4 && areaTilesInChunk == 1);
    CHECK(htp_ops_conv1x1_w8a8_hmx_chunk(
              5, 2, 3, &areaTileBase,
              &areaTilesInChunk) == AEE_EBADPARM);
    return true;
}

bool testHmxScaleBiasTable() {
    constexpr int inputChannels = 35;
    constexpr int outputChannels = 35;
    constexpr int inputTiles = 2;
    constexpr int outputTile = 1;
    std::vector<int8_t> rawWeight(
        static_cast<size_t>(inputChannels) * outputChannels, 0);
    rawWeight[static_cast<size_t>(32) * inputChannels + 0] = 2;
    rawWeight[static_cast<size_t>(32) * inputChannels + 31] = -3;
    rawWeight[static_cast<size_t>(32) * inputChannels + 32] = 4;
    rawWeight[static_cast<size_t>(32) * inputChannels + 34] = -5;
    std::vector<int8_t> packedWeight;
    CHECK(MNN::packW8A8Conv1x1Weight(
        packedWeight, rawWeight.data(), rawWeight.size(), inputChannels,
        outputChannels));

    std::vector<int32_t> bias(outputChannels, 0);
    std::vector<DSPRequantParam> requant(outputChannels);
    bias[32] = -123;
    bias[33] = 456;
    bias[34] = -789;
    requant[32] = MNN::makeRequantMultiplier(1.0 / 512.0);
    requant[33] = MNN::makeRequantMultiplier(2.0 / 512.0);
    requant[34] = MNN::makeRequantMultiplier(0.5 / 512.0);
    std::vector<uint32_t> table(64, 0xdeadbeefU);

    CHECK(htp_ops_prepare_hmx_w8a8_scale_bias_table_32(
              table.data(),
              reinterpret_cast<const uint8_t*>(packedWeight.data()),
              bias.data(), requant.data(), inputTiles, outputTile,
              inputChannels, outputChannels) == AEE_SUCCESS);
    CHECK(table[0] == 0x00403c00U);
    CHECK(table[1] == 0x00404000U);
    CHECK(table[2] == 0x00403800U);
    CHECK(static_cast<int32_t>(table[32]) ==
          -123 + 256 + 65536);
    CHECK(static_cast<int32_t>(table[33]) == 456 + 32768);
    CHECK(static_cast<int32_t>(table[34]) == -789 + 131072);
    for (int channel = 3; channel < 32; ++channel) {
        CHECK(table[channel] == 0);
        CHECK(table[32 + channel] == 0);
    }

    requant[32].outputZero = 1;
    CHECK(htp_ops_prepare_hmx_w8a8_scale_bias_table_32(
              table.data(),
              reinterpret_cast<const uint8_t*>(packedWeight.data()),
              bias.data(), requant.data(), inputTiles, outputTile,
              inputChannels, outputChannels) == AEE_EBADPARM);
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
        !testHmxRouteBoundary() ||
        !testHmxVtcmLayoutBoundary() ||
        !testHmxActivationAndWeightTails() ||
        !testHmxDmaChunkTail() ||
        !testHmxScaleBiasTable() ||
        !testInvalidContract()) {
        return 1;
    }
    std::printf("PASS: Hexagon W8A8 Conv1x1 contract\n");
    return 0;
}
