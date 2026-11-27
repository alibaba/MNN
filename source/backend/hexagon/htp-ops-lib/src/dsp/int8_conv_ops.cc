#include "dsp/int8_cnn.h"

#include <cstring>
#include <limits>

namespace {

size_t packedOffset(int batchIndex, int channel, int areaIndex, int batch,
                    int area, int pack) {
    return ((static_cast<size_t>(channel / pack) * batch +
             batchIndex) *
                area +
            areaIndex) *
               pack +
           channel % pack;
}

size_t packedWeightOffset(int outputChannel, int inputChannel,
                          int inputChannels) {
    const int inputTiles = (inputChannels + 31) / 32;
    const int outputTile = outputChannel / 32;
    const int outputLane = outputChannel % 32;
    const int inputTile = inputChannel / 32;
    const int inputLane = inputChannel % 32;
    const int inputPair = inputLane / 2;
    const int pairLane = outputLane * 2 + (inputLane & 1);
    return static_cast<size_t>(outputTile * inputTiles + inputTile) *
               1024 +
           static_cast<size_t>(inputPair / 2) * 128 +
           static_cast<size_t>(pairLane) * 2 + (inputPair & 1);
}

bool validConv1x1(const uint8_t* dst, const uint8_t* src,
                  const uint8_t* weight, const int32_t* bias,
                  const DSPRequantParam* requant,
                  const DSPConv2DInt8Param* p) {
    return dst != nullptr && src != nullptr && weight != nullptr &&
           bias != nullptr && requant != nullptr && p != nullptr &&
           p->batch > 0 && p->inputHeight > 0 && p->inputWidth > 0 &&
           p->inputChannels > 0 &&
           p->outputHeight == p->inputHeight &&
           p->outputWidth == p->inputWidth &&
           p->outputChannels > 0 && p->kernelY == 1 &&
           p->kernelX == 1 && p->strideY == 1 && p->strideX == 1 &&
           p->padY == 0 && p->padX == 0 && p->dilateY == 1 &&
           p->dilateX == 1 && p->groups == 1 &&
           p->channelPack > 0 && p->channelPack % 32 == 0 &&
           p->inputZero == 0 && p->outputZero == 0;
}

} // namespace

extern "C" AEEResult conv1x1_w8a8_reference(
    uint8_t* dst, const uint8_t* src, const uint8_t* weight,
    const int32_t* bias, const DSPRequantParam* requant,
    const DSPConv2DInt8Param* p) {
    if (!validConv1x1(dst, src, weight, bias, requant, p)) {
        return AEE_EBADPARM;
    }
    const int area = p->outputHeight * p->outputWidth;
    const int outputBlocks =
        (p->outputChannels + p->channelPack - 1) / p->channelPack;
    const size_t outputBytes = static_cast<size_t>(p->batch) *
                               outputBlocks * area * p->channelPack;
    std::memset(dst, 0, outputBytes);

    for (int b = 0; b < p->batch; ++b) {
        for (int a = 0; a < area; ++a) {
            for (int oc = 0; oc < p->outputChannels; ++oc) {
                int64_t sum = bias[oc];
                for (int ic = 0; ic < p->inputChannels; ++ic) {
                    const int32_t inputValue = static_cast<int8_t>(
                        src[packedOffset(
                            b, ic, a, p->batch, area,
                            p->channelPack)]);
                    const int32_t weightValue = static_cast<int8_t>(
                        weight[packedWeightOffset(
                            oc, ic, p->inputChannels)]);
                    sum += static_cast<int64_t>(inputValue) *
                           weightValue;
                }
                const int32_t accumulator =
                    sum < std::numeric_limits<int32_t>::min()
                        ? std::numeric_limits<int32_t>::min()
                        : (sum > std::numeric_limits<int32_t>::max()
                               ? std::numeric_limits<int32_t>::max()
                               : static_cast<int32_t>(sum));
                uint8_t output = 0;
                const AEEResult status = requantize_i8_reference(
                    &output, &accumulator, 1, requant + oc);
                if (status != AEE_SUCCESS) {
                    return status;
                }
                dst[packedOffset(
                    b, oc, a, p->batch, area,
                    p->channelPack)] = output;
            }
        }
    }
    return AEE_SUCCESS;
}

extern "C" AEEResult htp_ops_dispatch_conv2d_w8a8(
    uint8_t* dst, const uint8_t* src, const uint8_t* weight,
    const uint8_t* bias, const uint8_t* requant,
    const int32_t* params, int32_t paramCount) {
    if (dst == nullptr || src == nullptr || weight == nullptr ||
        bias == nullptr || requant == nullptr || params == nullptr ||
        paramCount !=
            static_cast<int32_t>(
                sizeof(DSPConv2DInt8Param) / sizeof(int32_t))) {
        return AEE_EBADPARM;
    }
    return htp_ops_conv1x1_w8a8(
        dst, src, weight, reinterpret_cast<const int32_t*>(bias),
        reinterpret_cast<const DSPRequantParam*>(requant),
        reinterpret_cast<const DSPConv2DInt8Param*>(params));
}
