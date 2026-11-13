#include "dsp/int8_cnn.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __hexagon__
#include "dsp/hvx_utils.h"
#include <hexagon_protos.h>
#include <hexagon_types.h>
#endif

static int valid_params(const DSPConv2DInt8Param* p) {
    return p != NULL && p->batch > 0 && p->inputHeight > 0 &&
           p->inputWidth > 0 && p->inputChannels > 0 &&
           p->outputHeight == p->inputHeight &&
           p->outputWidth == p->inputWidth &&
           p->outputChannels > 0 && p->kernelY == 1 &&
           p->kernelX == 1 && p->strideY == 1 && p->strideX == 1 &&
           p->padY == 0 && p->padX == 0 && p->dilateY == 1 &&
           p->dilateX == 1 && p->groups == 1 &&
           p->channelPack > 0 && p->channelPack % 32 == 0 &&
           p->inputZero == 0 && p->outputZero == 0;
}

#ifdef __hexagon__

static size_t packed_area_offset(int batchIndex, int channel, int areaIndex,
                                 int batch, int area, int pack) {
    return ((size_t)(channel / pack * batch + batchIndex) * area +
            areaIndex) *
               pack +
           channel % pack;
}

static uint32_t permute_activation_word(uint32_t value) {
    return (value & UINT32_C(0xff0000ff)) |
           ((value & UINT32_C(0x0000ff00)) << 8) |
           ((value & UINT32_C(0x00ff0000)) >> 8);
}

static uint32_t load_activation_group(
    const uint8_t* src, int batchIndex, int areaIndex, int channelBase,
    int channels, int batch, int area, int channelPack) {
    if (channelBase + 3 < channels &&
        channelBase % channelPack <= channelPack - 4) {
        uint32_t value = 0;
        memcpy(&value,
               src + packed_area_offset(
                         batchIndex, channelBase, areaIndex, batch, area,
                         channelPack),
               sizeof(value));
        return permute_activation_word(value);
    }
    uint32_t value = 0;
    uint8_t* bytes = (uint8_t*)&value;
    const int permutation[4] = {0, 2, 1, 3};
    for (int lane = 0; lane < 4; ++lane) {
        const int channel = channelBase + permutation[lane];
        if (channel < channels) {
            bytes[lane] =
                src[packed_area_offset(
                    batchIndex, channel, areaIndex, batch, area,
                    channelPack)];
        }
    }
    return value;
}

static AEEResult conv1x1_w8a8_hvx(
    uint8_t* dst, const uint8_t* src, const uint8_t* weight,
    const int32_t* bias, const DSPRequantParam* requant,
    const DSPConv2DInt8Param* p) {
    const int inputTiles = (p->inputChannels + 31) / 32;
    const int outputTiles = (p->outputChannels + 31) / 32;
    const int area = p->outputHeight * p->outputWidth;
    const int outputBlocks =
        (p->outputChannels + p->channelPack - 1) / p->channelPack;
    const size_t outputBytes =
        (size_t)p->batch * outputBlocks * area * p->channelPack;
    memset(dst, 0, outputBytes);

    for (int outputTile = 0; outputTile < outputTiles; ++outputTile) {
        const int outputChannel = outputTile * 32;
        const DSPRequantParam* tileRequant =
            requant + outputChannel;
        DSPRequantTile32Hvx preparedRequant;
        AEEResult status =
            htp_ops_prepare_requantize_i8_per_channel_32_hvx(
                &preparedRequant, tileRequant);
        if (status != AEE_SUCCESS) {
            return status;
        }
        const HVX_Vector tileBias = vmemu(bias + outputChannel);
        for (int batch = 0; batch < p->batch; ++batch) {
            for (int areaIndex = 0; areaIndex < area; ++areaIndex) {
                HVX_Vector accumulator = tileBias;
                for (int inputTile = 0; inputTile < inputTiles;
                     ++inputTile) {
                    const uint8_t* weightTile =
                        weight +
                        (size_t)(outputTile * inputTiles + inputTile) *
                            1024;
                    for (int group = 0; group < 8; ++group) {
                        const int inputChannel =
                            inputTile * 32 + group * 4;
                        const HVX_Vector activation = Q6_V_vsplat_R(
                            (int32_t)load_activation_group(
                                src, batch, areaIndex, inputChannel,
                                p->inputChannels, p->batch, area,
                                p->channelPack));
                        const HVX_Vector weightGroup =
                            vmemu(weightTile + group * 128);
                        accumulator = Q6_Vw_vrmpyacc_VwVbVb(
                            accumulator, weightGroup, activation);
                    }
                }
                uint8_t* output =
                    dst + packed_area_offset(
                              batch, outputChannel, areaIndex,
                              p->batch, area,
                              p->channelPack);
                status =
                    htp_ops_requantize_i8_per_channel_32_prepared_hvx(
                        output, accumulator, tileRequant,
                        &preparedRequant);
                if (status != AEE_SUCCESS) {
                    return status;
                }
            }
        }
    }
    return AEE_SUCCESS;
}

#endif

AEEResult htp_ops_conv1x1_w8a8(
    uint8_t* dst, const uint8_t* src, const uint8_t* weight,
    const int32_t* bias, const DSPRequantParam* requant,
    const DSPConv2DInt8Param* p) {
    if (dst == NULL || src == NULL || weight == NULL || bias == NULL ||
        requant == NULL || !valid_params(p)) {
        return AEE_EBADPARM;
    }
#ifdef __hexagon__
    // vrmpyacc wraps on overflow; retain the INT64 reference semantics for
    // channels or bias values that can exceed the INT32 accumulator range.
    const int64_t bound = (int64_t)p->inputChannels * 127 * 128;
    for (int channel = 0; channel < p->outputChannels; ++channel) {
        if ((int64_t)bias[channel] - bound < INT32_MIN ||
            (int64_t)bias[channel] + bound > INT32_MAX) {
            return conv1x1_w8a8_reference(dst, src, weight, bias, requant, p);
        }
    }
    return conv1x1_w8a8_hvx(
        dst, src, weight, bias, requant, p);
#else
    return conv1x1_w8a8_reference(
        dst, src, weight, bias, requant, p);
#endif
}
