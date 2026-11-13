#include "dsp/int8_cnn.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __hexagon__
#include "dsp/dma_utils.h"
#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_utils.h"
#include "dsp/vtcm_mgr.h"
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

static size_t packed_area_offset(int batchIndex, int channel, int areaIndex,
                                 int batch, int area, int pack) {
    return ((size_t)(channel / pack * batch + batchIndex) * area +
            areaIndex) *
               pack +
           channel % pack;
}

int32_t htp_ops_conv1x1_w8a8_hmx_eligible(
    const DSPRequantParam* requant, const DSPConv2DInt8Param* p) {
    if (requant == NULL || p == NULL || p->batch <= 0 ||
        p->inputHeight <= 0 || p->inputWidth <= 0 ||
        p->outputHeight != p->inputHeight ||
        p->outputWidth != p->inputWidth || p->inputChannels <= 0 ||
        p->outputChannels <= 0 || p->channelPack != 64 ||
        (int64_t)p->outputHeight * p->outputWidth <= 32 ||
        (int64_t)p->outputHeight * p->outputWidth > INT32_MAX) {
        return 0;
    }
    for (int32_t channel = 0; channel < p->outputChannels; ++channel) {
        const DSPRequantParam* param = requant + channel;
        if (param->multiplier <= 0 || param->outputZero != 0 ||
            param->clampMin < -128 || param->clampMax > 127 ||
            param->clampMin > param->clampMax) {
            return 0;
        }
    }
    return 1;
}

static uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

AEEResult htp_ops_conv1x1_w8a8_hmx_layout(
    int32_t inputChannels, uint32_t vtcmCapacity,
    DSPConv1x1W8A8HmxLayout* layout) {
    enum {
        kOperandTileBytes = 2048,
        kOutputTileBytes = 2048,
        kScaleTableBytes = 256,
        kWeightOffset = 4096
    };
    if (inputChannels <= 0 || layout == NULL) {
        return AEE_EBADPARM;
    }

    const uint64_t inputTiles =
        ((uint64_t)(uint32_t)inputChannels + 31u) / 32u;
    const uint64_t inputBlocks =
        ((uint64_t)(uint32_t)inputChannels + 63u) / 64u;
    const uint64_t weightBytes = inputTiles * kOperandTileBytes;
    const uint64_t rawWeightBytes = inputTiles * 1024u;
    const uint64_t rawWeightOffset = kWeightOffset + weightBytes;
    const uint64_t biasOffset = rawWeightOffset + rawWeightBytes;
    const uint64_t requantOffset = biasOffset + 32u * sizeof(int32_t);
    const uint64_t activationOffset = align_up_u64(
        requantOffset + 32u * sizeof(DSPRequantParam),
        kOperandTileBytes);
    const uint64_t activationBytesPerAreaTile = weightBytes;
    const uint64_t rawInputBytesPerAreaTile =
        inputBlocks * 64u * 32u;
    const uint64_t stagedOutputBytesPerAreaTile = 32u * 64u;
    const uint64_t bytesPerAreaTile =
        activationBytesPerAreaTile + rawInputBytesPerAreaTile +
        stagedOutputBytesPerAreaTile;

    if (inputTiles > UINT32_MAX || weightBytes > UINT32_MAX ||
        rawWeightOffset > UINT32_MAX || rawWeightBytes > UINT32_MAX ||
        biasOffset > UINT32_MAX || requantOffset > UINT32_MAX ||
        activationOffset > UINT32_MAX ||
        activationBytesPerAreaTile > UINT32_MAX ||
        rawInputBytesPerAreaTile > UINT32_MAX ||
        bytesPerAreaTile == 0 || bytesPerAreaTile > UINT32_MAX) {
        return AEE_EBADPARM;
    }
    if (activationOffset > vtcmCapacity ||
        bytesPerAreaTile > vtcmCapacity - activationOffset) {
        return AEE_ENOMEMORY;
    }

    const uint64_t maxAreaTiles =
        (vtcmCapacity - activationOffset) / bytesPerAreaTile;
    const uint64_t rawInputOffset =
        activationOffset + maxAreaTiles * activationBytesPerAreaTile;
    const uint64_t stagedOutputOffset =
        rawInputOffset + maxAreaTiles * rawInputBytesPerAreaTile;
    const uint64_t vtcmBytes =
        stagedOutputOffset + maxAreaTiles * stagedOutputBytesPerAreaTile;
    if (maxAreaTiles == 0 || maxAreaTiles > INT32_MAX ||
        rawInputOffset > UINT32_MAX ||
        stagedOutputOffset > UINT32_MAX || vtcmBytes > UINT32_MAX ||
        kOutputTileBytes + kScaleTableBytes > kWeightOffset) {
        return AEE_EBADPARM;
    }

    memset(layout, 0, sizeof(*layout));
    layout->inputTiles = (uint32_t)inputTiles;
    layout->rawOutputOffset = 0;
    layout->scaleTableOffset = kOutputTileBytes;
    layout->weightOffset = kWeightOffset;
    layout->weightBytes = (uint32_t)weightBytes;
    layout->rawWeightOffset = (uint32_t)rawWeightOffset;
    layout->rawWeightBytes = (uint32_t)rawWeightBytes;
    layout->biasOffset = (uint32_t)biasOffset;
    layout->requantOffset = (uint32_t)requantOffset;
    layout->activationOffset = (uint32_t)activationOffset;
    layout->activationBytesPerAreaTile =
        (uint32_t)activationBytesPerAreaTile;
    layout->rawInputBytesPerAreaTile =
        (uint32_t)rawInputBytesPerAreaTile;
    layout->stagedOutputBytesPerAreaTile =
        (uint32_t)stagedOutputBytesPerAreaTile;
    layout->maxAreaTilesPerChunk = (uint32_t)maxAreaTiles;
    layout->rawInputOffset = (uint32_t)rawInputOffset;
    layout->stagedOutputOffset = (uint32_t)stagedOutputOffset;
    layout->vtcmBytes = (uint32_t)vtcmBytes;
    return AEE_SUCCESS;
}

AEEResult htp_ops_conv1x1_w8a8_hmx_chunk(
    int32_t areaTiles, int32_t maxAreaTilesPerChunk, int32_t chunkIndex,
    int32_t* areaTileBase, int32_t* areaTilesInChunk) {
    if (areaTiles <= 0 || maxAreaTilesPerChunk <= 0 ||
        chunkIndex < 0 || areaTileBase == NULL ||
        areaTilesInChunk == NULL) {
        return AEE_EBADPARM;
    }
    const int64_t base =
        (int64_t)chunkIndex * maxAreaTilesPerChunk;
    if (base >= areaTiles) {
        return AEE_EBADPARM;
    }
    const int32_t remaining = areaTiles - (int32_t)base;
    *areaTileBase = (int32_t)base;
    *areaTilesInChunk =
        remaining < maxAreaTilesPerChunk
            ? remaining
            : maxAreaTilesPerChunk;
    return AEE_SUCCESS;
}

AEEResult htp_ops_pack_hmx_activation_u8_m32k64(
    uint8_t* dst, const uint8_t* src, int32_t batchIndex,
    int32_t batchCount, int32_t areaBase, int32_t validRows,
    int32_t channelBase, int32_t validChannels, int32_t channels,
    int32_t area, int32_t channelPack) {
    if (dst == NULL || src == NULL || batchIndex < 0 ||
        batchIndex >= batchCount || batchCount <= 0 || areaBase < 0 ||
        validRows < 0 || validRows > 32 ||
        areaBase + validRows > area || channelBase < 0 ||
        validChannels < 0 || validChannels > 32 ||
        channelBase + validChannels > channels || channels <= 0 ||
        area <= 0 || channelPack <= 0 || channelPack % 32 != 0) {
        return AEE_EBADPARM;
    }

    memset(dst, 0, 2048);
    for (int32_t row = 0; row < validRows; ++row) {
        for (int32_t channel = 0; channel < validChannels; ++channel) {
            const uint8_t value = src[packed_area_offset(
                batchIndex, channelBase + channel, areaBase + row,
                batchCount, area, channelPack)];
            dst[(size_t)row * 32 + channel] =
                (uint8_t)((int32_t)(int8_t)value + 128);
        }
    }
    return AEE_SUCCESS;
}

AEEResult htp_ops_pack_hmx_weight_s8_k64n32(
    uint8_t* dst, const uint8_t* packedWeight, int32_t inputTiles,
    int32_t outputTile, int32_t inputTileBase,
    int32_t validInputChannels, int32_t validOutputChannels) {
    if (dst == NULL || packedWeight == NULL || inputTiles <= 0 ||
        outputTile < 0 || inputTileBase < 0 ||
        inputTileBase >= inputTiles || validInputChannels < 0 ||
        validInputChannels > 64 || validOutputChannels < 0 ||
        validOutputChannels > 32 ||
        (validInputChannels + 31) / 32 >
            inputTiles - inputTileBase) {
        return AEE_EBADPARM;
    }

    memset(dst, 0, 2048);
    for (int32_t inputChannel = 0;
         inputChannel < validInputChannels; ++inputChannel) {
        const uint8_t* tile =
            packedWeight +
            ((size_t)outputTile * inputTiles + inputTileBase +
             inputChannel / 32) *
                1024;
        const int32_t tileInputChannel = inputChannel % 32;
        const int32_t inputPair = tileInputChannel / 2;
        for (int32_t outputChannel = 0;
             outputChannel < validOutputChannels; ++outputChannel) {
            const int32_t pairLane =
                outputChannel * 2 + (tileInputChannel & 1);
            const size_t sourceOffset =
                (size_t)(inputPair / 2) * 128 +
                (size_t)pairLane * 2 + (size_t)(inputPair & 1);
            const size_t destinationOffset =
                (size_t)(inputChannel / 4) * 128 +
                (size_t)(outputChannel / 4) * 16 +
                (size_t)(outputChannel & 3) * 4 +
                (size_t)(inputChannel & 3);
            dst[destinationOffset] = tile[sourceOffset];
        }
    }
    return AEE_SUCCESS;
}

static uint16_t hmx_w8a8_fp16_bits(float value) {
#if defined(__hexagon__) || defined(__arm__) || defined(__aarch64__)
    __fp16 half = (__fp16)value;
#else
    _Float16 half = (_Float16)value;
#endif
    uint16_t bits = 0;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
}

AEEResult htp_ops_prepare_hmx_w8a8_scale_bias_table_32(
    uint32_t* table, const uint8_t* packedWeight, const int32_t* bias,
    const DSPRequantParam* requant, int32_t inputTiles,
    int32_t outputTile, int32_t inputChannels,
    int32_t outputChannels) {
    if (table == NULL || packedWeight == NULL || bias == NULL ||
        requant == NULL || inputTiles <= 0 || outputTile < 0 ||
        inputChannels <= 0 ||
        inputTiles != (inputChannels + 31) / 32 ||
        outputChannels <= 0 ||
        outputTile >= (outputChannels + 31) / 32) {
        return AEE_EBADPARM;
    }

    memset(table, 0, 64 * sizeof(uint32_t));
    const int32_t outputBase = outputTile * 32;
    const int32_t validOutputChannels =
        outputChannels - outputBase < 32
            ? outputChannels - outputBase
            : 32;
#if defined(__hexagon__)
    int32_t weightSums[32] __attribute__((aligned(128)));
    HVX_Vector sum = Q6_V_vzero();
    const HVX_Vector ones = Q6_Vb_vsplat_R(1);
    for (int32_t tile = 0; tile < inputTiles; ++tile) {
        const uint8_t* tileWeight =
            packedWeight +
            ((size_t)outputTile * inputTiles + tile) * 1024;
        for (int32_t group = 0; group < 8; ++group) {
            const int32_t valid =
                inputChannels - tile * 32 - group * 4;
            const uint32_t validMask =
                valid >= 4
                    ? UINT32_MAX
                    : valid <= 0
                          ? 0
                          : valid == 3
                                ? UINT32_C(0x00ffffff)
                                : valid == 2
                                      ? UINT32_C(0x00ff00ff)
                                      : UINT32_C(0x000000ff);
            sum = Q6_Vw_vrmpyacc_VwVbVb(
                sum,
                Q6_V_vand_VV(
                    vmemu(tileWeight + group * 128),
                    Q6_V_vsplat_R((int32_t)validMask)),
                ones);
        }
    }
    vmem(weightSums) = sum;
#endif

    for (int32_t outputChannel = 0;
         outputChannel < validOutputChannels; ++outputChannel) {
        const int32_t globalOutputChannel =
            outputBase + outputChannel;
        const DSPRequantParam* param =
            requant + globalOutputChannel;
        if (param->multiplier <= 0 || param->outputZero != 0 ||
            param->clampMin < -128 || param->clampMax > 127 ||
            param->clampMin > param->clampMax) {
            return AEE_EBADPARM;
        }

        int32_t weightSum = 0;
#if defined(__hexagon__)
        weightSum = weightSums[outputChannel];
#else
        for (int32_t inputChannel = 0;
             inputChannel < inputChannels; ++inputChannel) {
            const uint8_t* tile =
                packedWeight +
                ((size_t)outputTile * inputTiles +
                 inputChannel / 32) *
                    1024;
            const int32_t tileInputChannel = inputChannel & 31;
            const int32_t inputPair = tileInputChannel / 2;
            const int32_t pairLane =
                outputChannel * 2 + (tileInputChannel & 1);
            const size_t sourceOffset =
                (size_t)(inputPair / 2) * 128 +
                (size_t)pairLane * 2 +
                (size_t)(inputPair & 1);
            weightSum += (int32_t)(int8_t)tile[sourceOffset];
        }
#endif

        const float scale =
            ldexpf((float)param->multiplier, param->shift - 31);
        const float encodedScale = scale * 512.0f;
        if (!isfinite(scale) || scale <= 0.0f ||
            !isfinite(encodedScale) ||
            128.0 / (double)scale > (double)INT32_MAX) {
            return AEE_EBADPARM;
        }
        const uint16_t scaleBits =
            hmx_w8a8_fp16_bits(encodedScale);
        if ((scaleBits & UINT16_C(0x7fff)) == 0 ||
            (scaleBits & UINT16_C(0x7c00)) ==
                UINT16_C(0x7c00)) {
            return AEE_EBADPARM;
        }

        const int64_t correctedBias =
            (int64_t)bias[globalOutputChannel] -
            (int64_t)128 * weightSum;
        const int64_t zeroPointBias =
            (int64_t)llroundf(128.0f / scale);
        const int64_t accumulatorBias =
            correctedBias + zeroPointBias;
        const int64_t accumulatorBound =
            (int64_t)inputChannels * 255 * 128;
        if (accumulatorBias < INT32_MIN ||
            accumulatorBias > INT32_MAX ||
            accumulatorBias - accumulatorBound < INT32_MIN ||
            accumulatorBias + accumulatorBound > INT32_MAX) {
            return AEE_EBADPARM;
        }
        table[outputChannel] =
            UINT32_C(0x00400000) | (uint32_t)scaleBits;
        table[32 + outputChannel] =
            (uint32_t)(int32_t)accumulatorBias;
    }
    return AEE_SUCCESS;
}

#ifdef __hexagon__

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

#if __HEXAGON_ARCH__ == 73 && defined(HTP_OPS_W8A8_HMX_ENABLE)

#define W8A8_HMX_MAX_VTCM_BYTES (1024u * 1024u)

static void conv1x1_w8a8_prepare_dma_descriptor(
    dma_desc_1d_t* descriptor, const void* src, void* dst,
    uint32_t bytes) {
    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->length = bytes;
    descriptor->type = DMA_DESC_TYPE_1D;
    descriptor->src_bypass = 0;
    descriptor->dst_bypass = 1;
    descriptor->ordered = 1;
    descriptor->dstate = DMA_DESC_DSTATE_PENDING;
    descriptor->src = (uint32_t)src;
    descriptor->dst = (uint32_t)dst;
}

static AEEResult repack_hmx_weight_s8_k32n32(
    uint8_t* dst, const uint8_t* src, int32_t validInputChannels,
    int32_t validOutputChannels) {
    if (dst == NULL || src == NULL || validInputChannels < 0 ||
        validInputChannels > 32 || validOutputChannels < 0 ||
        validOutputChannels > 32) {
        return AEE_EBADPARM;
    }

    memset(dst, 0, 2048);
    if (validInputChannels == 32 && validOutputChannels == 32) {
        const HVX_Vector keep =
            Q6_V_vsplat_R((int32_t)UINT32_C(0xff0000ff));
        const HVX_Vector byte1 =
            Q6_V_vsplat_R((int32_t)UINT32_C(0x0000ff00));
        const HVX_Vector byte2 =
            Q6_V_vsplat_R((int32_t)UINT32_C(0x00ff0000));
        for (int32_t group = 0; group < 8; ++group) {
            const HVX_Vector value = vmemu(src + group * 128);
            const HVX_Vector swapped = Q6_V_vor_VV(
                Q6_Vw_vasl_VwR(
                    Q6_V_vand_VV(value, byte1), 8),
                Q6_Vuw_vlsr_VuwR(
                    Q6_V_vand_VV(value, byte2), 8));
            vmem(dst + group * 128) = Q6_V_vor_VV(
                Q6_V_vand_VV(value, keep), swapped);
        }
        return AEE_SUCCESS;
    }

    for (int32_t outputChannel = 0;
         outputChannel < validOutputChannels; ++outputChannel) {
        for (int32_t inputChannel = 0;
             inputChannel < validInputChannels; ++inputChannel) {
            const int32_t inputPair = inputChannel / 2;
            const int32_t pairLane =
                outputChannel * 2 + (inputChannel & 1);
            const size_t sourceOffset =
                (size_t)(inputPair / 2) * 128 +
                (size_t)pairLane * 2 +
                (size_t)(inputPair & 1);
            const size_t destinationOffset =
                (size_t)(inputChannel / 4) * 128 +
                (size_t)(outputChannel / 4) * 16 +
                (size_t)(outputChannel & 3) * 4 +
                (size_t)(inputChannel & 3);
            dst[destinationOffset] = src[sourceOffset];
        }
    }
    return AEE_SUCCESS;
}

static __attribute__((noinline)) AEEResult
conv1x1_w8a8_hmx_pack_activation_chunk(
    uint8_t* activationTiles, uint8_t* rawInput,
    const uint8_t* src, const DSPConv2DInt8Param* p,
    int32_t batchIndex, int32_t areaTileBase,
    int32_t areaTilesInChunk, int32_t inputTiles,
    int32_t outputArea) {
    enum { kOperandTileBytes = 2048 };
    const int32_t chunkArea = areaTilesInChunk * 32;
    const int32_t areaBase = areaTileBase * 32;
    const int32_t validArea =
        outputArea - areaBase < chunkArea
            ? outputArea - areaBase
            : chunkArea;
    const int32_t blocks = (p->inputChannels + 63) / 64;

    for (int32_t block = 0; block < blocks; ++block) {
        uint8_t* const rawBlock =
            rawInput + (size_t)block * chunkArea * 64;
        dma_desc_1d_t descriptor __attribute__((aligned(64)));
        conv1x1_w8a8_prepare_dma_descriptor(
            &descriptor,
            src + packed_area_offset(
                      batchIndex, block * 64, areaBase, p->batch,
                      outputArea, p->channelPack),
            rawBlock, (uint32_t)validArea * 64);
        if (dma_submit_one(&descriptor) != 0 ||
            !dma_wait_for_idle()) {
            return AEE_EFAILED;
        }
        __asm__ __volatile__("" ::: "memory");
        if (validArea < chunkArea) {
            memset(rawBlock + (size_t)validArea * 64, 0,
                   (size_t)(chunkArea - validArea) * 64);
        }
    }

    for (int32_t areaTile = 0;
         areaTile < areaTilesInChunk; ++areaTile) {
        for (int32_t block = 0; block < blocks; ++block) {
            uint8_t* const tile0 =
                activationTiles +
                ((size_t)areaTile * inputTiles + block * 2) *
                    kOperandTileBytes;
            uint8_t* const tile1 =
                block * 2 + 1 < inputTiles
                    ? tile0 + kOperandTileBytes
                    : NULL;
            memset(tile0, 0, kOperandTileBytes);
            if (tile1 != NULL) {
                memset(tile1, 0, kOperandTileBytes);
            }
            const uint8_t* const rows =
                rawInput +
                ((size_t)block * chunkArea + areaTile * 32) * 64;
            for (int32_t row = 0; row < 32; row += 4) {
                const HVX_VectorPair packed = Q6_W_vdeal_VVR(
                    vmem(rows + (row + 2) * 64),
                    vmem(rows + row * 64), -32);
                vmem(tile0 + row * 32) = Q6_V_vxor_VV(
                    Q6_V_lo_W(packed), Q6_Vb_vsplat_R(128));
                if (tile1 != NULL) {
                    vmem(tile1 + row * 32) = Q6_V_vxor_VV(
                        Q6_V_hi_W(packed),
                        Q6_Vb_vsplat_R(128));
                }
            }
        }
    }
    return AEE_SUCCESS;
}

static __attribute__((noinline)) AEEResult
conv1x1_w8a8_hmx_prepare_output_tile(
    uint32_t* scaleTable, uint8_t* weightTiles,
    uint8_t* rawWeight, int32_t* stagedBias,
    DSPRequantParam* stagedRequant, const uint8_t* weight,
    const int32_t* bias, const DSPRequantParam* requant,
    const DSPConv2DInt8Param* p, int32_t inputTiles,
    int32_t outputTile) {
    enum { kOperandTileBytes = 2048 };
    const int32_t outputBase = outputTile * 32;
    const int32_t validOutputChannels =
        p->outputChannels - outputBase < 32
            ? p->outputChannels - outputBase
            : 32;
    dma_desc_1d_t descriptors[3] __attribute__((aligned(64)));
    conv1x1_w8a8_prepare_dma_descriptor(
        &descriptors[0],
        weight + (size_t)outputTile * inputTiles * 1024,
        rawWeight, (uint32_t)inputTiles * 1024);
    conv1x1_w8a8_prepare_dma_descriptor(
        &descriptors[1], bias + outputBase, stagedBias,
        (uint32_t)validOutputChannels * sizeof(int32_t));
    conv1x1_w8a8_prepare_dma_descriptor(
        &descriptors[2], requant + outputBase, stagedRequant,
        (uint32_t)validOutputChannels *
            sizeof(DSPRequantParam));
    descriptors[0].next = (uint32_t)&descriptors[1];
    descriptors[1].next = (uint32_t)&descriptors[2];
    if (dma_submit_one(&descriptors[0]) != 0 ||
        !dma_wait_for_idle()) {
        return AEE_EFAILED;
    }
    __asm__ __volatile__("" ::: "memory");

    AEEResult status =
        htp_ops_prepare_hmx_w8a8_scale_bias_table_32(
            scaleTable, rawWeight, stagedBias, stagedRequant,
            inputTiles, 0, p->inputChannels,
            validOutputChannels);
    for (int32_t inputTile = 0;
         inputTile < inputTiles && status == AEE_SUCCESS;
         ++inputTile) {
        const int32_t inputBase = inputTile * 32;
        const int32_t validInputChannels =
            p->inputChannels - inputBase < 32
                ? p->inputChannels - inputBase
                : 32;
        status = repack_hmx_weight_s8_k32n32(
            weightTiles +
                (size_t)inputTile * kOperandTileBytes,
            rawWeight + (size_t)inputTile * 1024,
            validInputChannels, validOutputChannels);
    }
    return status;
}

static __attribute__((noinline)) AEEResult
conv1x1_w8a8_hmx_execute_output_tile(
    uint8_t* dst, uint8_t* rawOutput,
    const uint32_t* scaleTable,
    const uint8_t* activationTiles,
    const uint8_t* weightTiles,
    const DSPRequantParam* requant,
    const DSPConv2DInt8Param* p, int32_t areaTileBase,
    int32_t areaTilesInChunk, int32_t inputTiles,
    int32_t outputArea, int32_t outputTile) {
    enum {
        kOperandTileBytes = 2048,
        kActivationControl = 0x77c,
        kWeightRange = 0x3ff,
        kOutputControl = 0x700
    };
    const int32_t outputBase = outputTile * 32;
    const int32_t validOutputChannels =
        p->outputChannels - outputBase < 32
            ? p->outputChannels - outputBase
            : 32;
    uint8_t maskBytes[128] __attribute__((aligned(128))) = {0};
    int8_t minBytes[128] __attribute__((aligned(128))) = {0};
    int8_t maxBytes[128] __attribute__((aligned(128))) = {0};
    const int32_t inner = outputBase & 63;
    for (int32_t row = 0; row < 2; ++row) {
        for (int32_t channel = 0;
             channel < validOutputChannels; ++channel) {
            const int32_t lane = row * 64 + inner + channel;
            maskBytes[lane] = 255;
            minBytes[lane] =
                (int8_t)requant[channel].clampMin;
            maxBytes[lane] =
                (int8_t)requant[channel].clampMax;
        }
    }
    const HVX_VectorPred mask = Q6_Q_vcmp_eq_VbVb(
        vmem(maskBytes), Q6_Vb_vsplat_R(-1));
    const HVX_Vector minimum = vmem(minBytes);
    const HVX_Vector maximum = vmem(maxBytes);
    hmx_set_output_scales(scaleTable);

    for (int32_t areaTile = 0;
         areaTile < areaTilesInChunk; ++areaTile) {
        const int32_t areaBase =
            (areaTileBase + areaTile) * 32;
        const int32_t validRows =
            outputArea - areaBase < 32
                ? outputArea - areaBase
                : 32;
        Q6_mxclracc();
        for (int32_t inputTile = 0;
             inputTile < inputTiles; ++inputTile) {
            Q6_activation_ub_mxmem_RR_deep_cm(
                (uintptr_t)(
                    activationTiles +
                    ((size_t)areaTile * inputTiles + inputTile) *
                        kOperandTileBytes),
                kActivationControl);
            Q6_weight_b_mxmem_RR(
                (uintptr_t)(
                    weightTiles +
                    (size_t)inputTile * kOperandTileBytes),
                kWeightRange);
        }
        Q6_cvt_ub_acc_R(0);
        __builtin_HEXAGON_M8_mxmem_cm(
            rawOutput, kOutputControl);
        __asm__ __volatile__("" ::: "memory");

        for (int32_t row = 0; row < validRows; row += 2) {
            const HVX_VectorPair pair = Q6_W_vshuff_VVR(
                Q6_V_vzero(),
                vmem(rawOutput + (row & ~3) * 32), -32);
            HVX_Vector packed =
                (row & 2) == 0 ? Q6_V_lo_W(pair)
                               : Q6_V_hi_W(pair);
            if (inner != 0) {
                packed = Q6_V_vror_VR(packed, 96);
            }
            HVX_Vector value = Q6_V_vxor_VV(
                packed, Q6_Vb_vsplat_R(128));
            value = Q6_Vb_vmax_VbVb(value, minimum);
            value = Q6_Vb_vmin_VbVb(value, maximum);
            HVX_VectorPred writeMask = mask;
            if (row + 1 == validRows) {
                writeMask = Q6_Q_and_QQ(
                    mask, Q6_Q_vsetq_R(64));
            }
            uint8_t* const output =
                dst + (size_t)(areaTile * 32 + row) * 64;
            vmem(output) = Q6_V_vmux_QVV(
                writeMask, value, vmem(output));
        }
    }
    return AEE_SUCCESS;
}

static AEEResult conv1x1_w8a8_hmx(
    uint8_t* dst, const uint8_t* src, const uint8_t* weight,
    const int32_t* bias, const DSPRequantParam* requant,
    const DSPConv2DInt8Param* p) {
    if (!vtcm_manager_is_acquired()) {
        return AEE_ENOMEMORY;
    }
    uint32_t vtcmCapacity = vtcm_manager_get_vtcm_size();
    if (vtcmCapacity > W8A8_HMX_MAX_VTCM_BYTES) {
        vtcmCapacity = W8A8_HMX_MAX_VTCM_BYTES;
    }
    DSPConv1x1W8A8HmxLayout layout;
    AEEResult status = htp_ops_conv1x1_w8a8_hmx_layout(
        p->inputChannels, vtcmCapacity, &layout);
    if (status != AEE_SUCCESS) {
        return status;
    }

    uint8_t* const staging =
        (uint8_t*)vtcm_manager_reserve_area(
            "conv1x1_w8a8_hmx", vtcmCapacity, 2048);
    const uintptr_t vtcmBase =
        (uintptr_t)vtcm_manager_get_vtcm_base();
    const uint32_t vtcmSize = vtcm_manager_get_vtcm_size();
    if (staging == NULL ||
        ((uintptr_t)staging & UINT32_C(2047)) != 0 ||
        (uintptr_t)staging < vtcmBase ||
        (uintptr_t)staging - vtcmBase > vtcmSize ||
        vtcmCapacity > vtcmSize - ((uintptr_t)staging - vtcmBase)) {
        return AEE_ENOMEMORY;
    }

    const int32_t inputTiles = (int32_t)layout.inputTiles;
    const int32_t outputTiles =
        (p->outputChannels + 31) / 32;
    const int32_t outputArea =
        p->outputHeight * p->outputWidth;
    const int32_t areaTiles = (outputArea + 31) / 32;
    const int32_t chunkCount =
        (areaTiles + (int32_t)layout.maxAreaTilesPerChunk - 1) /
        (int32_t)layout.maxAreaTilesPerChunk;
    uint8_t* const rawOutput =
        staging + layout.rawOutputOffset;
    uint32_t* const scaleTable =
        (uint32_t*)(staging + layout.scaleTableOffset);
    uint8_t* const weightTiles =
        staging + layout.weightOffset;
    uint8_t* const rawWeight =
        staging + layout.rawWeightOffset;
    int32_t* const stagedBias =
        (int32_t*)(staging + layout.biasOffset);
    DSPRequantParam* const stagedRequant =
        (DSPRequantParam*)(staging + layout.requantOffset);
    uint8_t* const activationTiles =
        staging + layout.activationOffset;
    uint8_t* const rawInput =
        staging + layout.rawInputOffset;
    uint8_t* const stagedOutput =
        staging + layout.stagedOutputOffset;

    hmx_manager_enable_execution();
    hmx_unit_acquire();
    for (int32_t batchIndex = 0;
         batchIndex < p->batch && status == AEE_SUCCESS;
         ++batchIndex) {
        for (int32_t chunkIndex = 0;
             chunkIndex < chunkCount && status == AEE_SUCCESS;
             ++chunkIndex) {
            int32_t areaTileBase = 0;
            int32_t areaTilesInChunk = 0;
            status = htp_ops_conv1x1_w8a8_hmx_chunk(
                areaTiles,
                (int32_t)layout.maxAreaTilesPerChunk,
                chunkIndex, &areaTileBase,
                &areaTilesInChunk);
            if (status != AEE_SUCCESS) {
                break;
            }
            status = conv1x1_w8a8_hmx_pack_activation_chunk(
                activationTiles, rawInput, src, p, batchIndex,
                areaTileBase, areaTilesInChunk, inputTiles,
                outputArea);
            for (int32_t outputTile = 0;
                 outputTile < outputTiles &&
                 status == AEE_SUCCESS;
                 ++outputTile) {
                if ((outputTile & 1) == 0) {
                    memset(
                        stagedOutput, 0,
                        (size_t)areaTilesInChunk *
                            layout.stagedOutputBytesPerAreaTile);
                }
                status =
                    conv1x1_w8a8_hmx_prepare_output_tile(
                        scaleTable, weightTiles, rawWeight,
                        stagedBias, stagedRequant, weight, bias,
                        requant, p, inputTiles, outputTile);
                if (status == AEE_SUCCESS) {
                    status =
                        conv1x1_w8a8_hmx_execute_output_tile(
                            stagedOutput, rawOutput, scaleTable,
                            activationTiles, weightTiles,
                            stagedRequant, p, areaTileBase,
                            areaTilesInChunk, inputTiles,
                            outputArea, outputTile);
                }
                if (status == AEE_SUCCESS &&
                    ((outputTile & 1) != 0 ||
                     outputTile + 1 == outputTiles)) {
                    const int32_t areaBase =
                        areaTileBase * 32;
                    const int32_t validArea =
                        outputArea - areaBase <
                                areaTilesInChunk * 32
                            ? outputArea - areaBase
                            : areaTilesInChunk * 32;
                    dma_desc_1d_t descriptor
                        __attribute__((aligned(64)));
                    conv1x1_w8a8_prepare_dma_descriptor(
                        &descriptor, stagedOutput,
                        dst + packed_area_offset(
                                  batchIndex,
                                  (outputTile / 2) * 64,
                                  areaBase, p->batch, outputArea,
                                  p->channelPack),
                        (uint32_t)validArea * 64);
                    descriptor.src_bypass = 1;
                    descriptor.dst_bypass = 0;
                    __asm__ __volatile__("" ::: "memory");
                    if (dma_submit_one(&descriptor) != 0 ||
                        !dma_wait_for_idle()) {
                        status = AEE_EFAILED;
                    }
                    __asm__ __volatile__("" ::: "memory");
                }
            }
        }
    }
    hmx_unit_release();
    hmx_manager_disable_execution();
    return status;
}

#endif

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
    // Both vector accumulators wrap on overflow; retain the INT64 reference.
    const int64_t bound = (int64_t)p->inputChannels * 127 * 128;
    for (int channel = 0; channel < p->outputChannels; ++channel) {
        if ((int64_t)bias[channel] - bound < INT32_MIN ||
            (int64_t)bias[channel] + bound > INT32_MAX) {
            return conv1x1_w8a8_reference(dst, src, weight, bias, requant, p);
        }
    }
#if __HEXAGON_ARCH__ == 73 && defined(HTP_OPS_W8A8_HMX_ENABLE)
    if (htp_ops_conv1x1_w8a8_hmx_eligible(requant, p)) {
        const AEEResult hmxStatus =
            conv1x1_w8a8_hmx(
                dst, src, weight, bias, requant, p);
        if (hmxStatus != AEE_ENOMEMORY &&
            hmxStatus != AEE_EBADPARM) {
            return hmxStatus;
        }
    }
#endif
    return conv1x1_w8a8_hvx(
        dst, src, weight, bias, requant, p);
#else
    return conv1x1_w8a8_reference(
        dst, src, weight, bias, requant, p);
#endif
}
