#ifndef DSP_INT8_CNN_H
#define DSP_INT8_CNN_H

#include <stdint.h>

#ifdef __hexagon__
#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <hexagon_types.h>
#else
typedef int AEEResult;
#ifndef AEE_SUCCESS
#define AEE_SUCCESS 0
#endif
#ifndef AEE_EBADPARM
#define AEE_EBADPARM 14
#endif
#ifndef AEE_ENOMEMORY
#define AEE_ENOMEMORY 4
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DSPRequantParam {
    int32_t multiplier;
    int32_t shift;
    int32_t outputZero;
    int32_t clampMin;
    int32_t clampMax;
} DSPRequantParam;

typedef struct DSPQuantizeParam {
    int32_t size;
    int32_t bytes;
    float scale;
    int32_t zero;
    int32_t batch;
    int32_t channels;
    int32_t area;
    int32_t pack;
} DSPQuantizeParam;

typedef struct DSPRequantizeCommandParam {
    int32_t size;
    DSPRequantParam requant;
} DSPRequantizeCommandParam;

typedef struct DSPConv2DInt8Param {
    int32_t batch;
    int32_t inputHeight;
    int32_t inputWidth;
    int32_t inputChannels;
    int32_t outputHeight;
    int32_t outputWidth;
    int32_t outputChannels;
    int32_t kernelY;
    int32_t kernelX;
    int32_t strideY;
    int32_t strideX;
    int32_t padY;
    int32_t padX;
    int32_t dilateY;
    int32_t dilateX;
    int32_t groups;
    int32_t channelPack;
    int32_t relu;
    int32_t relu6;
    int32_t inputZero;
    int32_t outputZero;
} DSPConv2DInt8Param;

typedef struct DSPConv1x1W8A8HmxLayout {
    uint32_t inputTiles;
    uint32_t rawOutputOffset;
    uint32_t scaleTableOffset;
    uint32_t weightOffset;
    uint32_t weightBytes;
    uint32_t rawWeightOffset;
    uint32_t rawWeightBytes;
    uint32_t biasOffset;
    uint32_t requantOffset;
    uint32_t activationOffset;
    uint32_t activationBytesPerAreaTile;
    uint32_t rawInputBytesPerAreaTile;
    uint32_t stagedOutputBytesPerAreaTile;
    uint32_t maxAreaTilesPerChunk;
    uint32_t rawInputOffset;
    uint32_t stagedOutputOffset;
    uint32_t vtcmBytes;
} DSPConv1x1W8A8HmxLayout;

AEEResult quantize_i8_reference(uint8_t* dst, const uint8_t* src,
                                int32_t size, int32_t srcBytes, float scale,
                                int32_t zero);
AEEResult dequantize_i8_reference(uint8_t* dst, const uint8_t* src,
                                  int32_t size, int32_t dstBytes, float scale,
                                  int32_t zero);
AEEResult requantize_i8_reference(uint8_t* dst, const int32_t* src,
                                  int32_t size,
                                  const DSPRequantParam* params);
AEEResult conv1x1_w8a8_reference(
    uint8_t* dst, const uint8_t* src, const uint8_t* weight,
    const int32_t* bias, const DSPRequantParam* requant,
    const DSPConv2DInt8Param* params);

AEEResult htp_ops_quantize_i8(uint8_t* dst, const uint8_t* src, int32_t size,
                              int32_t srcBytes, float scale, int32_t zero);
AEEResult htp_ops_quantize_i8_nc(
    uint8_t* dst, const uint8_t* src, int32_t size, int32_t srcBytes,
    float scale, int32_t zero, int32_t batch, int32_t channels,
    int32_t area, int32_t pack);
AEEResult htp_ops_dequantize_i8(uint8_t* dst, const uint8_t* src, int32_t size,
                                int32_t dstBytes, float scale, int32_t zero);
AEEResult htp_ops_requantize_i8(uint8_t* dst, const int32_t* src, int32_t size,
                                const DSPRequantParam* params);
AEEResult htp_ops_requantize_i8_per_channel_32(
    uint8_t* dst, const int32_t* src, const DSPRequantParam* params);
int32_t htp_ops_conv1x1_w8a8_hmx_eligible(
    const DSPRequantParam* requant, const DSPConv2DInt8Param* params);
AEEResult htp_ops_conv1x1_w8a8_hmx_layout(
    int32_t inputChannels, uint32_t vtcmCapacity,
    DSPConv1x1W8A8HmxLayout* layout);
AEEResult htp_ops_conv1x1_w8a8_hmx_chunk(
    int32_t areaTiles, int32_t maxAreaTilesPerChunk, int32_t chunkIndex,
    int32_t* areaTileBase, int32_t* areaTilesInChunk);
AEEResult htp_ops_pack_hmx_activation_u8_m32k64(
    uint8_t* dst, const uint8_t* src, int32_t batchIndex,
    int32_t batchCount, int32_t areaBase, int32_t validRows,
    int32_t channelBase, int32_t validChannels, int32_t channels,
    int32_t area, int32_t channelPack);
AEEResult htp_ops_pack_hmx_weight_s8_k64n32(
    uint8_t* dst, const uint8_t* packedWeight, int32_t inputTiles,
    int32_t outputTile, int32_t inputTileBase,
    int32_t validInputChannels, int32_t validOutputChannels);
AEEResult htp_ops_prepare_hmx_w8a8_scale_bias_table_32(
    uint32_t* table, const uint8_t* packedWeight, const int32_t* bias,
    const DSPRequantParam* requant, int32_t inputTiles,
    int32_t outputTile, int32_t inputChannels,
    int32_t outputChannels);
#ifdef __hexagon__
typedef struct DSPRequantTile32Hvx {
    HVX_Vector multiplier;
    HVX_Vector rightShift;
    HVX_Vector outputZero;
    HVX_Vector clampMin;
    HVX_Vector clampMax;
    int32_t useScalar;
} DSPRequantTile32Hvx;

AEEResult htp_ops_prepare_requantize_i8_per_channel_32_hvx(
    DSPRequantTile32Hvx* tile, const DSPRequantParam* params);
AEEResult htp_ops_requantize_i8_per_channel_32_prepared_hvx(
    uint8_t* dst, HVX_Vector src, const DSPRequantParam* params,
    const DSPRequantTile32Hvx* tile);
#endif
AEEResult htp_ops_conv1x1_w8a8(uint8_t* dst, const uint8_t* src,
                               const uint8_t* weight, const int32_t* bias,
                               const DSPRequantParam* requant,
                               const DSPConv2DInt8Param* params);
AEEResult htp_ops_dispatch_conv2d_w8a8(
    uint8_t* dst, const uint8_t* src, const uint8_t* weight,
    const uint8_t* bias, const uint8_t* requant, const int32_t* params,
    int32_t paramCount);

#ifdef __cplusplus
} // extern "C"

static_assert(sizeof(DSPRequantParam) == 20, "DSPRequantParam ABI changed");
static_assert(sizeof(DSPQuantizeParam) == 32, "DSPQuantizeParam ABI changed");
static_assert(sizeof(DSPRequantizeCommandParam) == 24,
              "DSPRequantizeCommandParam ABI changed");
static_assert(sizeof(DSPConv2DInt8Param) == 84, "DSPConv2DInt8Param ABI changed");
#endif

#endif
