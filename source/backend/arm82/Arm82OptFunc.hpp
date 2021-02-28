//
//  Arm82OptFunc.hpp
//  MNN
//
//  Created by MNN on 2019/02/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__
#ifndef Arm82OptFunc_hpp
#define Arm82OptFunc_hpp

#include "backend/arm82/Arm82Backend.hpp"
#include "core/Macro.h"

#define DST_XUNIT 8

#ifdef __cplusplus
extern "C" {
#endif

void MNNGemmFP16C8_UNIT(FLOAT16* dst, const FLOAT16* src, const FLOAT16* weight, const FLOAT16* bias, size_t src_loop,
                        size_t dst_step, size_t dst_loop, size_t relu, size_t relu6, size_t realDstCount);

void MNNShuffleChannelC8(FLOAT16* dst, const FLOAT16* src, size_t size, size_t halfFlag);
void MNNQuantizeFP16_UNIT4(FLOAT16* dst, const float* src, int size);
void MNNDequantizeFP16(float* dst, const int16_t* src, int size);

#ifdef __cplusplus
}
#endif

void MNNQuantizeFP16(FLOAT16* dst, const float* src, int size);

// nc4hw4 to nc8hw8(aka fp32 -> fp16), convete dataformat and data type
void MNNNC4HW4TONC8HW8(uint16_t* dest, const float* source, size_t plane, size_t channel);
// nc8hw8 to nc4hw4(aka fp16 -> fp32)
void MNNNC8HW8TONC4HW4(float* dest, const uint16_t* source, size_t plane, size_t channel);
// nchw to nc8hw8(aka fp32 -> fp16)
void MNNNCHWTONC8HW8(uint16_t* dest, const float* source, size_t plane, size_t channel);
// nc8hw8 to nchw(aka fp16 -> fp32)
void MNNNC8HW8TONCHW(float* dest, const uint16_t* source, size_t plane, size_t channel);

void MNNNC8HW8TONHWC(float* dest, const uint16_t* src, size_t plane, size_t channel);

void MNNNCHWTONC8HW8_NO_TYPE(uint16_t* dest, const uint16_t* source, size_t plane, size_t channel);
void MNNNC8HW8TONCHW_NO_TYPE(uint16_t* dest, const uint16_t* source, size_t plane, size_t channel);

template <typename TIN, typename TOUT, int UNIT>
void MNNPackUNIT(TOUT* dst, const TIN* src, size_t area, size_t depth) {
    int depthCUnit  = depth / UNIT;
    int depthRemain = depthCUnit * UNIT;
    int remain      = depth - depthRemain;
    int z, x, y;
    const TIN* srcChannel[UNIT];
    const TIN* srcOffset = src;
    for(z = 0; z < depthCUnit; ++z) {
        for(y = 0; y < UNIT; ++y) {
            srcChannel[y] = srcOffset + area * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < UNIT; ++y) {
                dst[0] = TOUT(srcChannel[y][0]);
                srcChannel[y]++;
                dst++;
            }
        }
        srcOffset += area * UNIT;
    }
    if(remain > 0){
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + area * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < remain; ++y) {
                dst[0] = TOUT(srcChannel[y][0]);
                srcChannel[y]++;
                dst++;
            }
            for(y = remain; y < UNIT; ++y) {
                dst[0] = 0;
                dst++;
            }
        }
    }
}

template <typename TIN, typename TOUT, int UNIT>
void MNNUnpackUNIT(TOUT* dst, const TIN* src, size_t area, size_t depth) {
    int depthCUnit  = depth / UNIT;
    int depthRemain = depthCUnit * UNIT;
    int remain      = depth - depthRemain;
    int z, x, y;
    const TIN* srcChannel[UNIT];
    const TIN* srcOffset = src;
    for(z = 0; z < depthCUnit; ++z) {
        for(y = 0; y < UNIT; ++y) {
            srcChannel[y] = srcOffset + y;
            for(x = 0; x < area; ++x) {
                dst[0] = TOUT(srcChannel[y][0]);
                srcChannel[y] += UNIT;
                dst++;
            }
        }
        srcOffset += area * UNIT;
    }
    if(remain > 0){
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + y;
            for(x = 0; x < area; ++x) {
                dst[0] = TOUT(srcChannel[y][0]);
                srcChannel[y] += UNIT;
                dst++;
            }
        }
    }
}

#endif

#endif
