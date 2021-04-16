//
//  CommonOptFunction.cpp
//  MNN
//
//  Created by MNN on 2018/09/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CommonOptFunction.h"
#include "ConvOpt.h"
#include "WinogradOptFunction.hpp"
#include <string.h>
#include <algorithm>
#include <math.h>
#include "math/Vec.hpp"
#include <vector>

#ifndef MNN_USE_NEON

void MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = 16;
    *lP = 1;
    *hP = 4;
}

template<typename DataType>
void MNNPackForMatMul_B_Template(DataType* dest, const DataType* source, size_t h, size_t l, bool transpose) {
    auto hP = h / 4;
    auto hR = hP * 4;
    if (hR != h) {
        ::memset(dest, 0, UP_DIV(h, 4)*4*l*sizeof(DataType));
    }
    if (!transpose) {
        for (int y=0; y<hP; ++y) {
            auto destY = dest + y * 4 * l;
            auto sourceY = source + y * 4;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 4 * x, sourceY + x * h, 4 * sizeof(DataType));
            }
        }
        auto hRemain = h - hR;
        if (hRemain > 0) {
            auto destY = dest + hP * 4 * l;
            auto sourceY = source + hP * 4;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 4 * x, sourceY + x * h, hRemain * sizeof(DataType));
            }
        }
        return;
    }
    MNNPackC4(dest, source, l, h);
}

void MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    MNNPackForMatMul_B_Template<float>(dest, source, h, l, transpose);
}

void MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias) {
    return MNNPackedMatMulRemain(C, A, B, 16, parameter, postParameters, bias);
    //return _AVX_MNNPackedMatMulFMA(C, A, B, parameter, cache);
}

void MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias) {
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto hRemain = parameter[4];
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride = bExtraStride + l * 4;
    auto hC4 = UP_DIV(h, 4);
    for (int y=0; y<hC4; ++y) {
        ::memset(C + y * cStride, 0, eSize * 4 * sizeof(float));
    }
    float alpha = 1.0f;
    float beta = 0.0f;
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
        alpha = postParameters[0];
        beta = postParameters[1];
    }

    for (int x=0; x<eSize; ++x) {
        auto dst = C + 4 * x;
        auto src = A + x;
        for (int y=0; y<hC4; ++y) {
            auto dstY = dst + y * cStride;
            auto weight = B + y * bStride;
            float summer[4] = {
                0.0f,
                0.0f,
                0.0f,
                0.0f,
            };
            if (nullptr != bias) {
                for (int v=0; v<4; ++v) {
                    summer[v] = bias[4 * y + v];
                }
            }
            for (int z=0; z<l; ++z) {
                auto aZ = src + z * 16;
                auto wZ = weight + z * 4;
                summer[0] += wZ[0] * aZ[0];
                summer[1] += wZ[1] * aZ[0];
                summer[2] += wZ[2] * aZ[0];
                summer[3] += wZ[3] * aZ[0];
            }
            for (int v=0; v<4; ++v) {
                auto dstValue = std::min(summer[v], maxValue);
                dstValue = std::max(dstValue, minValue);
                dstY[v] = dstValue;
            }
        }
    }
}
void MNNPackC4ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    int number = info[0];
    int eReal = info[1];
    int eDest = 16;
    int offset = info[3];

    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        auto dest = destOrigin + lOffset * eDest + eOffset;
        auto source = sourceGroup[n];

        auto lC4 = l / 4;
        auto lDiv = UP_DIV(l, 4);
        auto lRemain = lC4 * 4;
        for (int y=0; y<e; ++y) {
            auto yR = y % 16;
            for (int x=0; x<l; ++x) {
                auto xR = x % 4;
                auto xC = x / 4;
                dest[(x) * eDest + yR] = source[xC * eReal * 4 + y * 4 * offset + xR];
            }
        }
    }
}
#endif

#ifndef MNN_USE_SSE
#ifndef MNN_USE_NEON
void MNNTranspose32Bit(int32_t* dstO, const int32_t* srcO, int32_t* dim) {
    int w = dim[0];
    int h = dim[1];
    int srcStride = dim[2];
    int dstStride = dim[3];
    for (int i=0; i<h; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j=0; j<w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
}
#endif
void MNNFunctionInit() {
    // Do nothing
}
#endif

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

#define UNIT 4
using Vec4 = MNN::Math::Vec<float, 4>;

#ifndef MNN_USE_NEON

#ifndef MNN_USE_SSE

void MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        for (int j = 0; j < 4; ++j) {
            d[j] = s[j];
        }
    }
}

void MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        for (int j = 0; j < 4; ++j) {
            d[j] += s[j];
        }
    }
}

void MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    for (int j = 0; j < depthQuad; j++) {
        const float* slopeZ = slope + 4 * j;
        const float* srcZ   = src + 4 * j * sizeQuad;
        float* dstZ         = dst + 4 * j * sizeQuad;
        for (int i = 0; i < sizeQuad; i++) {
            for (int c = 0; c < 4; c++) {
                if (srcZ[4 * i + c] < 0) {
                    dstZ[4 * i + c] = srcZ[4 * i + c] * slopeZ[c];
                } else {
                    dstZ[4 * i + c] = srcZ[4 * i + c];
                }
            }
        }
    }
}
void MNNPackC4(float* dst, const float* src, size_t area, size_t depth) {
    int depthC4     = depth / 4;
    int depthRemain = depthC4 * 4;
    int remain      = depth - depthRemain;
    int z, x, y;
    const float* srcChannel[4];
    const float* srcOffset = src;
    for(z = 0; z < depthC4; ++z) {
        for(y = 0; y < 4; ++y) {
            srcChannel[y] = srcOffset + area * + y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < 4; ++y) {
                dst[0] = srcChannel[y][0];
                srcChannel[y]++;
                dst++;
            }
        }
        srcOffset += area * 4;
    }
    if(remain > 0){
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + area * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < remain; ++y) {
                dst[0] = srcChannel[y][0];
                srcChannel[y]++;
                dst++;
            }
            for(y = remain; y < 4; ++y) {
                dst[0] = 0;
                dst++;
            }
        }
    }
}

void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth) {
    int depthC4     = depth / 4;
    int depthRemain = depthC4 * 4;
    int remain      = depth - depthRemain;
    int z, x, y;
    const float* srcChannel[4];
    const float* srcOffset = src;
    for(z = 0; z < depthC4; ++z) {
        for(y = 0; y < 4; ++y) {
            srcChannel[y] = srcOffset + y;
            for(x = 0; x < area; ++x) {
                dst[0] = srcChannel[y][0];
                srcChannel[y] += 4;
                dst++;
            }
        }
        srcOffset += area * 4;
    }
    if(remain > 0){
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + y;
            for(x = 0; x < area; ++x) {
                dst[0] = srcChannel[y][0];
                srcChannel[y] += 4;
                dst++;
            }
        }
    }
}

void MNNExpC8(float* dest, const float* source, const float* parameters, size_t countC8) {
    auto count = countC8 * 8;
    auto param = parameters[0];
    float xLimit = 87;
    for (int i = 0; i < count; ++i) {
        auto x         = -source[i];
        x = ALIMAX(x, -xLimit);
        x = ALIMIN(x, xLimit);
        int div        = (x * parameters[1]);
        int div2       = (div + 127) << 23;
        auto xReamin   = x - div * param;
        float expBasic = *(float*)(&div2);
        auto t = xReamin;
        auto expRemain =
            ((((parameters[7] * t + parameters[6]) * t + parameters[5]) * t + parameters[4]) * t + parameters[3]) * t +
            parameters[2];
        dest[i] = expBasic * expRemain;
    }
}

void MNNReluInt8(int8_t* dst, const int8_t* src, size_t size) {
    int i;
    for (i = 0; i < size; ++i) {
        if (src[i] < 0) {
            dst[i] = 0;
        } else {
            dst[i] = src[i];
        }
    }
}
#endif // no MNN_USE_SSE

void MNNMaxFloat(float* input, float* maxBuffer, int32_t inputCountUnit) {
    for (int i = 0; i < inputCountUnit; i++) {
        for (int j = 0; j < UNIT; j++) {
            for (int m = 0; m < 2; m++) {
                maxBuffer[j] = std::max(input[i * UNIT * 2 + j * 2 + m], maxBuffer[j]);
            }
        }
    }
}
void MNNMinFloat(float* input, float* minBuffer, int32_t inputCountUnit) {
    for (int i = 0; i < inputCountUnit; i++) {
        for (int j = 0; j < UNIT; j++) {
            for (int m = 0; m < 2; m++) {
                minBuffer[j] = std::min(input[i * UNIT * 2 + j * 2 + m], minBuffer[j]);
            }
        }
    }
}
void MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                        size_t biasNumber) {
    for (int z = 0; z < biasNumber; ++z) {
        float* dstZ         = dst + planeNumber * 4 * z;
        const float* srcZ   = src + planeNumber * 4 * z;
        auto biasZ = Vec4::load(bias + 4 * z);
        auto alphaZ = Vec4::load(alpha + 4 * z);
        for (int p = 0; p < planeNumber; ++p) {
            float* dstX       = dstZ + 4 * p;
            const float* srcX = srcZ + 4 * p;
            Vec4::save(dstX, (Vec4::load(srcX) * alphaZ) + biasZ);
        }
    }
}



void MNNUInt8ToInt16WithOffsetC4Common(int16_t* dst, const uint8_t* src, size_t zeroPoint, size_t sizeQuad,
                                       size_t dstStride, size_t srcStride) {
    dstStride /= sizeof(int16_t);
    srcStride /= sizeof(uint8_t);
    for (int z = 0; z < sizeQuad; ++z) {
        auto dstZ = dst + dstStride * z;
        auto srcZ = src + srcStride * z;
        for (int j = 0; j < 4; ++j) {
            dstZ[j] = (int16_t)((int32_t)srcZ[j] - (int32_t)zeroPoint);
        }
    }
}

void MNNUInt8ToInt16WithOffsetC4Fast(int16_t* colAddr, const uint8_t* srcStart, size_t zeroPoint, size_t sizeQuad,
                                     size_t depthQuad, size_t dstZStep, size_t srcZStep) {
    dstZStep /= sizeof(int16_t);
    srcZStep /= sizeof(uint8_t);
    for (int sz = 0; sz < depthQuad; ++sz) {
        auto dstZ = colAddr + sz * dstZStep;
        auto srcZ = srcStart + sz * srcZStep;
        MNNUInt8ToInt16WithOffsetC4Common(dstZ, srcZ, zeroPoint, sizeQuad, 4 * sizeof(int16_t), 4 * sizeof(uint8_t));
    }
}

void MNNPowC8(float* dest, const float* source, const float* powfParam, size_t betaInt, size_t countC8) {
    const int count          = countC8 * 8;
    const float powfConstant = powfParam[6];
    for (int i = 0; i < count; ++i) {
        float result = 1, x, xInv = 1 / source[i];
        for (int j = 0; j < betaInt; result *= xInv, ++j)
            ;
        for (x = source[i]; x >= 1.25; x /= 1.5, result *= powfConstant)
            ;
        float t = x - 1;
        float powRemain =
            powfParam[0] +
            t * (powfParam[1] + t * (powfParam[2] + t * (powfParam[3] + t * (powfParam[4] + t * powfParam[5]))));
        result *= powRemain;
        dest[i] = result;
    }
}


#endif // no MNN_USE_NEON


void MNNPackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth) {
    int z, x;
    int cur = 0;
    memset(dst, 0, area * UP_DIV(depth, 4) * 4 * sizeof(uint8_t));
    for (z = 0; z < depth; ++z) {
        int plane         = z / 4;
        uint8_t* dstPlane = plane * area * 4 + dst;
        int offset        = z % 4;
        for (x = 0; x < area; ++x) {
            dstPlane[4 * x + offset] = src[cur++];
        }
    }
}

void MNNUnpackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth) {
    int x;
    int z;
    int cur = 0;
    for (z = 0; z < depth; ++z) {
        int plane               = z / 4;
        const uint8_t* srcPlane = plane * area * 4 + src;
        int offset              = z % 4;
        for (x = 0; x < area; ++x) {
            dst[cur++] = srcPlane[4 * x + offset];
        }
    }
}

void MNNUnpackTransposeUint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth) {
    if (depth == 4) {
        ::memcpy(dst, src, area * depth * sizeof(uint8_t));
        return;
    }
#ifdef MNN_USE_NEON
    if (depth == 3) {
        uint8x16x4_t rgba;
        rgba.val[3] = vdupq_n_u8(0);
        int sta     = 0;
        int staC16  = (int)area / 16;
        for (int i = 0; i < staC16; sta += 16, ++i) {
            auto rgb    = vld3q_u8(src + sta * 3);
            rgba.val[0] = rgb.val[0];
            rgba.val[1] = rgb.val[1];
            rgba.val[2] = rgb.val[2];
            vst4q_u8(dst + 4 * sta, rgba);
        }
        sta = staC16 * 16;

        for (; sta < area; ++sta) {
            auto s = src + sta * 3;
            auto d = dst + sta * 4;
            d[0]   = s[0];
            d[1]   = s[1];
            d[2]   = s[2];
            d[3]   = 0;
        }

        return;
    }
    if (depth == 1) {
        uint8x16x4_t rgba;
        rgba.val[1] = vdupq_n_u8(0);
        rgba.val[2] = vdupq_n_u8(0);
        rgba.val[3] = vdupq_n_u8(0);
        int sta     = 0;
        for (; sta < area; sta += 16) {
            rgba.val[0] = vld1q_u8(src + sta);
            vst4q_u8(dst + 4 * sta, rgba);
        }

        for (; sta < area; ++sta) {
            auto s = src + sta;
            auto d = dst + sta * 4;
            d[0]   = s[0];
            d[1]   = 0;
            d[2]   = 0;
            d[3]   = 0;
        }

        return;
    }
#endif
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = (src + hi * c);
        auto dstHeight = (dst + hi * 4);
        for (int ci = 0; ci < cDiv4; ++ci) {
            for (int i = 0; i < 4; ++i) {
                dstHeight[ci * area * 4 + i] = srcHeight[4 * ci + i];
            }
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + cAlign;
    auto dstAlign = dst + area * cAlign;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = srcAlign + hi * c;
        auto dstHeight = dstAlign + hi * 4;
        for (int i = 0; i < 4; ++i) {
            dstHeight[i] = 0;
        }
        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNUnpackTranspose(float* dst, const float* src, size_t area, size_t depth) {
#ifdef MNN_USE_NEON
    if (1 == depth) {
        auto zeroValue = vmovq_n_f32(0.0f);
        int areaC4     = (int)area / 4;
        int remain     = areaC4 * 4;
        for (int i = 0; i < areaC4; ++i) {
            auto srcCur   = src + 4 * i;
            auto dstCur   = dst + 16 * i;
            auto srcValue = vld1q_f32(srcCur);
            float32x4x4_t dstValue;
            dstValue.val[0] = srcValue;
            dstValue.val[1] = zeroValue;
            dstValue.val[2] = zeroValue;
            dstValue.val[3] = zeroValue;
            vst4q_f32(dstCur, dstValue);
        }
        for (int i = remain; i < area; ++i) {
            dst[4 * i + 0] = src[i];
            dst[4 * i + 1] = 0.0f;
            dst[4 * i + 2] = 0.0f;
            dst[4 * i + 3] = 0.0f;
        }
        return;
    }
    if (3 == depth) {
        auto zeroValue = vmovq_n_f32(0.0f);
        int areaC4     = (int)area / 4;
        int remain     = areaC4 * 4;
        for (int i = 0; i < areaC4; ++i) {
            auto srcCur   = src + 12 * i;
            auto dstCur   = dst + 16 * i;
            auto srcValue = vld3q_f32(srcCur);
            float32x4x4_t dstValue;
            dstValue.val[0] = srcValue.val[0];
            dstValue.val[1] = srcValue.val[1];
            dstValue.val[2] = srcValue.val[2];
            dstValue.val[3] = zeroValue;
            vst4q_f32(dstCur, dstValue);
        }
        for (int i = remain; i < area; ++i) {
            dst[4 * i + 0] = src[3 * i + 0];
            dst[4 * i + 1] = src[3 * i + 1];
            dst[4 * i + 2] = src[3 * i + 2];
            dst[4 * i + 3] = 0.0f;
        }
        return;
    }
#endif
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = src + hi * c;
        float* dstHeight       = dst + hi * 4;
        for (int ci = 0; ci < cDiv4; ++ci) {
            Vec4::save(dstHeight + 4 * ci * area, Vec4::load(srcHeight + 4 * ci));
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + cAlign;
    auto dstAlign = dst + area * cAlign;

#ifdef MNN_USE_NEON
    auto zeroVector = vdupq_n_f32(0.0f);
#endif

    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = srcAlign + hi * c;
        float* dstHeight       = dstAlign + hi * 4;
#ifdef MNN_USE_NEON
        vst1q_f32(dstHeight, zeroVector);
#else
        for (int i = 0; i < 4; ++i) {
            dstHeight[i] = 0;
        }
#endif
        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNPackTransposeUint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth) {
    if (1 == area) {
        ::memcpy(dst, src, depth * sizeof(uint8_t));
        return;
    }
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    if (cAlign == c) {
        int32_t* dst32       = (int32_t*)dst;
        const int32_t* src32 = (int32_t*)src;
        for (int hi = 0; hi < area; ++hi) {
            auto srcHeight = src32 + hi;
            auto dstHeight = dst32 + hi * cDiv4;
            for (int ci = 0; ci < cDiv4; ++ci) {
                dstHeight[ci] = srcHeight[ci * area];
            }
        }
        return;
    }

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = src + hi * 4;
        auto dstHeight = dst + hi * c;
        for (int ci = 0; ci < cDiv4; ++ci) {
            for (int i = 0; i < 4; ++i) {
                dstHeight[ci * 4 + i] = srcHeight[4 * ci * area + i];
            }
        }
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + area * cAlign;
    auto dstAlign = dst + cAlign;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = srcAlign + hi * 4;
        auto dstHeight = dstAlign + hi * c;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNPackTranspose(float* dst, const float* src, size_t area, size_t depth) {
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = src + hi * 4;
        float* dstHeight       = dst + hi * c;
        for (int ci = 0; ci < cDiv4; ++ci) {
            Vec4::save(dstHeight + 4 * ci, Vec4::load(srcHeight + 4 * ci * area));
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + area * cAlign;
    auto dstAlign = dst + cAlign;

    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = srcAlign + hi * 4;
        float* dstHeight       = dstAlign + hi * c;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNRelu6(float* dst, const float* src, size_t size) {
    int i;
    for (i = 0; i < size; ++i) {
        if (src[i] < 0) {
            dst[i] = 0;
        } else {
            dst[i] = src[i] < 6 ? src[i] : 6;
        }
    }
}

void MNNExp(float* dst, const float* src, size_t dataSize) {
    int countC8        = (int)dataSize / 8;
    if (countC8 > 0) {
        // Align to eight so asm is easier to write
        static float parameters[] = {
            (float)log(2.0f), 1.0f / (float)log(2.0f), 1.0f, 1.0f, 0.5f, 1.0f / 6.0f, 1.0f / 24.0f, 1.0f / 120.0f};
        MNNExpC8(dst, src, parameters, countC8);
    }
    int remain = countC8 * 8;
    auto param = log(2.0f);
    float xLimit = 87;
    for (int i = remain; i < dataSize; i++) {
        /*Origin Function*/
        //dst[i] = expf(-src[i]);
        /*Approciate Function*/

        auto x         = -src[i];
        x = ALIMAX(x, -xLimit);
        x = ALIMIN(x, xLimit);

        int div        = (x / param);
        int div2       = (div + 127) << 23;
        auto xReamin   = x - div * param;
        float expBasic = *(float*)(&div2);

        auto t         = xReamin;
        auto expRemain = ((((1.0f / 120 * t + 1.0f / 24) * t + 1.0f / 6) * t + 0.5f) * t + 1.0f) * t + 1.0f;
        dst[i]  = expBasic * expRemain;
    }
}

// Lambert's series with 7 divisions
// reference from
// https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
inline float tanhf_poly(float value) {
    if (value > 5.0) {
        return 1.0;
    } else if (value <= -5.0) {
        return -1.0;
    } else {
        float x2 = value * value;
        float a  = value * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
        float b  = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
        return a / b;
    }
}
void MNNTanh(float* dst, const float* src, size_t dataSize) {
    /* Origin Code
    for (int i = 0; i < dataSize; i++) {
        // outputData[i] = 1 - 2 / (expf(2 * inputData[i]) + 1);
        dst[i] = tanhf_poly(src[i]);
    }
     */
    for (int i = 0; i < dataSize; ++i) {
        dst[i] = src[i] + src[i];
    }
    MNNExp(dst, dst, dataSize);
    for (int i = 0; i < dataSize; i++) {
        // outputData[i] = 1 - 2 / (expf(2 * inputData[i]) + 1);
        auto expX2 = dst[i];
        dst[i] = (1.0f - expX2) / (1.0f + expX2);
    }
}

void MNNReluWithSlope(float* dst, const float* src, size_t sizeQuad, float slope) {
    float slopeValue[4];
    for (int i=0; i<4; ++i) {
        slopeValue[i] = slope;
    }
    MNNReluWithSlopeChannel(dst, src, slopeValue, sizeQuad, 1);
}

void MNNReluWithSlopeCommon(float* dst, const float* src, size_t size, float slope) {
    int sizeQuad = size / 4;
    int start = 0;
    if (sizeQuad > 0) {
        MNNReluWithSlope(dst, src, sizeQuad, slope);
        start = sizeQuad * 4;
    }
    for (int j = start; j < size; j++) {
        if (src[j] < 0) {
            dst[j] = src[j] * slope;
        } else {
            dst[j] = src[j];
        }
    }
}

void MNNHardSwishCommon(float* dst, const float* src, size_t size) {
    int sizeQuad = size / 4;
    int start = 0;
#ifdef MNN_USE_SSE
    if (sizeQuad > 0) {
        MNNHardSwish(dst, src, sizeQuad);
        start = sizeQuad * 4;
    }
#endif
#ifdef MNN_USE_NEON
    float32x4_t zero = vdupq_n_f32(0.f);
    float32x4_t three = vdupq_n_f32(3.f);
    float32x4_t six = vdupq_n_f32(6.f);
    float32x4_t divsix = vdupq_n_f32(1.0f/6.f);
    for (int i = 0; i < sizeQuad; i++) {
        auto x = vld1q_f32(src + 4 * i);
        auto y = vmulq_f32(vmulq_f32(x, vminq_f32(vmaxq_f32(vaddq_f32(x, three), zero), six)), divsix);
        vst1q_f32(dst + 4 * i, y);
    }
    start = sizeQuad * 4;
#endif
    for (int j = start; j < size; j++) {
        if (src[j] <= -3) {
            dst[j] = 0;
        } else if (src[j] >= 3){
            dst[j] = src[j];
        } else {
            dst[j] = src[j] * (src[j] + 3) / 6.f;
        }
    }
}

void MNNScaleAndAddBiasScalar(float* dst, const float* src, float bias, float alpha, size_t number) {
    int numberC4 = (int)number / 4;
    int start = 0;
    if (numberC4 > 0) {
        float biasC4[4] = {
            bias,
            bias,
            bias,
            bias
        };
        float alphaC4[4] = {
            alpha,
            alpha,
            alpha,
            alpha
        };
        MNNScaleAndAddBias(dst, src, biasC4, alphaC4, numberC4, 1);
        start = numberC4 * 4;
    }
    for (int i=start; i<number; ++i) {
        dst[i] = src[i] * alpha + bias;
    }
}
void MNNAxByClamp(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t bStride, size_t height, const float* parameters) {
    int widthC4 = (int)width / 4;
    if (widthC4 > 0) {
        auto minF = Vec4(parameters[2]);
        auto maxF = Vec4(parameters[3]);
        auto alpha = Vec4(parameters[0]);
        auto beta = Vec4(parameters[1]);
        for (int y = 0; y < height; ++y) {
            auto a = A + aStride * y;
            auto b = B + bStride * y;
            auto c = C + cStride * y;
            for (int x = 0; x < width; ++x) {
                auto av = Vec4::load(a + 4 * x);
                auto bv = Vec4::load(b + 4 * x);
                auto cv = av * alpha + bv * beta;
                cv = Vec4::min(cv, maxF);
                cv = Vec4::max(cv, minF);
                Vec4::save(c + 4 * x, cv);
            }
        }
        width = width - 4*widthC4;
        C = C + widthC4 * 4;
        A = A + widthC4 * 4;
        B = B + widthC4 * 4;
    }
    if (width > 0) {
        auto minF = parameters[2];
        auto maxF = parameters[3];
        auto alpha = parameters[0];
        auto beta = parameters[1];
        for (int y = 0; y < height; ++y) {
            auto a = A + aStride * y;
            auto b = B + bStride * y;
            auto c = C + cStride * y;
            for (int x = 0; x < width; ++x) {
                auto av = a[x];
                auto bv = b[x];
                auto cv = av * alpha + bv * beta;
                cv = std::min(cv, maxF);
                cv = std::max(cv, minF);
                c[x] = cv;
            }
        }
    }
}
#ifndef MNN_USE_NEON
void MNNAxByClampBroadcastUnit(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters) {
    auto minF = Vec4(parameters[2]);
    auto maxF = Vec4(parameters[3]);
    auto beta = Vec4(parameters[1]);
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + 4 * y;
        auto bv = Vec4::load(b);
        auto c = C + cStride * y;
        for (int x = 0; x < width; ++x) {
            auto av = Vec4::load(a + 4 * x);
            auto cv = av + bv * beta;
            cv = Vec4::min(cv, maxF);
            cv = Vec4::max(cv, minF);
            Vec4::save(c + 4 * x, cv);
        }
    }
}
void MNNVectorTop1Float(float* input, float* maxValue, int32_t* maxIndex, size_t inputCountUnit) {
    float maxV = input[0];
    int maxIdx = 0;
    for (int i = 0; i < inputCountUnit; i++) {
        int offset = i * UNIT;
        for (int j = 0; j < UNIT; j++) {
            if (input[offset + j] > maxV) {
                maxV = input[offset + j];
                maxIdx = offset + j;
            }
        }
    }
    maxValue[0] = maxV;
    maxIndex[0] = maxIdx;
}

void MNNVectorTop1Int32(int32_t* input, int32_t* maxValue, int32_t* maxIndex, size_t inputCountUnit) {
    int32_t maxV = input[0];
    int maxIdx = 0;
    for (int i = 0; i < inputCountUnit; i++) {
        int offset = i * UNIT;
        for (int j = 0; j < UNIT; j++) {
            if (input[offset + j] > maxV) {
                maxV = input[offset + j];
                maxIdx = offset + j;
            }
        }
    }
    maxValue[0] = maxV;
    maxIndex[0] = maxIdx;
}

#endif

#ifndef MNN_USE_SSE

void MNNComputeMatMulForE_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) {
    auto l = param->l;
    auto h = param->h;
    auto numberThread = param->numberThread;
    auto lC4 = l / 4;
    auto lR = lC4 * 4;
    if (param->BTranspose) {
        for (int y=tId; y<h; y+=numberThread) {
            Vec4 sumValue = Vec4(0.0f);
            auto by = B + y * l;
            for (int x=0; x<lC4; ++x) {
                sumValue = sumValue + Vec4::load(A + x * 4) * Vec4::load(by + x * 4);
            }
            float sumRemain = 0.0f;
            for (int x=lR; x<l; ++x) {
                sumRemain = sumRemain + A[x] * by[x];
            }
            if (nullptr != biasPtr) {
                sumRemain += biasPtr[y];
            }
            C[y] = sumRemain + sumValue[0] + sumValue[1] + sumValue[2] + sumValue[3];
        }
    } else {
        auto hC4 = h / 4;
        auto hR = hC4 * 4;
        for (int y=tId; y<hC4; y+=numberThread) {
            auto bs = B + 4 * y;
            Vec4 sumValue = Vec4(0.0f);
            if (biasPtr != nullptr) {
                sumValue = Vec4::load(biasPtr + 4 * y);
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                sumValue = sumValue + Vec4(A[x]) * Vec4::load(bs + h * x);
            }
            Vec4::save(C + 4 * y, sumValue);
        }
        for (int y=hR + tId; y<h; y+=numberThread) {
            auto bs = B + y;
            float sumValue = 0.0f;
            if (biasPtr != nullptr) {
                sumValue = biasPtr[y];
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                sumValue = sumValue + A[x] * bs[h * x];
            }
            C[y] = sumValue;
        }
    }
}
#endif

void MNNPackC4Int16(int16_t* dst, const int16_t* src, size_t area, size_t depth) {
    int z, x;
    int cur = 0;
    memset(dst, 0, area * UP_DIV(depth, 4) * 4 * sizeof(int16_t));
    for (z = 0; z < depth; ++z) {
        int plane       = z / 4;
        int16_t* dstPlane = plane * area * 4 + dst;
        int offset      = z % 4;
        for (x = 0; x < area; ++x) {
            dstPlane[4 * x + offset] = src[cur++];
        }
    }
}

void MNNUnpackC4Int16(int16_t* dst, const int16_t* src, size_t area, size_t depth) {
    int x;
    int z;
    int cur = 0;
    for (z = 0; z < depth; ++z) {
        int plane             = z / 4;
        const int16_t* srcPlane = plane * area * 4 + src;
        int offset            = z % 4;
        for (x = 0; x < area; ++x) {
            dst[cur++] = srcPlane[4 * x + offset];
        }
    }
}

void MNNUnpackTransposeInt16(int16_t* dst, const int16_t* src, size_t area, size_t depth) {
    if (depth == 4) {
        ::memcpy(dst, src, area * depth * sizeof(int16_t));
        return;
    }
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = (src + hi * c);
        auto dstHeight = (dst + hi * 4);
        for (int ci = 0; ci < cDiv4; ++ci) {
            for (int i = 0; i < 4; ++i) {
                dstHeight[ci * area * 4 + i] = srcHeight[4 * ci + i];
            }
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + cAlign;
    auto dstAlign = dst + area * cAlign;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = srcAlign + hi * c;
        auto dstHeight = dstAlign + hi * 4;
        for (int i = 0; i < 4; ++i) {
            dstHeight[i] = 0;
        }
        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}
void MNNPackTransposeInt16(int16_t* dst, const int16_t* src, size_t area, size_t depth) {
    if (1 == area) {
        ::memcpy(dst, src, depth * sizeof(int16_t));
        return;
    }
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    if (cAlign == c) {
        int64_t* dst32       = (int64_t*)dst;
        const int64_t* src32 = (int64_t*)src;
        for (int hi = 0; hi < area; ++hi) {
            auto srcHeight = src32 + hi;
            auto dstHeight = dst32 + hi * cDiv4;
            for (int ci = 0; ci < cDiv4; ++ci) {
                dstHeight[ci] = srcHeight[ci * area];
            }
        }
        return;
    }

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = src + hi * 4;
        auto dstHeight = dst + hi * c;
        for (int ci = 0; ci < cDiv4; ++ci) {
            for (int i = 0; i < 4; ++i) {
                dstHeight[ci * 4 + i] = srcHeight[4 * ci * area + i];
            }
        }
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + area * cAlign;
    auto dstAlign = dst + cAlign;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = srcAlign + hi * 4;
        auto dstHeight = dstAlign + hi * c;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNCopyC4Int16WithStride(const float* sourceF, float* destF, size_t srcStride, size_t dstStride, size_t count) {
    auto source = (int16_t*)sourceF;
    auto dest = (int16_t*)destF;
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        *(int64_t*)(d) = *((int64_t*)s);
    }
}


void MNNSin(float* dst, const float* src, size_t dataSize) {
    for (int i = 0; i < dataSize; i++) {
        dst[i] = sinf(src[i]);
    }
}

void MNNSigmoid(float* dst, const float* src, size_t dataSize) {
    MNNExp(dst, src, dataSize);
    for (int i = 0; i < dataSize; ++i) {
        dst[i] = 1.0f / (1.0f + dst[i]);
    }
}

/**
 Modified from https://github.com/alibaba/MNN/pull/1359
 Thanks for https://github.com/hroken
 */
void MNNSigmoidLowp(float* dst, const float* src, size_t dataSize) {
    MNNExp(dst, src, dataSize);
#ifdef MNN_USE_NEON
    int dataC4 = (int)dataSize / 4;
    if(dataC4 > 0) {
        // neon optimization for sigmid cpu
        float32x4_t value = vdupq_n_f32(1.0f);
        float32x4_t out = vld1q_f32(dst);
        for (int i = 1; i < dataC4; ++i) {
            out = vrecpeq_f32(vaddq_f32(value,out));
            vst1q_f32(dst ,out);
            dst += 4;
            out = vld1q_f32(dst);
        }
        out = vrecpeq_f32(vaddq_f32(value,out));
        vst1q_f32(dst, out);
        dataSize = dataSize - 4 * dataC4;
    }
#endif
    for (int i = 0; i < dataSize; ++i) {
        dst[i] = 1.0f / (1.0f + dst[i]);
    }
}
extern "C" {
void MNNConvDwF23MulTransUnit(float **cacheLine, const float *weigth, float *dest, size_t ow);
}

void MNNMultiAndDestTransformCommon23(float **cacheLine, const float *weigth, float *dest, int cacheLineSize, int ow) {
    int unit = ow / 2;
    MNN_ASSERT(cacheLineSize >= 1);
    for (int x = 0; x < unit; ++x) {
        auto offset = 4 * 4 * x;
        int i = 0;
        Vec4 m0     = Vec4::load(weigth + i * 16 + 4 * 0) * Vec4::load(cacheLine[i] + offset + 4 * 0);
        Vec4 m1     = Vec4::load(weigth + i * 16 + 4 * 1) * Vec4::load(cacheLine[i] + offset + 4 * 1);
        Vec4 m2     = Vec4::load(weigth + i * 16 + 4 * 2) * Vec4::load(cacheLine[i] + offset + 4 * 2);
        Vec4 m3     = Vec4::load(weigth + i * 16 + 4 * 3) * Vec4::load(cacheLine[i] + offset + 4 * 3);

        for (i = 1; i < cacheLineSize; ++i) {
            m0 = m0 + Vec4::load(weigth + i * 16 + 4 * 0) * Vec4::load(cacheLine[i] + offset + 4 * 0);
            m1 = m1 + Vec4::load(weigth + i * 16 + 4 * 1) * Vec4::load(cacheLine[i] + offset + 4 * 1);
            m2 = m2 + Vec4::load(weigth + i * 16 + 4 * 2) * Vec4::load(cacheLine[i] + offset + 4 * 2);
            m3 = m3 + Vec4::load(weigth + i * 16 + 4 * 3) * Vec4::load(cacheLine[i] + offset + 4 * 3);
        }

        auto o0 = m0 + m1 + m2;
        auto o1 = m1 - m2 + m3;
        Vec4::save(dest + 8 * x + 0 * 4, o0);
        Vec4::save(dest + 8 * x + 1 * 4, o1);
    }
    if (unit * 2 < ow) {
        auto offset = 4 * 4 * unit;
        int i = 0;
        Vec4 m0     = Vec4::load(weigth + i * 16 + 4 * 0) * Vec4::load(cacheLine[i] + offset + 4 * 0);
        Vec4 m1     = Vec4::load(weigth + i * 16 + 4 * 1) * Vec4::load(cacheLine[i] + offset + 4 * 1);
        Vec4 m2     = Vec4::load(weigth + i * 16 + 4 * 2) * Vec4::load(cacheLine[i] + offset + 4 * 2);

        for (i = 1; i < cacheLineSize; ++i) {
            m0 = m0 + Vec4::load(weigth + i * 16 + 4 * 0) * Vec4::load(cacheLine[i] + offset + 4 * 0);
            m1 = m1 + Vec4::load(weigth + i * 16 + 4 * 1) * Vec4::load(cacheLine[i] + offset + 4 * 1);
            m2 = m2 + Vec4::load(weigth + i * 16 + 4 * 2) * Vec4::load(cacheLine[i] + offset + 4 * 2);
        }

        auto o0 = m0 + m1 + m2;
        Vec4::save(dest + 8 * unit + 0 * 4, o0);
    }
}
extern "C" {
void MNNConvDwF23SourceTransUnit(const float *source, float *dest, size_t unit);
}

void MNNSourceTransformCommonF23(const float *source, float *dest, int unit, int iw, int pad, int su, int eu) {
    for (int x = 0; x < su; ++x) {
        auto dstX = dest + 4 * 4 * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;

        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);

        Vec4 v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec4::load(source + 4 * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec4::save(dstX + 4 * 0, m0);
        Vec4::save(dstX + 4 * 1, m1);
        Vec4::save(dstX + 4 * 2, m2);
        Vec4::save(dstX + 4 * 3, m3);
    }
    MNNConvDwF23SourceTransUnit(source + 4 * (su * 2 - pad), dest + 4 * 4 * su, eu - su);

    for (int x = eu; x < unit; ++x) {
        auto dstX = dest + 4 * 4 * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;

        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);

        Vec4 v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec4::load(source + 4 * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec4::save(dstX + 4 * 0, m0);
        Vec4::save(dstX + 4 * 1, m1);
        Vec4::save(dstX + 4 * 2, m2);
        Vec4::save(dstX + 4 * 3, m3);
    }
}

#ifndef MNN_USE_NEON
void MNNConvDwF23MulTransUnit(float **cacheLine, const float *weigth, float *dest, size_t ow) {
    int unit = ow / 2;
    auto w00 = Vec4::load(weigth + 0 * 16 + 4 * 0);
    auto w01 = Vec4::load(weigth + 0 * 16 + 4 * 1);
    auto w02 = Vec4::load(weigth + 0 * 16 + 4 * 2);
    auto w03 = Vec4::load(weigth + 0 * 16 + 4 * 3);
    auto w10 = Vec4::load(weigth + 1 * 16 + 4 * 0);
    auto w11 = Vec4::load(weigth + 1 * 16 + 4 * 1);
    auto w12 = Vec4::load(weigth + 1 * 16 + 4 * 2);
    auto w13 = Vec4::load(weigth + 1 * 16 + 4 * 3);
    auto w20 = Vec4::load(weigth + 2 * 16 + 4 * 0);
    auto w21 = Vec4::load(weigth + 2 * 16 + 4 * 1);
    auto w22 = Vec4::load(weigth + 2 * 16 + 4 * 2);
    auto w23 = Vec4::load(weigth + 2 * 16 + 4 * 3);
    for (int x = 0; x < unit; ++x) {
        auto offset = 4 * 4 * x;
        int i = 0;
        Vec4 m0     = w00 * Vec4::load(cacheLine[0] + offset + 4 * 0);
        Vec4 m1     = w01 * Vec4::load(cacheLine[0] + offset + 4 * 1);
        Vec4 m2     = w02 * Vec4::load(cacheLine[0] + offset + 4 * 2);
        Vec4 m3     = w03 * Vec4::load(cacheLine[0] + offset + 4 * 3);

        m0 = m0 + w10 * Vec4::load(cacheLine[1] + offset + 4 * 0);
        m1 = m1 + w11 * Vec4::load(cacheLine[1] + offset + 4 * 1);
        m2 = m2 + w12 * Vec4::load(cacheLine[1] + offset + 4 * 2);
        m3 = m3 + w13 * Vec4::load(cacheLine[1] + offset + 4 * 3);

        m0 = m0 + w20 * Vec4::load(cacheLine[2] + offset + 4 * 0);
        m1 = m1 + w21 * Vec4::load(cacheLine[2] + offset + 4 * 1);
        m2 = m2 + w22 * Vec4::load(cacheLine[2] + offset + 4 * 2);
        m3 = m3 + w23 * Vec4::load(cacheLine[2] + offset + 4 * 3);

        auto o0 = m0 + m1 + m2;
        auto o1 = m1 - m2 + m3;
        Vec4::save(dest + 8 * x + 0 * 4, o0);
        Vec4::save(dest + 8 * x + 1 * 4, o1);
    }
    if (unit * 2 < ow) {
        auto offset = 4 * 4 * unit;
        Vec4 m0     = w00 * Vec4::load(cacheLine[0] + offset + 4 * 0);
        Vec4 m1     = w01 * Vec4::load(cacheLine[0] + offset + 4 * 1);
        Vec4 m2     = w02 * Vec4::load(cacheLine[0] + offset + 4 * 2);

        m0 = m0 + w10 * Vec4::load(cacheLine[1] + offset + 4 * 0);
        m1 = m1 + w11 * Vec4::load(cacheLine[1] + offset + 4 * 1);
        m2 = m2 + w12 * Vec4::load(cacheLine[1] + offset + 4 * 2);

        m0 = m0 + w20 * Vec4::load(cacheLine[2] + offset + 4 * 0);
        m1 = m1 + w21 * Vec4::load(cacheLine[2] + offset + 4 * 1);
        m2 = m2 + w22 * Vec4::load(cacheLine[2] + offset + 4 * 2);
        auto o0 = m0 + m1 + m2;
        Vec4::save(dest + 8 * unit + 0 * 4, o0);
    }
}
void MNNConvDwF23SourceTransUnit(const float *source, float *dest, size_t unit) {
    if (unit <= 0) {
        return;
    }
    Vec4 v0 = Vec4::load(source + 4 * 0);
    Vec4 v1 = Vec4::load(source + 4 * 1);
    Vec4 v2;
    Vec4 v3;
    source += 8;

    for (int x = 0; x < unit; ++x) {
        v2 = Vec4::load(source + 0 * 4);
        v3 = Vec4::load(source + 1 * 4);
        auto m0 = v0 - v2;
        auto m1 = v1 + v2;
        auto m2 = v2 - v1;
        auto m3 = v3 - v1;

        Vec4::save(dest + 4 * 0, m0);
        Vec4::save(dest + 4 * 1, m1);
        Vec4::save(dest + 4 * 2, m2);
        Vec4::save(dest + 4 * 3, m3);

        source += 8;
        dest += 16;

        v0 = v2;
        v1 = v3;
    }
}
#endif

namespace MNN {

static CoreFunctions* gCoreFunction = nullptr;

void MNNCoreFunctionInit() {
    gCoreFunction = new CoreFunctions;
    // MatMul
    gCoreFunction->MNNGetMatMulPackMode = MNNGetMatMulPackMode;
    gCoreFunction->MNNPackC4ForMatMul_A = MNNPackC4ForMatMul_A;
    gCoreFunction->MNNPackForMatMul_B = MNNPackForMatMul_B;
    gCoreFunction->MNNPackedMatMul = MNNPackedMatMul;
    gCoreFunction->MNNPackedMatMulRemain = MNNPackedMatMulRemain;

    // Lowp
    gCoreFunction->MNNFp32ToLowp = nullptr;
    gCoreFunction->MNNLowpToFp32 = nullptr;
    gCoreFunction->bytes = 4;// sizeof(float)

    // Packed Function
    gCoreFunction->pack = 4;
    gCoreFunction->MNNPackCUnit = MNNPackC4;
    gCoreFunction->MNNUnpackCUnit = MNNUnpackC4;
    
    // FIXME: MNNPackTranspose and MNNUnpackTranspose is reverted
    gCoreFunction->MNNUnpackCUnitTranspose = MNNPackTranspose;
    gCoreFunction->MNNPackCUnitTranspose = MNNUnpackTranspose;
    gCoreFunction->MNNAxByClampBroadcastUnit = MNNAxByClampBroadcastUnit;
    gCoreFunction->MNNConvRunForLineDepthwise = MNNConvRunForLineDepthwise;
    gCoreFunction->MNNConvRunForUnitDepthWise = MNNConvRunForUnitDepthWise;
    gCoreFunction->MNNSourceTransformCommonF23 = MNNSourceTransformCommonF23;
    gCoreFunction->MNNConvDwF23MulTransUnit = MNNConvDwF23MulTransUnit;
    gCoreFunction->MNNMultiAndDestTransformCommon23 = MNNMultiAndDestTransformCommon23;
    gCoreFunction->MNNMatrixAdd = MNNMatrixAdd;
    gCoreFunction->MNNMatrixSub = MNNMatrixSub;
    gCoreFunction->MNNStrassenMergeCFunction = MNNStrassenMergeCFunction;
    gCoreFunction->penalty = 1.5f;
    gCoreFunction->MNNScaleAndAddBias = MNNScaleAndAddBias;
    gCoreFunction->MNNAddC4WithStride = MNNAddC4WithStride;
    gCoreFunction->MNNCopyC4WithStride = MNNCopyC4WithStride;
    
    gCoreFunction->chooseWinoSourceTransform = WinogradFunction::chooseSourceTransform;
    gCoreFunction->chooseWinoDestTransform = WinogradFunction::chooseDestTransform;
    gCoreFunction->MNNDeconvRunForLineDepthwise = MNNDeconvRunForLineDepthwise;
    gCoreFunction->MNNDeconvRunForUnitDepthWise = MNNDeconvRunForUnitDepthWise;
    MNNFunctionInit();
}
CoreFunctions* MNNGetCoreFunctions() {
    return gCoreFunction;
}
};
