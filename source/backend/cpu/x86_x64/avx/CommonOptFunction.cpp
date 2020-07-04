//
//  CommonOptFunction.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include <limits>
#include <string.h>
#include <algorithm>
#include <vector>
void _AVX_MNNAddBias(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    if (planeNumber == 0) {
        return;
    }
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm256_broadcast_ps((const __m128 *)(bias + 4 * z));
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber - 1; p += 2) {
            auto dstV = _mm256_add_ps(_mm256_loadu_ps(dst_z + 4 * p), biasV);
            _mm256_storeu_ps(dst_z + 4 * p, dstV);
        }
        if (planeNumber % 2 == 1) {
            _mm256_zeroall();
            auto biasV = _mm_loadu_ps(bias + 4 * z);
            auto dstV = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * (planeNumber - 1)), biasV);
            _mm_storeu_ps(dst_z + 4 * (planeNumber - 1), dstV);
        }
    }
    _mm256_zeroall();
}

void _AVX_MNNAddBiasRelu(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    if (planeNumber == 0) {
        return;
    }
    auto maxV = _mm256_set1_ps(0.0f);
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm256_broadcast_ps((const __m128 *)(bias + 4 * z));
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber - 1; p += 2) {
            auto dstV = _mm256_add_ps(_mm256_loadu_ps(dst_z + 4 * p), biasV);
            dstV      = _mm256_max_ps(dstV, maxV);
            _mm256_storeu_ps(dst_z + 4 * p, dstV);
        }
        if (planeNumber % 2 == 1) {
            _mm256_zeroall();
            auto biasV = _mm_loadu_ps(bias + 4 * z);
            auto dstV  = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * (planeNumber - 1)), biasV);
            dstV       = _mm_max_ps(dstV, _mm_set1_ps(0.0f));
            _mm_storeu_ps(dst_z + 4 * (planeNumber - 1), dstV);
            maxV = _mm256_set1_ps(0.0f);
        }
    }
    _mm256_zeroall();
}

void _AVX_MNNAddBiasRelu6(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    if (planeNumber == 0) {
        return;
    }
    auto maxV = _mm256_set1_ps(0.0f);
    auto minV = _mm256_set1_ps(6.0f);
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm256_broadcast_ps((const __m128 *)(bias + 4 * z));
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber - 1; p += 2) {
            auto dstV = _mm256_add_ps(_mm256_loadu_ps(dst_z + 4 * p), biasV);
            dstV      = _mm256_max_ps(dstV, maxV);
            dstV      = _mm256_min_ps(dstV, minV);
            _mm256_storeu_ps(dst_z + 4 * p, dstV);
        }
        if (planeNumber % 2 == 1) {
            _mm256_zeroall();
            auto biasV = _mm_loadu_ps(bias + 4 * z);
            auto dstV  = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * (planeNumber - 1)), biasV);
            dstV       = _mm_min_ps(_mm_max_ps(dstV, _mm_set1_ps(0.0f)), _mm_set1_ps(6.0f));
            _mm_storeu_ps(dst_z + 4 * (planeNumber - 1), dstV);
            maxV = _mm256_set1_ps(0.0f);
            minV = _mm256_set1_ps(6.0f);
        }
    }
    _mm256_zeroall();
}


static void _computeUnit(float* dst, const float* weight, const float* A, int l) {
    auto z0 = _mm256_set1_ps(0.0f);
    auto z1 = _mm256_set1_ps(0.0f);
    auto z2 = _mm256_set1_ps(0.0f);
    auto z3 = _mm256_set1_ps(0.0f);
    auto z4 = _mm256_set1_ps(0.0f);
    auto z5 = _mm256_set1_ps(0.0f);
    auto z6 = _mm256_set1_ps(0.0f);
    auto z7 = _mm256_set1_ps(0.0f);
    auto z8 = _mm256_set1_ps(0.0f);
    auto z9 = _mm256_set1_ps(0.0f);
    auto z10 = _mm256_set1_ps(0.0f);
    auto z11 = _mm256_set1_ps(0.0f);
    
    for (int sy=0; sy<l; ++sy) {
        auto s0 = _mm256_loadu_ps(A + sy * 16);
        auto s1 = _mm256_loadu_ps(A + sy * 16 + 8);
        auto w0 = _mm256_broadcast_ss(weight + sy * 6 + 0);
        auto w1 = _mm256_broadcast_ss(weight + sy * 6 + 1);
        z0 = _mm256_add_ps(_mm256_mul_ps(s0, w0), z0);
        z1 = _mm256_add_ps(_mm256_mul_ps(s1, w0), z1);
        z2 = _mm256_add_ps(_mm256_mul_ps(s0, w1), z2);
        z3 = _mm256_add_ps(_mm256_mul_ps(s1, w1), z3);
        w0 = _mm256_broadcast_ss(weight + sy * 6 + 2);
        w1 = _mm256_broadcast_ss(weight + sy * 6 + 3);
        z4 = _mm256_add_ps(_mm256_mul_ps(s0, w0), z4);
        z5 = _mm256_add_ps(_mm256_mul_ps(s1, w0), z5);
        z6 = _mm256_add_ps(_mm256_mul_ps(s0, w1), z6);
        z7 = _mm256_add_ps(_mm256_mul_ps(s1, w1), z7);
        w0 = _mm256_broadcast_ss(weight + sy * 6 + 4);
        w1 = _mm256_broadcast_ss(weight + sy * 6 + 5);
        z8 = _mm256_add_ps(_mm256_mul_ps(s0, w0), z8);
        z9 = _mm256_add_ps(_mm256_mul_ps(s1, w0), z9);
        z10 = _mm256_add_ps(_mm256_mul_ps(s0, w1), z10);
        z11 = _mm256_add_ps(_mm256_mul_ps(s1, w1), z11);
    }
    _mm256_storeu_ps(dst + 8 * 0, z0);
    _mm256_storeu_ps(dst + 8 * 1, z1);
    _mm256_storeu_ps(dst + 8 * 2, z2);
    _mm256_storeu_ps(dst + 8 * 3, z3);
    _mm256_storeu_ps(dst + 8 * 4, z4);
    _mm256_storeu_ps(dst + 8 * 5, z5);
    _mm256_storeu_ps(dst + 8 * 6, z6);
    _mm256_storeu_ps(dst + 8 * 7, z7);
    _mm256_storeu_ps(dst + 8 * 8, z8);
    _mm256_storeu_ps(dst + 8 * 9, z9);
    _mm256_storeu_ps(dst + 8 * 10, z10);
    _mm256_storeu_ps(dst + 8 * 11, z11);
}

static void _computeUnit_4(float* dst, const float* weight, const float* A, int l) {
    auto z0 = _mm256_set1_ps(0.0f);
    auto z1 = _mm256_set1_ps(0.0f);
    auto z2 = _mm256_set1_ps(0.0f);
    auto z3 = _mm256_set1_ps(0.0f);
    auto z4 = _mm256_set1_ps(0.0f);
    auto z5 = _mm256_set1_ps(0.0f);
    auto z6 = _mm256_set1_ps(0.0f);
    auto z7 = _mm256_set1_ps(0.0f);
    
    for (int sy=0; sy<l; ++sy) {
        auto s0 = _mm256_loadu_ps(A + sy * 16);
        auto s1 = _mm256_loadu_ps(A + sy * 16 + 8);
        auto w0 = _mm256_broadcast_ss(weight + sy * 6 + 0);
        auto w1 = _mm256_broadcast_ss(weight + sy * 6 + 1);
        z0 = _mm256_add_ps(_mm256_mul_ps(s0, w0), z0);
        z1 = _mm256_add_ps(_mm256_mul_ps(s1, w0), z1);
        z2 = _mm256_add_ps(_mm256_mul_ps(s0, w1), z2);
        z3 = _mm256_add_ps(_mm256_mul_ps(s1, w1), z3);
        w0 = _mm256_broadcast_ss(weight + sy * 6 + 2);
        w1 = _mm256_broadcast_ss(weight + sy * 6 + 3);
        z4 = _mm256_add_ps(_mm256_mul_ps(s0, w0), z4);
        z5 = _mm256_add_ps(_mm256_mul_ps(s1, w0), z5);
        z6 = _mm256_add_ps(_mm256_mul_ps(s0, w1), z6);
        z7 = _mm256_add_ps(_mm256_mul_ps(s1, w1), z7);
    }
    _mm256_storeu_ps(dst + 8 * 0, z0);
    _mm256_storeu_ps(dst + 8 * 1, z1);
    _mm256_storeu_ps(dst + 8 * 2, z2);
    _mm256_storeu_ps(dst + 8 * 3, z3);
    _mm256_storeu_ps(dst + 8 * 4, z4);
    _mm256_storeu_ps(dst + 8 * 5, z5);
    _mm256_storeu_ps(dst + 8 * 6, z6);
    _mm256_storeu_ps(dst + 8 * 7, z7);
}
static void _computeUnit_2(float* dst, const float* weight, const float* A, int l) {
    auto z0 = _mm256_set1_ps(0.0f);
    auto z1 = _mm256_set1_ps(0.0f);
    auto z2 = _mm256_set1_ps(0.0f);
    auto z3 = _mm256_set1_ps(0.0f);
    
    for (int sy=0; sy<l; ++sy) {
        auto s0 = _mm256_loadu_ps(A + sy * 16);
        auto s1 = _mm256_loadu_ps(A + sy * 16 + 8);
        auto w0 = _mm256_broadcast_ss(weight + sy * 6 + 0);
        auto w1 = _mm256_broadcast_ss(weight + sy * 6 + 1);
        z0 = _mm256_add_ps(_mm256_mul_ps(s0, w0), z0);
        z1 = _mm256_add_ps(_mm256_mul_ps(s1, w0), z1);
        z2 = _mm256_add_ps(_mm256_mul_ps(s0, w1), z2);
        z3 = _mm256_add_ps(_mm256_mul_ps(s1, w1), z3);
    }
    _mm256_storeu_ps(dst + 8 * 0, z0);
    _mm256_storeu_ps(dst + 8 * 1, z1);
    _mm256_storeu_ps(dst + 8 * 2, z2);
    _mm256_storeu_ps(dst + 8 * 3, z3);
}

static void _computeUnit_1(float* dst, const float* weight, const float* A, int l) {
    auto z0 = _mm256_set1_ps(0.0f);
    auto z1 = _mm256_set1_ps(0.0f);
    auto z2 = _mm256_set1_ps(0.0f);
    auto z3 = _mm256_set1_ps(0.0f);
    
    for (int sy=0; sy<l; ++sy) {
        auto s0 = _mm256_loadu_ps(A + sy * 16);
        auto s1 = _mm256_loadu_ps(A + sy * 16 + 8);
        auto w0 = _mm256_broadcast_ss(weight + sy * 6 + 0);
        z0 = _mm256_add_ps(_mm256_mul_ps(s0, w0), z0);
        z1 = _mm256_add_ps(_mm256_mul_ps(s1, w0), z1);
    }
    _mm256_storeu_ps(dst + 8 * 0, z0);
    _mm256_storeu_ps(dst + 8 * 1, z1);
}
static void _save16x4(float* dst, const float* src, const float* postParameters, const float* biasPtr) {
    auto m00 = _mm_loadu_ps(src + 4 * 0);
    auto m01 = _mm_loadu_ps(src + 4 * 1);
    auto m02 = _mm_loadu_ps(src + 4 * 2);
    auto m03 = _mm_loadu_ps(src + 4 * 3);

    auto m10 = _mm_loadu_ps(src + 4 * 4);
    auto m11 = _mm_loadu_ps(src + 4 * 5);
    auto m12 = _mm_loadu_ps(src + 4 * 6);
    auto m13 = _mm_loadu_ps(src + 4 * 7);

    auto m20 = _mm_loadu_ps(src + 4 *  8);
    auto m21 = _mm_loadu_ps(src + 4 *  9);
    auto m22 = _mm_loadu_ps(src + 4 * 10);
    auto m23 = _mm_loadu_ps(src + 4 * 11);

    auto m30 = _mm_loadu_ps(src + 4 * 12);
    auto m31 = _mm_loadu_ps(src + 4 * 13);
    auto m32 = _mm_loadu_ps(src + 4 * 14);
    auto m33 = _mm_loadu_ps(src + 4 * 15);
    
    _MM_TRANSPOSE4_PS(m00, m10, m20, m30);
    _MM_TRANSPOSE4_PS(m01, m11, m21, m31);
    _MM_TRANSPOSE4_PS(m02, m12, m22, m32);
    _MM_TRANSPOSE4_PS(m03, m13, m23, m33);

    if (nullptr != postParameters) {
        auto minValue = _mm_set1_ps(postParameters[2]);
        auto maxValue = _mm_set1_ps(postParameters[3]);
        //auto beta = _mm_set1_ps(postParameters[1]); // TODO
        if (nullptr != biasPtr) {
            auto biasV = _mm_loadu_ps(biasPtr);
            m00 = _mm_add_ps(m00, biasV);
            m01 = _mm_add_ps(m01, biasV);
            m02 = _mm_add_ps(m02, biasV);
            m03 = _mm_add_ps(m03, biasV);

            m10 = _mm_add_ps(m10, biasV);
            m11 = _mm_add_ps(m11, biasV);
            m12 = _mm_add_ps(m12, biasV);
            m13 = _mm_add_ps(m13, biasV);

            m20 = _mm_add_ps(m20, biasV);
            m21 = _mm_add_ps(m21, biasV);
            m22 = _mm_add_ps(m22, biasV);
            m23 = _mm_add_ps(m23, biasV);

            m30 = _mm_add_ps(m30, biasV);
            m31 = _mm_add_ps(m31, biasV);
            m32 = _mm_add_ps(m32, biasV);
            m33 = _mm_add_ps(m33, biasV);
        }
        m00 = _mm_max_ps(m00, minValue);
        m01 = _mm_max_ps(m01, minValue);
        m02 = _mm_max_ps(m02, minValue);
        m03 = _mm_max_ps(m03, minValue);
        
        m10 = _mm_max_ps(m10, minValue);
        m11 = _mm_max_ps(m11, minValue);
        m12 = _mm_max_ps(m12, minValue);
        m13 = _mm_max_ps(m13, minValue);

        m20 = _mm_max_ps(m20, minValue);
        m21 = _mm_max_ps(m21, minValue);
        m22 = _mm_max_ps(m22, minValue);
        m23 = _mm_max_ps(m23, minValue);
        
        m30 = _mm_max_ps(m30, minValue);
        m31 = _mm_max_ps(m31, minValue);
        m32 = _mm_max_ps(m32, minValue);
        m33 = _mm_max_ps(m33, minValue);
        
        m00 = _mm_min_ps(m00, maxValue);
        m01 = _mm_min_ps(m01, maxValue);
        m02 = _mm_min_ps(m02, maxValue);
        m03 = _mm_min_ps(m03, maxValue);
        
        m10 = _mm_min_ps(m10, maxValue);
        m11 = _mm_min_ps(m11, maxValue);
        m12 = _mm_min_ps(m12, maxValue);
        m13 = _mm_min_ps(m13, maxValue);

        m20 = _mm_min_ps(m20, maxValue);
        m21 = _mm_min_ps(m21, maxValue);
        m22 = _mm_min_ps(m22, maxValue);
        m23 = _mm_min_ps(m23, maxValue);
        
        m30 = _mm_min_ps(m30, maxValue);
        m31 = _mm_min_ps(m31, maxValue);
        m32 = _mm_min_ps(m32, maxValue);
        m33 = _mm_min_ps(m33, maxValue);
    }
    _mm_store_ps(dst + 4 * 0, m00);
    _mm_store_ps(dst + 4 * 1, m10);
    _mm_store_ps(dst + 4 * 2, m20);
    _mm_store_ps(dst + 4 * 3, m30);
    
    _mm_store_ps(dst + 4 * 4, m01);
    _mm_store_ps(dst + 4 * 5, m11);
    _mm_store_ps(dst + 4 * 6, m21);
    _mm_store_ps(dst + 4 * 7, m31);

    _mm_store_ps(dst + 4 * 8, m02);
    _mm_store_ps(dst + 4 * 9, m12);
    _mm_store_ps(dst + 4 * 10, m22);
    _mm_store_ps(dst + 4 * 11, m32);

    _mm_store_ps(dst + 4 * 12, m03);
    _mm_store_ps(dst + 4 * 13, m13);
    _mm_store_ps(dst + 4 * 14, m23);
    _mm_store_ps(dst + 4 * 15, m33);
}

static void _save8x4(float* dst, const float* src, const float* postParameters, const float* biasPtr) {
    auto m00 = _mm_loadu_ps(src + 4 * 0);
    auto m01 = _mm_loadu_ps(src + 4 * 1);

    auto m10 = _mm_loadu_ps(src + 4 * 2);
    auto m11 = _mm_loadu_ps(src + 4 * 3);

    auto m20 = _mm_loadu_ps(src + 4 * 4);
    auto m21 = _mm_loadu_ps(src + 4 * 5);

    auto m30 = _mm_loadu_ps(src + 4 * 6);
    auto m31 = _mm_loadu_ps(src + 4 * 7);
    
    _MM_TRANSPOSE4_PS(m00, m10, m20, m30);
    _MM_TRANSPOSE4_PS(m01, m11, m21, m31);

    if (nullptr != postParameters) {
        auto minValue = _mm_set1_ps(postParameters[2]);
        auto maxValue = _mm_set1_ps(postParameters[3]);
        //auto beta = _mm_set1_ps(postParameters[1]); // TODO
        if (nullptr != biasPtr) {
            auto biasV = _mm_loadu_ps(biasPtr);
            m00 = _mm_add_ps(m00, biasV);
            m01 = _mm_add_ps(m01, biasV);

            m10 = _mm_add_ps(m10, biasV);
            m11 = _mm_add_ps(m11, biasV);

            m20 = _mm_add_ps(m20, biasV);
            m21 = _mm_add_ps(m21, biasV);

            m30 = _mm_add_ps(m30, biasV);
            m31 = _mm_add_ps(m31, biasV);
        }
        m00 = _mm_max_ps(m00, minValue);
        m01 = _mm_max_ps(m01, minValue);
        
        m10 = _mm_max_ps(m10, minValue);
        m11 = _mm_max_ps(m11, minValue);

        m20 = _mm_max_ps(m20, minValue);
        m21 = _mm_max_ps(m21, minValue);
        
        m30 = _mm_max_ps(m30, minValue);
        m31 = _mm_max_ps(m31, minValue);
        
        m00 = _mm_min_ps(m00, maxValue);
        m01 = _mm_min_ps(m01, maxValue);
        
        m10 = _mm_min_ps(m10, maxValue);
        m11 = _mm_min_ps(m11, maxValue);

        m20 = _mm_min_ps(m20, maxValue);
        m21 = _mm_min_ps(m21, maxValue);
        
        m30 = _mm_min_ps(m30, maxValue);
        m31 = _mm_min_ps(m31, maxValue);
    }
    _mm_store_ps(dst + 4 * 0, m00);
    _mm_store_ps(dst + 4 * 1, m10);
    _mm_store_ps(dst + 4 * 2, m20);
    _mm_store_ps(dst + 4 * 3, m30);
    
    _mm_store_ps(dst + 4 * 4, m01);
    _mm_store_ps(dst + 4 * 5, m11);
    _mm_store_ps(dst + 4 * 6, m21);
    _mm_store_ps(dst + 4 * 7, m31);
}

void _AVX_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, float* cache, const float* postParameters, const float* bias) {
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride = bExtraStride + l * 6;
    auto hC4 = UP_DIV(h, 4);
    int hDiv = hC4 / 3;
    for (int y=0; y<hDiv; ++y) {
        for (int p=0; p<2; ++p) {
            auto weight = B + (y * 2 + p) * bStride;
            auto dst = cache + p * 16 * 6;
            _computeUnit(dst, weight, A, l);
        }
        for (int p=0; p<3; ++p) {
            auto dst = C + (y * 3 + p) * cStride;
            auto src = cache + 16 * 4 * p;
            if (nullptr == bias) {
                _save16x4(dst, src, postParameters, bias);
            } else {
                _save16x4(dst, src, postParameters, bias + (y * 3 + p) * 4);
            }
        }
    }
    auto hRemain = hC4 % 3;
    auto lastHStart = hDiv*3;
    if (hRemain == 2) {
        auto realHRemain = (int)h - (int)hDiv * 12;
        // First
        {
             auto weight = B + (hDiv * 2 + 0) * bStride;
             auto dst = cache;
            _computeUnit(dst, weight, A, l);
            realHRemain -= 6;
        }
        // Second, max only two
        {
            auto weight = B + (hDiv * 2 + 1) * bStride;
            auto dst = cache + 1 * 16 * 6;
            ::memset(dst, 0, 16 * 6 * sizeof(float));
            if (realHRemain == 2) {
                _computeUnit_2(dst, weight, A, l);
            } else if (realHRemain == 1) {
                _computeUnit_1(dst, weight, A, l);
            }
        }
        for (int p=0; p<2; ++p) {
            auto dst = C + (hDiv * 3 + p) * cStride;
            auto src = cache + 16 * 4 * p;
            if (nullptr == bias) {
                _save16x4(dst, src, postParameters, bias);
            } else {
                _save16x4(dst, src, postParameters, bias + (hDiv * 3 + p) * 4);
            }
        }
    }
    if (hRemain == 1) {
        // First
        {
             auto weight = B + (hDiv * 2 + 0) * bStride;
             auto dst = cache;
            _computeUnit_4(dst, weight, A, l);
        }
        for (int p=0; p<1; ++p) {
            auto dst = C + (hDiv * 3 + p) * cStride;
            auto src = cache + 16 * 4 * p;
            if (nullptr == bias) {
                _save16x4(dst, src, postParameters, bias);
            } else {
                _save16x4(dst, src, postParameters, bias + (hDiv * 3 + p) * 4);
            }
        }
    }
}
void _AVX_MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, float* cache, const float* postParameters, const float* bias) {
    auto h = parameter[2];
    auto cStride = parameter[3] / sizeof(float);
    auto hC4 = UP_DIV(h, 4);
    auto CTemp = cache + 16 * 6 * 2;
    size_t tempParamters[] = {
        parameter[0],
        parameter[1],
        parameter[2],
        16 * 4 * sizeof(float),
        parameter[4],
        parameter[5],
    };
    _AVX_MNNPackedMatMul(CTemp, A, B, tempParamters, cache, postParameters, bias);
    for (int y=0; y<hC4; ++y) {
        ::memcpy(C + y * cStride, CTemp + 64 * y, eSize * 4 * sizeof(float));
    }
}

static void _computeUnitFMA(float* dst, const float* weight, const float* A, int l) {
    auto s0 = _mm256_loadu_ps(A + 0 * 16);
    auto s1 = _mm256_loadu_ps(A + 0 * 16 + 8);
    auto w0 = _mm256_broadcast_ss(weight + 0 * 6 + 0);
    auto w1 = _mm256_broadcast_ss(weight + 0 * 6 + 1);
    auto z0 = _mm256_mul_ps(s0, w0);
    auto z1 = _mm256_mul_ps(s1, w0);
    auto z2 = _mm256_mul_ps(s0, w1);
    auto z3 = _mm256_mul_ps(s1, w1);
    w0 = _mm256_broadcast_ss(weight + 0 * 6 + 2);
    w1 = _mm256_broadcast_ss(weight + 0 * 6 + 3);
    auto z4 = _mm256_mul_ps(s0, w0);
    auto z5 = _mm256_mul_ps(s1, w0);
    auto z6 = _mm256_mul_ps(s0, w1);
    auto z7 = _mm256_mul_ps(s1, w1);
    w0 = _mm256_broadcast_ss(weight + 0 * 6 + 4);
    w1 = _mm256_broadcast_ss(weight + 0 * 6 + 5);
    auto z8 = _mm256_mul_ps(s0, w0);
    auto z9 = _mm256_mul_ps(s1, w0);
    auto z10 = _mm256_mul_ps(s0, w1);
    auto z11 = _mm256_mul_ps(s1, w1);

    for (int sy=1; sy<l; ++sy) {
        auto s0 = _mm256_loadu_ps(A + sy * 16);
        auto s1 = _mm256_loadu_ps(A + sy * 16 + 8);
        auto w0 = _mm256_broadcast_ss(weight + sy * 6 + 0);
        auto w1 = _mm256_broadcast_ss(weight + sy * 6 + 1);
        z0 = _mm256_fmadd_ps(s0, w0, z0);
        z1 = _mm256_fmadd_ps(s1, w0, z1);
        z2 = _mm256_fmadd_ps(s0, w1, z2);
        z3 = _mm256_fmadd_ps(s1, w1, z3);
        w0 = _mm256_broadcast_ss(weight + sy * 6 + 2);
        w1 = _mm256_broadcast_ss(weight + sy * 6 + 3);
        z4 = _mm256_fmadd_ps(s0, w0, z4);
        z5 = _mm256_fmadd_ps(s1, w0, z5);
        z6 = _mm256_fmadd_ps(s0, w1, z6);
        z7 = _mm256_fmadd_ps(s1, w1, z7);
        w0 = _mm256_broadcast_ss(weight + sy * 6 + 4);
        w1 = _mm256_broadcast_ss(weight + sy * 6 + 5);
        z8 = _mm256_fmadd_ps(s0, w0, z8);
        z9 = _mm256_fmadd_ps(s1, w0, z9);
        z10 = _mm256_fmadd_ps(s0, w1, z10);
        z11 = _mm256_fmadd_ps(s1, w1, z11);
    }
    _mm256_storeu_ps(dst + 8 * 0, z0);
    _mm256_storeu_ps(dst + 8 * 1, z1);
    _mm256_storeu_ps(dst + 8 * 2, z2);
    _mm256_storeu_ps(dst + 8 * 3, z3);
    _mm256_storeu_ps(dst + 8 * 4, z4);
    _mm256_storeu_ps(dst + 8 * 5, z5);
    _mm256_storeu_ps(dst + 8 * 6, z6);
    _mm256_storeu_ps(dst + 8 * 7, z7);
    _mm256_storeu_ps(dst + 8 * 8, z8);
    _mm256_storeu_ps(dst + 8 * 9, z9);
    _mm256_storeu_ps(dst + 8 * 10, z10);
    _mm256_storeu_ps(dst + 8 * 11, z11);
}

static void _computeUnit_4FMA(float* dst, const float* weight, const float* A, int l) {
    auto z0 = _mm256_set1_ps(0.0f);
    auto z1 = _mm256_set1_ps(0.0f);
    auto z2 = _mm256_set1_ps(0.0f);
    auto z3 = _mm256_set1_ps(0.0f);
    auto z4 = _mm256_set1_ps(0.0f);
    auto z5 = _mm256_set1_ps(0.0f);
    auto z6 = _mm256_set1_ps(0.0f);
    auto z7 = _mm256_set1_ps(0.0f);
    
    for (int sy=0; sy<l; ++sy) {
        auto s0 = _mm256_loadu_ps(A + sy * 16);
        auto s1 = _mm256_loadu_ps(A + sy * 16 + 8);
        auto w0 = _mm256_broadcast_ss(weight + sy * 6 + 0);
        auto w1 = _mm256_broadcast_ss(weight + sy * 6 + 1);
        z0 = _mm256_fmadd_ps(s0, w0, z0);
        z1 = _mm256_fmadd_ps(s1, w0, z1);
        z2 = _mm256_fmadd_ps(s0, w1, z2);
        z3 = _mm256_fmadd_ps(s1, w1, z3);
        w0 = _mm256_broadcast_ss(weight + sy * 6 + 2);
        w1 = _mm256_broadcast_ss(weight + sy * 6 + 3);
        z4 = _mm256_fmadd_ps(s0, w0, z4);
        z5 = _mm256_fmadd_ps(s1, w0, z5);
        z6 = _mm256_fmadd_ps(s0, w1, z6);
        z7 = _mm256_fmadd_ps(s1, w1, z7);
    }
    _mm256_storeu_ps(dst + 8 * 0, z0);
    _mm256_storeu_ps(dst + 8 * 1, z1);
    _mm256_storeu_ps(dst + 8 * 2, z2);
    _mm256_storeu_ps(dst + 8 * 3, z3);
    _mm256_storeu_ps(dst + 8 * 4, z4);
    _mm256_storeu_ps(dst + 8 * 5, z5);
    _mm256_storeu_ps(dst + 8 * 6, z6);
    _mm256_storeu_ps(dst + 8 * 7, z7);
}
static void _computeUnit_2FMA(float* dst, const float* weight, const float* A, int l) {
    auto z0 = _mm256_set1_ps(0.0f);
    auto z1 = _mm256_set1_ps(0.0f);
    auto z2 = _mm256_set1_ps(0.0f);
    auto z3 = _mm256_set1_ps(0.0f);
    
    for (int sy=0; sy<l; ++sy) {
        auto s0 = _mm256_loadu_ps(A + sy * 16);
        auto s1 = _mm256_loadu_ps(A + sy * 16 + 8);
        auto w0 = _mm256_broadcast_ss(weight + sy * 6 + 0);
        auto w1 = _mm256_broadcast_ss(weight + sy * 6 + 1);
        z0 = _mm256_fmadd_ps(s0, w0, z0);
        z1 = _mm256_fmadd_ps(s1, w0, z1);
        z2 = _mm256_fmadd_ps(s0, w1, z2);
        z3 = _mm256_fmadd_ps(s1, w1, z3);
    }
    _mm256_storeu_ps(dst + 8 * 0, z0);
    _mm256_storeu_ps(dst + 8 * 1, z1);
    _mm256_storeu_ps(dst + 8 * 2, z2);
    _mm256_storeu_ps(dst + 8 * 3, z3);
}
static void _computeUnit_1FMA(float* dst, const float* weight, const float* A, int l) {
    auto z0 = _mm256_set1_ps(0.0f);
    auto z1 = _mm256_set1_ps(0.0f);
    
    for (int sy=0; sy<l; ++sy) {
        auto s0 = _mm256_loadu_ps(A + sy * 16);
        auto s1 = _mm256_loadu_ps(A + sy * 16 + 8);
        auto w0 = _mm256_broadcast_ss(weight + sy * 6 + 0);
        z0 = _mm256_fmadd_ps(s0, w0, z0);
        z1 = _mm256_fmadd_ps(s1, w0, z1);
    }
    _mm256_storeu_ps(dst + 8 * 0, z0);
    _mm256_storeu_ps(dst + 8 * 1, z1);
}

void _AVX_MNNPackedMatMulFMA(float* C, const float* A, const float* B, const size_t* parameter, float* cache, const float* postParameters, const float* bias) {
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride = bExtraStride + l * 6;
    auto hC4 = UP_DIV(h, 4);
    int hDiv = hC4 / 3;
    for (int y=0; y<hDiv; ++y) {
        for (int p=0; p<2; ++p) {
            auto weight = B + (y * 2 + p) * bStride;
            auto dst = cache + p * 16 * 6;
            _computeUnitFMA(dst, weight, A, l);
        }
        for (int p=0; p<3; ++p) {
            auto dst = C + (y * 3 + p) * cStride;
            auto src = cache + 16 * 4 * p;
            if (nullptr == bias) {
                _save16x4(dst, src, postParameters, bias);
            } else {
                _save16x4(dst, src, postParameters, bias + (y * 3 + p) * 4);
            }
        }
    }
    auto hRemain = hC4 % 3;
    auto lastHStart = hDiv*3;
    if (hRemain == 2) {
        auto realHRemain = (int)h - (int)hDiv * 12;
        // First
        {
             auto weight = B + (hDiv * 2 + 0) * bStride;
             auto dst = cache;
            _computeUnitFMA(dst, weight, A, l);
            realHRemain -= 6;
        }
        // Second, max only two
        {
            auto weight = B + (hDiv * 2 + 1) * bStride;
            auto dst = cache + 1 * 16 * 6;
            ::memset(dst, 0, 16 * 6 * sizeof(float));
            if (realHRemain == 2) {
                _computeUnit_2FMA(dst, weight, A, l);
            } else if (realHRemain == 1) {
                _computeUnit_1FMA(dst, weight, A, l);
            }
        }
        for (int p=0; p<2; ++p) {
            auto dst = C + (hDiv * 3 + p) * cStride;
            auto src = cache + 16 * 4 * p;
            if (nullptr == bias) {
                _save16x4(dst, src, postParameters, bias);
            } else {
                _save16x4(dst, src, postParameters, bias + (hDiv * 3 + p) * 4);
            }
        }
    }
    if (hRemain == 1) {
        // First
        {
             auto weight = B + (hDiv * 2 + 0) * bStride;
             auto dst = cache;
            _computeUnit_4FMA(dst, weight, A, l);
        }
        for (int p=0; p<1; ++p) {
            auto dst = C + (hDiv * 3 + p) * cStride;
            auto src = cache + 16 * 4 * p;
            if (nullptr == bias) {
                _save16x4(dst, src, postParameters, bias);
            } else {
                _save16x4(dst, src, postParameters, bias + (hDiv * 3 + p) * 4);
            }
        }
    }
}

static void _computeUnitFMA_8(float* dst, const float* weight, const float* A, int l) {
    auto s0 = _mm256_loadu_ps(A + 0 * 16);
    auto w0 = _mm256_broadcast_ss(weight + 0 * 6 + 0);
    auto w1 = _mm256_broadcast_ss(weight + 0 * 6 + 1);
    auto z0 = _mm256_mul_ps(s0, w0);
    auto z2 = _mm256_mul_ps(s0, w1);
    w0 = _mm256_broadcast_ss(weight + 0 * 6 + 2);
    w1 = _mm256_broadcast_ss(weight + 0 * 6 + 3);
    auto z4 = _mm256_mul_ps(s0, w0);
    auto z6 = _mm256_mul_ps(s0, w1);
    w0 = _mm256_broadcast_ss(weight + 0 * 6 + 4);
    w1 = _mm256_broadcast_ss(weight + 0 * 6 + 5);
    auto z8 = _mm256_mul_ps(s0, w0);
    auto z10 = _mm256_mul_ps(s0, w1);

    for (int sy=1; sy<l; ++sy) {
        auto s0 = _mm256_loadu_ps(A + sy * 16);
        auto w0 = _mm256_broadcast_ss(weight + sy * 6 + 0);
        auto w1 = _mm256_broadcast_ss(weight + sy * 6 + 1);
        z0 = _mm256_fmadd_ps(s0, w0, z0);
        z2 = _mm256_fmadd_ps(s0, w1, z2);
        w0 = _mm256_broadcast_ss(weight + sy * 6 + 2);
        w1 = _mm256_broadcast_ss(weight + sy * 6 + 3);
        z4 = _mm256_fmadd_ps(s0, w0, z4);
        z6 = _mm256_fmadd_ps(s0, w1, z6);
        w0 = _mm256_broadcast_ss(weight + sy * 6 + 4);
        w1 = _mm256_broadcast_ss(weight + sy * 6 + 5);
        z8 = _mm256_fmadd_ps(s0, w0, z8);
        z10 = _mm256_fmadd_ps(s0, w1, z10);
    }
    _mm256_storeu_ps(dst + 8 * 0, z0);
    _mm256_storeu_ps(dst + 8 * 1, z2);
    _mm256_storeu_ps(dst + 8 * 2, z4);
    _mm256_storeu_ps(dst + 8 * 3, z6);
    _mm256_storeu_ps(dst + 8 * 4, z8);
    _mm256_storeu_ps(dst + 8 * 5, z10);
}

static void _computeUnit_4FMA_8(float* dst, const float* weight, const float* A, int l) {
    auto z0 = _mm256_set1_ps(0.0f);
    auto z2 = _mm256_set1_ps(0.0f);
    auto z4 = _mm256_set1_ps(0.0f);
    auto z6 = _mm256_set1_ps(0.0f);
    
    for (int sy=0; sy<l; ++sy) {
        auto s0 = _mm256_loadu_ps(A + sy * 16);
        auto w0 = _mm256_broadcast_ss(weight + sy * 6 + 0);
        auto w1 = _mm256_broadcast_ss(weight + sy * 6 + 1);
        z0 = _mm256_fmadd_ps(s0, w0, z0);
        z2 = _mm256_fmadd_ps(s0, w1, z2);
        w0 = _mm256_broadcast_ss(weight + sy * 6 + 2);
        w1 = _mm256_broadcast_ss(weight + sy * 6 + 3);
        z4 = _mm256_fmadd_ps(s0, w0, z4);
        z6 = _mm256_fmadd_ps(s0, w1, z6);
    }
    _mm256_storeu_ps(dst + 8 * 0, z0);
    _mm256_storeu_ps(dst + 8 * 1, z2);
    _mm256_storeu_ps(dst + 8 * 2, z4);
    _mm256_storeu_ps(dst + 8 * 3, z6);
}
static void _computeUnit_2FMA_8(float* dst, const float* weight, const float* A, int l) {
    auto z0 = _mm256_set1_ps(0.0f);
    auto z2 = _mm256_set1_ps(0.0f);
    
    for (int sy=0; sy<l; ++sy) {
        auto s0 = _mm256_loadu_ps(A + sy * 16);
        auto w0 = _mm256_broadcast_ss(weight + sy * 6 + 0);
        auto w1 = _mm256_broadcast_ss(weight + sy * 6 + 1);
        z0 = _mm256_fmadd_ps(s0, w0, z0);
        z2 = _mm256_fmadd_ps(s0, w1, z2);
    }
    _mm256_storeu_ps(dst + 8 * 0, z0);
    _mm256_storeu_ps(dst + 8 * 1, z2);
}
static void _computeUnit_1FMA_8(float* dst, const float* weight, const float* A, int l) {
    auto z0 = _mm256_set1_ps(0.0f);
    
    for (int sy=0; sy<l; ++sy) {
        auto s0 = _mm256_loadu_ps(A + sy * 16);
        auto w0 = _mm256_broadcast_ss(weight + sy * 6 + 0);
        z0 = _mm256_fmadd_ps(s0, w0, z0);
    }
    _mm256_storeu_ps(dst + 8 * 0, z0);
}


static void _AVX_MNNPackedMatMulFMA_8(float* C, const float* A, const float* B, const size_t* parameter, float* cache, const float* postParameters, const float* bias) {
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride = bExtraStride + l * 6;
    auto hC4 = UP_DIV(h, 4);
    int hDiv = hC4 / 3;
    for (int y=0; y<hDiv; ++y) {
        for (int p=0; p<2; ++p) {
            auto weight = B + (y * 2 + p) * bStride;
            auto dst = cache + p * 8 * 6;
            _computeUnitFMA_8(dst, weight, A, l);
        }
        for (int p=0; p<3; ++p) {
            auto dst = C + (y * 3 + p) * cStride;
            auto src = cache + 8 * 4 * p;
            if (nullptr == bias) {
                _save8x4(dst, src, postParameters, bias);
            } else {
                _save8x4(dst, src, postParameters, bias + (y * 3 + p) * 4);
            }
        }
    }
    auto hRemain = hC4 % 3;
    auto lastHStart = hDiv*3;
    if (hRemain == 2) {
        auto realHRemain = (int)h - (int)hDiv * 12;
        // First
        {
             auto weight = B + (hDiv * 2 + 0) * bStride;
             auto dst = cache;
            _computeUnitFMA_8(dst, weight, A, l);
            realHRemain -= 6;
        }
        // Second, max only two
        {
            auto weight = B + (hDiv * 2 + 1) * bStride;
            auto dst = cache + 1 * 8 * 6;
            ::memset(dst, 0, 8 * 6 * sizeof(float));
            if (realHRemain == 2) {
                _computeUnit_2FMA_8(dst, weight, A, l);
            } else if (realHRemain == 1) {
                _computeUnit_1FMA_8(dst, weight, A, l);
            }
        }
        for (int p=0; p<2; ++p) {
            auto dst = C + (hDiv * 3 + p) * cStride;
            auto src = cache + 8 * 4 * p;
            if (nullptr == bias) {
                _save8x4(dst, src, postParameters, bias);
            } else {
                _save8x4(dst, src, postParameters, bias + (hDiv * 3 + p) * 4);
            }
        }
    }
    if (hRemain == 1) {
        // First
        {
             auto weight = B + (hDiv * 2 + 0) * bStride;
             auto dst = cache;
            _computeUnit_4FMA_8(dst, weight, A, l);
        }
        for (int p=0; p<1; ++p) {
            auto dst = C + (hDiv * 3 + p) * cStride;
            auto src = cache + 8 * 4 * p;
            if (nullptr == bias) {
                _save8x4(dst, src, postParameters, bias);
            } else {
                _save8x4(dst, src, postParameters, bias + (hDiv * 3 + p) * 4);
            }
        }
    }
}
void _AVX_MNNPackedMatMulRemainFMA(float* C_, const float* A_, const float* B, size_t eSize, const size_t* parameter, float* cache, const float* postParameters, const float* bias) {
    auto h = parameter[2];
    auto cStride = parameter[3] / sizeof(float);
    auto hC4 = UP_DIV(h, 4);
    auto CTemp = cache + 16 * 6 * 2;
    size_t tempParamters[] = {
        parameter[0],
        parameter[1],
        parameter[2],
        16 * 4 * sizeof(float),
        parameter[4],
        parameter[5],
    };
    if (eSize > 8) {
        _AVX_MNNPackedMatMulFMA(CTemp, A_, B, tempParamters, cache, postParameters, bias);
    } else {
        _AVX_MNNPackedMatMulFMA_8(CTemp, A_, B, tempParamters, cache, postParameters, bias);
    }
    for (int y=0; y<hC4; ++y) {
        ::memcpy(C_ + y * cStride, CTemp + 64 * y, eSize * 4 * sizeof(float));
    }
}
