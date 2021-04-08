#include "FunctionSummary.hpp"
#include "Gemm24_4_4.hpp"
#include "core/Macro.h"
#include "math/Vec.hpp"
#include <limits>
#include <string.h>
#include <algorithm>
#include <vector>
// TODO: this function is not implemented for avx512 yet.
void AVX512GemmPostTreat(float* C, size_t eSize, const size_t* parameter, const float* postParameters,
                       const float* bias) {
    if (nullptr == postParameters) {
        return;
    }
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    auto minValue     = _mm_broadcast_ss(postParameters + 2);
    auto maxValue     = _mm_broadcast_ss(postParameters + 3);
    int eC2           = eSize / 2;
    int eR            = eSize % 2;
    auto minV2        = _mm256_broadcast_ss(postParameters + 2);
    auto maxV2        = _mm256_broadcast_ss(postParameters + 3);
    if (nullptr != bias) {
        if (eR > 0) {
            for (int y = 0; y < hC4; ++y) {
                auto biasValue = _mm_loadu_ps(bias + 4 * y);
                auto bias2     = _mm256_broadcast_ps((__m128*)(bias + 4 * y));
                auto dst       = C + y * cStride;
                for (int x = 0; x < eC2; ++x) {
                    auto sum = _mm256_add_ps(bias2, _mm256_loadu_ps(dst));
                    sum      = _mm256_max_ps(sum, minV2);
                    sum      = _mm256_min_ps(sum, maxV2);
                    _mm256_storeu_ps(dst, sum);
                    dst += 8;
                }
                auto sum = _mm_add_ps(biasValue, _mm_loadu_ps(dst));
                sum      = _mm_max_ps(sum, minValue);
                sum      = _mm_min_ps(sum, maxValue);
                _mm_storeu_ps(dst, sum);
            }
        } else {
            for (int y = 0; y < hC4; ++y) {
                auto biasValue = _mm_loadu_ps(bias + 4 * y);
                auto bias2     = _mm256_broadcast_ps((__m128*)(bias + 4 * y));
                auto dst       = C + y * cStride;
                for (int x = 0; x < eC2; ++x) {
                    auto sum = _mm256_add_ps(bias2, _mm256_loadu_ps(dst));
                    sum      = _mm256_max_ps(sum, minV2);
                    sum      = _mm256_min_ps(sum, maxV2);
                    _mm256_storeu_ps(dst, sum);
                    dst += 8;
                }
            }
        }
    } else {
        if (eR > 0) {
            for (int y = 0; y < hC4; ++y) {
                auto dst = C + y * cStride;
                for (int x = 0; x < eC2; ++x) {
                    auto sum = _mm256_loadu_ps(dst);
                    sum      = _mm256_max_ps(sum, minV2);
                    sum      = _mm256_min_ps(sum, maxV2);
                    _mm256_storeu_ps(dst, sum);
                    dst += 8;
                }
                auto sum = _mm_loadu_ps(dst);
                sum      = _mm_max_ps(sum, minValue);
                sum      = _mm_min_ps(sum, maxValue);
                _mm_storeu_ps(dst, sum);
            }
        } else {
            for (int y = 0; y < hC4; ++y) {
                auto dst = C + y * cStride;
                for (int x = 0; x < eC2; ++x) {
                    auto sum = _mm256_loadu_ps(dst);
                    sum      = _mm256_max_ps(sum, minV2);
                    sum      = _mm256_min_ps(sum, maxV2);
                    _mm256_storeu_ps(dst, sum);
                    dst += 8;
                }
            }
        }
    }
}

#ifdef MNN_X86_USE_ASM
extern "C" {
void _AVX512_MNNGemmFloatUnitMainFMA(float* C, const float* A, const float* B, const size_t* parameter, size_t hC4);
void _AVX512_MNNGemmFloatUnit16(float* C, const float* A, const float* B, const size_t* parameter, size_t hC4);
}
#endif


void _AVX512_MNNPackC4ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];
    int pOffset = 4 * offset;

    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        auto lC4         = l / 4;
        auto lDiv        = UP_DIV(l, 4);
        auto lOC = lOffset / 4;
        auto lOR = lOffset % 4;
        auto source = sourceGroup[n];
        auto dest = destOrigin + eOffset * 4 + lOC * eDest * 4;
        if (lOR == 0) {
            // Fast way
            int alignLC4 = lDiv;
            for (int x=0; x<lDiv; ++x) {
                auto destX = dest + x * eDest * 4;
                auto srcX = source + x * eReal * 4;
                for (int y=0; y<e; ++y) {
                    _mm_storeu_ps(destX + 4 * y, _mm_loadu_ps(srcX + 4 * y * offset));
                }
            }
            continue;
        }
        for (int x=0; x<l; ++x) {
            auto dl = lOR + x;
            auto dlC = dl / 4;
            auto dlR = dl % 4;
            auto xC = x / 4;
            auto xR = x % 4;
            auto destX = dest + dlC * eDest * 4 + dlR;
            auto srcX = source + xC * eReal * 4 + xR;
            for (int y=0; y<e; ++y) {
                destX[y*4] = srcX[y*4*offset];
            }
        }
    }
}

void _AVX512_MNNPackForMatMul_B(float* destF, const float* sourceF, size_t h, size_t l, bool transpose) {
    auto dest = destF;
    auto source = sourceF;
    auto lC4 = UP_DIV(l, 4);
    auto hC4 = UP_DIV(h, 4);
    int sYstride = 1;
    int sXstride = h;
    if (transpose) {
        sYstride = l;
        sXstride = 1;
    }
    int l4 = l / 4;
    int h4 = h / 4;
    int lR = l % 4;
    int hR = h % 4;
    if (transpose) {
        for (int y = 0; y < h4; ++y) {
            auto srcY0 = source + (4 * y + 0) * l;
            auto srcY1 = source + (4 * y + 1) * l;
            auto srcY2 = source + (4 * y + 2) * l;
            auto srcY3 = source + (4 * y + 3) * l;
            auto dstY = dest + 16 * y * lC4;
            for (int x = 0; x < l4; ++x) {
                _mm_storeu_ps(dstY + 4 * 0, _mm_loadu_ps(srcY0));
                _mm_storeu_ps(dstY + 4 * 1, _mm_loadu_ps(srcY1));
                _mm_storeu_ps(dstY + 4 * 2, _mm_loadu_ps(srcY2));
                _mm_storeu_ps(dstY + 4 * 3, _mm_loadu_ps(srcY3));
                srcY0 += 4;
                srcY1 += 4;
                srcY2 += 4;
                srcY3 += 4;
                dstY += 16;
            }
            if (lR > 0) {
                float temp[16];
                ::memset(temp, 0, sizeof(float) * 16);
                ::memcpy(temp + 4 * 0, srcY0, lR * sizeof(float));
                ::memcpy(temp + 4 * 1, srcY1, lR * sizeof(float));
                ::memcpy(temp + 4 * 2, srcY2, lR * sizeof(float));
                ::memcpy(temp + 4 * 3, srcY3, lR * sizeof(float));
                ::memcpy(dstY, temp, sizeof(float) * 16);
            }
        }
        if (hR > 0) {
            auto srcY0 = source + (4 * h4 + 0) * l;
            auto srcY1 = source + (4 * h4 + 1) * l;
            auto srcY2 = source + (4 * h4 + 2) * l;
            auto dstY = dest + 16 * h4 * lC4;
            auto zero = _mm_set1_ps(0.0f);
            switch (hR) {
                case 3: {
                    for (int x = 0; x < l4; ++x) {
                        _mm_storeu_ps(dstY + 4 * 0, _mm_loadu_ps(srcY0));
                        _mm_storeu_ps(dstY + 4 * 1, _mm_loadu_ps(srcY1));
                        _mm_storeu_ps(dstY + 4 * 2, _mm_loadu_ps(srcY2));
                        _mm_storeu_ps(dstY + 4 * 3, zero);
                        srcY0 += 4;
                        srcY1 += 4;
                        srcY2 += 4;
                        dstY += 16;
                    }
                    break;
                }
                case 2: {
                    for (int x = 0; x < l4; ++x) {
                        _mm_storeu_ps(dstY + 4 * 0, _mm_loadu_ps(srcY0));
                        _mm_storeu_ps(dstY + 4 * 1, _mm_loadu_ps(srcY1));
                        _mm_storeu_ps(dstY + 4 * 2, zero);
                        _mm_storeu_ps(dstY + 4 * 3, zero);
                        srcY0 += 4;
                        srcY1 += 4;
                        dstY += 16;
                    }
                    break;
                }
                case 1: {
                    for (int x = 0; x < l4; ++x) {
                        _mm_storeu_ps(dstY + 4 * 0, _mm_loadu_ps(srcY0));
                        _mm_storeu_ps(dstY + 4 * 1, zero);
                        _mm_storeu_ps(dstY + 4 * 2, zero);
                        _mm_storeu_ps(dstY + 4 * 3, zero);
                        srcY0 += 4;
                        dstY += 16;
                    }
                    break;
                }
                default:
                    break;
            }
            if (lR > 0) {
                float temp[16];
                ::memset(temp, 0, sizeof(float) * 16);
                ::memcpy(temp + 4 * 0, srcY0, lR * sizeof(float));
                if (hR >= 1) {
                    ::memcpy(temp + 4 * 1, srcY1, lR * sizeof(float));
                }
                if (hR >= 2) {
                    ::memcpy(temp + 4 * 2, srcY2, lR * sizeof(float));
                }
                ::memcpy(dstY, temp, sizeof(float) * 16);
            }
        }
        return;
    }
    
    
    // No Transpose
    for (int x = 0; x < l4; ++x) {
        auto srcX0 = source + (4 * x + 0) * h;
        auto srcX1 = source + (4 * x + 1) * h;
        auto srcX2 = source + (4 * x + 2) * h;
        auto srcX3 = source + (4 * x + 3) * h;
        auto dstX = dest + 16 * x;
        for (int y = 0; y < h4; ++y) {
            auto p0 = _mm_loadu_ps(srcX0);
            auto p1 = _mm_loadu_ps(srcX1);
            auto p2 = _mm_loadu_ps(srcX2);
            auto p3 = _mm_loadu_ps(srcX3);
            _MM_TRANSPOSE4_PS(p0, p1, p2, p3);
            _mm_storeu_ps(dstX + 4 * 0, p0);
            _mm_storeu_ps(dstX + 4 * 1, p1);
            _mm_storeu_ps(dstX + 4 * 2, p2);
            _mm_storeu_ps(dstX + 4 * 3, p3);
            srcX0 += 4;
            srcX1 += 4;
            srcX2 += 4;
            srcX3 += 4;
            dstX += 16 * lC4;
        }
        if (hR > 0) {
            float temp[16];
            ::memset(temp, 0, sizeof(float) * 16);
            ::memcpy(temp + 4 * 0, srcX0, hR * sizeof(float));
            ::memcpy(temp + 4 * 1, srcX1, hR * sizeof(float));
            ::memcpy(temp + 4 * 2, srcX2, hR * sizeof(float));
            ::memcpy(temp + 4 * 3, srcX3, hR * sizeof(float));
            auto p0 = _mm_loadu_ps(temp + 4 * 0);
            auto p1 = _mm_loadu_ps(temp + 4 * 1);
            auto p2 = _mm_loadu_ps(temp + 4 * 2);
            auto p3 = _mm_loadu_ps(temp + 4 * 3);
            _MM_TRANSPOSE4_PS(p0, p1, p2, p3);
            _mm_storeu_ps(dstX + 4 * 0, p0);
            _mm_storeu_ps(dstX + 4 * 1, p1);
            _mm_storeu_ps(dstX + 4 * 2, p2);
            _mm_storeu_ps(dstX + 4 * 3, p3);
        }
    }
    if (lR > 0) {
        auto zero = _mm_set1_ps(0.0f);
        auto srcX0 = source + (4 * l4 + 0) * h;
        auto srcX1 = source + (4 * l4 + 1) * h;
        auto srcX2 = source + (4 * l4 + 2) * h;
        auto dstX = dest + 16 * l4;
        switch (lR) {
            case 3: {
                for (int y = 0; y < h4; ++y) {
                    auto p0 = _mm_loadu_ps(srcX0);
                    auto p1 = _mm_loadu_ps(srcX1);
                    auto p2 = _mm_loadu_ps(srcX2);
                    auto p3 = zero;
                    _MM_TRANSPOSE4_PS(p0, p1, p2, p3);
                    _mm_storeu_ps(dstX + 4 * 0, p0);
                    _mm_storeu_ps(dstX + 4 * 1, p1);
                    _mm_storeu_ps(dstX + 4 * 2, p2);
                    _mm_storeu_ps(dstX + 4 * 3, p3);
                    srcX0 += 4;
                    srcX1 += 4;
                    srcX2 += 4;
                    dstX += 16 * lC4;
                }
                break;
            }
            case 2: {
                for (int y = 0; y < h4; ++y) {
                    auto p0 = _mm_loadu_ps(srcX0);
                    auto p1 = _mm_loadu_ps(srcX1);
                    auto p2 = zero;
                    auto p3 = zero;
                    _MM_TRANSPOSE4_PS(p0, p1, p2, p3);
                    _mm_storeu_ps(dstX + 4 * 0, p0);
                    _mm_storeu_ps(dstX + 4 * 1, p1);
                    _mm_storeu_ps(dstX + 4 * 2, p2);
                    _mm_storeu_ps(dstX + 4 * 3, p3);
                    srcX0 += 4;
                    srcX1 += 4;
                    srcX2 += 4;
                    dstX += 16 * lC4;
                }
                break;
            }
            case 1: {
                for (int y = 0; y < h4; ++y) {
                    auto p0 = _mm_loadu_ps(srcX0);
                    auto p1 = zero;
                    auto p2 = zero;
                    auto p3 = zero;
                    _MM_TRANSPOSE4_PS(p0, p1, p2, p3);
                    _mm_storeu_ps(dstX + 4 * 0, p0);
                    _mm_storeu_ps(dstX + 4 * 1, p1);
                    _mm_storeu_ps(dstX + 4 * 2, p2);
                    _mm_storeu_ps(dstX + 4 * 3, p3);
                    srcX0 += 4;
                    srcX1 += 4;
                    srcX2 += 4;
                    dstX += 16 * lC4;
                }
                break;
            }
            default:
                break;
        }
        if (hR > 0) {
            float temp[16];
            ::memset(temp, 0, sizeof(float) * 16);
            ::memcpy(temp + 4 * 0, srcX0, hR * sizeof(float));
            if (lR > 1) {
                ::memcpy(temp + 4 * 1, srcX1, hR * sizeof(float));
            }
            if (lR > 2) {
                ::memcpy(temp + 4 * 2, srcX2, hR * sizeof(float));
            }
            auto p0 = _mm_loadu_ps(temp + 4 * 0);
            auto p1 = _mm_loadu_ps(temp + 4 * 1);
            auto p2 = _mm_loadu_ps(temp + 4 * 2);
            auto p3 = _mm_loadu_ps(temp + 4 * 3);
            _MM_TRANSPOSE4_PS(p0, p1, p2, p3);
            _mm_storeu_ps(dstX + 4 * 0, p0);
            _mm_storeu_ps(dstX + 4 * 1, p1);
            _mm_storeu_ps(dstX + 4 * 2, p2);
            _mm_storeu_ps(dstX + 4 * 3, p3);
        }
    }
}

void _AVX512_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias) {
    auto h       = parameter[2];
    auto hC4     = UP_DIV(h, 4);
    auto cStride = parameter[3] / sizeof(float);
#ifdef MNN_X86_USE_ASM
    _AVX512_MNNGemmFloatUnitMainFMA(C, A, B, parameter, hC4);
#else
    _AVX512_MNNPackedMatMul_24(C, A, B, parameter);
#endif
    AVX512GemmPostTreat(C, 24, parameter, postParameters, bias);
}

void _AVX512_MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias) {
    auto h       = parameter[2];
    auto hC4     = UP_DIV(h, 4);
    auto oriSize = eSize;
    auto oC      = C;
    //FUNC_PRINT(eSize);
#ifdef MNN_X86_USE_ASM
    if (16 <= eSize) {
        _AVX512_MNNGemmFloatUnit16(C, A, B, parameter, hC4);
        eSize -= 16;
        C += 16 * 4;
        A += 16 * 4;
    }
#endif
    _AVX512_MNNPackednMatMulRemainCommon_4(C, A, B, eSize, parameter, postParameters, bias);
    AVX512GemmPostTreat(oC, oriSize, parameter, postParameters, bias);
}
