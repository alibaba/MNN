#ifdef MNN_USE_SSE
#include "../x86_x64/sse/FunctionSummary.hpp"
#include "../x86_x64/avx/FunctionSummary.hpp"
#include "../x86_x64/avxfma/FunctionSummary.hpp"
#include "../x86_x64/avx512/FunctionSummary.hpp"
#endif
#include "core/Macro.h"
#if defined(MNN_USE_NEON)
#include "../arm/FunctionSummary.hpp"
#endif

#include "BF16Functions.hpp"
#include "../compute/CommonOptFunction.h"
#include "../CPURuntime.hpp"
#include "VecHalf.hpp"
#include "math/Vec.hpp"
using BFVec4 = MNN::Math::VecHalf<4>;
using Vec4 = MNN::Math::Vec<float, 4>;
namespace MNN {
// The Function Will be Called in init
void registerBF16Backend() {
    BF16Functions::init();
}
// just for reference BF16 converting of c++ code, not for arm or sse.
inline int16_t MNNFP32ToBF16(float fp32Value) {
    int32_t* s32Value = (int32_t*)(&fp32Value);
    return (int16_t)((*s32Value) >> 16);
}
inline float MNNLowpToFp32(int16_t s16Value) {
    int32_t s32Value = ((int32_t)s16Value) << 16;
    float* fp32Value = (float*)(&s32Value);
    return *fp32Value;
}

static void _MNNFp32ToLowp(const float* src, int16_t* dst, size_t size) {
    int sizeC4 = size / 4;
    for (int i = 0; i < sizeC4; ++i) {
        auto srcV = Vec4::load(src);
        auto dstV = BFVec4(std::move(srcV.value));
        BFVec4::save(dst, dstV);
        src+=4;
        dst+=4;
    }
    int sizeRemain = size % 4;
    if (sizeRemain > 0) {
        float srcTemp[4];
        int64_t dstTemp[1];
        ::memcpy(srcTemp, src, sizeRemain * sizeof(float));
        auto srcV = Vec4::load(srcTemp);
        auto dstV = BFVec4(std::move(srcV.value));
        BFVec4::save((int16_t*)dstTemp, dstV);
        ::memcpy(dst, dstTemp, sizeRemain * sizeof(int16_t));
    }
}
static void _MNNLowpToFp32(const int16_t* src, float* dst, size_t size) {
    int sizeC4 = size / 4;
    for (int i = 0; i < sizeC4; ++i) {
        auto srcV = BFVec4::load(src);
        auto dstV = Vec4(std::move(srcV.value));
        Vec4::save(dst, dstV);
        src+=4;
        dst+=4;
    }
    int sizeRemain = size % 4;
    if (sizeRemain > 0) {
        int64_t srcTemp[2];
        float dstTemp[4];
        ::memcpy(srcTemp, src, sizeRemain * sizeof(int16_t));
        auto srcV = BFVec4::load((int16_t*)srcTemp);
        auto dstV = Vec4(std::move(srcV.value));
        Vec4::save(dstTemp, dstV);
        ::memcpy(dst, dstTemp, sizeRemain * sizeof(float));
    }
}

#if defined(MNN_USE_NEON)
// todo: search for proper value for bf16
void NEON_MNNGetMatMulPackMode_BF16(int* eP, int* lP, int* hP) {
    *eP = 12;
    *lP = 1;
#ifdef __aarch64__
    *hP = 8;
#else
    *hP = 4;
#endif
}

#ifdef __aarch64__
#define EP 12
#define HP 8
#define LP 4
void ARMV86_MNNGetMatMulPackMode_BF16(int* eP, int* lP, int* hP) {
    *eP = EP;
    *hP = HP;
    *lP = LP;
}
void ARMV86_MNNPackForMatMul_B_BF16(float* destF, const float* sourceF, size_t h, size_t l, bool transpose) {
    // [l, h] -> [h/hp, l/lp, hp, lp]
    auto dest = (int16_t*)destF;
    auto source = (const int32_t*)sourceF;
    auto lCP = UP_DIV(l, LP);
    auto hCP = UP_DIV(h, HP);
    int sYstride = 1;
    int sXstride = h;
    if (transpose) {
        sYstride = l;
        sXstride = 1;
    }
    ::memset(dest, 0, lCP * hCP * sizeof(int16_t) * HP * LP);
    for (int y = 0; y < h; ++y) {
        int yC = y / HP;
        int yR = y % HP;
        for (int x = 0; x < l; ++x) {
            int xC = x / LP;
            int xR = x % LP;
            dest[xR + yR * LP + xC * HP * LP + yC * HP * LP * lCP] = source[sXstride * x + sYstride * y] >> 16;
        }
    }
}
void ARMV86_MNNPackC4ForMatMul_A_BF16(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    // [l/4, e, 4] -> [l/4, ep, 4]
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];
    if (1 == number) {
        int l = el[1];
        if (l % 8 != 0) {
            auto lAigin = UP_DIV(l, LP) * LP;
            ::memset(destOrigin, 0, eDest * lAigin * sizeof(int16_t));
        }
    }

    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        auto lC4         = l / 4;
        auto lDiv        = UP_DIV(l, 4);
        auto lOC = lOffset / LP;
        auto lOR = lOffset % LP;
        auto source = (const float*)(sourceGroup[n]);
        auto sourceInt = (const int32_t*)(source);
        auto dest = ((int16_t*)destOrigin) + eOffset * LP + lOC * eDest * LP;
        if (lOR == 0) {
            // [l/4, e, 4] -> [l/4, ep, 4]
            for (int x = 0; x < lDiv; ++x) {
                auto destX = dest + x * eDest * 4;
                auto srcX  = source + x * eReal * 4;
                for (int y = 0; y < e; ++y) {
                    auto srcV = Vec4::load(srcX + y * offset * 4);
                    auto dstV = BFVec4(std::move(srcV.value));
                    BFVec4::save((int16_t*)(destX + 4*y), dstV);
                }
            }
            continue;
        }
        for (int x = 0; x < l; ++x) {
            auto dl = lOR + x;
            auto dlC = dl / LP;
            auto dlR = dl % LP;
            auto xC = x / LP;
            auto xR = x % LP;
            auto destX = dest + dlC * eDest * LP + dlR;
            auto srcX = sourceInt + xC * eReal * LP + xR;
            for (int y = 0; y < e; ++y) {
                destX[y * 4] = srcX[y * 4 * offset] >> 16;
            }
        }
    }
}
#undef EP
#undef HP
#undef LP
void NEON_MNNPackForMatMul_B_BF16(float* destFloat, const float* sourceFloat, size_t h, size_t l, bool transpose) {
    auto hP         = (int)h / 8;
    auto hR         = (int)hP * 8;
    int16_t* dest   = (int16_t*)destFloat;
    const float* source = sourceFloat;
    if (!transpose) {
        for (int y = 0; y < hP; ++y) {
            auto destY   = dest + y * 8 * l;
            auto sourceY = source + y * 8;
            for (int x = 0; x < l; ++x) {
                auto s0 = Vec4::load(sourceY + x * h + 0);
                auto s1 = Vec4::load(sourceY + x * h + 4);
                auto d0 = BFVec4(std::move(s0.value));
                auto d1 = BFVec4(std::move(s1.value));
                BFVec4::save(destY + 8 * x + 0, d0);
                BFVec4::save(destY + 8 * x + 4, d1);
            }
        }
        auto hRemain = h - hR;
        if (hRemain > 0) {
            auto destY   = dest + hP * 8 * l;
            auto sourceY = source + hP * 8;
            float sTmp[8];
            ::memset(sTmp, 0, sizeof(sTmp));
            for (int x = 0; x < l; ++x) {
                ::memcpy(sTmp, sourceY + x * h, hRemain * sizeof(float));
                auto s0 = Vec4::load(sTmp + 0);
                auto s1 = Vec4::load(sTmp + 4);
                auto d0 = BFVec4(std::move(s0.value));
                auto d1 = BFVec4(std::move(s1.value));
                BFVec4::save(destY + 8 * x + 0, d0);
                BFVec4::save(destY + 8 * x + 4, d1);
            }
        }
        return;
    }
    if (hR != h) {
        ::memset(dest, 0, UP_DIV(h, 8) * 8 * l * sizeof(int16_t));
    }
    auto sourceInt32 = (const int32_t*)source;
#if 0
    // Origin C++ code
    for (int y = 0; y < h; ++y) {
        auto yR = y % 8;
        auto yC = y / 8;
        for (int x = 0; x < l; ++x) {
            dest[x * 8 + yR + yC * 8 * l] = sourceInt32[x + y * l] >> 16;
        }
    }
    return;
#endif
    int lC8 = (int)l / 8;
    auto lR = lC8 * 8;
    if (hP > 0 && lC8 > 0) {
        MNNPackC8_BF16(destFloat, sourceFloat, l, h);
    }
    for (int y = hR; y < h; ++y) {
        auto yR = y % 8;
        auto yC = hP;
        for (int x = 0; x < l; ++x) {
            dest[x * 8 + yR + yC * 8 * l] = sourceInt32[x + y * l] >> 16;
        }
    }
    for (int y = 0; y < hR; ++y) {
        auto yR = y % 8;
        auto yC = y / 8;
        for (int x = lR; x < l; ++x) {
            dest[x * 8 + yR + yC * 8 * l] = sourceInt32[x + y * l] >> 16;
        }
    }
}

#else
void NEON_MNNPackForMatMul_B_BF16(float* destFloat, const float* sourceFloat, size_t h, size_t l, bool transpose) {
    int16_t* dest   = (int16_t*)destFloat;
    const float* source = sourceFloat;
    if (!transpose) {
        auto hP = h / 4;
        auto hR = hP * 4;
        if (hR != h) {
            ::memset(dest, 0, UP_DIV(h, 4) * 4 * l * sizeof(int16_t));
        }
        for (int y = 0; y < hP; ++y) {
            auto destY = dest + y * 4 * l;
            auto sourceY = source + y * 4;
            for (int x = 0; x < l; ++x) {
                auto s0 = Vec4::load(sourceY + x * h + 0);
                auto d0 = BFVec4(std::move(s0.value));
                BFVec4::save(destY + 4 * x + 0, d0);
            }
        }
        auto hRemain = h - hR;
        if (hRemain > 0) {
            auto destY = dest + hP * 4 * l;
            auto sourceY = source + hP * 4;
            for (int x = 0; x < l; ++x) {
                auto s0 = Vec4::load(sourceY + x * h + 0);
                auto d0 = BFVec4(std::move(s0.value));
                BFVec4::save(destY + 4 * x + 0, d0);
            }
        }
        return;
    }
#if 0
    auto sourceInt32 = (const int32_t*)source;
    // Origin C++ code
    ::memset(dest, 0, UP_DIV(h, 4) * 4 * l * sizeof(int16_t));

    for (int y = 0; y < h; ++y) {
        auto yR = y % 4;
        auto yC = y / 4;
        for (int x = 0; x < l; ++x) {
            dest[x * 4 + yR + yC * 4 * l] = sourceInt32[x + y * l] >> 16;
        }
    }
    return;
#endif
    int offset[2] = {
        (int)l,
        (int)l,
    };
    MNNPackC4_BF16(destFloat, sourceFloat, l, h, offset);
}
#endif // __aarch64__
#endif

#if 0
void MNNPackC4ForMatMul_ABF16(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];
    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        auto dest = (int16_t*)destOrigin + lOffset * eDest + eOffset;
        auto source = (int32_t*)sourceGroup[n];

        for (int y=0; y<e; ++y) {
            auto yR = y % eDest;
            for (int x=0; x<l; ++x) {
                auto xR = x % 4;
                auto xC = x / 4;
                dest[(x) * eDest + yR] = source[xC * eReal * 4 + y * 4 * offset + xR] >> 16;
            }
        }
    }
}
#endif

static CoreFunctions* gInstance = nullptr;
bool BF16Functions::init() {
#if !defined(MNN_USE_NEON)
    return false;
#else
    gInstance = new CoreFunctions;
    *gInstance = *MNNGetCoreFunctions();
    gInstance->MNNFp32ToLowp = _MNNFp32ToLowp;
    gInstance->MNNLowpToFp32 = _MNNLowpToFp32;
    gInstance->matmulBytes = 2;

    gInstance->MNNPackForMatMul_B = NEON_MNNPackForMatMul_B_BF16;
    gInstance->MNNGetMatMulPackMode = NEON_MNNGetMatMulPackMode_BF16;
    gInstance->MNNPackC4ForMatMul_A = NEON_MNNPackC4ForMatMul_A_BF16;
    gInstance->MNNPackedMatMul = NEON_MNNPackedMatMul_BF16;
    gInstance->MNNPackedMatMulRemain = NEON_MNNPackedMatMulRemain_BF16;
#ifdef __aarch64__
    const MNNCPUInfo& gCPUInfo = *MNNGetCPUInfo();
    gInstance->supportFp16arith = gCPUInfo.fp16arith;
    gInstance->supportSDot = gCPUInfo.dot;
    gInstance->supportI8mm = gCPUInfo.i8mm;
    if (gInstance->supportI8mm) {
        gInstance->MNNPackForMatMul_B = ARMV86_MNNPackForMatMul_B_BF16;
        gInstance->MNNPackC4ForMatMul_A = ARMV86_MNNPackC4ForMatMul_A_BF16;
        gInstance->MNNGetMatMulPackMode = ARMV86_MNNGetMatMulPackMode_BF16;
        gInstance->MNNPackedMatMul = ARMV86_MNNPackedMatMul_BF16;
        gInstance->MNNPackedMatMulRemain = ARMV86_MNNPackedMatMulRemain_BF16;
    }
#endif
    gInstance->MNNPackedMatMul_int4 = nullptr;
    gInstance->MNNPackedMatMul_int8 = nullptr;
    // TODO: raw cpu version of bf16
    return true;
#endif
}

CoreFunctions* BF16Functions::get() {
    return gInstance;
}
};
