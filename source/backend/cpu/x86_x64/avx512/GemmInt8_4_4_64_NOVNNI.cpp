#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include "GemmInt8Macro.h"

#define mnn_mm512_dpbusds_epi32(x, y, z) mnn_mm512_dpbusds_epi32_replace(x, y, z, one)
static inline __m512i mnn_mm512_dpbusds_epi32_replace(__m512i dst, __m512i src, __m512i W0,  __m512i oneValue) {
    auto w0 = _mm512_mask_set1_epi8(W0, 0x5555555555555555, 0);
    auto w1 = _mm512_mask_set1_epi8(W0, 0xaaaaaaaaaaaaaaaa, 0);
    auto s0 = _mm512_maddubs_epi16(src, w0);
    auto s1 = _mm512_maddubs_epi16(src, w1);
    auto p0 = _mm512_madd_epi16(s0, oneValue);
    auto p1 = _mm512_madd_epi16(s1, oneValue);
    dst = _mm512_add_epi32(dst, p0);
    dst = _mm512_add_epi32(dst, p1);
    return dst;
}

#define MATMULCOREFUNC_NAME _AVX512_NO_VNNI_4_4_64
#define MATMULCOREFUNC_NAME_W4 _AVX512_NO_VNNI_4_4_64_w4
#include "Matmul_4_4_64.inl"