#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include "GemmInt8Macro.h"

#define mnn_mm512_dpbusds_epi32(x, y, z) mnn_mm512_dpbusds_epi32_replace_fast(x, y, z, one)
static inline __m512i mnn_mm512_dpbusds_epi32_replace_fast(__m512i dst, __m512i src, __m512i W0,  __m512i oneValue) {
    auto s0 = _mm512_maddubs_epi16(src, W0);
    auto p0 = _mm512_madd_epi16(s0, oneValue);
    dst = _mm512_add_epi32(dst, p0);
    return dst;
}

#define MATMULCOREFUNC_NAME _AVX512_NO_VNNI_4_4_64_7bit
#include "Matmul_4_4_64.inl"