#include <riscv_vector.h>
#include "../../compute/CommonOptFunction.h"

// Evaluate sigmoid in registers. The exponential uses the same ln(2)
// range reduction as MNNExp, without materializing or summing exp(-x).
template <bool low>
static inline vfloat32m4_t siluVector(vfloat32m4_t x, size_t vl) {
    auto z = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(__riscv_vfneg_v_f32m4(x, vl), -87, vl), 87, vl);
    auto n = __riscv_vfcvt_x_f_v_i32m4(__riscv_vfmul_vf_f32m4(z, 1.4426950408889634f, vl), vl);
    auto r = __riscv_vfnmsac_vf_f32m4(z, 0.6931471805599453f, __riscv_vfcvt_f_x_v_f32m4(n, vl), vl);
    auto y = __riscv_vfmv_v_f_f32m4(1.f / 720.f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, r, __riscv_vfmv_v_f_f32m4(1.f / 120.f, vl), vl);
    y = __riscv_vfmadd_vv_f32m4(y, r, __riscv_vfmv_v_f_f32m4(1.f / 24.f, vl), vl);
    y = __riscv_vfmadd_vv_f32m4(y, r, __riscv_vfmv_v_f_f32m4(1.f / 6.f, vl), vl);
    y = __riscv_vfmadd_vv_f32m4(y, r, __riscv_vfmv_v_f_f32m4(.5f, vl), vl);
    const auto one = __riscv_vfmv_v_f_f32m4(1, vl);
    y = __riscv_vfmadd_vv_f32m4(y, r, one, vl);
    y = __riscv_vfmadd_vv_f32m4(y, r, one, vl);
    auto power = __riscv_vreinterpret_v_i32m4_f32m4(__riscv_vsll_vx_i32m4(__riscv_vadd_vx_i32m4(n, 127, vl), 23, vl));
    auto denom = __riscv_vfmacc_vv_f32m4(one, y, power, vl);
    auto inv = __riscv_vfrec7_v_f32m4(denom, vl);
    auto two = __riscv_vfmv_v_f_f32m4(2, vl);
    inv = __riscv_vfmul_vv_f32m4(inv, __riscv_vfnmsac_vv_f32m4(two, denom, inv, vl), vl);
    if (!low)
        inv = __riscv_vfmul_vv_f32m4(inv, __riscv_vfnmsac_vv_f32m4(two, denom, inv, vl), vl);
    auto result = __riscv_vfmul_vv_f32m4(x, inv, vl);
    return __riscv_vfmerge_vfm_f32m4(result, -0.0f, __riscv_vmflt_vf_f32m4_b8(x, -87.f, vl), vl);
}

template <bool low>
static void mulSilu(float* dst, const float* src, const float* up, size_t size) {
    while (size > 0) {
        const size_t vl = __riscv_vsetvl_e32m4(size);
        auto x = __riscv_vle32_v_f32m4(src, vl);
        auto value = siluVector<low>(x, vl);
        value = __riscv_vfmul_vv_f32m4(value, __riscv_vle32_v_f32m4(up, vl), vl);
        up += vl;
        __riscv_vse32_v_f32m4(dst, value, vl);
        src += vl;
        dst += vl;
        size -= vl;
    }
}

void MNNSiLu_RVV(float* dst, const float* src, size_t dataSize) {
    float offset[4] = {-1.0f, 0.0f, 0.0f, 0.0f};
    ::MNNExp(dst, src, offset, dataSize);
    size_t n = dataSize;
    float* dstPtr = dst;
    const float* srcPtr = src;
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vExp = __riscv_vle32_v_f32m4(dstPtr, vl);
        vfloat32m4_t vSrc = __riscv_vle32_v_f32m4(srcPtr, vl);
        vfloat32m4_t vDenom = __riscv_vfadd_vf_f32m4(vExp, 1.0f, vl);
        vfloat32m4_t vRcp = __riscv_vfrec7_v_f32m4(vDenom, vl);
        // Newton-Raphson x2: rcp *= (2 - denom * rcp)
        vfloat32m4_t vErr = __riscv_vfmul_vv_f32m4(vDenom, vRcp, vl);
        vErr = __riscv_vfrsub_vf_f32m4(vErr, 2.0f, vl);
        vRcp = __riscv_vfmul_vv_f32m4(vRcp, vErr, vl);
        vErr = __riscv_vfmul_vv_f32m4(vDenom, vRcp, vl);
        vErr = __riscv_vfrsub_vf_f32m4(vErr, 2.0f, vl);
        vRcp = __riscv_vfmul_vv_f32m4(vRcp, vErr, vl);
        vfloat32m4_t vResult = __riscv_vfmul_vv_f32m4(vSrc, vRcp, vl);
        __riscv_vse32_v_f32m4(dstPtr, vResult, vl);
        dstPtr += vl;
        srcPtr += vl;
        n -= vl;
    }
}

void MNNSiLuLowp_RVV(float* dst, const float* src, size_t dataSize) {
    float offset[4] = {-1.0f, 0.0f, 0.0f, 0.0f};
    ::MNNExp(dst, src, offset, dataSize);
    size_t n = dataSize;
    float* dstPtr = dst;
    const float* srcPtr = src;
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vExp = __riscv_vle32_v_f32m4(dstPtr, vl);
        vfloat32m4_t vSrc = __riscv_vle32_v_f32m4(srcPtr, vl);
        vfloat32m4_t vDenom = __riscv_vfadd_vf_f32m4(vExp, 1.0f, vl);
        vfloat32m4_t vRcp = __riscv_vfrec7_v_f32m4(vDenom, vl);
        vfloat32m4_t vErr = __riscv_vfmul_vv_f32m4(vDenom, vRcp, vl);
        vErr = __riscv_vfrsub_vf_f32m4(vErr, 2.0f, vl);
        vRcp = __riscv_vfmul_vv_f32m4(vRcp, vErr, vl);
        vfloat32m4_t vResult = __riscv_vfmul_vv_f32m4(vSrc, vRcp, vl);
        __riscv_vse32_v_f32m4(dstPtr, vResult, vl);
        dstPtr += vl;
        srcPtr += vl;
        n -= vl;
    }
}

void MNNMulSiLu_RVV(float* dst, const float* gate, const float* up, size_t size, bool low) {
    if (low)
        mulSilu<true>(dst, gate, up, size);
    else
        mulSilu<false>(dst, gate, up, size);
}

// Stream four packed channels directly from the input. The three carry
// samples are the persistent causal-convolution state for this channel.
void MNNConv4SiLuC4_RVV(float* out, const float* input, float* state, const float* weight, size_t len) {
    for (size_t c = 0; c < 4; ++c) {
        auto samples = input + c;
        const auto w = weight + c * 4;
        auto history = state + c * 3;
        for (size_t row = 0; row < len;) {
            auto ptr = samples + row * 4;
            const auto vl = __riscv_vsetvl_e32m4(len - row);
            auto x = __riscv_vlse32_v_f32m4(ptr, 4 * sizeof(float), vl);
            auto prev1 = __riscv_vfslide1up_vf_f32m4(x, row ? ptr[-4] : history[2], vl);
            auto prev2 = __riscv_vfslide1up_vf_f32m4(prev1, row ? ptr[-8] : history[1], vl);
            auto prev3 = __riscv_vfslide1up_vf_f32m4(prev2, row ? ptr[-12] : history[0], vl);
            auto sum = __riscv_vfmul_vf_f32m4(prev3, w[0], vl);
            sum = __riscv_vfmacc_vf_f32m4(sum, w[1], prev2, vl);
            sum = __riscv_vfmacc_vf_f32m4(sum, w[2], prev1, vl);
            sum = __riscv_vfmacc_vf_f32m4(sum, w[3], x, vl);
            __riscv_vse32_v_f32m4(out + c * len + row, siluVector<false>(sum, vl), vl);
            row += vl;
        }
        for (size_t i = 0; i < 3; ++i)
            history[i] = len + i < 3 ? history[len + i] : samples[(len + i - 3) * 4];
    }
}
