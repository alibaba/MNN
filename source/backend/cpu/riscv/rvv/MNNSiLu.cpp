#include <riscv_vector.h>
#include <cstring>
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

template <bool low>
static inline vfloat32m2_t siluVectorM2(vfloat32m2_t x, size_t vl) {
    auto z = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(__riscv_vfneg_v_f32m2(x, vl), -87, vl), 87, vl);
    auto n = __riscv_vfcvt_x_f_v_i32m2(__riscv_vfmul_vf_f32m2(z, 1.4426950408889634f, vl), vl);
    auto r = __riscv_vfnmsac_vf_f32m2(z, 0.6931471805599453f, __riscv_vfcvt_f_x_v_f32m2(n, vl), vl);
    auto y = __riscv_vfmv_v_f_f32m2(1.f / 720.f, vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, __riscv_vfmv_v_f_f32m2(1.f / 120.f, vl), vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, __riscv_vfmv_v_f_f32m2(1.f / 24.f, vl), vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, __riscv_vfmv_v_f_f32m2(1.f / 6.f, vl), vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, __riscv_vfmv_v_f_f32m2(.5f, vl), vl);
    const auto one = __riscv_vfmv_v_f_f32m2(1, vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, one, vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, one, vl);
    auto power = __riscv_vreinterpret_v_i32m2_f32m2(__riscv_vsll_vx_i32m2(__riscv_vadd_vx_i32m2(n, 127, vl), 23, vl));
    auto denom = __riscv_vfmacc_vv_f32m2(one, y, power, vl);
    auto inv = __riscv_vfrec7_v_f32m2(denom, vl);
    auto two = __riscv_vfmv_v_f_f32m2(2, vl);
    inv = __riscv_vfmul_vv_f32m2(inv, __riscv_vfnmsac_vv_f32m2(two, denom, inv, vl), vl);
    if (!low)
        inv = __riscv_vfmul_vv_f32m2(inv, __riscv_vfnmsac_vv_f32m2(two, denom, inv, vl), vl);
    auto result = __riscv_vfmul_vv_f32m2(x, inv, vl);
    return __riscv_vfmerge_vfm_f32m2(result, -0.0f, __riscv_vmflt_vf_f32m2_b16(x, -87.f, vl), vl);
}

// Stream four packed channels with one unit-stride segment load per step.
template <bool low>
static void conv4SiLuC4Impl(float* out, const float* input, float* state, const float* weight, size_t len) {
    for (size_t row = 0; row < len;) {
        const auto vl = __riscv_vsetvl_e32m2(len - row);
        auto seg = __riscv_vlseg4e32_v_f32m2x4(input + row * 4, vl);
        auto runChannel = [&](size_t c, vfloat32m2_t x) {
            const auto w = weight + c * 4;
            const auto history = state + c * 3;
            const float* ptr = input + row * 4 + c;
            const float h2 = row >= 1 ? ptr[-4] : history[2];
            const float h1 = row >= 2 ? ptr[-8] : history[1 + row];
            const float h0 = row >= 3 ? ptr[-12] : history[row];
            auto prev1 = __riscv_vfslide1up_vf_f32m2(x, h2, vl);
            auto prev2 = __riscv_vfslide1up_vf_f32m2(prev1, h1, vl);
            auto prev3 = __riscv_vfslide1up_vf_f32m2(prev2, h0, vl);
            auto sum = __riscv_vfmul_vf_f32m2(prev3, w[0], vl);
            sum = __riscv_vfmacc_vf_f32m2(sum, w[1], prev2, vl);
            sum = __riscv_vfmacc_vf_f32m2(sum, w[2], prev1, vl);
            sum = __riscv_vfmacc_vf_f32m2(sum, w[3], x, vl);
            __riscv_vse32_v_f32m2(out + c * len + row, siluVectorM2<low>(sum, vl), vl);
        };
        runChannel(0, __riscv_vget_v_f32m2x4_f32m2(seg, 0));
        runChannel(1, __riscv_vget_v_f32m2x4_f32m2(seg, 1));
        runChannel(2, __riscv_vget_v_f32m2x4_f32m2(seg, 2));
        runChannel(3, __riscv_vget_v_f32m2x4_f32m2(seg, 3));
        row += vl;
    }
    for (size_t c = 0; c < 4; ++c) {
        const auto samples = input + c;
        auto history = state + c * 3;
        for (size_t i = 0; i < 3; ++i)
            history[i] = len + i < 3 ? history[len + i] : samples[(len + i - 3) * 4];
    }
}

void MNNConv4SiLuC4_RVV(float* out, const float* input, float* state, const float* weight, size_t len, bool low) {
    if (low)
        conv4SiLuC4Impl<true>(out, input, state, weight, len);
    else
        conv4SiLuC4Impl<false>(out, input, state, weight, len);
}

// Causal K=4 depthwise conv + SiLU in NC4HW4 [len, 4] layout using e32m2 without
// vrgather or vslideup (pre-tiled 4-channel weights in wBuf[256], zero vector spills).
template <bool low>
static void conv4SiLuC4SweepImpl(float* out, const float* input, const float* history, const float* weight, size_t len,
                                 bool isFirstTile, float* wBuf) {
    for (int tap = 0; tap < 4; ++tap) {
        uint32_t a, b, c, d;
        ::memcpy(&a, weight + tap, 4);
        ::memcpy(&b, weight + 4 + tap, 4);
        ::memcpy(&c, weight + 8 + tap, 4);
        ::memcpy(&d, weight + 12 + tap, 4);
        const uint64_t lo = (uint64_t(b) << 32) | a;
        const uint64_t hi = (uint64_t(d) << 32) | c;
        uint64_t* dst64 = reinterpret_cast<uint64_t*>(wBuf + tap * 64);
        for (int i = 0; i < 16; ++i) {
            dst64[2 * i] = lo;
            dst64[2 * i + 1] = hi;
        }
    }
    const size_t maxVl = __riscv_vsetvlmax_e32m2();
    const size_t wVl = maxVl < 64 ? maxVl : 64;
    auto vw0 = __riscv_vle32_v_f32m2(wBuf + 0, wVl);
    auto vw1 = __riscv_vle32_v_f32m2(wBuf + 64, wVl);
    auto vw2 = __riscv_vle32_v_f32m2(wBuf + 128, wVl);
    auto vw3 = __riscv_vle32_v_f32m2(wBuf + 192, wVl);
    size_t startOff = 0;
    if (isFirstTile) {
        const size_t head = len < 3 ? len : 3;
        float pad[6 * 4];
        for (size_t c = 0; c < 4; ++c) {
            pad[0 * 4 + c] = history[c * 3 + 0];
            pad[1 * 4 + c] = history[c * 3 + 1];
            pad[2 * 4 + c] = history[c * 3 + 2];
        }
        for (size_t r = 0; r < head; ++r) {
            for (size_t c = 0; c < 4; ++c) {
                pad[(3 + r) * 4 + c] = input[r * 4 + c];
            }
        }
        float tmp[12];
        for (size_t r = 0; r < head; ++r) {
            for (size_t c = 0; c < 4; ++c) {
                const auto w = weight + c * 4;
                tmp[r * 4 + c] = pad[r * 4 + c] * w[0] + pad[(r + 1) * 4 + c] * w[1] +
                                 pad[(r + 2) * 4 + c] * w[2] + pad[(r + 3) * 4 + c] * w[3];
            }
        }
        for (size_t off = 0; off < head * 4;) {
            const size_t vl = __riscv_vsetvl_e32m2(head * 4 - off);
            __riscv_vse32_v_f32m2(out + off, siluVectorM2<low>(__riscv_vle32_v_f32m2(tmp + off, vl), vl), vl);
            off += vl;
        }
        startOff = head * 4;
    }
    const size_t total = len * 4;
    for (size_t off = startOff; off < total;) {
        const size_t rem = total - off;
        const size_t vl = __riscv_vsetvl_e32m2(rem < 64 ? rem : 64);
        auto sum = __riscv_vfmul_vv_f32m2(__riscv_vle32_v_f32m2(input + off - 12, vl), vw0, vl);
        sum = __riscv_vfmacc_vv_f32m2(sum, __riscv_vle32_v_f32m2(input + off - 8, vl), vw1, vl);
        sum = __riscv_vfmacc_vv_f32m2(sum, __riscv_vle32_v_f32m2(input + off - 4, vl), vw2, vl);
        sum = __riscv_vfmacc_vv_f32m2(sum, __riscv_vle32_v_f32m2(input + off, vl), vw3, vl);
        __riscv_vse32_v_f32m2(out + off, siluVectorM2<low>(sum, vl), vl);
        off += vl;
    }
}

void MNNConv4SiLuC4Sweep_RVV(float* out, const float* input, const float* history, const float* weight, size_t len,
                             bool isFirstTile, bool low, float* wBuf) {
    if (low)
        conv4SiLuC4SweepImpl<true>(out, input, history, weight, len, isFirstTile, wBuf);
    else
        conv4SiLuC4SweepImpl<false>(out, input, history, weight, len, isFirstTile, wBuf);
}

void MNNConv4SiLuC4KeepC4_RVV(float* out, const float* input, float* state, const float* weight, size_t len, bool low) {
    alignas(64) float wBuf[256];
    MNNConv4SiLuC4Sweep_RVV(out, input, state, weight, len, true, low, wBuf);
    for (size_t c = 0; c < 4; ++c) {
        const auto samples = input + c;
        auto history = state + c * 3;
        for (size_t i = 0; i < 3; ++i)
            history[i] = len + i < 3 ? history[len + i] : samples[(len + i - 3) * 4];
    }
}
