#include "MNNRvvPrefill.hpp"
#include <riscv_vector.h>
#include <math.h>

void MNNNorm_RVV(float* dst, const float* src, const float* gamma, const float* beta, float epsilon, size_t size,
                 bool RMSNorm) {
    float mean = 0.0f;
    if (!RMSNorm) {
        size_t n = size;
        const float* p = src;
        vfloat32m1_t vAcc = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t v = __riscv_vle32_v_f32m4(p, vl);
            vAcc = __riscv_vfredusum_vs_f32m4_f32m1(v, vAcc, vl);
            p += vl;
            n -= vl;
        }
        mean = __riscv_vfmv_f_s_f32m1_f32(vAcc) / (float)size;
    }

    // Phase 1: square_sum reduction
    size_t n = size;
    const float* srcPtr = src;
    vfloat32m1_t vSqSum = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    if (RMSNorm) {
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t vSrc = __riscv_vle32_v_f32m4(srcPtr, vl);
            vfloat32m4_t vSq = __riscv_vfmul_vv_f32m4(vSrc, vSrc, vl);
            vSqSum = __riscv_vfredusum_vs_f32m4_f32m1(vSq, vSqSum, vl);
            srcPtr += vl;
            n -= vl;
        }
    } else {
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t vSrc = __riscv_vle32_v_f32m4(srcPtr, vl);
            vfloat32m4_t vDiff = __riscv_vfsub_vf_f32m4(vSrc, mean, vl);
            vfloat32m4_t vSq = __riscv_vfmul_vv_f32m4(vDiff, vDiff, vl);
            vSqSum = __riscv_vfredusum_vs_f32m4_f32m1(vSq, vSqSum, vl);
            srcPtr += vl;
            n -= vl;
        }
    }
    float square_sum = __riscv_vfmv_f_s_f32m1_f32(vSqSum);

    // Phase 2: rsqrt (scalar)
    float variable = 1.0f / sqrtf(square_sum / (float)size + epsilon);

    // Phase 3: normalize + scale
    n = size;
    srcPtr = src;
    float* dstPtr = dst;
    if (RMSNorm && gamma && !beta) {
        const float* gammaPtr = gamma;
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t vSrc = __riscv_vle32_v_f32m4(srcPtr, vl);
            vfloat32m4_t vG = __riscv_vle32_v_f32m4(gammaPtr, vl);
            vfloat32m4_t vNorm = __riscv_vfmul_vf_f32m4(vSrc, variable, vl);
            vNorm = __riscv_vfmul_vv_f32m4(vNorm, vG, vl);
            __riscv_vse32_v_f32m4(dstPtr, vNorm, vl);
            srcPtr += vl;
            gammaPtr += vl;
            dstPtr += vl;
            n -= vl;
        }
    } else if (gamma && beta) {
        const float* gammaPtr = gamma;
        const float* betaPtr = beta;
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t vSrc = __riscv_vle32_v_f32m4(srcPtr, vl);
            vfloat32m4_t vG = __riscv_vle32_v_f32m4(gammaPtr, vl);
            vfloat32m4_t vB = __riscv_vle32_v_f32m4(betaPtr, vl);
            vfloat32m4_t vNorm = __riscv_vfsub_vf_f32m4(vSrc, mean, vl);
            vNorm = __riscv_vfmul_vf_f32m4(vNorm, variable, vl);
            vNorm = __riscv_vfmadd_vv_f32m4(vNorm, vG, vB, vl);
            __riscv_vse32_v_f32m4(dstPtr, vNorm, vl);
            srcPtr += vl;
            gammaPtr += vl;
            betaPtr += vl;
            dstPtr += vl;
            n -= vl;
        }
    } else if (gamma) {
        const float* gammaPtr = gamma;
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t vSrc = __riscv_vle32_v_f32m4(srcPtr, vl);
            vfloat32m4_t vG = __riscv_vle32_v_f32m4(gammaPtr, vl);
            vfloat32m4_t vNorm = __riscv_vfsub_vf_f32m4(vSrc, mean, vl);
            vNorm = __riscv_vfmul_vf_f32m4(vNorm, variable, vl);
            vNorm = __riscv_vfmul_vv_f32m4(vNorm, vG, vl);
            __riscv_vse32_v_f32m4(dstPtr, vNorm, vl);
            srcPtr += vl;
            gammaPtr += vl;
            dstPtr += vl;
            n -= vl;
        }
    } else {
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t vSrc = __riscv_vle32_v_f32m4(srcPtr, vl);
            vfloat32m4_t vNorm = __riscv_vfsub_vf_f32m4(vSrc, mean, vl);
            vNorm = __riscv_vfmul_vf_f32m4(vNorm, variable, vl);
            __riscv_vse32_v_f32m4(dstPtr, vNorm, vl);
            srcPtr += vl;
            dstPtr += vl;
            n -= vl;
        }
    }
}

// Each lane accumulates one NC4HW4 lane across channels. Adjacent four-lane
// groups are independent tokens, including when the residual is an output.
void MNNRmsNormC4_RVV(float* out, float* sum, const float* src, const float* residual, const float* gamma,
                      const float* beta, float epsilon, size_t rows, size_t cols, int id, int threads) {
    MNN_ASSERT(threads > 0);
    if (threads <= 0) return;
    const size_t tile = __riscv_vsetvlmax_e32m4() / 4;
    for (size_t row = id * tile; row < rows; row += threads * tile) {
        const size_t count = (rows - row < tile ? rows - row : tile) * 4;
        const auto lane = __riscv_vid_v_u32m4(count);
        auto squared = __riscv_vfmv_v_f_f32m4(0, count);
        for (size_t c = 0; c < cols; c += 4) {
            const size_t offset = c * rows + row * 4;
            auto x = __riscv_vle32_v_f32m4(src + offset, count);
            if (residual) {
                x = __riscv_vfadd_vv_f32m4(x, __riscv_vle32_v_f32m4(residual + offset, count), count);
                __riscv_vse32_v_f32m4(sum + offset, x, count);
            }
            squared = __riscv_vfmacc_vv_f32m4(squared, x, x, count);
        }
        squared = __riscv_vfadd_vv_f32m4(squared, __riscv_vslidedown_vx_f32m4(squared, 1, count), count);
        squared = __riscv_vfadd_vv_f32m4(squared, __riscv_vslidedown_vx_f32m4(squared, 2, count), count);
        auto rowSum = __riscv_vrgather_vv_f32m4(squared, __riscv_vand_vx_u32m4(lane, ~3u, count), count);
        auto denom = __riscv_vfadd_vf_f32m4(__riscv_vfdiv_vf_f32m4(rowSum, float(cols), count), epsilon, count);
        auto inv = __riscv_vfrdiv_vf_f32m4(__riscv_vfsqrt_v_f32m4(denom, count), 1, count);
        auto chan = __riscv_vand_vx_u32m4(lane, 3, count);
        const float* base = residual ? sum : src;
        for (size_t c = 0; c < cols; c += 4) {
            const size_t offset = c * rows + row * 4;
            auto x = __riscv_vfmul_vv_f32m4(__riscv_vle32_v_f32m4(base + offset, count), inv, count);
            if (gamma && beta) {
                auto g = __riscv_vrgather_vv_f32m4(__riscv_vle32_v_f32m4(gamma + c, 4), chan, count);
                auto b = __riscv_vrgather_vv_f32m4(__riscv_vle32_v_f32m4(beta + c, 4), chan, count);
                x = __riscv_vfmacc_vv_f32m4(b, x, g, count);
            }
            __riscv_vse32_v_f32m4(out + offset, x, count);
        }
    }
}

namespace MNN {
void MNNRvvInitPackedNorm(CoreFunctions* core) {
    static auto* const registered = core;
    MNN_ASSERT(registered == core);
    static const auto fallback = core->MNNNormPacked;
    core->MNNNormPacked = [](float* out, float* sum, const float* src, const float* residual, const float* gamma,
                             const float* beta, float epsilon, size_t rows, size_t cols, bool rms, int id, int threads) {
        if (rms && rows >= 16 && cols >= 4 && cols % 4 == 0) {
            MNNRmsNormC4_RVV(out, sum, src, residual, gamma, beta, epsilon, rows, cols, id, threads);
            return;
        }
        fallback(out, sum, src, residual, gamma, beta, epsilon, rows, cols, rms, id, threads);
    };
}
} // namespace MNN
