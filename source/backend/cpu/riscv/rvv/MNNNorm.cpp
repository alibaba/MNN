#include <riscv_vector.h>
#include <math.h>

extern "C" {

void MNNNorm(float* dst, const float* src, const float* gamma, const float* beta, float epsilon, size_t size,
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

} // extern "C"
