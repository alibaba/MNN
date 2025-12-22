#include <riscv_vector.h>
#include <cfloat>

void MNNSoftmax(float *dest, const float *source, size_t size) {
    size_t n = size;
    const float *sourcePtr = source;
    float *destPtr = dest;
    float maxValue = -FLT_MAX;
    vfloat32m1_t maxVecValue = __riscv_vfmv_s_f_f32m1(maxValue, 1);

    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vSrc = __riscv_vle32_v_f32m8(sourcePtr, vl);
        maxVecValue = __riscv_vfredmax_vs_f32m8_f32m1(vSrc, maxVecValue, vl);
        sourcePtr += vl;
        n -= vl;
    }

    maxValue = __riscv_vfmv_f_s_f32m1_f32(maxVecValue);
    const float param = 0.6931471805599453f;
    const float xLimit = 87.0f;
    float sumValue = 0.f;
    vfloat32m1_t sumVecValue = __riscv_vfmv_s_f_f32m1(sumValue, 1);
    n = size;
    sourcePtr = source;
    destPtr = dest;

    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vA = __riscv_vle32_v_f32m8(sourcePtr, vl);
        vA = __riscv_vfsub_vf_f32m8(vA, maxValue, vl);
        vA = __riscv_vfmax_vf_f32m8(vA, -xLimit, vl);
        vA = __riscv_vfmin_vf_f32m8(vA, xLimit, vl);

        vfloat32m8_t vB = __riscv_vfdiv_vf_f32m8(vA, param, vl);
        vint32m8_t vBI = __riscv_vfcvt_x_f_v_i32m8(vB, vl);

        vfloat32m8_t vC = __riscv_vreinterpret_v_i32m8_f32m8(
            __riscv_vsll_vx_i32m8(
                __riscv_vadd_vx_i32m8(vBI, 127, vl), 23, vl));

        vB = __riscv_vfcvt_f_x_v_f32m8(vBI, vl);
        vB = __riscv_vfnmsub_vf_f32m8(vB, param, vA, vl);

        vA = __riscv_vfmv_v_f_f32m8(1.0f / 120.0f, vl);
        vA = __riscv_vfmul_vv_f32m8(vA, vB, vl);
        vA = __riscv_vfadd_vf_f32m8(vA, 1.0f / 24.0f, vl);
        vA = __riscv_vfmul_vv_f32m8(vA, vB, vl);
        vA = __riscv_vfadd_vf_f32m8(vA, 1.0f / 6.0f, vl);
        vA = __riscv_vfmul_vv_f32m8(vA, vB, vl);
        vA = __riscv_vfadd_vf_f32m8(vA, 0.5f, vl);
        vA = __riscv_vfmul_vv_f32m8(vA, vB, vl);
        vA = __riscv_vfadd_vf_f32m8(vA, 1.0f, vl);
        vA = __riscv_vfmul_vv_f32m8(vA, vB, vl);
        vA = __riscv_vfadd_vf_f32m8(vA, 1.0f, vl);

        vA = __riscv_vfmul_vv_f32m8(vC, vA, vl);
        __riscv_vse32_v_f32m8(destPtr, vA, vl);
        sumVecValue = __riscv_vfredosum_vs_f32m8_f32m1(vA, sumVecValue, vl);

        sourcePtr += vl;
        destPtr += vl;
        n -= vl;
    }

    sumValue = __riscv_vfmv_f_s_f32m1_f32(sumVecValue);
    float sumInv = 1.0f / sumValue;
    n = size;
    destPtr = dest;

    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vDest = __riscv_vle32_v_f32m8(destPtr, vl);
        vDest = __riscv_vfmul_vf_f32m8(vDest, sumInv, vl);
        __riscv_vse32_v_f32m8(destPtr, vDest, vl);
        destPtr += vl;
        n -= vl;
    }
}
