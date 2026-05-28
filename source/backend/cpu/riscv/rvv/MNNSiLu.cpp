#include <riscv_vector.h>

extern "C" {

extern void MNNExp(float* dst, const float* src, float* offset, size_t dataSize);

void MNNSiLu(float* dst, const float* src, size_t dataSize) {
    float offset[4] = {-1.0f, 0.0f, 0.0f, 0.0f};
    MNNExp(dst, src, offset, dataSize);
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

void MNNSiLuLowp(float* dst, const float* src, size_t dataSize) {
    float offset[4] = {-1.0f, 0.0f, 0.0f, 0.0f};
    MNNExp(dst, src, offset, dataSize);
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

} // extern "C"
