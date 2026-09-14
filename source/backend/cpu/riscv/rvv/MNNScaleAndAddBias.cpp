#include "MNNRvvC4Functions.hpp"
#include <riscv_vector.h>

// The generic kernel lives in CommonOptFunction.cpp and is declared with C
// linkage there, so the call back into it needs the same linkage. Narrow planes
// cannot amortize the coefficient gathers, so they are handed back to it instead
// of being force-fed through a vector path.
extern "C" void MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha,
                                   size_t planeNumber, size_t biasNumber);

void MNNScaleAndAddBias_RVV(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                            size_t biasNumber) {
    if (planeNumber == 0 || biasNumber == 0) {
        return;
    }
    // Below this width the two coefficient gathers per plane cost more than the
    // vectorised arithmetic saves; the generic strided kernel is faster.
    if (planeNumber < 16) {
        MNNScaleAndAddBias(dst, src, bias, alpha, planeNumber, biasNumber);
        return;
    }

    const size_t maxVl = __riscv_vsetvlmax_e32m4();
    const vuint32m4_t channel = __riscv_vand_vx_u32m4(__riscv_vid_v_u32m4(maxVl), 3, maxVl);
    const size_t count = planeNumber * 4;
    for (size_t z = 0; z < biasNumber; ++z) {
        const vfloat32m1_t alphaC4 = __riscv_vle32_v_f32m1(alpha + 4 * z, 4);
        const vfloat32m1_t biasC4 = __riscv_vle32_v_f32m1(bias + 4 * z, 4);
        const vfloat32m4_t alphaRepeated =
            __riscv_vrgather_vv_f32m4(__riscv_vlmul_ext_v_f32m1_f32m4(alphaC4), channel, maxVl);
        const vfloat32m4_t biasRepeated =
            __riscv_vrgather_vv_f32m4(__riscv_vlmul_ext_v_f32m1_f32m4(biasC4), channel, maxVl);
        const float* srcZ = src + z * count;
        float* dstZ = dst + z * count;
        size_t remaining = count;
        while (remaining > 0) {
            // Bound AVL by VLMAX so VL stays a multiple of four, including the tail.
            const size_t vl = __riscv_vsetvl_e32m4(remaining < maxVl ? remaining : maxVl);
            vfloat32m4_t data = __riscv_vle32_v_f32m4(srcZ, vl);
            data = __riscv_vfmul_vv_f32m4(data, alphaRepeated, vl);
            data = __riscv_vfadd_vv_f32m4(data, biasRepeated, vl);
            __riscv_vse32_v_f32m4(dstZ, data, vl);
            srcZ += vl;
            dstZ += vl;
            remaining -= vl;
        }
    }
}
