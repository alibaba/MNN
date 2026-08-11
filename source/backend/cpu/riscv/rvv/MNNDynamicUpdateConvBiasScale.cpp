#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

void MNNDynamicUpdateConvBiasScale_RVV(float* newbias, float* oldbias, float* weightKernelSum, float* inputBias,
                                       size_t ocQuad) {
    int ocUp4 = 4 * ocQuad;
    float ib = inputBias[0];

    size_t vl;
    for (size_t i = 0; i < ocUp4; i += vl) {
        vl = __riscv_vsetvl_e32m4(ocUp4 - i);

        vfloat32m4_t v_old = __riscv_vle32_v_f32m4(oldbias + i, vl);
        vfloat32m4_t v_wks = __riscv_vle32_v_f32m4(weightKernelSum + i, vl);

        vfloat32m4_t v_new = __riscv_vfmacc_vf_f32m4(v_old, ib, v_wks, vl);

        __riscv_vse32_v_f32m4(newbias + i, v_new, vl);
    }
}
