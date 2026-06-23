#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

void MNNQuantScaleFP32_RVV(float* absmax, float* quant_scale, float* dequant_scale, size_t thread, size_t batch) {
    size_t vl;
    for (size_t i = 0; i < batch; i += vl) {
        vl = __riscv_vsetvl_e32m4(batch - i);

        vfloat32m4_t v_max = __riscv_vfmv_v_f_f32m4(0.0f, vl);

        for (size_t t = 0; t < thread; ++t) {
            vfloat32m4_t v_val = __riscv_vle32_v_f32m4(absmax + t * batch + i, vl);
            v_max = __riscv_vfmax_vv_f32m4(v_max, v_val, vl);
        }

        vbool8_t mask = __riscv_vmflt_vf_f32m4_b8(v_max, 1e-7f, vl);

        vfloat32m4_t v_127 = __riscv_vfmv_v_f_f32m4(127.0f, vl);
        vfloat32m4_t v_qscale = __riscv_vfdiv_vv_f32m4(v_127, v_max, vl);
        vfloat32m4_t v_dqscale = __riscv_vfdiv_vf_f32m4(v_max, 127.0f, vl);

        v_qscale = __riscv_vfmerge_vfm_f32m4(v_qscale, 1.0f, mask, vl);
        v_dqscale = __riscv_vfmerge_vfm_f32m4(v_dqscale, 1.0f, mask, vl);

        __riscv_vse32_v_f32m4(quant_scale + i, v_qscale, vl);
        __riscv_vse32_v_f32m4(dequant_scale + i, v_dqscale, vl);
    }
}
