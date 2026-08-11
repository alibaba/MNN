#include <riscv_vector.h>
void MNNAbsMaxFP32_RVV(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack) {
    size_t stride = pack * realSize;
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    for (size_t i = 0; i < realSize; ++i) {
        vfloat32m1_t max_v = __riscv_vfmv_v_f_f32m1(0.f, vlmax);
        for (size_t c = 0; c < src_depth_quad; ++c) {
            const float* src = source + c * stride + i * pack;
            size_t n = pack;
            size_t vl;
            for (; n > 0; n -= vl, src += vl) {
                vl = __riscv_vsetvl_e32m1(n);
                vfloat32m1_t v = __riscv_vle32_v_f32m1(src, vl);
                vfloat32m1_t v_abs = __riscv_vfabs_v_f32m1(v, vl);
                max_v = __riscv_vfmax_vv_f32m1(max_v, v_abs, vl);
            }
        }
        vfloat32m1_t res = __riscv_vfredmax_vs_f32m1_f32m1(max_v, __riscv_vfmv_s_f_f32m1(0.0f, vlmax), vlmax);
        float absmaxVal = __riscv_vfmv_f_s_f32m1_f32(res);
        absmax[i] = absmaxVal;
    }
}
