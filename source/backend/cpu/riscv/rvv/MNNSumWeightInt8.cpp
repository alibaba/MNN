#include <riscv_vector.h>
#include <cstring>
#include <cstdint>

void MNNSumWeightInt8_RVV(float* kernelsum, int8_t* source, size_t outside, size_t reduceAxis, size_t hP, size_t lP) {
    size_t inside = hP * lP;
    size_t stride0 = inside * reduceAxis;
    for (size_t i = 0; i < outside; ++i) {
        float* dst = kernelsum + i * hP;
        memset(dst, 0, hP * sizeof(float));
        for (size_t j = 0; j < reduceAxis; ++j) {
            for (size_t k = 0; k < hP; ++k) {
                const int8_t* src_ptr = source + i * stride0 + j * inside + k * lP;
                size_t x = 0;
                int sum = 0;
                while (x < lP) {
                    size_t vl = __riscv_vsetvl_e8m1(lP - x);
                    vint8m1_t v8 = __riscv_vle8_v_i8m1(src_ptr + x, vl);
                    vint16m2_t v16 = __riscv_vwcvt_x_x_v_i16m2(v8, vl);
                    vint32m4_t v32 = __riscv_vwcvt_x_x_v_i32m4(v16, vl);
                    vint32m1_t vzero = __riscv_vmv_s_x_i32m1(0, vl);
                    vint32m1_t vres = __riscv_vredsum_vs_i32m4_i32m1(v32, vzero, vl);
                    sum += __riscv_vmv_x_s_i32m1_i32(vres);
                    x += vl;
                }
                dst[k] += (float)sum;
            }
        }
    }
}
