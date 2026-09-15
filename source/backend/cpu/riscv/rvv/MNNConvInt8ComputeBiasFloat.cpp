#include <riscv_vector.h>
#include <cstddef>
#include <cstdint>

void MNNConvInt8ComputeBiasFloat_RVV(float* dst, const int32_t* bias, const float* weightScale, float inputScale,
                                     float outputScale, size_t size) {
    // Operation order must match the scalar path in CPUConvolution.cpp:
    //     bias * weightScale * inputScale / outputScale
    // The ratio is deliberately not precomputed: folding inputScale/outputScale
    // into a single ratio rounds once up front and can differ in the last bit
    // from the two-step form the scalar path uses. Multiplying by 1.0f is also
    // not short-circuited, because inputScale == outputScale does not make the
    // two extra operations a no-op in floating point.
    const bool needScale = (inputScale != 0.0f && outputScale != 0.0f);
    size_t offset = 0;
    while (offset < size) {
        const size_t vl = __riscv_vsetvl_e32m8(size - offset);
        const vint32m8_t biasValue = __riscv_vle32_v_i32m8(bias + offset, vl);
        vfloat32m8_t value = __riscv_vfcvt_f_x_v_f32m8(biasValue, vl);
        const vfloat32m8_t scale = __riscv_vle32_v_f32m8(weightScale + offset, vl);
        value = __riscv_vfmul_vv_f32m8(value, scale, vl);
        if (needScale) {
            value = __riscv_vfmul_vf_f32m8(value, inputScale, vl);
            value = __riscv_vfdiv_vf_f32m8(value, outputScale, vl);
        }
        __riscv_vse32_v_f32m8(dst + offset, value, vl);
        offset += vl;
    }
}
