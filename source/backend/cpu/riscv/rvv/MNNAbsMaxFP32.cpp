#include <riscv_vector.h>
#include <cstddef>

void MNNAbsMaxFP32_RVV(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack) {
    if (realSize == 1) {
        // Channels are contiguous for a single position. Vectorizing positions here
        // would leave only one active lane and repeat strided loads for every channel.
        const size_t maxVl = __riscv_vsetvlmax_e32m8();
        vfloat32m8_t maximum = __riscv_vfmv_v_f_f32m8(0.0f, maxVl);
        const size_t count = src_depth_quad * static_cast<size_t>(pack);
        for (size_t offset = 0; offset < count;) {
            const size_t vl = __riscv_vsetvl_e32m8(count - offset);
            const vfloat32m8_t value = __riscv_vle32_v_f32m8(source + offset, vl);
            // Preserve maxima in inactive lanes of the final partial vector.
            maximum = __riscv_vfmax_vv_f32m8_tu(maximum, maximum, __riscv_vfabs_v_f32m8(value, vl), vl);
            offset += vl;
        }
        const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        absmax[0] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(maximum, zero, maxVl));
        return;
    }
    const size_t planeStride = static_cast<size_t>(pack) * realSize;
    const ptrdiff_t srcStrideBytes = static_cast<ptrdiff_t>(pack) * sizeof(float);

    for (size_t i = 0; i < realSize;) {
        const size_t vl = __riscv_vsetvl_e32m8(realSize - i);
        const vfloat32m8_t zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        __riscv_vse32_v_f32m8(absmax + i, zero, vl);
        i += vl;
    }

    for (size_t c = 0; c < src_depth_quad; ++c) {
        const float* srcC = source + c * planeStride;
        for (size_t i = 0; i < realSize;) {
            const size_t vl = __riscv_vsetvl_e32m8(realSize - i);
            const float* src = srcC + i * pack;
            vfloat32m8_t maxValue = __riscv_vle32_v_f32m8(absmax + i, vl);
            for (int k = 0; k < pack; ++k) {
                const vfloat32m8_t value = __riscv_vlse32_v_f32m8(src + k, srcStrideBytes, vl);
                maxValue = __riscv_vfmax_vv_f32m8(maxValue, __riscv_vfabs_v_f32m8(value, vl), vl);
            }
            __riscv_vse32_v_f32m8(absmax + i, maxValue, vl);
            i += vl;
        }
    }
}
