#include <cstddef>
#include <riscv_vector.h>

void MNNPackInt8C2_RVV(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    const ptrdiff_t srcAreaStride = static_cast<ptrdiff_t>(areaOffset[0]);
    const ptrdiff_t dstAreaStride = static_cast<ptrdiff_t>(areaOffset[1]);
    const size_t depthC2 = (depth + 1) / 2;

    for (size_t z = 0; z < depthC2; ++z) {
        const size_t cBase = z * 2;
        float* dstZ = dst + static_cast<ptrdiff_t>(z) * dstAreaStride * 2;
        size_t valid = depth - cBase;
        if (valid > 2)
            valid = 2;

        const float* src0 = src + static_cast<ptrdiff_t>(cBase) * srcAreaStride;
        size_t x = 0;
        while (x < area) {
            const size_t vl = __riscv_vsetvl_e32m4(area - x);
            vfloat32m4x2_t values = __riscv_vcreate_v_f32m4x2(
                __riscv_vle32_v_f32m4(src0 + x, vl),
                valid == 2 ? __riscv_vle32_v_f32m4(src0 + srcAreaStride + x, vl) : __riscv_vfmv_v_f_f32m4(0.0f, vl));
            __riscv_vsseg2e32_v_f32m4x2(dstZ + 2 * x, values, vl);
            x += vl;
        }
    }
}
