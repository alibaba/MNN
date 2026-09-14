#include "MNNRvvC4Functions.hpp"
#include <riscv_vector.h>

void MNNReluWithSlopeChannel_RVV(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    if (sizeQuad == 0 || depthQuad == 0) {
        return;
    }

    // A single C4 needs no coefficient replication.
    if (sizeQuad == 1) {
        const size_t vl = __riscv_vsetvl_e32m1(4);
        for (size_t z = 0; z < depthQuad; ++z) {
            const vfloat32m1_t slopeC4 = __riscv_vle32_v_f32m1(slope + 4 * z, vl);
            vfloat32m1_t data = __riscv_vle32_v_f32m1(src + 4 * z, vl);
            const vbool32_t negative = __riscv_vmflt_vf_f32m1_b32(data, 0.0f, vl);
            data = __riscv_vfmul_vv_f32m1_mu(negative, data, data, slopeC4, vl);
            __riscv_vse32_v_f32m1(dst + 4 * z, data, vl);
        }
        return;
    }

    const size_t maxVl = __riscv_vsetvlmax_e32m4();
    const vuint32m4_t channel = __riscv_vand_vx_u32m4(__riscv_vid_v_u32m4(maxVl), 3, maxVl);
    const size_t count = sizeQuad * 4;
    for (size_t z = 0; z < depthQuad; ++z) {
        const vfloat32m1_t slopeC4 = __riscv_vle32_v_f32m1(slope + 4 * z, 4);
        const vfloat32m4_t slopeRepeated =
            __riscv_vrgather_vv_f32m4(__riscv_vlmul_ext_v_f32m1_f32m4(slopeC4), channel, maxVl);
        const float* srcZ = src + z * count;
        float* dstZ = dst + z * count;
        size_t remaining = count;
        while (remaining > 0) {
            // Bound AVL by VLMAX so every iteration starts at the C4 coefficient boundary.
            const size_t vl = __riscv_vsetvl_e32m4(remaining < maxVl ? remaining : maxVl);
            vfloat32m4_t data = __riscv_vle32_v_f32m4(srcZ, vl);
            const vbool8_t negative = __riscv_vmflt_vf_f32m4_b8(data, 0.0f, vl);
            // Only negative lanes multiply; signed zeros, positive values and NaNs remain unchanged.
            data = __riscv_vfmul_vv_f32m4_mu(negative, data, data, slopeRepeated, vl);
            __riscv_vse32_v_f32m4(dstZ, data, vl);
            srcZ += vl;
            dstZ += vl;
            remaining -= vl;
        }
    }
}
