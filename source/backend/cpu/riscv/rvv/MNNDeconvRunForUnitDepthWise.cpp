#include <riscv_vector.h>
#include "backend/cpu/compute/ConvOpt.h"
#include <cstddef>

void MNNDeconvRunForUnitDepthWise_RVV(const float* dst, float* src, const float* weight, size_t fw, size_t fh,
                                  size_t weight_y_step, size_t dilateX_step, size_t dilateY_step) {
    if (fw == 0 || fh == 0) {
        return;
    }

    // The contiguous path below builds the repeated C4 pixel with an indexed
    // load (vloxei32). That is a one-off setup cost, so the path only pays off
    // once there is enough work to amortise it: measured on the C920-class
    // board the setup is roughly 70ns, so fh=1 with fw=16..32 came out at
    // 0.54x-0.87x, while fw*fh >= 48 stayed between 1.12x and 2.69x. Below the
    // threshold the unit-stride path is both faster and gather-free.
    if (dilateX_step == 4 && fw >= 16 && fw * fh >= 48) {
        const size_t total = fw * 4;
        const size_t vlmax = __riscv_vsetvlmax_e32m8();
        const size_t fullEnd = total & ~(vlmax - 1);

        const vuint32m8_t index = __riscv_vid_v_u32m8(vlmax);
        const vuint32m8_t channel = __riscv_vremu_vx_u32m8(index, 4, vlmax);
        const vuint32m8_t byteOffset = __riscv_vsll_vx_u32m8(channel, 2, vlmax);
        const vfloat32m8_t dstValue = __riscv_vloxei32_v_f32m8(dst, byteOffset, vlmax);

        for (size_t fy = 0; fy < fh; ++fy) {
            float* srcY = src + fy * dilateY_step;
            const float* weightY = weight + fy * weight_y_step;

            size_t x = 0;
            for (; x < fullEnd; x += vlmax) {
                vfloat32m8_t srcValue = __riscv_vle32_v_f32m8(srcY + x, vlmax);
                const vfloat32m8_t weightValue = __riscv_vle32_v_f32m8(weightY + x, vlmax);
                srcValue = __riscv_vfmacc_vv_f32m8(srcValue, dstValue, weightValue, vlmax);
                __riscv_vse32_v_f32m8(srcY + x, srcValue, vlmax);
            }
            if (x < total) {
                const size_t vl = __riscv_vsetvl_e32m8(total - x);
                vfloat32m8_t srcValue = __riscv_vle32_v_f32m8(srcY + x, vl);
                const vfloat32m8_t weightValue = __riscv_vle32_v_f32m8(weightY + x, vl);
                srcValue = __riscv_vfmacc_vv_f32m8(srcValue, dstValue, weightValue, vl);
                __riscv_vse32_v_f32m8(srcY + x, srcValue, vl);
            }
        }
        return;
    }

    // Update one C4 pixel at a time: unit-stride loads/stores avoid expensive
    // wide strided accesses for dilated filters and small filter widths.
    const size_t vl = __riscv_vsetvl_e32m1(4);
    const vfloat32m1_t dstValue = __riscv_vle32_v_f32m1(dst, vl);
    for (size_t fy = 0; fy < fh; ++fy) {
        float* srcY = src + fy * dilateY_step;
        const float* weightY = weight + fy * weight_y_step;
        for (size_t fx = 0; fx < fw; ++fx) {
            float* pixel = srcY + fx * dilateX_step;
            vfloat32m1_t value = __riscv_vle32_v_f32m1(pixel, vl);
            const vfloat32m1_t w = __riscv_vle32_v_f32m1(weightY + fx * 4, vl);
            value = __riscv_vfmacc_vv_f32m1(value, dstValue, w, vl);
            __riscv_vse32_v_f32m1(pixel, value, vl);
        }
    }
}
