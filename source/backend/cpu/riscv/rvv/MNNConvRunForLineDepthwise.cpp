#include <riscv_vector.h>
#include <cstddef>

// Registered as CoreFunctions::MNNConvRunForLineDepthwise, and only inside the
// supportRVV branch, so the scalar definition in compute/ConvOpt.cpp is what a CPU
// without the V extension keeps executing. A same-named definition here would have
// C++ linkage while ConvOpt.h declares the generic entry point extern "C": both
// copies would be present in the library and every call site would keep resolving
// to the generic one, leaving this kernel unreachable.
void MNNConvRunForLineDepthwise_RVV(float* dst, const float* src, const float* weight, size_t width,
                                    size_t src_w_setup, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step,
                                    size_t height, size_t srcHStep, size_t dstHStep, const float* bias,
                                    const float* parameters) {
    const float minV = parameters[0];
    const float maxV = parameters[1];
    const ptrdiff_t srcByteStride = static_cast<ptrdiff_t>(src_w_setup) * sizeof(float);
    const ptrdiff_t dstByteStride = 4 * sizeof(float);

    // Adjacent output pixels consume adjacent C4 groups when the input stride
    // is one. Load and store those groups together; four segments require an
    // LMUL of at most two. Keep the wider strided path for other input strides.
    if (src_w_setup == 4) {
        for (size_t y = 0; y < height; ++y) {
            const float* srcY = src + y * srcHStep;
            float* dstY = dst + y * dstHStep;
            for (size_t dx = 0; dx < width;) {
                const size_t vl = __riscv_vsetvl_e32m2(width - dx);
                vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(bias[0], vl);
                vfloat32m2_t acc1 = __riscv_vfmv_v_f_f32m2(bias[1], vl);
                vfloat32m2_t acc2 = __riscv_vfmv_v_f_f32m2(bias[2], vl);
                vfloat32m2_t acc3 = __riscv_vfmv_v_f_f32m2(bias[3], vl);
                const float* srcBase = srcY + dx * 4;
                const float* weightPtr = weight;

                for (size_t fy = 0; fy < fh; ++fy) {
                    const float* srcFy = srcBase + fy * dilateY_step;
                    for (size_t fx = 0; fx < fw; ++fx) {
                        const vfloat32m2x4_t values = __riscv_vlseg4e32_v_f32m2x4(srcFy + fx * dilateX_step, vl);
                        acc0 = __riscv_vfmacc_vf_f32m2(acc0, weightPtr[0], __riscv_vget_v_f32m2x4_f32m2(values, 0), vl);
                        acc1 = __riscv_vfmacc_vf_f32m2(acc1, weightPtr[1], __riscv_vget_v_f32m2x4_f32m2(values, 1), vl);
                        acc2 = __riscv_vfmacc_vf_f32m2(acc2, weightPtr[2], __riscv_vget_v_f32m2x4_f32m2(values, 2), vl);
                        acc3 = __riscv_vfmacc_vf_f32m2(acc3, weightPtr[3], __riscv_vget_v_f32m2x4_f32m2(values, 3), vl);
                        weightPtr += 4;
                    }
                }

                acc0 = __riscv_vfmax_vf_f32m2(acc0, minV, vl);
                acc1 = __riscv_vfmax_vf_f32m2(acc1, minV, vl);
                acc2 = __riscv_vfmax_vf_f32m2(acc2, minV, vl);
                acc3 = __riscv_vfmax_vf_f32m2(acc3, minV, vl);
                acc0 = __riscv_vfmin_vf_f32m2(acc0, maxV, vl);
                acc1 = __riscv_vfmin_vf_f32m2(acc1, maxV, vl);
                acc2 = __riscv_vfmin_vf_f32m2(acc2, maxV, vl);
                acc3 = __riscv_vfmin_vf_f32m2(acc3, maxV, vl);
                const vfloat32m2x4_t result = __riscv_vcreate_v_f32m2x4(acc0, acc1, acc2, acc3);
                __riscv_vsseg4e32_v_f32m2x4(dstY + dx * 4, result, vl);
                dx += vl;
            }
        }
        return;
    }

    for (size_t y = 0; y < height; ++y) {
        const float* srcY = src + y * srcHStep;
        float* dstY = dst + y * dstHStep;
        for (size_t dx = 0; dx < width;) {
            const size_t vl = __riscv_vsetvl_e32m8(width - dx);

            for (int c = 0; c < 4; ++c) {
                vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(bias[c], vl);
                const float* srcBase = srcY + dx * src_w_setup + c;
                const float* weightPtr = weight + c;

                for (size_t fy = 0; fy < fh; ++fy) {
                    const float* srcFy = srcBase + fy * dilateY_step;
                    for (size_t fx = 0; fx < fw; ++fx) {
                        const vfloat32m8_t srcValue =
                            __riscv_vlse32_v_f32m8(srcFy + fx * dilateX_step, srcByteStride, vl);
                        acc = __riscv_vfmacc_vf_f32m8(acc, *weightPtr, srcValue, vl);
                        weightPtr += 4;
                    }
                }

                acc = __riscv_vfmax_vf_f32m8(acc, minV, vl);
                acc = __riscv_vfmin_vf_f32m8(acc, maxV, vl);
                __riscv_vsse32_v_f32m8(dstY + dx * 4 + c, dstByteStride, acc, vl);
            }
            dx += vl;
        }
    }
}
