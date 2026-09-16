#include <math.h>
#include <riscv_vector.h>
#include <stddef.h>

namespace MNN {
namespace CV {
struct Point;
} // namespace CV
} // namespace MNN

static inline float clampBilinearCoordinate(float value, float maxValue) {
    if (value < 0.0f) {
        return 0.0f;
    }
    return value > maxValue ? maxValue : value;
}

// This kernel evaluates the blend in FP32, while the generic body in
// ImageProcessFunction.cpp computes the c10 term as `yF * (1.0 - xF) * c10`
// with `1.0 - xF` in double. The two agree everywhere except on inputs that land
// exactly on a rounding boundary, where the FP32 result can come out one grey
// level lower - pinned by the boundary case in
// test/backend/cpu/RVVImageProcessTest.cpp. The four weights below still keep the
// scalar expression order so the divergence stays at that single step.
static inline unsigned char bilinearPixel(unsigned char c00, unsigned char c01, unsigned char c10, unsigned char c11,
                                          float xF, float yF) {
    const float w0 = (1.0f - xF) * (1.0f - yF);
    const float w1 = xF * (1.0f - yF);
    const float w2 = yF * (1.0f - xF);
    const float w3 = xF * yF;
    float value = w0 * c00 + w1 * c01 + w2 * c10 + w3 * c11;
    value = value < 0.0f ? 0.0f : (value > 255.0f ? 255.0f : value);
    return (unsigned char)__builtin_roundf(value);
}

// roundf() rounds half away from zero. After the clamp the range is [0, 255],
// so adding 0.5f and truncating would nearly match it - nearly, because the
// addition itself rounds and can push a value that sits just below a tie onto
// the far side. Truncate and compare the exact fraction instead.
static inline vuint8mf2_t roundAndPack(vfloat32m2_t value, size_t vl) {
    const vuint32m2_t truncated = __riscv_vfcvt_rtz_xu_f_v_u32m2(value, vl);
    const vfloat32m2_t truncatedBack = __riscv_vfcvt_f_xu_v_f32m2(truncated, vl);
    const vfloat32m2_t fraction = __riscv_vfsub_vv_f32m2(value, truncatedBack, vl);
    const vbool16_t roundUp = __riscv_vmfge_vf_f32m2_b16(fraction, 0.5f, vl);
    const vuint32m2_t rounded = __riscv_vadd_vx_u32m2_mu(roundUp, truncated, truncated, 1, vl);
    const vuint16m1_t packed16 = __riscv_vncvt_x_x_w_u16m1(rounded, vl);
    return __riscv_vncvt_x_x_w_u8mf2(packed16, vl);
}

void MNNSamplerC4Bilinear_RVV(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                              size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    (void)capacity;
    dest += 4 * sta;
    const float* pointData = reinterpret_cast<const float*>(points);
    float currentX = pointData[0];
    float currentY = pointData[1];
    const float dx = pointData[2];
    const float dy = pointData[3];
    const float xMax = (float)(iw - 1);
    const float yMax = (float)(ih - 1);
    const size_t vl = __riscv_vsetvl_e8mf2(4);

    if (vl < 4) {
        // A core narrow enough that e8mf2 cannot hold four lanes would drop
        // channels from every pixel. Such cores are unusual, but silently
        // returning wrong colours is worse than running the scalar path.
        for (size_t i = 0; i < count; ++i) {
            const float y = clampBilinearCoordinate(currentY, yMax);
            const float x = clampBilinearCoordinate(currentX, xMax);
            const int y0 = (int)y;
            const int x0 = (int)x;
            const int y1 = (int)__builtin_ceilf(y);
            const int x1 = (int)__builtin_ceilf(x);
            const float xF = x - (float)x0;
            const float yF = y - (float)y0;
            const unsigned char* row0 = source + (size_t)y0 * yStride;
            const unsigned char* row1 = source + (size_t)y1 * yStride;
            for (size_t c = 0; c < 4; ++c) {
                dest[4 * i + c] =
                    bilinearPixel(row0[4 * x0 + c], row0[4 * x1 + c], row1[4 * x0 + c], row1[4 * x1 + c], xF, yF);
            }
            currentY += dy;
            currentX += dx;
        }
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        const float y = clampBilinearCoordinate(currentY, yMax);
        const float x = clampBilinearCoordinate(currentX, xMax);
        const int y0 = (int)y;
        const int x0 = (int)x;
        const int y1 = (int)__builtin_ceilf(y);
        const int x1 = (int)__builtin_ceilf(x);
        const float xF = x - (float)x0;
        const float yF = y - (float)y0;

        const vuint32m2_t c00 =
            __riscv_vzext_vf4_u32m2(__riscv_vle8_v_u8mf2(source + (size_t)y0 * yStride + 4 * (size_t)x0, vl), vl);
        const vuint32m2_t c01 =
            __riscv_vzext_vf4_u32m2(__riscv_vle8_v_u8mf2(source + (size_t)y0 * yStride + 4 * (size_t)x1, vl), vl);
        const vuint32m2_t c10 =
            __riscv_vzext_vf4_u32m2(__riscv_vle8_v_u8mf2(source + (size_t)y1 * yStride + 4 * (size_t)x0, vl), vl);
        const vuint32m2_t c11 =
            __riscv_vzext_vf4_u32m2(__riscv_vle8_v_u8mf2(source + (size_t)y1 * yStride + 4 * (size_t)x1, vl), vl);

        const float w0 = (1.0f - xF) * (1.0f - yF);
        const float w1 = xF * (1.0f - yF);
        const float w2 = yF * (1.0f - xF);
        const float w3 = xF * yF;

        vfloat32m2_t value = __riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_xu_v_f32m2(c00, vl), w0, vl);
        value = __riscv_vfadd_vv_f32m2(value, __riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_xu_v_f32m2(c01, vl), w1, vl), vl);
        value = __riscv_vfadd_vv_f32m2(value, __riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_xu_v_f32m2(c10, vl), w2, vl), vl);
        value = __riscv_vfadd_vv_f32m2(value, __riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_xu_v_f32m2(c11, vl), w3, vl), vl);
        value = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(value, 0.0f, vl), 255.0f, vl);
        __riscv_vse8_v_u8mf2(dest + 4 * i, roundAndPack(value, vl), vl);

        currentY += dy;
        currentX += dx;
    }
}
