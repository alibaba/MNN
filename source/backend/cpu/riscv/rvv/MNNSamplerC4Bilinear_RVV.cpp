#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "MNNSamplerC4Bilinear_RVV.hpp"

namespace MNN {
namespace CV {
struct Point;
} // namespace CV
} // namespace MNN

// A lane carries one whole C4 pixel, not one channel.
//
// This kernel previously pinned vl to 4 and let the four lanes be the four
// channels of a single pixel. That caps the vector at one output pixel per
// iteration however wide the hardware is, and the load pattern that follows from
// it is wasteful: a lane only wants one byte, while an indexed load always moves
// a whole 32-bit element, so three quarters of every fetch is thrown away.
//
// Here each lane needs exactly one 32-bit C4 word - the element size an indexed
// load moves anyway. Nothing is fetched and discarded, and one vector operation
// covers vl output pixels instead of one. The price is the unpack below: a word
// holds all four channels, so each channel is shifted out into its own value, the
// blend runs once per channel, and the four results are shifted back into a word
// for a single store.
//
// The coordinates are advanced with the same step-by-step accumulation the
// generic body uses, so the sampled positions match it exactly and the only
// remaining divergence is the documented FP64/FP32 one described below. Deriving
// lane coordinates as `start + lane * delta` would be marginally more accurate
// but would no longer agree with the reference, and a position landing near an
// integer could then select different neighbours and drift by far more than one
// grey level.

// e32m4 holds VLEN / 8 lanes, so this buffer covers VLEN up to 2048 bits. A core
// wider than that takes the scalar fallback rather than overflowing the stack
// buffer.
static constexpr size_t kMaxLanes = 256;

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
// test/backend/cpu/RVVImageProcessTest.cpp.
//
// roundf() rounds half away from zero. After the clamp the range is [0, 255], so
// adding 0.5f and truncating would nearly match it - nearly, because the addition
// itself rounds and can push a value that sits just below a tie onto the far
// side. Truncate and compare the exact fraction instead.
static inline vuint32m4_t roundAndPack(vfloat32m4_t value, size_t vl) {
    const vuint32m4_t truncated = __riscv_vfcvt_rtz_xu_f_v_u32m4(value, vl);
    const vfloat32m4_t truncatedBack = __riscv_vfcvt_f_xu_v_f32m4(truncated, vl);
    const vfloat32m4_t fraction = __riscv_vfsub_vv_f32m4(value, truncatedBack, vl);
    const vbool8_t roundUp = __riscv_vmfge_vf_f32m4_b8(fraction, 0.5f, vl);
    return __riscv_vadd_vx_u32m4_mu(roundUp, truncated, truncated, 1, vl);
}

// One lane's byte out of a packed C4 word.
static inline vfloat32m4_t extractChannelAsFloat(vuint32m4_t word, uint32_t shift, size_t vl) {
    const vuint32m4_t byte = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(word, shift, vl), 0xFF, vl);
    return __riscv_vfcvt_f_xu_v_f32m4(byte, vl);
}

// Used when the vector path cannot be entered at all. Keeps the generic body's
// arithmetic so the two agree bit for bit.
//
// `points` is read through a float pointer, exactly as the vector path does: this
// translation unit only forward-declares MNN::CV::Point so that it stays free of
// MNN headers, and Point is a pair of floats.
static void samplerC4BilinearScalar(const unsigned char* source, unsigned char* dest, const float* pointData,
                                    size_t count, size_t iw, size_t ih, size_t yStride) {
    const float dy = pointData[3];
    const float dx = pointData[2];
    const float xMax = (float)(iw - 1);
    const float yMax = (float)(ih - 1);
    float currentX = pointData[0];
    float currentY = pointData[1];
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
            const float w0 = (1.0f - xF) * (1.0f - yF);
            const float w1 = xF * (1.0f - yF);
            const float w2 = yF * (1.0f - xF);
            const float w3 = xF * yF;
            float value = w0 * row0[4 * x0 + c] + w1 * row0[4 * x1 + c] + w2 * row1[4 * x0 + c] + w3 * row1[4 * x1 + c];
            value = value < 0.0f ? 0.0f : (value > 255.0f ? 255.0f : value);
            dest[4 * i + c] = (unsigned char)__builtin_roundf(value);
        }
        currentY += dy;
        currentX += dx;
    }
}

void MNNSamplerC4Bilinear_RVV(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                              size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    (void)capacity;
    dest += 4 * sta;
    const float* pointData = reinterpret_cast<const float*>(points);
    const size_t vlmax = __riscv_vsetvlmax_e32m4();
    const bool addressingSafe = MNN::CV::RVV::canUseC4BilinearIndexedLoad(source, iw, ih, yStride);

    // The vector path indexes aligned C4 words with 32-bit byte offsets. The
    // complete row-plus-column offset must fit, not just the row component.
    if (vlmax > kMaxLanes || vlmax == 0 || !addressingSafe) {
        samplerC4BilinearScalar(source, dest, pointData, count, iw, ih, yStride);
        return;
    }

    float currentX = pointData[0];
    float currentY = pointData[1];
    const float dx = pointData[2];
    const float dy = pointData[3];
    const float xMax = (float)(iw - 1);
    const float yMax = (float)(ih - 1);
    const uint32_t iwMinus1 = (uint32_t)(iw - 1);
    const uint32_t ihMinus1 = (uint32_t)(ih - 1);
    const uint32_t yStrideU = (uint32_t)yStride;
    const uint32_t* sourceWords = reinterpret_cast<const uint32_t*>(source);

    float xBuffer[kMaxLanes];
    float yBuffer[kMaxLanes];

    for (size_t i = 0; i < count;) {
        const size_t vl = __riscv_vsetvl_e32m4(count - i);
        // The generic body advances one output pixel at a time; building the lane
        // coordinates the same way keeps the two on identical positions.
        for (size_t lane = 0; lane < vl; ++lane) {
            xBuffer[lane] = clampBilinearCoordinate(currentX, xMax);
            yBuffer[lane] = clampBilinearCoordinate(currentY, yMax);
            currentX += dx;
            currentY += dy;
        }

        const vfloat32m4_t vx = __riscv_vle32_v_f32m4(xBuffer, vl);
        const vfloat32m4_t vy = __riscv_vle32_v_f32m4(yBuffer, vl);

        const vint32m4_t x0i = __riscv_vfcvt_rtz_x_f_v_i32m4(vx, vl);
        const vint32m4_t y0i = __riscv_vfcvt_rtz_x_f_v_i32m4(vy, vl);
        const vfloat32m4_t xF = __riscv_vfsub_vv_f32m4(vx, __riscv_vfcvt_f_x_v_f32m4(x0i, vl), vl);
        const vfloat32m4_t yF = __riscv_vfsub_vv_f32m4(vy, __riscv_vfcvt_f_x_v_f32m4(y0i, vl), vl);

        // ceil() without a round-up conversion: both coordinates are clamped into
        // [0, max], so ceil(v) equals trunc(v) plus one exactly when v carries a
        // fractional part.
        const vuint32m4_t ux0 = __riscv_vreinterpret_v_i32m4_u32m4(x0i);
        const vuint32m4_t uy0 = __riscv_vreinterpret_v_i32m4_u32m4(y0i);
        const vbool8_t xRoundedUp = __riscv_vmfgt_vf_f32m4_b8(xF, 0.0f, vl);
        const vbool8_t yRoundedUp = __riscv_vmfgt_vf_f32m4_b8(yF, 0.0f, vl);
        vuint32m4_t ux1 = __riscv_vadd_vx_u32m4_mu(xRoundedUp, ux0, ux0, 1, vl);
        vuint32m4_t uy1 = __riscv_vadd_vx_u32m4_mu(yRoundedUp, uy0, uy0, 1, vl);
        ux1 = __riscv_vminu_vx_u32m4(ux1, iwMinus1, vl);
        uy1 = __riscv_vminu_vx_u32m4(uy1, ihMinus1, vl);

        const vfloat32m4_t v1mxF = __riscv_vfrsub_vf_f32m4(xF, 1.0f, vl);
        const vfloat32m4_t v1myF = __riscv_vfrsub_vf_f32m4(yF, 1.0f, vl);
        const vfloat32m4_t w0 = __riscv_vfmul_vv_f32m4(v1mxF, v1myF, vl);
        const vfloat32m4_t w1 = __riscv_vfmul_vv_f32m4(xF, v1myF, vl);
        const vfloat32m4_t w2 = __riscv_vfmul_vv_f32m4(yF, v1mxF, vl);
        const vfloat32m4_t w3 = __riscv_vfmul_vv_f32m4(xF, yF, vl);

        const vuint32m4_t row0 = __riscv_vmul_vx_u32m4(uy0, yStrideU, vl);
        const vuint32m4_t row1 = __riscv_vmul_vx_u32m4(uy1, yStrideU, vl);
        const vuint32m4_t col0 = __riscv_vsll_vx_u32m4(ux0, 2, vl);
        const vuint32m4_t col1 = __riscv_vsll_vx_u32m4(ux1, 2, vl);

        // One 32-bit C4 word per lane per neighbour: the fetched element is used
        // in full, which is what makes the indexed load pay for itself.
        const vuint32m4_t w00 = __riscv_vluxei32_v_u32m4(sourceWords, __riscv_vadd_vv_u32m4(row0, col0, vl), vl);
        const vuint32m4_t w01 = __riscv_vluxei32_v_u32m4(sourceWords, __riscv_vadd_vv_u32m4(row0, col1, vl), vl);
        const vuint32m4_t w10 = __riscv_vluxei32_v_u32m4(sourceWords, __riscv_vadd_vv_u32m4(row1, col0, vl), vl);
        const vuint32m4_t w11 = __riscv_vluxei32_v_u32m4(sourceWords, __riscv_vadd_vv_u32m4(row1, col1, vl), vl);

        // The weights are per pixel, so all four channels reuse them.
        vuint32m4_t packed = __riscv_vmv_v_x_u32m4(0, vl);
        for (uint32_t channel = 0; channel < 4; ++channel) {
            const uint32_t shift = 8 * channel;
            const vfloat32m4_t c00 = extractChannelAsFloat(w00, shift, vl);
            const vfloat32m4_t c01 = extractChannelAsFloat(w01, shift, vl);
            const vfloat32m4_t c10 = extractChannelAsFloat(w10, shift, vl);
            const vfloat32m4_t c11 = extractChannelAsFloat(w11, shift, vl);

            vfloat32m4_t value = __riscv_vfmul_vv_f32m4(c00, w0, vl);
            value = __riscv_vfadd_vv_f32m4(value, __riscv_vfmul_vv_f32m4(c01, w1, vl), vl);
            value = __riscv_vfadd_vv_f32m4(value, __riscv_vfmul_vv_f32m4(c10, w2, vl), vl);
            value = __riscv_vfadd_vv_f32m4(value, __riscv_vfmul_vv_f32m4(c11, w3, vl), vl);
            value = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(value, 0.0f, vl), 255.0f, vl);

            packed = __riscv_vor_vv_u32m4(packed, __riscv_vsll_vx_u32m4(roundAndPack(value, vl), shift, vl), vl);
        }

        // Lane L owns output pixel i + L, whose four channels are bytes
        // 4L..4L+3, so the group is a single run of 4 * vl bytes. A byte store
        // carries no alignment requirement, and the byte view of u32m4 is u8m4.
        __riscv_vse8_v_u8m4(dest + 4 * i, __riscv_vreinterpret_v_u32m4_u8m4(packed), 4 * vl);
        i += vl;
    }
}
