#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
// Round half away from zero without changing frm: an explicit-RMM conversion
// can leave surrounding scalar arithmetic under RMM in GCC's generated loop.
// Truncation plus an exact fractional comparison also handles values just below a tie.
static inline vint32m4_t roundQuantized(vfloat32m4_t value, size_t vl) {
    value = __riscv_vfmax_vf_f32m4(value, -128.0f, vl);
    value = __riscv_vfmin_vf_f32m4(value, 127.0f, vl);
    vint32m4_t quant = __riscv_vfcvt_rtz_x_f_v_i32m4(value, vl);
    const vfloat32m4_t integer = __riscv_vfcvt_f_x_v_f32m4(quant, vl);
    const vfloat32m4_t fraction = __riscv_vfsub_vv_f32m4(value, integer, vl);
    const vbool8_t up = __riscv_vmfge_vf_f32m4_b8(fraction, 0.5f, vl);
    const vbool8_t down = __riscv_vmfle_vf_f32m4_b8(fraction, -0.5f, vl);
    quant = __riscv_vmerge_vvm_i32m4(quant, __riscv_vadd_vx_i32m4(quant, 1, vl), up, vl);
    return __riscv_vmerge_vvm_i32m4(quant, __riscv_vsub_vx_i32m4(quant, 1, vl), down, vl);
}

static inline int upDiv(int x, int y) {
    return (x + y - 1) / y;
}

static inline int roundUp(int x, int y) {
    return upDiv(x, y) * y;
}

void MNNQuantAttentionKey_RVV(int8_t* dst, const float* source, float* sumKeyPtr, float* maxKeyPtr, int32_t* params) {
    const int32_t kvNumHead = params[0];
    const int32_t seqLen = params[1];
    const int32_t headDim = params[2];
    const int32_t blockNum = params[3];
    const int32_t lP = params[5];
    const int32_t hP = params[6];
    const int32_t pastLength = params[7];
    const int32_t kvHeadIdx = params[8];

    if (seqLen <= 0 || headDim <= 0 || blockNum <= 0 || lP <= 0 || hP <= 0) {
        return;
    }

    const int32_t blockL = upDiv(headDim, blockNum);
    const int32_t weightStride1 = roundUp(blockL, lP) * hP;
    const int32_t weightStride2 = lP * hP;
    const int32_t packedWeightStride1 = weightStride1 + 2 * 4 * hP;

    if (seqLen > 1) {
        for (int s = 0; s < seqLen; ++s) {
            const float* keySrc = source + s * kvNumHead * headDim + kvHeadIdx * headDim;
            size_t d = 0;
            while (d < static_cast<size_t>(headDim)) {
                const size_t vl = __riscv_vsetvl_e32m8(static_cast<size_t>(headDim) - d);
                const vfloat32m8_t srcVec = __riscv_vle32_v_f32m8(keySrc + d, vl);
                vfloat32m8_t maxVec = __riscv_vle32_v_f32m8(maxKeyPtr + d, vl);
                maxVec = __riscv_vfmax_vv_f32m8(maxVec, srcVec, vl);
                __riscv_vse32_v_f32m8(maxKeyPtr + d, maxVec, vl);
                d += vl;
            }
        }
    }

    for (int s = 0; s < seqLen; ++s) {
        const float* keySrc = source + s * kvNumHead * headDim + kvHeadIdx * headDim;
        const float init = keySrc[0] - maxKeyPtr[0];
        vfloat32m1_t minReduced = __riscv_vfmv_s_f_f32m1(init, 1);
        vfloat32m1_t maxReduced = __riscv_vfmv_s_f_f32m1(init, 1);

        size_t d = 0;
        while (d < static_cast<size_t>(headDim)) {
            const size_t vl = __riscv_vsetvl_e32m8(static_cast<size_t>(headDim) - d);
            const vfloat32m8_t srcVec = __riscv_vle32_v_f32m8(keySrc + d, vl);
            const vfloat32m8_t maxKeyVec = __riscv_vle32_v_f32m8(maxKeyPtr + d, vl);
            const vfloat32m8_t keyData = __riscv_vfsub_vv_f32m8(srcVec, maxKeyVec, vl);
            minReduced = __riscv_vfredmin_vs_f32m8_f32m1(keyData, minReduced, vl);
            maxReduced = __riscv_vfredmax_vs_f32m8_f32m1(keyData, maxReduced, vl);
            d += vl;
        }

        const float minKey = __riscv_vfmv_f_s_f32m1_f32(minReduced);
        const float maxKey = __riscv_vfmv_f_s_f32m1_f32(maxReduced);
        const float range = maxKey - minKey;
        const float scale = range / 255.0f;
        const float bias = minKey + 128.0f * range / 255.0f;

        const int outIndex = (pastLength + s) / hP;
        const int inIndex = (pastLength + s) % hP;
        float sumKey = 0.0f;

        for (int k = 0; k < blockNum; ++k) {
            int8_t* weightDst = dst + outIndex * blockNum * packedWeightStride1 + k * packedWeightStride1;
            float* scaleDst = reinterpret_cast<float*>(weightDst + weightStride1);
            float* biasDst = scaleDst + hP;
            scaleDst[inIndex] = scale;
            biasDst[inIndex] = bias;

            const float* currentKeyBlock = keySrc + k * blockL;
            const float* currentMaxBlock = maxKeyPtr + k * blockL;
            const int validLength = headDim - k * blockL < blockL ? headDim - k * blockL : blockL;
            size_t blockOffset = 0;
            while (static_cast<int>(blockOffset) < validLength) {
                const int j = static_cast<int>(blockOffset) % lP;
                const size_t contiguous = static_cast<size_t>(lP - j);
                const size_t remain = static_cast<size_t>(validLength) - blockOffset;
                const size_t request = remain < contiguous ? remain : contiguous;
                const size_t vl = __riscv_vsetvl_e32m4(request);

                const vfloat32m4_t srcVec = __riscv_vle32_v_f32m4(currentKeyBlock + blockOffset, vl);
                const vfloat32m4_t maxKeyVec = __riscv_vle32_v_f32m4(currentMaxBlock + blockOffset, vl);
                vfloat32m4_t value = __riscv_vfsub_vv_f32m4(srcVec, maxKeyVec, vl);
                value = __riscv_vfsub_vf_f32m4(value, minKey, vl);
                // Keep the scalar division/multiplication order: a reciprocal can
                // change which side of a rounding boundary a value lies on.
                if (range > 0.0f) {
                    value = __riscv_vfdiv_vf_f32m4(value, range, vl);
                    value = __riscv_vfmacc_vf_f32m4(__riscv_vfmv_v_f_f32m4(-128.0f, vl), 255.0f, value, vl);
                } else {
                    value = __riscv_vfmv_v_f_f32m4(-128.0f, vl);
                }

                vint32m4_t quant = roundQuantized(value, vl);
                quant = __riscv_vmax_vx_i32m4(quant, -128, vl);
                quant = __riscv_vmin_vx_i32m4(quant, 127, vl);

                const vint16m2_t quant16 = __riscv_vncvt_x_x_w_i16m2(quant, vl);
                const vint8m1_t quant8 = __riscv_vncvt_x_x_w_i8m1(quant16, vl);
                const int i = static_cast<int>(blockOffset) / lP;
                int8_t* dstPtr = weightDst + i * weightStride2 + inIndex * lP + j;
                __riscv_vse8_v_i8m1(dstPtr, quant8, vl);
                // The KV correction sums dequantized elements in source order.
                // Reducing integers then applying scale/bias changes FP32 rounding.
                for (size_t lane = 0; lane < vl; ++lane) {
                    sumKey += fmaf((float)dstPtr[lane], scale, bias);
                }
                blockOffset += vl;
            }
            // A final block may have fewer source dimensions than blockL.
            // Do not read the next head (or beyond the source/max buffers).
            for (; blockOffset < static_cast<size_t>(blockL); ++blockOffset) {
                weightDst[blockOffset / lP * weightStride2 + inIndex * lP + blockOffset % lP] = 0;
            }
        }

        sumKeyPtr[outIndex * hP + inIndex] = sumKey;
    }
}
#endif
