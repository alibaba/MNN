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

static inline size_t minSize(size_t a, size_t b) {
    return a < b ? a : b;
}

void MNNQuantAttentionValue_RVV(int8_t* dst, const float* source, float* valueSum, int32_t* params) {
    const int32_t kvNumHead = params[0];
    const int32_t seqLen = params[1];
    const int32_t headDim = params[2];
    const int32_t blockNum = params[3];
    const int32_t maxLength = params[4];
    const int32_t lP = params[5];
    const int32_t hP = params[6];
    const int32_t pastLength = params[7];
    const int32_t kvHeadIdx = params[8];
    const int32_t flashAttentionBlockKv = params[9];

    if (seqLen <= 0 || headDim <= 0 || blockNum <= 0 || flashAttentionBlockKv <= 0 || lP <= 0 || hP <= 0) {
        return;
    }

    const int32_t weightStride2 = lP * hP;
    const int32_t weightStride1 = upDiv(flashAttentionBlockKv, lP) * weightStride2;
    const int32_t packedStride1 = weightStride1 + 2 * hP * static_cast<int32_t>(sizeof(float));
    const int32_t packedStride0 = upDiv(headDim, hP) * packedStride1;
    const int32_t srcStride0 = kvNumHead * headDim;
    const ptrdiff_t srcStrideBytes = static_cast<ptrdiff_t>(srcStride0 * sizeof(float));
    const int32_t roundedHeadDim = roundUp(headDim, hP);
    const float* sourceFp32 = source;
    const int32_t srcHeadOffset = kvHeadIdx * headDim;

    if (pastLength == 0) {
        for (int d = 0; d < headDim; ++d) {
            float* scalePtr = reinterpret_cast<float*>(dst + (d / hP) * packedStride1 + weightStride1) + (d % hP);
            float* biasPtr = scalePtr + hP;

            const float* srcBase = sourceFp32 + srcHeadOffset + d;
            float dMax = srcBase[0];
            float dMin = srcBase[0];
            size_t s = 0;
            while (s < static_cast<size_t>(seqLen)) {
                const size_t vl = __riscv_vsetvl_e32m8(static_cast<size_t>(seqLen) - s);
                const vfloat32m8_t data = __riscv_vlse32_v_f32m8(srcBase + s * srcStride0, srcStrideBytes, vl);
                const vfloat32m1_t minReduced =
                    __riscv_vfredmin_vs_f32m8_f32m1(data, __riscv_vfmv_s_f_f32m1(dMin, 1), vl);
                const vfloat32m1_t maxReduced =
                    __riscv_vfredmax_vs_f32m8_f32m1(data, __riscv_vfmv_s_f_f32m1(dMax, 1), vl);
                dMin = __riscv_vfmv_f_s_f32m1_f32(minReduced);
                dMax = __riscv_vfmv_f_s_f32m1_f32(maxReduced);
                s += vl;
            }

            const float range = dMax - dMin;
            if (range < 1e-6) {
                scalePtr[0] = 0.0f;
                biasPtr[0] = dMax;
            } else {
                const float scale = range / 255.0f;
                scalePtr[0] = scale;
                biasPtr[0] = fmaf(128.0f, scale, dMin);
            }
        }
    }

    if (pastLength == 0 || (pastLength % flashAttentionBlockKv) == 0) {
        const int32_t blockCount = upDiv(maxLength, flashAttentionBlockKv);
        const int32_t headBlock = upDiv(headDim, hP);
        for (int k = 0; k < blockCount; ++k) {
            for (int r = 0; r < headBlock; ++r) {
                float* scalePtr = reinterpret_cast<float*>(dst + k * packedStride0 + r * packedStride1 + weightStride1);
                float* biasPtr = scalePtr + hP;
                const float* baseScale = reinterpret_cast<const float*>(dst + r * packedStride1 + weightStride1);
                const float* baseBias = baseScale + hP;
                for (int c = 0; c < hP; ++c) {
                    scalePtr[c] = baseScale[c];
                    biasPtr[c] = baseBias[c];
                }
            }
        }
    }

    for (int d = 0; d < headDim; ++d) {
        const int idxBase = (d / hP) * packedStride1 + (d % hP) * lP;
        int8_t* dstBase = dst + idxBase;
        float* scaleBase = reinterpret_cast<float*>(dst + (d / hP) * packedStride1 + weightStride1) + (d % hP);
        float* biasBase = scaleBase + hP;
        float* sumBase = valueSum + (d / hP) * hP + (d % hP);
        const float scale = scaleBase[0];
        const float bias = biasBase[0];
        const float qscale = scale < 1e-6 ? 0.0f : 1.0f / scale;
        const float qbias = scale < 1e-6 ? 0.0f : -bias / scale;
        const float* srcBase = sourceFp32 + srcHeadOffset + d;

        size_t s = 0;
        while (s < static_cast<size_t>(seqLen)) {
            const int kvSeqIndex = static_cast<int>(s) + pastLength;
            const int seqInBlock = kvSeqIndex % flashAttentionBlockKv;
            const int seqInLP = seqInBlock % lP;
            const size_t contiguousLP = static_cast<size_t>(lP - seqInLP);
            const size_t contiguousBlock = static_cast<size_t>(flashAttentionBlockKv - seqInBlock);
            const size_t remain = static_cast<size_t>(seqLen) - s;
            const size_t request = minSize(remain, minSize(contiguousLP, contiguousBlock));
            const size_t vl = __riscv_vsetvl_e32m4(request);

            const int idxInner =
                (kvSeqIndex / flashAttentionBlockKv) * packedStride0 + (seqInBlock / lP) * weightStride2 + seqInLP;
            const int idxSum = (kvSeqIndex / flashAttentionBlockKv) * roundedHeadDim;
            vfloat32m4_t value = __riscv_vlse32_v_f32m4(srcBase + s * srcStride0, srcStrideBytes, vl);
            value = __riscv_vfmacc_vf_f32m4(__riscv_vfmv_v_f_f32m4(qbias, vl), qscale, value, vl);

            vint32m4_t quant = roundQuantized(value, vl);
            quant = __riscv_vmax_vx_i32m4(quant, -128, vl);
            quant = __riscv_vmin_vx_i32m4(quant, 127, vl);

            const vint16m2_t quant16 = __riscv_vncvt_x_x_w_i16m2(quant, vl);
            const vint8m1_t quant8 = __riscv_vncvt_x_x_w_i8m1(quant16, vl);
            __riscv_vse8_v_i8m1(dstBase + idxInner, quant8, vl);
            // Preserve the scalar dequantization and accumulation order, including
            // an existing partial-block sum during decode/cache appends.
            for (size_t lane = 0; lane < vl; ++lane) {
                sumBase[idxSum] += fmaf(static_cast<float>(dstBase[idxInner + lane]), scale, bias);
            }
            s += vl;
        }
    }
}
#endif
