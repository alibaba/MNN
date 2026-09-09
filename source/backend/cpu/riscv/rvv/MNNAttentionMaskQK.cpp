#include <riscv_vector.h>
#include <stddef.h>

#ifndef UP_DIV
#define UP_DIV(x, y) (((x) + (y) - 1) / (y))
#endif
#ifndef ROUND_UP
#define ROUND_UP(x, y) ((((x) + (y) - 1) / (y)) * (y))
#endif

void MNNAttentionMaskQK_RVV(float* qkPacked, const float* scale, size_t seqLen, size_t processedKvSeq, int pack,
                            int kvSeqLen, int kvoffset, int padKvSeqLen, const float* sinksPtr, const float* maskPtr,
                            size_t maskElementSize, bool scaleApplied, bool isLowerTriangular) {
    (void)sinksPtr;
    if (isLowerTriangular && scaleApplied) {
        return;
    }

    const float scaleVal = scale[0];
    const size_t qkSize = ROUND_UP(processedKvSeq, pack) * seqLen;
    if (isLowerTriangular) {
        size_t offset = 0;
        while (offset < qkSize) {
            const size_t vl = __riscv_vsetvl_e32m8(qkSize - offset);
            vfloat32m8_t data = __riscv_vle32_v_f32m8(qkPacked + offset, vl);
            data = __riscv_vfmul_vf_f32m8(data, scaleVal, vl);
            __riscv_vse32_v_f32m8(qkPacked + offset, data, vl);
            offset += vl;
        }
        return;
    }

    if (maskPtr == nullptr) {
        return;
    }

    const bool fullMask = maskElementSize == (seqLen + padKvSeqLen) * (kvSeqLen + padKvSeqLen);
    const int gapLen = fullMask ? 0 : static_cast<int>(kvSeqLen - seqLen);
    const int maskCols = fullMask ? kvSeqLen + padKvSeqLen : static_cast<int>(seqLen) + padKvSeqLen;
    const size_t kvBlockCount = UP_DIV(processedKvSeq, pack);

    for (size_t i = 0; i < kvBlockCount; ++i) {
        float* blockDataPtr = qkPacked + i * seqLen * pack;
        const int kvStart = kvoffset + static_cast<int>(i) * pack;
        const int maskEnd = gapLen + maskCols;
        int scaleEnd = pack;
        if (kvStart + scaleEnd > maskEnd) {
            // Scalar path scales the current lane before checking the mask bound and breaking.
            scaleEnd = kvStart >= maskEnd ? 1 : maskEnd - kvStart + 1;
        }
        if (scaleEnd < 0) {
            scaleEnd = 0;
        }
        if (scaleEnd > pack) {
            scaleEnd = pack;
        }
        int validBegin = 0;
        if (kvStart < gapLen) {
            validBegin = gapLen - kvStart;
        }
        int validEnd = pack;
        if (kvStart + validEnd > maskEnd) {
            validEnd = maskEnd - kvStart;
        }
        if (validBegin < 0) {
            validBegin = 0;
        }
        if (validEnd > pack) {
            validEnd = pack;
        }
        for (size_t j = 0; j < seqLen; ++j) {
            float* dataPtr = blockDataPtr + j * pack;
            const float* currentMaskRow = maskPtr + j * maskCols;

            // A complete mask tile can be scaled and added in one load/store pass.
            if (validBegin == 0 && validEnd == pack) {
                for (int lane = 0; lane < pack;) {
                    const size_t vl = __riscv_vsetvl_e32m8(pack - lane);
                    vfloat32m8_t data = __riscv_vle32_v_f32m8(dataPtr + lane, vl);
                    if (!scaleApplied) {
                        data = __riscv_vfmul_vf_f32m8(data, scaleVal, vl);
                    }
                    const vfloat32m8_t mask = __riscv_vle32_v_f32m8(currentMaskRow + kvStart - gapLen + lane, vl);
                    data = __riscv_vfadd_vv_f32m8(data, mask, vl);
                    __riscv_vse32_v_f32m8(dataPtr + lane, data, vl);
                    lane += vl;
                }
                continue;
            }
            if (!scaleApplied && scaleEnd > 0) {
                int remain = scaleEnd;
                float* ptr = dataPtr;
                while (remain > 0) {
                    const size_t vl = __riscv_vsetvl_e32m8(static_cast<size_t>(remain));
                    vfloat32m8_t data = __riscv_vle32_v_f32m8(ptr, vl);
                    data = __riscv_vfmul_vf_f32m8(data, scaleVal, vl);
                    __riscv_vse32_v_f32m8(ptr, data, vl);
                    ptr += vl;
                    remain -= static_cast<int>(vl);
                }
            }

            if (validBegin >= validEnd) {
                continue;
            }

            int remain = validEnd - validBegin;
            float* dst = dataPtr + validBegin;
            const float* src = currentMaskRow + kvStart + validBegin - gapLen;
            while (remain > 0) {
                const size_t vl = __riscv_vsetvl_e32m8(static_cast<size_t>(remain));
                vfloat32m8_t data = __riscv_vle32_v_f32m8(dst, vl);
                const vfloat32m8_t mask = __riscv_vle32_v_f32m8(src, vl);
                data = __riscv_vfadd_vv_f32m8(data, mask, vl);
                __riscv_vse32_v_f32m8(dst, data, vl);
                dst += vl;
                src += vl;
                remain -= static_cast<int>(vl);
            }
        }
    }
}
