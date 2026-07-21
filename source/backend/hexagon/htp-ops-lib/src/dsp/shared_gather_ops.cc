#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdint.h>
#include <string.h>

extern "C" {

static inline void make_int4_lut_fp16(uint16_t* lut, const __fp16* scalePtr) {
    const __fp16 scale = *scalePtr;
    for (int q = 0; q < 16; ++q) {
        union {
            __fp16 h;
            uint16_t u;
        } v;
        v.h = (__fp16)(q - 8) * scale;
        lut[q] = v.u;
    }
}

static inline void make_int4_lut_fp32(float* lut, float scale) {
    for (int q = 0; q < 16; ++q) {
        lut[q] = (q - 8) * scale;
    }
}

static inline void store_int4_group_fp16(__fp16* dstRow, int channelBase, int ic, const uint8_t* src,
                                         const uint16_t* lut) {
    const uint8_t b0 = src[0];
    const uint8_t b1 = src[1];
    const uint8_t b2 = src[2];
    const uint8_t b3 = src[3];
    uint16_t* dst = reinterpret_cast<uint16_t*>(dstRow) + channelBase;
    if (channelBase + 7 < ic) {
        const uint64_t lo = (uint64_t)lut[b0 & 0x0F] |
                            ((uint64_t)lut[b2 & 0x0F] << 16) |
                            ((uint64_t)lut[b1 & 0x0F] << 32) |
                            ((uint64_t)lut[b3 & 0x0F] << 48);
        const uint64_t hi = (uint64_t)lut[b0 >> 4] |
                            ((uint64_t)lut[b2 >> 4] << 16) |
                            ((uint64_t)lut[b1 >> 4] << 32) |
                            ((uint64_t)lut[b3 >> 4] << 48);
        reinterpret_cast<uint64_t*>(dst)[0] = lo;
        reinterpret_cast<uint64_t*>(dst)[1] = hi;
        return;
    }

    const uint8_t q[8] = {
        (uint8_t)(b0 & 0x0F), (uint8_t)(b2 & 0x0F), (uint8_t)(b1 & 0x0F), (uint8_t)(b3 & 0x0F),
        (uint8_t)(b0 >> 4),   (uint8_t)(b2 >> 4),   (uint8_t)(b1 >> 4),   (uint8_t)(b3 >> 4),
    };
    for (int c = 0; c < 8 && channelBase + c < ic; ++c) {
        dst[c] = lut[q[c]];
    }
}

static inline void store_int4_group_fp32(float* dstRow, int channelBase, int ic, const uint8_t* src,
                                         const float* lut) {
    const uint8_t b0 = src[0];
    const uint8_t b1 = src[1];
    const uint8_t b2 = src[2];
    const uint8_t b3 = src[3];
    if (channelBase + 7 < ic) {
        dstRow[channelBase + 0] = lut[b0 & 0x0F];
        dstRow[channelBase + 1] = lut[b2 & 0x0F];
        dstRow[channelBase + 2] = lut[b1 & 0x0F];
        dstRow[channelBase + 3] = lut[b3 & 0x0F];
        dstRow[channelBase + 4] = lut[b0 >> 4];
        dstRow[channelBase + 5] = lut[b2 >> 4];
        dstRow[channelBase + 6] = lut[b1 >> 4];
        dstRow[channelBase + 7] = lut[b3 >> 4];
        return;
    }

    const uint8_t q[8] = {
        (uint8_t)(b0 & 0x0F), (uint8_t)(b2 & 0x0F), (uint8_t)(b1 & 0x0F), (uint8_t)(b3 & 0x0F),
        (uint8_t)(b0 >> 4),   (uint8_t)(b2 >> 4),   (uint8_t)(b1 >> 4),   (uint8_t)(b3 >> 4),
    };
    for (int c = 0; c < 8 && channelBase + c < ic; ++c) {
        dstRow[channelBase + c] = lut[q[c]];
    }
}

static void shared_gather_int4(__fp16* outputFp16, float* outputFp32, const int32_t* indices, const uint8_t* weight,
                               int32_t selectSize, int32_t ic, int32_t oc, int32_t bytes) {
    const int icP = (ic + 31) / 32;
    const int ocP = (oc + 31) / 32;
    const size_t quantSize = (size_t)icP * ocP * 32 * 16;
    const __fp16* scales = reinterpret_cast<const __fp16*>(weight + quantSize);
    for (int i = 0; i < selectSize; ++i) {
        const int32_t index = indices[i];
        if (bytes == 2) {
            __fp16* dstRow = outputFp16 + (size_t)i * ic;
            if (index < 0 || index >= oc) {
                memset(dstRow, 0, (size_t)ic * sizeof(__fp16));
                continue;
            }
            uint16_t lut[16];
            make_int4_lut_fp16(lut, scales + index);
            const int tileY = index / 32;
            const int yi = index % 32;
            for (int x = 0; x < icP; ++x) {
                const uint8_t* tilePtr = weight + ((size_t)tileY * icP + x) * 32 * 16;
                const uint8_t* rowPtr = tilePtr + yi * 4;
                const int channelBase = x * 32;
                for (int g = 0; g < 4; ++g) {
                    store_int4_group_fp16(dstRow, channelBase + g * 8, ic, rowPtr + g * 128, lut);
                }
            }
            continue;
        }

        float* dstRow = outputFp32 + (size_t)i * ic;
        if (index < 0 || index >= oc) {
            memset(dstRow, 0, (size_t)ic * sizeof(float));
            continue;
        }
        float lut[16];
        make_int4_lut_fp32(lut, (float)scales[index]);
        const int tileY = index / 32;
        const int yi = index % 32;
        for (int x = 0; x < icP; ++x) {
            const uint8_t* tilePtr = weight + ((size_t)tileY * icP + x) * 32 * 16;
            const uint8_t* rowPtr = tilePtr + yi * 4;
            const int channelBase = x * 32;
            for (int g = 0; g < 4; ++g) {
                store_int4_group_fp32(dstRow, channelBase + g * 8, ic, rowPtr + g * 128, lut);
            }
        }
    }
}

static void shared_gather_int4_raw(__fp16* outputFp16, float* outputFp32, const int32_t* indices, const uint8_t* weight,
                                   int32_t selectSize, int32_t ic, int32_t oc, int32_t bytes) {
    const size_t rowBytes = ((size_t)ic + 1) / 2;
    const __fp16* scales = reinterpret_cast<const __fp16*>(weight + rowBytes * oc);

    for (int i = 0; i < selectSize; ++i) {
        const int32_t index = indices[i];
        if (bytes == 2) {
            __fp16* dstRow = outputFp16 + (size_t)i * ic;
            if (index < 0 || index >= oc) {
                memset(dstRow, 0, (size_t)ic * sizeof(__fp16));
                continue;
            }
            const uint8_t* srcRow = weight + (size_t)index * rowBytes;
            uint16_t lut[16];
            make_int4_lut_fp16(lut, scales + index);
            const int pairCount = ic / 2;
            for (int p = 0; p < pairCount; ++p) {
                const uint8_t val = srcRow[p];
                reinterpret_cast<uint16_t*>(dstRow)[2 * p] = lut[val >> 4];
                reinterpret_cast<uint16_t*>(dstRow)[2 * p + 1] = lut[val & 0x0F];
            }
            if (ic & 1) {
                const uint8_t val = srcRow[pairCount];
                reinterpret_cast<uint16_t*>(dstRow)[ic - 1] = lut[val >> 4];
            }
            continue;
        }

        float* dstRow = outputFp32 + (size_t)i * ic;
        if (index < 0 || index >= oc) {
            memset(dstRow, 0, (size_t)ic * sizeof(float));
            continue;
        }
        const uint8_t* srcRow = weight + (size_t)index * rowBytes;
        float lut[16];
        make_int4_lut_fp32(lut, (float)scales[index]);
        const int pairCount = ic / 2;
        for (int p = 0; p < pairCount; ++p) {
            const uint8_t val = srcRow[p];
            dstRow[2 * p] = lut[val >> 4];
            dstRow[2 * p + 1] = lut[val & 0x0F];
        }
        if (ic & 1) {
            const uint8_t val = srcRow[pairCount];
            dstRow[ic - 1] = lut[val >> 4];
        }
    }
}

AEEResult htp_ops_shared_gather(uint8_t* dst, uint8_t* indices_ptr, uint8_t* weight_ptr, int32_t selectSize,
                                int32_t ic, int32_t oc, int32_t bytes, int32_t isInt4) {
    if (dst == nullptr || indices_ptr == nullptr || weight_ptr == nullptr) {
        return AEE_EBADPARM;
    }
    if (selectSize <= 0 || ic <= 0 || oc <= 0) {
        return 0;
    }
    if (bytes != 2 && bytes != 4) {
        return AEE_EBADPARM;
    }

    const int32_t* indices = reinterpret_cast<const int32_t*>(indices_ptr);
    if (isInt4 == 2) {
        shared_gather_int4_raw(reinterpret_cast<__fp16*>(dst), reinterpret_cast<float*>(dst), indices,
                               reinterpret_cast<const uint8_t*>(weight_ptr), selectSize, ic, oc, bytes);
        return 0;
    }
    if (isInt4 != 0) {
        shared_gather_int4(reinterpret_cast<__fp16*>(dst), reinterpret_cast<float*>(dst), indices,
                           reinterpret_cast<const uint8_t*>(weight_ptr), selectSize, ic, oc, bytes);
        return 0;
    }

    const __fp16* weight = reinterpret_cast<const __fp16*>(weight_ptr);
    const int icP = (ic + 31) / 32;

    if (bytes == 2) {
        __fp16* output = reinterpret_cast<__fp16*>(dst);
        for (int i = 0; i < selectSize; ++i) {
            __fp16* dstRow = output + (size_t)i * ic;
            const int32_t index = indices[i];
            if (index < 0 || index >= oc) {
                memset(dstRow, 0, (size_t)ic * sizeof(__fp16));
                continue;
            }
            const int tileY = index / 32;
            const int yi = index % 32;
            for (int x = 0; x < icP; ++x) {
                const __fp16* tile = weight + ((size_t)tileY * icP + x) * 32 * 32;
                const int remain = ic - x * 32;
                const int channelCount = remain < 32 ? remain : 32;
                __fp16* dstBlock = dstRow + x * 32;
                int pairCount = channelCount / 2;
                for (int p = 0; p < pairCount; ++p) {
                    const __fp16* srcPair = tile + p * 64 + yi * 2;
                    dstBlock[2 * p] = srcPair[0];
                    dstBlock[2 * p + 1] = srcPair[1];
                }
                if (channelCount & 1) {
                    dstBlock[channelCount - 1] = *(tile + pairCount * 64 + yi * 2);
                }
            }
        }
        return 0;
    }

    float* output = reinterpret_cast<float*>(dst);
    for (int i = 0; i < selectSize; ++i) {
        float* dstRow = output + (size_t)i * ic;
        const int32_t index = indices[i];
        if (index < 0 || index >= oc) {
            memset(dstRow, 0, (size_t)ic * sizeof(float));
            continue;
        }
        const int tileY = index / 32;
        const int yi = index % 32;
        for (int x = 0; x < icP; ++x) {
            const __fp16* tile = weight + ((size_t)tileY * icP + x) * 32 * 32;
            const int remain = ic - x * 32;
            const int channelCount = remain < 32 ? remain : 32;
            float* dstBlock = dstRow + x * 32;
            int pairCount = channelCount / 2;
            for (int p = 0; p < pairCount; ++p) {
                const __fp16* srcPair = tile + p * 64 + yi * 2;
                dstBlock[2 * p] = (float)srcPair[0];
                dstBlock[2 * p + 1] = (float)srcPair[1];
            }
            if (channelCount & 1) {
                dstBlock[channelCount - 1] = (float)*(tile + pairCount * 64 + yi * 2);
            }
        }
    }
    return 0;
}

} // extern "C"
