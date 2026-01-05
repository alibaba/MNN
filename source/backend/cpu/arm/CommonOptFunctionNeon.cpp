#include "core/Macro.h"
#include "../compute/CommonOptFunction.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#include "./FunctionSummary.hpp"
#include "core/MemoryFormater.h"
extern "C" {
void MNNTranspose32Bit4x4(int32_t* dstO, const int32_t* srcO, int32_t* dim);
void MNNTranspose16Bit8x8(int16_t* dstO, const int16_t* srcO, int32_t* dim);
}

static inline float vmaxvq_f32_compat(float32x4_t v) {
    #if defined(__aarch64__)
        return vmaxvq_f32(v);
    #else
        float32x2_t p = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
        p = vpmax_f32(p, p);
        return vget_lane_f32(p, 0);
    #endif
    }

    static inline float vminvq_f32_compat(float32x4_t v) {
    #if defined(__aarch64__)
        return vminvq_f32(v);
    #else
        float32x2_t step1 = vpmin_f32(vget_low_f32(v), vget_high_f32(v));
        step1 = vpmin_f32(step1, step1);
        return vget_lane_f32(step1, 0);
    #endif
    }

    static inline float vaddvq_f32_compat(float32x4_t v) {
    #if defined(__aarch64__)
        return vaddvq_f32(v);
    #else
        float32x2_t p = vpadd_f32(vget_low_f32(v), vget_high_f32(v));
        p = vpadd_f32(p, p);
        return vget_lane_f32(p, 0);
    #endif
    }

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#ifdef __aarch64__
void MNNQuantAttentionKey(int8_t* dst, const float* source, float* sumKeyPtr, float* maxKeyPtr, int32_t* params) {
    int32_t kvNumHead = params[0];
    int32_t seqLen = params[1];
    int32_t headDim = params[2];
    int32_t blockNum = params[3];
    int32_t eP = params[4];
    int32_t lP = params[5];
    int32_t hP = params[6];
    int32_t pastLength = params[7];
    int32_t kvHeadIdx = params[8];

    auto blockHeadDim = UP_DIV(headDim, blockNum);
    auto weightStride1 = ROUND_UP(blockHeadDim, lP) * hP;
    auto weightStride2 = lP * hP;
    auto packedWeightStride1 = weightStride1 + 2 * 4 * hP;

    int8_t tempBuffer[8];

    if (seqLen > 1) {
        // get max
        for (int s = 0; s < seqLen; ++s) {
            const float* keySrc = source + s * kvNumHead * headDim + kvHeadIdx * headDim;
            int d = 0;
            for (; d <= headDim - 8; d += 8) {
                float32x4_t max_vec0 = vld1q_f32(maxKeyPtr + d);
                float32x4_t max_vec1 = vld1q_f32(maxKeyPtr + d + 4);
                float32x4_t src_vec0 = vld1q_f32(keySrc + d);
                float32x4_t src_vec1 = vld1q_f32(keySrc + d + 4);
                max_vec0 = vmaxq_f32(max_vec0, src_vec0);
                max_vec1 = vmaxq_f32(max_vec1, src_vec1);
                vst1q_f32(maxKeyPtr + d, max_vec0);
                vst1q_f32(maxKeyPtr + d + 4, max_vec1);
            }
            for (; d <= headDim - 4; d += 4) {
                float32x4_t max_vec = vld1q_f32(maxKeyPtr + d);
                float32x4_t src_vec = vld1q_f32(keySrc + d);
                max_vec = vmaxq_f32(max_vec, src_vec);
                vst1q_f32(maxKeyPtr + d, max_vec);
            }
            for (; d < headDim; ++d) {
                maxKeyPtr[d] = ALIMAX(maxKeyPtr[d], keySrc[d]);
            }
        }
    }

    for (int s = 0; s < seqLen; s++) {
        const float* keySrc = source + s * kvNumHead * headDim + kvHeadIdx * headDim;

        float32x4_t min_vec = vdupq_n_f32(keySrc[0] - maxKeyPtr[0]);
        float32x4_t max_vec = vdupq_n_f32(keySrc[0] - maxKeyPtr[0]);

        int d = 0;
        for (; d <= headDim - 4; d += 4) {
            float32x4_t src_vec = vld1q_f32(keySrc + d);
            float32x4_t max_key_vec = vld1q_f32(maxKeyPtr + d);
            float32x4_t keydata_vec = vsubq_f32(src_vec, max_key_vec);

            min_vec = vminq_f32(min_vec, keydata_vec);
            max_vec = vmaxq_f32(max_vec, keydata_vec);
        }
        // Reduction
        float minKey = vminvq_f32_compat(min_vec);
        float maxKey = vmaxvq_f32_compat(max_vec);

        // remain headDim
        for (; d < headDim; ++d) {
            auto keydata = keySrc[d] - maxKeyPtr[d];
            minKey = ALIMIN(minKey, keydata);
            maxKey = ALIMAX(maxKey, keydata);
        }

        int outIndex = (pastLength + s) / hP;
        int inIndex  = (pastLength + s) % hP;

        float range = maxKey - minKey;
        float quantScaleVal = 0;
        float biasVal = minKey + 128.0f * (range) / 255.0f;
        if (range <= 1e-6f) {
            quantScaleVal = 0.0f;
        } else {
            quantScaleVal = 255.0f / range;
        }

        for (int k = 0; k < blockNum; ++k) {
            int8_t* weightDstBase = dst + outIndex * blockNum * packedWeightStride1 + k * packedWeightStride1;
            float* scaleDst = (float*)(weightDstBase + weightStride1);
            float* biasDst = scaleDst + hP;

            scaleDst[inIndex] = range / 255.f;
            biasDst[inIndex] = biasVal;

            float32x4_t scaleVec = vdupq_n_f32(quantScaleVal);
            float32x4_t negBiasVec = vdupq_n_f32(-minKey);
            float32x4_t neg128Vec = vdupq_n_f32(-128.0f);

            const float* currentKeyBlock = keySrc + k * blockHeadDim;
            const float* currentMaxBlock = maxKeyPtr + k * blockHeadDim;

            int32x4_t sumInt32_0 = vdupq_n_s32(0);
            int32x4_t sumInt32_1 = vdupq_n_s32(0);
            int headDimIdx = 0;
            for (; headDimIdx <= blockHeadDim - 8; headDimIdx += 8) {
                float32x4_t srcVec0 = vld1q_f32(currentKeyBlock + headDimIdx);
                float32x4_t srcVec1 = vld1q_f32(currentKeyBlock + headDimIdx + 4);
                float32x4_t maxVec0 = vld1q_f32(currentMaxBlock + headDimIdx);
                float32x4_t maxVec1 = vld1q_f32(currentMaxBlock + headDimIdx + 4);

                float32x4_t keyData0 = vsubq_f32(srcVec0, maxVec0);
                float32x4_t keyData1 = vsubq_f32(srcVec1, maxVec1);

                keyData0 = vaddq_f32(keyData0, negBiasVec);
                keyData1 = vaddq_f32(keyData1, negBiasVec);

                keyData0 = vmulq_f32(keyData0, scaleVec);
                keyData1 = vmulq_f32(keyData1, scaleVec);

                keyData0 = vaddq_f32(keyData0, neg128Vec);
                keyData1 = vaddq_f32(keyData1, neg128Vec);

                int32x4_t s32_0 = vcvtaq_s32_f32(keyData0);
                int32x4_t s32_1 = vcvtaq_s32_f32(keyData1);

                sumInt32_0 = vaddq_s32(sumInt32_0, s32_0);
                sumInt32_1 = vaddq_s32(sumInt32_1, s32_1);

                int16x4_t s16_0 = vmovn_s32(s32_0);
                int16x4_t s16_1 = vmovn_s32(s32_1);

                int16x8_t s16Combined = vcombine_s16(s16_0, s16_1);

                int8x8_t s8Vec = vqmovn_s16(s16Combined);

                if (lP == 8) {
                    int i = headDimIdx / lP;
                    int8_t* dstPtr = weightDstBase + i * weightStride2 + inIndex * lP;
                    vst1_s8(dstPtr, s8Vec);
                } else if (lP == 4) {
                    vst1_s8(tempBuffer, s8Vec);
                    int iLow = headDimIdx / lP;
                    int iHigh = (headDimIdx + 4) / lP;

                    int8_t* dstPtrLow = weightDstBase + iLow * weightStride2 + inIndex * lP;
                    int8_t* dstPtrHigh = weightDstBase + iHigh * weightStride2 + inIndex * lP;

                    std::memcpy(dstPtrLow, tempBuffer, 4);
                    std::memcpy(dstPtrHigh, tempBuffer + 4, 4);
                } else {
                    vst1_s8(tempBuffer, s8Vec);
                    for (int k = 0; k < 8; ++k) {
                        int headDimCurr = headDimIdx + k;
                        int i = headDimCurr / lP;
                        int j = headDimCurr % lP;
                        weightDstBase[i * weightStride2 + inIndex * lP + j] = tempBuffer[k];
                    }
                }
            }

            int32_t sumInt32 = vaddvq_s32(sumInt32_0) + vaddvq_s32(sumInt32_1);

            // remain L
            for (; headDimIdx < blockHeadDim; ++headDimIdx) {
                int i = headDimIdx / lP;
                int j = headDimIdx % lP;
                float keyVal = currentKeyBlock[headDimIdx] - currentMaxBlock[headDimIdx];
                float quant_val = (keyVal - minKey) * quantScaleVal - 128.0f;
                int32_t rounded_val = static_cast<int32_t>(roundf(quant_val));
                int8_t finalVal = static_cast<int8_t>(std::max(-128, std::min(127, rounded_val)));
                weightDstBase[i * weightStride2 + inIndex * lP + j] = finalVal;
                sumInt32 += finalVal;
            }

            // store sum
            sumKeyPtr[outIndex * hP + inIndex] = (float)sumInt32 * range / 255.f + (minKey * blockHeadDim + 128.0f * range * blockHeadDim / 255.0f);
        }
    }
}

void MNNQuantAttentionValue(int8_t* dst, const float* source, float* valueSum, int32_t* params) {
    // float   value src : [kvSeq,kvNumHead,headDim]
    // int8_t  value dest: [updiv(maxLength,flashAttentionBlockKv), updiv(headDim,hp),updiv(flashAttentionBlockKv,lp),hp,lp]
    // float   value sum: [updiv(maxLength,flashAttentionBlockKv), roundup(headDim,hp)]
    int32_t kvNumHead = params[0];
    int32_t seqLen = params[1];
    int32_t headDim = params[2];
    int32_t blockNum = params[3];
    int32_t maxLength = params[4];

    int32_t lP = params[5];
    int32_t hP = params[6];
    int32_t pastLength = params[7];
    int32_t kvHeadIdx = params[8];

    int32_t flashAttentionBlockKv = params[9];

    auto blockKvseq = UP_DIV(seqLen + pastLength, blockNum);
    auto weightStride2 = lP * hP;
    auto weightStride1 = UP_DIV(flashAttentionBlockKv, lP) * weightStride2;

    auto packedStride1 = (int)(weightStride1 + 2 * hP * sizeof(float));
    auto packedStride0 = UP_DIV(headDim, hP) * packedStride1;

    auto srcStride0 = kvNumHead * headDim;

    auto sourceFp32 = (float*)source;

    // quant scale & bias
    if (pastLength == 0) {
        for (int d = 0; d < headDim; ++d) {
            float* scalePtr = (float*)(dst + (d / hP) * packedStride1 + weightStride1) + (d % hP);
            float* biasPtr = scalePtr + hP;

            // find min,max
            float dMax = sourceFp32[d + kvHeadIdx * headDim];
            float dMin = dMax;
            for (int s = 0; s < seqLen; ++s) {
                float data = sourceFp32[s * srcStride0 + d + kvHeadIdx * headDim];
                dMax = ALIMAX(dMax, data);
                dMin = ALIMIN(dMin, data);
            }

            // scale & bias
            float range = dMax - dMin;
            if (range < 1e-6) {
                scalePtr[0] = 0.f;
                biasPtr[0] = dMax;
            } else {
                float scale = range / 255.f;
                float bias  = range / 255.f * 128.f + dMin;
                scalePtr[0] = scale;
                biasPtr[0] = bias;
            }
        }
    }

    // copy the scale&bias to each blockKv
    //                                    pastLength == 0: First time prefill
    // pastLength % flashAttentionBlockKv == 0: Open a new blockKv
    if (pastLength == 0 || (pastLength % flashAttentionBlockKv) == 0) {
        int32_t d0 = UP_DIV(maxLength, flashAttentionBlockKv);
        int32_t d1 = UP_DIV(headDim, hP);
        for (int k = 0; k < d0; ++k) {
            for (int r = 0; r < d1; ++r) {
                float* scalePtr = (float*)(dst + k * packedStride0 + r * packedStride1 + weightStride1);
                float* biasPtr  = scalePtr + hP;
                memcpy(scalePtr, dst + r * packedStride1 + weightStride1, hP * sizeof(float));
                memcpy(biasPtr, dst + r * packedStride1 + weightStride1 + hP * sizeof(float), hP * sizeof(float));
            }
        }
    }

    // Quant fp16
    for (int d = 0; d < headDim; ++d) {
        // dst address
        int idxBase = (d / hP) * packedStride1 + (d % hP) * lP;
        int8_t*   dstBase = dst + idxBase;
        float*  scaleBase = (float*)(dst + (d / hP) * packedStride1 + weightStride1) + (d % hP);
        float*   biasBase = scaleBase + hP;
        float*   sumBase = valueSum + (d / hP) * hP + (d % hP);

        float qscale = scaleBase[0] < 1e-6 ? 0 : 1.0f / scaleBase[0];
        float qbias = scaleBase[0] < 1e-6 ? 0 : (-biasBase[0] / scaleBase[0]);
        // quant
        for (int s = 0; s < seqLen; ++s) {
            int kvSeqIndx = s + pastLength;
            int idxInner = (kvSeqIndx / flashAttentionBlockKv) * packedStride0 + (kvSeqIndx % flashAttentionBlockKv) / lP * weightStride2 + (kvSeqIndx % flashAttentionBlockKv) % lP;
            float xf = sourceFp32[s * srcStride0 + d + kvHeadIdx * headDim];
            int8_t xq = ALIMAX(ALIMIN(127, static_cast<int32_t>(roundf(xf * qscale + qbias))), -128);
            dstBase[idxInner] = xq;

            // sum
            int idxSum = (kvSeqIndx / flashAttentionBlockKv) * ROUND_UP(headDim, hP);
            sumBase[idxSum] += ((float)xq * scaleBase[0] + biasBase[0]);
        }
    }
}
#endif // __aarch64__
#endif // MNN_SUPPORT_TRANSFORMER_FUSE

void MNNTranspose32Bit(int32_t* dstO, const int32_t* srcO, int32_t* dim) {
    int w = dim[0];
    int h = dim[1];
    auto wC4 = w / 4;
    auto hC4 = h / 4;
    int srcStride = dim[2];
    int dstStride = dim[3];
    if (wC4 > 0 && hC4 > 0) {
        MNNTranspose32Bit4x4(dstO, srcO, dim);
    }
    // Down
    for (int i=hC4 * 4; i<h; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j=0; j<w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
    // Right
    for (int i=0; i<hC4 * 4; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j=wC4 * 4; j<w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
}

void MNNTranspose16Bit(int16_t* dstO, const int16_t* srcO, int32_t* dim) {
    int w = dim[0];
    int h = dim[1];
    auto wC8 = w / 8;
    auto hC8 = h / 8;
    int srcStride = dim[2];
    int dstStride = dim[3];
    if (wC8 > 0 && hC8 > 0) {
        MNNTranspose16Bit8x8(dstO, srcO, dim);
    }

    // Down
    for (int i = hC8 * 8; i < h; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j = 0; j < w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
    // Right
    for (int i = 0; i < hC8 * 8; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j = wC8 * 8; j < w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
}


#define EXP_APPROX_MIN_INPUT vdupq_n_f32(-88.0f)
#define EXP_APPROX_MAX_INPUT vdupq_n_f32(88.0f)
#define EXP_APPROX_LN2         vdupq_n_f32(0.69314718056f)  // ln(2)
#define EXP_APPROX_LN2_INV     vdupq_n_f32(1.44269504089f)   // 1/ln(2)
// Fourth-order polynomial approximation coefficients of exp(r):
// P(x) = c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
#define EXP_APPROX_C4          vdupq_n_f32(0.0416624f)
#define EXP_APPROX_C3          vdupq_n_f32(0.166665f)
#define EXP_APPROX_C2          vdupq_n_f32(0.500000f)
#define EXP_APPROX_C1          vdupq_n_f32(1.0f)
#define EXP_APPROX_C0          vdupq_n_f32(1.0f)

#ifndef __aarch64__
static inline float32x4_t vrndaq_f32_compat(float32x4_t x) {
    float32x4_t sign = vbslq_f32(vdupq_n_u32(0x80000000), x, vdupq_n_f32(0.0f));
    return vcvtq_f32_s32(vcvtq_s32_f32(vaddq_f32(x, vbslq_f32(vcltq_f32(x, vdupq_n_f32(0.0f)), vdupq_n_f32(-0.5f), vdupq_n_f32(0.5f)))));
}
#endif

static inline float32x4_t expApprox(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, EXP_APPROX_MIN_INPUT), EXP_APPROX_MAX_INPUT);

    float32x4_t k_float;
    float32x4_t r;
    float32x4_t exp_r;
#if defined(__aarch64__)
    k_float = vrndaq_f32(vmulq_f32(x, EXP_APPROX_LN2_INV));

    // r = x - k * ln(2)
    r = vfmsq_f32(x, k_float, EXP_APPROX_LN2);

    // P(r) = (c0 + c2*r^2 + c4*r^4) + r*(c1 + c3*r^2)
    float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t p_odd = vfmaq_f32(EXP_APPROX_C1, EXP_APPROX_C3, r2);

    float32x4_t p_even = vfmaq_f32(EXP_APPROX_C0, EXP_APPROX_C2, r2);
    p_even = vfmaq_f32(p_even, EXP_APPROX_C4, vmulq_f32(r2, r2));
    exp_r = vfmaq_f32(p_even, p_odd, r);
#else

    k_float = vrndaq_f32_compat(vmulq_f32(x, EXP_APPROX_LN2_INV));


    r = vsubq_f32(x, vmulq_f32(k_float, EXP_APPROX_LN2));

    // 2. c0 + r*(c1 + r*(c2 + r*(c3 + r*c4)))
    exp_r = vmlaq_f32(EXP_APPROX_C3, EXP_APPROX_C4, r); // c3 + c4*r
    exp_r = vmlaq_f32(EXP_APPROX_C2, exp_r, r);         // c2 + r*(...)
    exp_r = vmlaq_f32(EXP_APPROX_C1, exp_r, r);         // c1 + r*(...)
    exp_r = vmlaq_f32(EXP_APPROX_C0, exp_r, r);         // c0 + r*(...)

#endif

    int32x4_t k_int = vcvtq_s32_f32(k_float);
    int32x4_t k_shifted = vshlq_n_s32(k_int, 23);
    return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(exp_r), k_shifted));
}

void MNNExpC8(float* dst, const float* src, float* offset, const float* parameters, size_t countC8) {
    float32x4_t maxVec = vdupq_n_f32(offset[2]);
    float32x4_t sumVec0 = vdupq_n_f32(0);
    float32x4_t sumVec1 = vdupq_n_f32(0);

    float32x4_t c0 = vdupq_n_f32(offset[0]);
    float32x4_t c1 = vdupq_n_f32(offset[1]);


    for (int i = 0; i < countC8; ++i) {
        float32x4_t srcVec0 = vld1q_f32(src);
        float32x4_t srcVec1 = vld1q_f32(src + 4);
        auto subVec0 = vaddq_f32(vmulq_f32(srcVec0, c0), maxVec);
        auto subVec1 = vaddq_f32(vmulq_f32(srcVec1, c0), maxVec);
        auto expVec0 = vaddq_f32(expApprox(subVec0), c1);
        auto expVec1 = vaddq_f32(expApprox(subVec1), c1);
        vst1q_f32(dst, expVec0);
        vst1q_f32(dst + 4, expVec1);
        sumVec0 = vaddq_f32(sumVec0, expVec0);
        sumVec1 = vaddq_f32(sumVec1, expVec1);

        src += 8;
        dst += 8;
    }

    sumVec0 = vaddq_f32(sumVec0, sumVec1);
    float32x2_t sumP = vpadd_f32(vget_low_f32(sumVec0), vget_high_f32(sumVec0));
    sumP = vpadd_f32(sumP, sumP);
    offset[3] += vget_lane_f32(sumP, 0);
}


void MNNExp(float* destPtr, const float* srcPtr, float* offset, size_t size) {
    float32x4_t maxVec = vdupq_n_f32(-offset[2]);
    float32x4_t sumVec0 = vdupq_n_f32(0);
    float32x4_t sumVec1 = vdupq_n_f32(0);
    if (offset[0] == 1.f && offset[1] == 0.f) {
        while (size >= 8) {
            float32x4_t srcVec0 = vld1q_f32(srcPtr);
            float32x4_t srcVec1 = vld1q_f32(srcPtr + 4);
            auto subVec0 = vsubq_f32(srcVec0, maxVec);
            auto subVec1 = vsubq_f32(srcVec1, maxVec);
            auto expVec0 = expApprox(subVec0);
            auto expVec1 = expApprox(subVec1);
            vst1q_f32(destPtr, expVec0);
            vst1q_f32(destPtr + 4, expVec1);
            sumVec0 = vaddq_f32(sumVec0, expVec0);
            sumVec1 = vaddq_f32(sumVec1, expVec1);
            srcPtr += 8;
            destPtr += 8;
            size -= 8;

        }
        while (size >= 4) {
            float32x4_t srcVec0 = vld1q_f32(srcPtr);
            auto subVec0 = vsubq_f32(srcVec0, maxVec);
            auto expVec0 = expApprox(subVec0);
            sumVec0 = vaddq_f32(sumVec0, expVec0);
            vst1q_f32(destPtr, expVec0);
            srcPtr += 4;
            destPtr += 4;
            size -= 4;
        }
        //merge
        sumVec0 = vaddq_f32(sumVec0, sumVec1);
        float32x2_t sumP = vpadd_f32(vget_low_f32(sumVec0), vget_high_f32(sumVec0));
        sumP = vpadd_f32(sumP, sumP);
        auto newSum = vget_lane_f32(sumP, 0);
        if (size > 0) {
            float tmp[4];
            memcpy(tmp, srcPtr, size * sizeof(float));
            float32x4_t srcVec0 = vld1q_f32(tmp);
            auto subVec0 = vsubq_f32(srcVec0, maxVec);
            auto expVec0 = expApprox(subVec0);
            vst1q_f32(tmp, expVec0);
            for (int i = 0; i < size; ++i) {
                newSum += tmp[i];
                destPtr[i] = tmp[i];
            }
        }
        offset[3] += newSum;
    } else {
        float32x4_t c0 = vdupq_n_f32(offset[0]);
        float32x4_t c1 = vdupq_n_f32(offset[1]);
        while (size >= 8) {
            float32x4_t srcVec0 = vld1q_f32(srcPtr);
            float32x4_t srcVec1 = vld1q_f32(srcPtr + 4);
            auto subVec0 = vsubq_f32(vmulq_f32(srcVec0, c0), maxVec);
            auto subVec1 = vsubq_f32(vmulq_f32(srcVec1, c0), maxVec);
            auto expVec0 = vaddq_f32(expApprox(subVec0), c1);
            auto expVec1 = vaddq_f32(expApprox(subVec1), c1);
            vst1q_f32(destPtr, expVec0);
            vst1q_f32(destPtr + 4, expVec1);
            sumVec0 = vaddq_f32(sumVec0, expVec0);
            sumVec1 = vaddq_f32(sumVec1, expVec1);
            srcPtr += 8;
            destPtr += 8;
            size -= 8;

        }
        while (size >= 4) {
            float32x4_t srcVec0 = vld1q_f32(srcPtr);
            auto subVec0 = vsubq_f32(vmulq_f32(srcVec0, c0), maxVec);
            auto expVec0 = vaddq_f32(expApprox(subVec0), c1);
            sumVec0 = vaddq_f32(sumVec0, expVec0);
            vst1q_f32(destPtr, expVec0);
            srcPtr += 4;
            destPtr += 4;
            size -= 4;
        }
        //merge
        sumVec0 = vaddq_f32(sumVec0, sumVec1);
        float32x2_t sumP = vpadd_f32(vget_low_f32(sumVec0), vget_high_f32(sumVec0));
        sumP = vpadd_f32(sumP, sumP);
        auto newSum = vget_lane_f32(sumP, 0);
        if (size > 0) {
            float tmp[4];
            memcpy(tmp, srcPtr, size * sizeof(float));
            float32x4_t srcVec0 = vld1q_f32(tmp);
            auto subVec0 = vsubq_f32(vmulq_f32(srcVec0, c0), maxVec);
            auto expVec0 = vaddq_f32(expApprox(subVec0), c1);
            vst1q_f32(tmp, expVec0);
            for (int i = 0; i < size; ++i) {
                newSum += tmp[i];
                destPtr[i] = tmp[i];
            }
        }
        offset[3] += newSum;
    }
}


static inline void transposeAndStore4x4(const float* srcRowPtrs[4], float* dstColBase, size_t dstColStride) {
    float32x4_t row0 = vld1q_f32(srcRowPtrs[0]);
    float32x4_t row1 = vld1q_f32(srcRowPtrs[1]);
    float32x4_t row2 = vld1q_f32(srcRowPtrs[2]);
    float32x4_t row3 = vld1q_f32(srcRowPtrs[3]);

    // Step 1: Transpose 2x2 blocks of 2-element vectors
    float32x4x2_t t01 = vtrnq_f32(row0, row1);
    float32x4x2_t t23 = vtrnq_f32(row2, row3);

    // Step 2: Combine the results to get the full transpose
    float32x4_t col0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
    float32x4_t col1 = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
    float32x4_t col2 = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
    float32x4_t col3 = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));

    vst1q_f32(dstColBase, col0);
    vst1q_f32(dstColBase + dstColStride, col1);
    vst1q_f32(dstColBase + 2 * dstColStride, col2);
    vst1q_f32(dstColBase + 3 * dstColStride, col3);
}

inline void smartCopy(void* dest, const void* src, size_t size) {
    switch (size) {
        case 1:
            *(uint8_t*)dest = *(const uint8_t*)src;
            break;
        case 2:
            *(uint16_t*)dest = *(const uint16_t*)src;
            break;
        case 4:
            *(uint32_t*)dest = *(const uint32_t*)src;
            break;
        case 8:
            *(uint64_t*)dest = *(const uint64_t*)src;
            break;
        default:
            ::memcpy(dest, src, size);
            break;
    }
}

void MNNPackForMatMul_A(float* dst, const float* src, size_t E, size_t L, size_t eP, size_t lP, size_t bytes) {
    if (E == 0 || L == 0) {
        return;
    }

    // source [E, L] <=> [E/eP, eP, L/lP, lP]
    // dest   [E/eP, L/lP, eP, lP]

    if (lP > 1) {
        auto eU = UP_DIV(E, eP);
        auto lU = UP_DIV(L, lP);
        const size_t lC = L / lP;
        const size_t lR = L % lP;
        const size_t copySizeBytes = (size_t)lP * bytes;

        const size_t srcStride0 = (size_t)L * bytes;
        const size_t dstStride0 = (size_t)lU * eP * lP * bytes;
        const size_t dstStride1 = eP * lP * bytes;
        const size_t dstStride2 = lP * bytes;

        for (int i = 0; i < eU; ++i) {
            const size_t xC = ALIMIN(eP, E - i * eP);
            const uint8_t* APtr = (uint8_t*)src + (i * eP) * srcStride0;
            uint8_t* ADst = (uint8_t*)dst + i * dstStride0;

            if (lC > 0) {
                for (int x = 0; x < xC; ++x) {
                    auto srcBase = APtr + x * srcStride0;
                    auto destBase = ADst + x * dstStride2;

                    for (int yy = 0; yy < lC; ++yy) {
                        auto srcPtr = srcBase + (size_t)yy * copySizeBytes;
                        auto destPtr = destBase + (size_t)yy * dstStride1;

                        smartCopy(destPtr, srcPtr, copySizeBytes);
                    }
                }
            }

            if (lR > 0) {
                const size_t remainderCopyBytes = (size_t)lR * bytes;

                for (int x = 0; x < xC; ++x) {
                    auto srcPtr = APtr + x * srcStride0 + lC * lP * bytes;
                    auto destPtr = ADst + lC * dstStride1 + x * dstStride2;

                    ::memcpy(destPtr, srcPtr, remainderCopyBytes);
                    ::memset(destPtr + remainderCopyBytes, 0, copySizeBytes - remainderCopyBytes);
                }
            }
        }
        return;
    }

    const int kTileS = 4; // Tiling size for E dimension
    const int kTileK = 4; // Tiling size for L dimension
    const size_t dstSOuterStride = L * eP;

    int s = 0;
    for (; s + kTileS <= E; s += kTileS) {
        const int sOuter = s / eP;
        const int sInner = s % eP;
        if (sInner + kTileS > eP) {
            break;
        }

        float* dstSBase = dst + sOuter * dstSOuterStride + sInner;
        const float* srcRowPtrs[kTileS];
        srcRowPtrs[0] = src + (s + 0) * L;
        srcRowPtrs[1] = src + (s + 1) * L;
        srcRowPtrs[2] = src + (s + 2) * L;
        srcRowPtrs[3] = src + (s + 3) * L;

        int k = 0;
        for (; k + kTileK <= L; k += kTileK) {
            const float* currentSrcPtrs[kTileS];
            currentSrcPtrs[0] = srcRowPtrs[0] + k;
            currentSrcPtrs[1] = srcRowPtrs[1] + k;
            currentSrcPtrs[2] = srcRowPtrs[2] + k;
            currentSrcPtrs[3] = srcRowPtrs[3] + k;
            float* dstKBase = dstSBase + k * eP;
            transposeAndStore4x4(currentSrcPtrs, dstKBase, eP);
        }

        for (; k < L; ++k) {
            float buffer[kTileS] = {
                srcRowPtrs[0][k],
                srcRowPtrs[1][k],
                srcRowPtrs[2][k],
                srcRowPtrs[3][k]
            };
            vst1q_f32(dstSBase + k * eP, vld1q_f32(buffer));
        }
    }

    for (; s < E; ++s) {
        const int sOuter = s / eP;
        const int sInner = s % eP;
        const float* srcRow = src + s * L;
        float* dstSBase = dst + sOuter * dstSOuterStride + sInner;
        for (int k = 0; k < L; ++k) {
            dstSBase[k * eP] = srcRow[k];
        }
    }
}


void MNNSoftmaxFp32_Pack1(float* softmaxDst, const float* softmaxSrc, float* runningMax, float* runningSum, float* updateScale, int outside, int reduceSize, int kvSeqOffset, int validOffset, bool mask) {

    for (int k = 0; k < outside; ++k) {
        int currentValidSize = reduceSize;
        bool isRowValid = true;

        if (mask) {
            if (kvSeqOffset > k + validOffset) {
                isRowValid = false;
                currentValidSize = 0;
                if (updateScale) updateScale[k] = 1.0f;
            } else {
                currentValidSize = ALIMIN(reduceSize, k + (validOffset + 1) - kvSeqOffset);
            }
        }

        float* dstRow = softmaxDst + k * reduceSize;

        if (!isRowValid || currentValidSize == 0) {
            memset(dstRow, 0, reduceSize * sizeof(float));
            continue;
        }

        const float* srcRow = softmaxSrc + k * reduceSize;

        // 1. Find max
        float oldMax = std::numeric_limits<float>::lowest();
        if (runningMax) oldMax = runningMax[k];

        float32x4_t maxVec = vdupq_n_f32(std::numeric_limits<float>::lowest());

        int i = 0;
        // Unroll 4 vectors (16 floats) per iteration
        for (; i <= currentValidSize - 16; i += 16) {
            float32x4_t v0 = vld1q_f32(srcRow + i + 0);
            float32x4_t v1 = vld1q_f32(srcRow + i + 4);
            float32x4_t v2 = vld1q_f32(srcRow + i + 8);
            float32x4_t v3 = vld1q_f32(srcRow + i + 12);

            maxVec = vmaxq_f32(maxVec, v0);
            maxVec = vmaxq_f32(maxVec, v1);
            maxVec = vmaxq_f32(maxVec, v2);
            maxVec = vmaxq_f32(maxVec, v3);
        }
        // Handle remaining blocks of 4
        for (; i <= currentValidSize - 4; i += 4) {
            maxVec = vmaxq_f32(maxVec, vld1q_f32(srcRow + i));
        }

        // Reduction
        float newMax = vmaxvq_f32_compat(maxVec);

        // Scalar Tail
        for (; i < currentValidSize; ++i) {
            newMax = ALIMAX(newMax, srcRow[i]);
        }

        float finalMax = ALIMAX(oldMax, newMax);
        float32x4_t finalMaxVec = vdupq_n_f32(finalMax);

        // 2. Exp & Sum & Store (4-way Unroll)
        float sum = 0.0f;
        float32x4_t sumVec = vdupq_n_f32(0.0f);

        i = 0;
        // Unroll 4 vectors (16 floats)
        for (; i <= currentValidSize - 16; i += 16) {
            float32x4_t v0 = vld1q_f32(srcRow + i + 0);
            float32x4_t v1 = vld1q_f32(srcRow + i + 4);
            float32x4_t v2 = vld1q_f32(srcRow + i + 8);
            float32x4_t v3 = vld1q_f32(srcRow + i + 12);

            // Sub Max
            v0 = vsubq_f32(v0, finalMaxVec);
            v1 = vsubq_f32(v1, finalMaxVec);
            v2 = vsubq_f32(v2, finalMaxVec);
            v3 = vsubq_f32(v3, finalMaxVec);

            // Exp (Expensive operation, pipeline parallelism helps here)
            v0 = expApprox(v0);
            v1 = expApprox(v1);
            v2 = expApprox(v2);
            v3 = expApprox(v3);

            // Accumulate Sum
            sumVec = vaddq_f32(sumVec, v0);
            sumVec = vaddq_f32(sumVec, v1);
            sumVec = vaddq_f32(sumVec, v2);
            sumVec = vaddq_f32(sumVec, v3);

            // Store (Temporary exp values)
            vst1q_f32(dstRow + i + 0, v0);
            vst1q_f32(dstRow + i + 4, v1);
            vst1q_f32(dstRow + i + 8, v2);
            vst1q_f32(dstRow + i + 12, v3);
        }

        // Remaining blocks of 4
        for (; i <= currentValidSize - 4; i += 4) {
            float32x4_t v = vld1q_f32(srcRow + i);
            v = vsubq_f32(v, finalMaxVec);
            v = expApprox(v);
            sumVec = vaddq_f32(sumVec, v);
            vst1q_f32(dstRow + i, v);
        }

        // Scalar Tail
        for (; i < currentValidSize; ++i) {
            float val = expf(srcRow[i] - finalMax);
            sum += val;
            dstRow[i] = val;
        }

        sum += vaddvq_f32_compat(sumVec);

        if (currentValidSize < reduceSize) {
            memset(dstRow + currentValidSize, 0, (reduceSize - currentValidSize) * sizeof(float));
        }

        // 3. Update running max & sum or Normalize
        if (runningMax && runningSum && updateScale) {
            float scaleForSum = expf(oldMax - finalMax);
            runningSum[k] = runningSum[k] * scaleForSum + sum;
            runningMax[k] = finalMax;
            updateScale[k] = scaleForSum;
        } else {
            if (runningMax && runningSum) {
                sum += runningSum[k] * expf(oldMax - finalMax);
            }
            float scale = 1.0f / (sum + 1e-20f);
            float32x4_t scaleVec = vdupq_n_f32(scale);

            i = 0;
            // Unroll 4 vectors
            for (; i <= currentValidSize - 16; i += 16) {
                float32x4_t v0 = vld1q_f32(dstRow + i + 0);
                float32x4_t v1 = vld1q_f32(dstRow + i + 4);
                float32x4_t v2 = vld1q_f32(dstRow + i + 8);
                float32x4_t v3 = vld1q_f32(dstRow + i + 12);

                vst1q_f32(dstRow + i + 0, vmulq_f32(v0, scaleVec));
                vst1q_f32(dstRow + i + 4, vmulq_f32(v1, scaleVec));
                vst1q_f32(dstRow + i + 8, vmulq_f32(v2, scaleVec));
                vst1q_f32(dstRow + i + 12, vmulq_f32(v3, scaleVec));
            }

            for (; i <= currentValidSize - 4; i += 4) {
                float32x4_t v = vld1q_f32(dstRow + i);
                vst1q_f32(dstRow + i, vmulq_f32(v, scaleVec));
            }

            for (; i < currentValidSize; ++i) {
                dstRow[i] *= scale;
            }
        }
    }
}

void MNNSoftmaxFp32_Pack4(float* softmaxDst, const float* softmaxSrc, float* runningMax, float* runningSum, float* updateScale, int outside, int reduceSize, int kvSeqOffset, int validOffset, bool mask) {
    const int packUnit = 4;
    int reduceSizeOuter = UP_DIV(reduceSize, packUnit);
    int stride0 = outside * packUnit;

    for (int k = 0; k < outside; k += 4) {
        int count = ALIMIN(4, outside - k);

        int validTotalLen[4];
        int fullBlocks[4];
        int remain[4];
        bool isRowValid[4];

        for (int i = 0; i < count; ++i) {
            int currentK = k + i;
            if (mask && kvSeqOffset > currentK + validOffset) {
                isRowValid[i] = false;
                validTotalLen[i] = 0;
                if (updateScale) updateScale[currentK] = 1.0f;
            } else {
                isRowValid[i] = true;
                validTotalLen[i] = mask ? ALIMIN(reduceSize, currentK + (validOffset + 1) - kvSeqOffset) : reduceSize;
            }

            fullBlocks[i] = validTotalLen[i] / packUnit;
            remain[i] = validTotalLen[i] % packUnit;
        }

        float currentMax[4];
        float32x4_t vecMaxAccum[4];
        float minVal = std::numeric_limits<float>::lowest();
        float32x4_t minVec = vdupq_n_f32(minVal);

        for (int i = 0; i < count; ++i) {
            currentMax[i] = runningMax ? runningMax[k + i] : minVal;
            vecMaxAccum[i] = minVec;
        }

        for (int j = 0; j < reduceSizeOuter; ++j) {
            auto blockSrcBase = softmaxSrc + j * stride0 + k * packUnit;

            for (int i = 0; i < count; ++i) {
                if (!isRowValid[i]) continue;

                if (j < fullBlocks[i]) {
                    float32x4_t val = vld1q_f32(blockSrcBase + i * packUnit);
                    vecMaxAccum[i] = vmaxq_f32(vecMaxAccum[i], val);
                } else if (j == fullBlocks[i] && remain[i] > 0) {
                    auto ptr = blockSrcBase + i * packUnit;
                    for (int p = 0; p < remain[i]; ++p) {
                        currentMax[i] = ALIMAX(currentMax[i], ptr[p]);
                    }
                }
            }
        }

        // Finalize Max
        float32x4_t finalMaxVec[4];
        for (int i = 0; i < count; ++i) {
            if (!isRowValid[i]) {
                    finalMaxVec[i] = vdupq_n_f32(0.0f);
                    continue;
            }
            float maxInVec = vmaxvq_f32_compat(vecMaxAccum[i]);
            currentMax[i] = ALIMAX(currentMax[i], maxInVec);
            finalMaxVec[i] = vdupq_n_f32(currentMax[i]);
        }

        float currentSum[4] = {0.0f};
        float32x4_t vecSumAccum[4];
        for (int i = 0; i < count; ++i) vecSumAccum[i] = vdupq_n_f32(0.0f);

        for (int j = 0; j < reduceSizeOuter; ++j) {
            auto blockSrcBase = softmaxSrc + j * stride0 + k * packUnit;
            auto blockDstBase = softmaxDst + j * stride0 + k * packUnit;

            for (int i = 0; i < count; ++i) {
                if (!isRowValid[i]) {
                    memset(blockDstBase + i * packUnit, 0, sizeof(float) * 4);
                    continue;
                }

                auto dstPtr = blockDstBase + i * packUnit;

                if (j < fullBlocks[i]) {
                    auto srcPtr = blockSrcBase + i * packUnit;
                    float32x4_t val = vld1q_f32(srcPtr);

                    val = vsubq_f32(val, finalMaxVec[i]);

                    val = expApprox(val);
                    vecSumAccum[i] = vaddq_f32(vecSumAccum[i], val);
                    vst1q_f32(dstPtr, val);

                } else if (j == fullBlocks[i] && remain[i] > 0) {
                    auto srcPtr = blockSrcBase + i * packUnit;
                    for (int p = 0; p < remain[i]; ++p) {
                        float val = expf(srcPtr[p] - currentMax[i]);
                        currentSum[i] += val;
                        dstPtr[p] = val;
                    }
                    memset(dstPtr + remain[i], 0, (packUnit - remain[i]) * sizeof(float));
                } else {
                    memset(dstPtr, 0, sizeof(float) * 4);
                }
            }
        }

        for (int i = 0; i < count; ++i) {
            currentSum[i] += vaddvq_f32_compat(vecSumAccum[i]);
        }

        for (int i = 0; i < count; ++i) {
            int currentK = k + i;
            if (!isRowValid[i]) continue;

            float scale;
            if (runningMax && runningSum && updateScale) {
                float oldMax = runningMax[currentK];
                float scaleForSum = expf(oldMax - currentMax[i]);
                runningSum[currentK] = runningSum[currentK] * scaleForSum + currentSum[i];
                runningMax[currentK] = currentMax[i];
                updateScale[currentK] = scaleForSum;
                continue;
            } else {
                if (runningMax && runningSum) {
                    currentSum[i] += runningSum[currentK] * expf(runningMax[currentK] - currentMax[i]);
                }
                scale = 1.0f / (currentSum[i] + 1e-20f);
            }

            float32x4_t scaleVec = vdupq_n_f32(scale);

            // Normalize Pass
            for (int j = 0; j < reduceSizeOuter; ++j) {
                if (j > fullBlocks[i] || (j == fullBlocks[i] && remain[i] == 0)) {
                    continue;
                }

                auto dstPtr = softmaxDst + j * stride0 + k * packUnit + i * packUnit;

                if (j < fullBlocks[i]) {
                    float32x4_t val = vld1q_f32(dstPtr);
                    val = vmulq_f32(val, scaleVec);
                    vst1q_f32(dstPtr, val);
                } else {
                    // Tail
                    for (int p = 0; p < remain[i]; ++p) {
                        dstPtr[p] *= scale;
                    }
                }
            }
        }
    }
}

void MNNSoftmax(float* softmaxDst, const float* softmaxSrc, float* runningMax, float* runningSum, float* updateScale, int outside, int reduceSize, int kvSeqOffset, int validOffset, int pack, bool mask) {

    // source shape: [reduceSizeOuter, outside, reduceSizeInner]
    // for C4, [up_div(reduceSize,4), outside,4] => reduceSizeOuter=up_div(reduceSize,4), reduceSizeInner=4
    // for C,  [outside, reduceSize]             => reduceSizeOuter=1, reduceSizeInner=reduceSize

    if (pack == 4) {
        MNNSoftmaxFp32_Pack4(softmaxDst, softmaxSrc, runningMax, runningSum, updateScale, outside, reduceSize, kvSeqOffset, validOffset, mask);
        return;
    }
    if (pack == 1) {
        MNNSoftmaxFp32_Pack1(softmaxDst, softmaxSrc, runningMax, runningSum, updateScale, outside, reduceSize, kvSeqOffset, validOffset, mask);
        return;
    }
    MNN_ERROR("MNNSoftmax not support pack != 1 and pack != 4\n");
    return;
}

#ifndef MNN_USE_NEON

void MNNPackedSparseMatMulEpx1(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, unsigned int* NNZMap, int* dataOffsetMap) {

    auto eP = parameter[0] / sizeof(float);
    MNN_ASSERT((eP & 0x03) == 0); // In sparse calculate, eP should be evenly divided by 4
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto hRemain = parameter[4];
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride = bExtraStride + l * 4;
    auto aStride = eP * l; // sizeof(float);
    auto hC4 = UP_DIV(h, 4);
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
    }
    const float32x4_t vmin = vld1q_dup_f32(&minValue);
    const float32x4_t vmax = vld1q_dup_f32(&maxValue);

    // MNN_PRINT("NEON MNNPackedSparseMatMul eP:%lu, eSize:%lu, l:%lu, h:%lu, cStride:%lu, aStride:%lu\n", eP, eSize, l, h, cStride, aStride);

    const float* a = A;
    size_t ie = 0;
    for (ie = 0; ie + eP <= eSize; ie += eP) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (auto ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float32x4_t vacc0123 = vld1q_dup_f32(&initValue);
            float32x4_t vacc4567 = vacc0123;
            float32x4_t vacc89AB = vacc0123;
            float32x4_t vaccCDEF = vacc0123;
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                const float32x4_t va4567 = vld1q_f32(a + 4);
                const float32x4_t va89AB = vld1q_f32(a + 8);
                const float32x4_t vaCDEF = vld1q_f32(a + 12);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_dup_f32(w);
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;

                vacc0123 = vfmaq_f32(vacc0123, va0123, w4);
                vacc4567 = vfmaq_f32(vacc4567, va4567, w4);
                vacc89AB = vfmaq_f32(vacc89AB, va89AB, w4);
                vaccCDEF = vfmaq_f32(vaccCDEF, vaCDEF, w4);

            }
            vacc0123 = vminq_f32(vacc0123, vmax);
            vacc4567 = vminq_f32(vacc4567, vmax);
            vacc89AB = vminq_f32(vacc89AB, vmax);
            vaccCDEF = vminq_f32(vaccCDEF, vmax);
            vacc0123 = vmaxq_f32(vacc0123, vmin);
            vacc4567 = vmaxq_f32(vacc4567, vmin);
            vacc89AB = vmaxq_f32(vacc89AB, vmin);
            vaccCDEF = vmaxq_f32(vaccCDEF, vmin);

            // how to store faster: st4 / transpose /
            vst1q_lane_f32(c, vacc0123, 0);
            vst1q_lane_f32(c + 4, vacc0123, 1);
            vst1q_lane_f32(c + 4 * 2, vacc0123, 2);
            vst1q_lane_f32(c + 4 * 3, vacc0123, 3);
            vst1q_lane_f32(c+  4 * 4, vacc4567, 0);
            vst1q_lane_f32(c + 4 * 5, vacc4567, 1);
            vst1q_lane_f32(c + 4 * 6, vacc4567, 2);
            vst1q_lane_f32(c + 4 * 7, vacc4567, 3);
            vst1q_lane_f32(c + 4 * 8, vacc89AB, 0);
            vst1q_lane_f32(c + 4 * 9, vacc89AB, 1);
            vst1q_lane_f32(c + 4 * 10, vacc89AB, 2);
            vst1q_lane_f32(c + 4 * 11, vacc89AB, 3);
            vst1q_lane_f32(c + 4 * 12, vaccCDEF, 0);
            vst1q_lane_f32(c + 4 * 13, vaccCDEF, 1);
            vst1q_lane_f32(c + 4 * 14, vaccCDEF, 2);
            vst1q_lane_f32(c + 4 * 15, vaccCDEF, 3);
        }
        a += aStride;
    }
    // const float* blockA = A + ie * l;
    if (eSize & 0x08) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (auto ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float32x4_t vacc0123 = vld1q_dup_f32(&initValue);
            float32x4_t vacc4567 = vacc0123;

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                const float32x4_t va4567 = vld1q_f32(a + 4);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_dup_f32(w);
                // MNN_PRINT("8-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-7]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {8});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc0123 = vfmaq_f32(vacc0123, va0123, w4);
                vacc4567 = vfmaq_f32(vacc4567, va4567, w4);
            }
            vacc0123 = vminq_f32(vacc0123, vmax);
            vacc4567 = vminq_f32(vacc4567, vmax);
            vacc0123 = vmaxq_f32(vacc0123, vmin);
            vacc4567 = vmaxq_f32(vacc4567, vmin);

            // how to store faster: st4 / transpose /
            vst1q_lane_f32(c, vacc0123, 0);
            vst1q_lane_f32(c + 4, vacc0123, 1);
            vst1q_lane_f32(c + 4 * 2, vacc0123, 2);
            vst1q_lane_f32(c + 4 * 3, vacc0123, 3);
            vst1q_lane_f32(c + 4 * 4, vacc4567, 0);
            vst1q_lane_f32(c + 4 * 5, vacc4567, 1);
            vst1q_lane_f32(c + 4 * 6, vacc4567, 2);
            vst1q_lane_f32(c + 4 * 7, vacc4567, 3);
        }
        ie += 8;
        a += 8;
    }

    if (eSize & 0x04) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (auto ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float32x4_t vacc0123 = vld1q_dup_f32(&initValue);

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_dup_f32(w);
                // MNN_PRINT("4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-3]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {4});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc0123 = vfmaq_f32(vacc0123, va0123, w4);
            }
            vacc0123 = vminq_f32(vacc0123, vmax);
            vacc0123 = vmaxq_f32(vacc0123, vmin);

            // how to store faster: st4 / transpose /
            vst1q_lane_f32(c, vacc0123, 0);
            vst1q_lane_f32(c + 4, vacc0123, 1);
            vst1q_lane_f32(c + 4 * 2, vacc0123, 2);
            vst1q_lane_f32(c + 4 * 3, vacc0123, 3);
        }
        ie += 4;
        a += 4;
    }
    if (eSize & 0x02) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (auto ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float32x2_t vacc01 = vld1_dup_f32(&initValue);

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float32x2_t va01 = vld1_f32(a);
                // __builtin_prefetch(a + aStride);

                float32x2_t w2 = vld1_dup_f32(w);
                // MNN_PRINT("2-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-1]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {2});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc01 = vfma_f32(vacc01, va01, w2);
            }
            vacc01 = vmin_f32(vacc01, vget_low_f32(vmax));
            vacc01 = vmax_f32(vacc01, vget_low_f32(vmin));
            // how to store faster: st4 / transpose /
            vst1_lane_f32(c, vacc01, 0);
            vst1_lane_f32(c + 4, vacc01, 1);
        }
        ie += 2;
        a += 2;
    }
    if (eSize & 0x01) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (auto ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float oneW = *w++;
                // MNN_PRINT("1-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {1});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
        }
        ie += 1;
        // a += 1;
    }

    return;
}

void MNNPackedSparseMatMulEpx4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias,  unsigned int* NNZMap, int* dataOffsetMap) {

    auto eP = parameter[0] / sizeof(float);
    MNN_ASSERT((eP & 0x03) == 0); // In sparse calculate, eP should be evenly divided by 4
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto hRemain = parameter[4];
    auto bExtraStride = parameter[5] / sizeof(float);
    // auto bStride = bExtraStride + l * 4;
    auto aStride = eP * l; // sizeof(float);
    auto hC4 = UP_DIV(h, 4);
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
    }
    const float32x4_t vmin = vld1q_dup_f32(&minValue);
    const float32x4_t vmax = vld1q_dup_f32(&maxValue);
    const int sparseBlockOC = 4;
    // MNN_PRINT("NEON MNNPackedSparseMatMul eP:%lu, eSize:%lu, l:%lu, h:%lu, cStride:%lu, aStride:%lu\n", eP, eSize, l, h, cStride, aStride);

    const float* a = A;
    size_t ie = 0;
    for (ie = 0; ie + eP <= eSize; ie += eP) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;

            // tobe merged in to weight data
            float32x4_t vacc0c4 = nullptr != bias ? vld1q_f32(bias + ih) : vdupq_n_f32(0);
            float32x4_t vacc1c4 = vacc0c4;
            float32x4_t vacc2c4 = vacc0c4;
            float32x4_t vacc3c4 = vacc0c4;
            float32x4_t vacc4c4 = vacc0c4;
            float32x4_t vacc5c4 = vacc0c4;
            float32x4_t vacc6c4 = vacc0c4;
            float32x4_t vacc7c4 = vacc0c4;
            float32x4_t vacc8c4 = vacc0c4;
            float32x4_t vacc9c4 = vacc0c4;
            float32x4_t vacc10c4 = vacc0c4;
            float32x4_t vacc11c4 = vacc0c4;
            float32x4_t vacc12c4 = vacc0c4;
            float32x4_t vacc13c4 = vacc0c4;
            float32x4_t vacc14c4 = vacc0c4;
            float32x4_t vacc15c4 = vacc0c4;
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                const float32x4_t va4567 = vld1q_f32(a + 4);
                const float32x4_t va89AB = vld1q_f32(a + 8);
                const float32x4_t vaCDEF = vld1q_f32(a + 12);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_f32(w);

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w += 4;
                a = a + diff;
                vacc0c4 = vfmaq_laneq_f32(vacc0c4, w4, va0123, 0);
                vacc4c4 = vfmaq_laneq_f32(vacc4c4, w4, va4567, 0);
                vacc8c4 = vfmaq_laneq_f32(vacc8c4, w4, va89AB, 0);
                vacc12c4 = vfmaq_laneq_f32(vacc12c4, w4, vaCDEF, 0);

                vacc1c4 = vfmaq_laneq_f32(vacc1c4, w4, va0123, 1);
                vacc5c4 = vfmaq_laneq_f32(vacc5c4, w4, va4567, 1);
                vacc9c4 = vfmaq_laneq_f32(vacc9c4, w4, va89AB, 1);
                vacc13c4 = vfmaq_laneq_f32(vacc13c4, w4, vaCDEF, 1);

                vacc2c4 = vfmaq_laneq_f32(vacc2c4, w4, va0123, 2);
                vacc6c4 = vfmaq_laneq_f32(vacc6c4, w4, va4567, 2);
                vacc10c4 = vfmaq_laneq_f32(vacc10c4, w4, va89AB, 2);
                vacc14c4 = vfmaq_laneq_f32(vacc14c4, w4, vaCDEF, 2);

                vacc3c4 = vfmaq_laneq_f32(vacc3c4, w4, va0123, 3);
                vacc7c4 = vfmaq_laneq_f32(vacc7c4, w4, va4567, 3);
                vacc11c4 = vfmaq_laneq_f32(vacc11c4, w4, va89AB, 3);
                vacc15c4 = vfmaq_laneq_f32(vacc15c4, w4, vaCDEF, 3);
            }
            vacc0c4 = vminq_f32(vacc0c4, vmax);
            vacc1c4 = vminq_f32(vacc1c4, vmax);
            vacc2c4 = vminq_f32(vacc2c4, vmax);
            vacc3c4 = vminq_f32(vacc3c4, vmax);
            vacc4c4 = vminq_f32(vacc4c4, vmax);
            vacc5c4 = vminq_f32(vacc5c4, vmax);
            vacc6c4 = vminq_f32(vacc6c4, vmax);
            vacc7c4 = vminq_f32(vacc7c4, vmax);
            vacc8c4 = vminq_f32(vacc8c4, vmax);
            vacc9c4 = vminq_f32(vacc9c4, vmax);
            vacc10c4 = vminq_f32(vacc10c4, vmax);
            vacc11c4 = vminq_f32(vacc11c4, vmax);
            vacc12c4 = vminq_f32(vacc12c4, vmax);
            vacc13c4 = vminq_f32(vacc13c4, vmax);
            vacc14c4 = vminq_f32(vacc14c4, vmax);
            vacc15c4 = vminq_f32(vacc15c4, vmax);

            vacc0c4 = vmaxq_f32(vacc0c4, vmin);
            vacc1c4 = vmaxq_f32(vacc1c4, vmin);
            vacc2c4 = vmaxq_f32(vacc2c4, vmin);
            vacc3c4 = vmaxq_f32(vacc3c4, vmin);
            vacc4c4 = vmaxq_f32(vacc4c4, vmin);
            vacc5c4 = vmaxq_f32(vacc5c4, vmin);
            vacc6c4 = vmaxq_f32(vacc6c4, vmin);
            vacc7c4 = vmaxq_f32(vacc7c4, vmin);
            vacc8c4 = vmaxq_f32(vacc8c4, vmin);
            vacc9c4 = vmaxq_f32(vacc9c4, vmin);
            vacc10c4 = vmaxq_f32(vacc10c4, vmin);
            vacc11c4 = vmaxq_f32(vacc11c4, vmin);
            vacc12c4 = vmaxq_f32(vacc12c4, vmin);
            vacc13c4 = vmaxq_f32(vacc13c4, vmin);
            vacc14c4 = vmaxq_f32(vacc14c4, vmin);
            vacc15c4 = vmaxq_f32(vacc15c4, vmin);

            // vacc is continuous along c
            vst1q_f32(c, vacc0c4);
            vst1q_f32(c + 4 , vacc1c4);
            vst1q_f32(c + 4 * 2 , vacc2c4);
            vst1q_f32(c + 4 * 3 , vacc3c4);
            vst1q_f32(c + 4 * 4 , vacc4c4);
            vst1q_f32(c + 4 * 5 , vacc5c4);
            vst1q_f32(c + 4 * 6 , vacc6c4);
            vst1q_f32(c + 4 * 7 , vacc7c4);
            vst1q_f32(c + 4 * 8 , vacc8c4);
            vst1q_f32(c + 4 * 9 , vacc9c4);
            vst1q_f32(c + 4 * 10 , vacc10c4);
            vst1q_f32(c + 4 * 11 , vacc11c4);
            vst1q_f32(c + 4 * 12 , vacc12c4);
            vst1q_f32(c + 4 * 13 , vacc13c4);
            vst1q_f32(c + 4 * 14 , vacc14c4);
            vst1q_f32(c + 4 * 15 , vacc15c4);
        }
        blockC += (h >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float32x4_t vacc0123 = vld1q_dup_f32(&initValue);
            float32x4_t vacc4567 = vacc0123;
            float32x4_t vacc89AB = vacc0123;
            float32x4_t vaccCDEF = vacc0123;
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                const float32x4_t va4567 = vld1q_f32(a + 4);
                const float32x4_t va89AB = vld1q_f32(a + 8);
                const float32x4_t vaCDEF = vld1q_f32(a + 12);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_dup_f32(w);
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;

                vacc0123 = vfmaq_f32(vacc0123, va0123, w4);
                vacc4567 = vfmaq_f32(vacc4567, va4567, w4);
                vacc89AB = vfmaq_f32(vacc89AB, va89AB, w4);
                vaccCDEF = vfmaq_f32(vaccCDEF, vaCDEF, w4);

            }
            vacc0123 = vminq_f32(vacc0123, vmax);
            vacc4567 = vminq_f32(vacc4567, vmax);
            vacc89AB = vminq_f32(vacc89AB, vmax);
            vaccCDEF = vminq_f32(vaccCDEF, vmax);
            vacc0123 = vmaxq_f32(vacc0123, vmin);
            vacc4567 = vmaxq_f32(vacc4567, vmin);
            vacc89AB = vmaxq_f32(vacc89AB, vmin);
            vaccCDEF = vmaxq_f32(vaccCDEF, vmin);

            // how to store faster: st4 / transpose /
            vst1q_lane_f32(c, vacc0123, 0);
            vst1q_lane_f32(c + 4, vacc0123, 1);
            vst1q_lane_f32(c + 4 * 2, vacc0123, 2);
            vst1q_lane_f32(c + 4 * 3, vacc0123, 3);
            vst1q_lane_f32(c+  4 * 4, vacc4567, 0);
            vst1q_lane_f32(c + 4 * 5, vacc4567, 1);
            vst1q_lane_f32(c + 4 * 6, vacc4567, 2);
            vst1q_lane_f32(c + 4 * 7, vacc4567, 3);
            vst1q_lane_f32(c + 4 * 8, vacc89AB, 0);
            vst1q_lane_f32(c + 4 * 9, vacc89AB, 1);
            vst1q_lane_f32(c + 4 * 10, vacc89AB, 2);
            vst1q_lane_f32(c + 4 * 11, vacc89AB, 3);
            vst1q_lane_f32(c + 4 * 12, vaccCDEF, 0);
            vst1q_lane_f32(c + 4 * 13, vaccCDEF, 1);
            vst1q_lane_f32(c + 4 * 14, vaccCDEF, 2);
            vst1q_lane_f32(c + 4 * 15, vaccCDEF, 3);
        }
        a += aStride;
    }
    // const float* blockA = A + ie * l;
    if (eSize & 0x08) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;
            float32x4_t vacc0c4 = nullptr != bias ? vld1q_f32(bias + ih) : vdupq_n_f32(0.f);
            float32x4_t vacc1c4 = vacc0c4;
            float32x4_t vacc2c4 = vacc0c4;
            float32x4_t vacc3c4 = vacc0c4;
            float32x4_t vacc4c4 = vacc0c4;
            float32x4_t vacc5c4 = vacc0c4;
            float32x4_t vacc6c4 = vacc0c4;
            float32x4_t vacc7c4 = vacc0c4;

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                const float32x4_t va4567 = vld1q_f32(a + 4);
                const float32x4_t va89AB = vld1q_f32(a + 8);
                const float32x4_t vaCDEF = vld1q_f32(a + 12);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_f32(w);

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w += 4;
                a = a + diff;
                vacc0c4 = vfmaq_laneq_f32(vacc0c4, w4, va0123, 0);
                vacc4c4 = vfmaq_laneq_f32(vacc4c4, w4, va4567, 0);

                vacc1c4 = vfmaq_laneq_f32(vacc1c4, w4, va0123, 1);
                vacc5c4 = vfmaq_laneq_f32(vacc5c4, w4, va4567, 1);

                vacc2c4 = vfmaq_laneq_f32(vacc2c4, w4, va0123, 2);
                vacc6c4 = vfmaq_laneq_f32(vacc6c4, w4, va4567, 2);

                vacc3c4 = vfmaq_laneq_f32(vacc3c4, w4, va0123, 3);
                vacc7c4 = vfmaq_laneq_f32(vacc7c4, w4, va4567, 3);
            }
            vacc0c4 = vminq_f32(vacc0c4, vmax);
            vacc1c4 = vminq_f32(vacc1c4, vmax);
            vacc2c4 = vminq_f32(vacc2c4, vmax);
            vacc3c4 = vminq_f32(vacc3c4, vmax);
            vacc4c4 = vminq_f32(vacc4c4, vmax);
            vacc5c4 = vminq_f32(vacc5c4, vmax);
            vacc6c4 = vminq_f32(vacc6c4, vmax);
            vacc7c4 = vminq_f32(vacc7c4, vmax);

            vacc0c4 = vmaxq_f32(vacc0c4, vmin);
            vacc1c4 = vmaxq_f32(vacc1c4, vmin);
            vacc2c4 = vmaxq_f32(vacc2c4, vmin);
            vacc3c4 = vmaxq_f32(vacc3c4, vmin);
            vacc4c4 = vmaxq_f32(vacc4c4, vmin);
            vacc5c4 = vmaxq_f32(vacc5c4, vmin);
            vacc6c4 = vmaxq_f32(vacc6c4, vmin);
            vacc7c4 = vmaxq_f32(vacc7c4, vmin);

            // vacc is continuous along c
            vst1q_f32(c, vacc0c4);
            vst1q_f32(c + 4 , vacc1c4);
            vst1q_f32(c + 4 * 2 , vacc2c4);
            vst1q_f32(c + 4 * 3 , vacc3c4);
            vst1q_f32(c + 4 * 4 , vacc4c4);
            vst1q_f32(c + 4 * 5 , vacc5c4);
            vst1q_f32(c + 4 * 6 , vacc6c4);
            vst1q_f32(c + 4 * 7 , vacc7c4);
        }
        blockC += (h >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float32x4_t vacc0123 = vld1q_dup_f32(&initValue);
            float32x4_t vacc4567 = vacc0123;

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                const float32x4_t va4567 = vld1q_f32(a + 4);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_dup_f32(w);
                // MNN_PRINT("8-4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-7]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {8});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc0123 = vfmaq_f32(vacc0123, va0123, w4);
                vacc4567 = vfmaq_f32(vacc4567, va4567, w4);
            }
            vacc0123 = vminq_f32(vacc0123, vmax);
            vacc4567 = vminq_f32(vacc4567, vmax);
            vacc0123 = vmaxq_f32(vacc0123, vmin);
            vacc4567 = vmaxq_f32(vacc4567, vmin);

            // how to store faster: st4 / transpose /
            vst1q_lane_f32(c, vacc0123, 0);
            vst1q_lane_f32(c + 4, vacc0123, 1);
            vst1q_lane_f32(c + 4 * 2, vacc0123, 2);
            vst1q_lane_f32(c + 4 * 3, vacc0123, 3);
            vst1q_lane_f32(c + 4 * 4, vacc4567, 0);
            vst1q_lane_f32(c + 4 * 5, vacc4567, 1);
            vst1q_lane_f32(c + 4 * 6, vacc4567, 2);
            vst1q_lane_f32(c + 4 * 7, vacc4567, 3);
        }
        ie += 8;
        a += 8;
    }

    if (eSize & 0x04) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;
            float32x4_t vacc0c4 = nullptr != bias ? vld1q_f32(bias + ih) : vdupq_n_f32(0);
            float32x4_t vacc1c4 = vacc0c4;
            float32x4_t vacc2c4 = vacc0c4;
            float32x4_t vacc3c4 = vacc0c4;
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_f32(w);

                // MNN_PRINT("4-4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w += 4;
                a = a + diff;
                vacc0c4 = vfmaq_laneq_f32(vacc0c4, w4, va0123, 0);
                vacc1c4 = vfmaq_laneq_f32(vacc1c4, w4, va0123, 1);
                vacc2c4 = vfmaq_laneq_f32(vacc2c4, w4, va0123, 2);
                vacc3c4 = vfmaq_laneq_f32(vacc3c4, w4, va0123, 3);
            }
            vacc0c4 = vminq_f32(vacc0c4, vmax);
            vacc1c4 = vminq_f32(vacc1c4, vmax);
            vacc2c4 = vminq_f32(vacc2c4, vmax);
            vacc3c4 = vminq_f32(vacc3c4, vmax);
            vacc0c4 = vmaxq_f32(vacc0c4, vmin);
            vacc1c4 = vmaxq_f32(vacc1c4, vmin);
            vacc2c4 = vmaxq_f32(vacc2c4, vmin);
            vacc3c4 = vmaxq_f32(vacc3c4, vmin);
            // vacc is continuous along c
            vst1q_f32(c, vacc0c4);
            vst1q_f32(c + 4 , vacc1c4);
            vst1q_f32(c + 4 * 2 , vacc2c4);
            vst1q_f32(c + 4 * 3 , vacc3c4);
        }
        blockC += (h >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float32x4_t vacc0123 = vld1q_dup_f32(&initValue);

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_dup_f32(w);
                // MNN_PRINT("4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-3]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {4});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc0123 = vfmaq_f32(vacc0123, va0123, w4);
            }
            vacc0123 = vminq_f32(vacc0123, vmax);
            vacc0123 = vmaxq_f32(vacc0123, vmin);

            // how to store faster: st4 / transpose /
            vst1q_lane_f32(c, vacc0123, 0);
            vst1q_lane_f32(c + 4, vacc0123, 1);
            vst1q_lane_f32(c + 4 * 2, vacc0123, 2);
            vst1q_lane_f32(c + 4 * 3, vacc0123, 3);
        }
        ie += 4;
        a += 4;
    }
    if (eSize & 0x02) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;
            float32x4_t vacc0c4 = nullptr != bias ? vld1q_f32(bias + ih) : vdupq_n_f32(0.f);
            float32x4_t vacc1c4 = vacc0c4;
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float32x2_t va01 = vld1_f32(a);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_f32(w);

                // MNN_PRINT("2-4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w += 4;
                a = a + diff;
                vacc0c4 = vfmaq_lane_f32(vacc0c4, w4, va01, 0);
                vacc1c4 = vfmaq_lane_f32(vacc1c4, w4, va01, 1);
            }
            vacc0c4 = vminq_f32(vacc0c4, vmax);
            vacc1c4 = vminq_f32(vacc1c4, vmax);
            vacc0c4 = vmaxq_f32(vacc0c4, vmin);
            vacc1c4 = vmaxq_f32(vacc1c4, vmin);
            // vacc is continuous along c
            vst1q_f32(c, vacc0c4);
            vst1q_f32(c + 4, vacc1c4);
        }
        blockC += (h >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float32x2_t vacc01 = vld1_dup_f32(&initValue);

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float32x2_t va01 = vld1_f32(a);
                // __builtin_prefetch(a + aStride);

                float32x2_t w2 = vld1_dup_f32(w);
                // MNN_PRINT("2-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-1]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {2});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc01 = vfma_f32(vacc01, va01, w2);
            }
            vacc01 = vmin_f32(vacc01, vget_low_f32(vmax));
            vacc01 = vmax_f32(vacc01, vget_low_f32(vmin));
            // how to store faster: st4 / transpose /
            vst1_lane_f32(c, vacc01, 0);
            vst1_lane_f32(c + 4, vacc01, 1);
        }
        ie += 2;
        a += 2;
    }
    if (eSize & 0x01) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;
            float32x4_t vacc0c4 = nullptr != bias ? vld1q_f32(bias + ih) : vdupq_n_f32(0);
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float32x2_t va01 = vld1_f32(a);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_f32(w);

                // MNN_PRINT("1-4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w += 4;
                a = a + diff;
                vacc0c4 = vfmaq_lane_f32(vacc0c4, w4, va01, 0);
            }
            vacc0c4 = vminq_f32(vacc0c4, vmax);
            vacc0c4 = vmaxq_f32(vacc0c4, vmin);
            // vacc is continuous along c
            vst1q_f32(c, vacc0c4);
        }
        blockC += (h >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float oneW = *w++;
                // MNN_PRINT("1-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {1});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
        }
        ie += 1;
        // a += 1;
    }

    return;
}

#endif

void MNNGetSparseMatMulPackMode(int* eP, int *lP, int* hP) {
#ifdef __aarch64__
    *eP = 16;
#else
    *eP = 8; // total vector number is 16, we choose to use 8 for output.
#endif
    *lP = 1;
    *hP = 4;
    // hp is corresponding to sparse block along right matrix colum dimension. in ramdom sparse, it is 1.
    return;
}


void MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = 12;
    *lP = 1;
#ifdef __aarch64__
    *hP = 8;
#else
    *hP = 4;
#endif
}

#ifdef __aarch64__

// input shape is (l, h) when transpose=false, else input shape is (h, l)
// output shape is (UP_DIV(h, 8), l, 8)
void MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t kernelsize, size_t ic, bool transpose) {
    auto hP = (int)h / 8;
    auto hR = (int)hP * 8;
    auto l = kernelsize * ic;
    if (hR != h) {
        ::memset(dest, 0, UP_DIV(h, 8)*8*l*sizeof(float));
    }
    if (!transpose) {
        for (int y=0; y<hP; ++y) {
            auto destY = dest + y * 8 * l;
            auto sourceY = source + y * 8;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 8 * x, sourceY + x * h, 8 * sizeof(float));
            }
        }
        auto hRemain = h - hR;
        if (hRemain > 0) {
            auto destY = dest + hP * 8 * l;
            auto sourceY = source + hP * 8;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 8 * x, sourceY + x * h, hRemain * sizeof(float));
            }
        }
        return;
    }
    int lC8 = (int)l / 8;
    auto lR = lC8 * 8;
    if (hP > 0 && lC8 > 0) {
        MNNPackC8(dest, source, l, h);
    }
    for (int y=hR; y<h; ++y) {
        auto yR = y % 8;
        auto yC = hP;
        for (int x=0; x<l; ++x) {
            dest[x * 8 + yR + yC * 8 * l] = source[x + y * l];
        }
    }
    for (int y=0; y<hR; ++y) {
        auto yR = y % 8;
        auto yC = y / 8;
        for (int x=lR; x<l; ++x) {
            dest[x * 8 + yR + yC * 8 * l] = source[x + y * l];
        }
    }
}
#else
void MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t kernelsize, size_t ic, bool transpose) {
    auto l = kernelsize * ic;
    if (!transpose) {
        auto hP = h / 4;
        auto hR = hP * 4;
        if (hR != h) {
            ::memset(dest, 0, UP_DIV(h, 4)*4*l*sizeof(float));
        }
        for (int y=0; y<hP; ++y) {
            auto destY = dest + y * 4 * l;
            auto sourceY = source + y * 4;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 4 * x, sourceY + x * h, 4 * sizeof(float));
            }
        }
        auto hRemain = h - hR;
        if (hRemain > 0) {
            auto destY = dest + hP * 4 * l;
            auto sourceY = source + hP * 4;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 4 * x, sourceY + x * h, hRemain * sizeof(float));
            }
        }
        return;
    }
    int offset[] = {
        (int)l, (int)l
    };
    MNNPackC4(dest, source, l, h, offset);
}
#endif


#endif
