// Scalar references based on CommonOptFunction.cpp at bef71b9756a2c77549eddbe33eb97290e3b16602.
// The key reference additionally zero-pads missing tail dimensions, excluding them from the sum.
// This standalone regression is enabled explicitly; ordinary test builds contain no extra main.
#ifdef MNN_RVV_QUANT_TEST_MAIN
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#define UP_DIV(x, y) (((x) + (y) - 1) / (y))
#define ROUND_UP(x, y) (UP_DIV(x, y) * (y))
#define ALIMAX(x, y) std::max((x), (y))
#define ALIMIN(x, y) std::min((x), (y))
void MNNQuantAttentionKey_RVV(int8_t*, const float*, float*, float*, int32_t*);
void MNNQuantAttentionValue_RVV(int8_t*, const float*, float*, int32_t*);
void MNNQuantAttentionKeyReference(int8_t* dst, const float* source, float* sumKeyPtr, float* maxKeyPtr,
                                   int32_t* params) {
    int32_t kvNumHead = params[0];
    int32_t seqLen = params[1];
    int32_t headDim = params[2];
    int32_t blockNum = params[3];
    int32_t eP = params[4];
    int32_t lP = params[5];
    int32_t hP = params[6];
    int32_t pastLength = params[7];
    int32_t kvHeadIdx = params[8];

    auto blockL = UP_DIV(headDim, blockNum);
    auto weightStride1 = ROUND_UP(blockL, lP) * hP;
    auto weightStride2 = lP * hP;
    auto packedWeightStride1 = weightStride1 + 2 * 4 * hP;

    if (seqLen > 1) {
        // get max
        for (int s = 0; s < seqLen; ++s) {
            const float* keySrc = source + s * kvNumHead * headDim + kvHeadIdx * headDim;
            for (int d = 0; d < headDim; d++) {
                maxKeyPtr[d] = ALIMAX(maxKeyPtr[d], keySrc[d]);
            }
        }
    }

    for (int s = 0; s < seqLen; s++) {
        const float* keySrc = source + s * kvNumHead * headDim + kvHeadIdx * headDim;
        float minKey, maxKey;
        minKey = keySrc[0] - maxKeyPtr[0];
        maxKey = keySrc[0] - maxKeyPtr[0];
        for (int d = 1; d < headDim; d++) {
            auto keydata = keySrc[d] - maxKeyPtr[d];
            minKey = ALIMIN(minKey, keydata);
            maxKey = ALIMAX(maxKey, keydata);
        }

        int outIndex = (pastLength + s) / hP;
        int inIndex = (pastLength + s) % hP;

        float sumKey = 0;
        for (int k = 0; k < blockNum; ++k) {
            int8_t* weightDst = dst + outIndex * blockNum * packedWeightStride1 + k * packedWeightStride1;
            float* scaleDst = (float*)(weightDst + weightStride1);
            float* biasDst = scaleDst + hP;

            scaleDst[inIndex] = (maxKey - minKey) / 255.0f;
            biasDst[inIndex] = minKey + 128.f * (maxKey - minKey) / 255.f;

            for (int d = 0; d < blockL; d++) {
                int i = d / lP;
                int j = d % lP;

                if (d + k * blockL >= headDim) {
                    weightDst[i * weightStride2 + inIndex * lP + j] = 0;
                    continue;
                }
                int int8v =
                    (int)(roundf(fmaf((keySrc[d + k * blockL] - maxKeyPtr[d + k * blockL] - minKey) / (maxKey - minKey),
                                      255.0f, -128.0f)));
                weightDst[i * weightStride2 + inIndex * lP + j] = int8v;
                sumKey += fmaf((float)int8v, scaleDst[inIndex], biasDst[inIndex]);
            }
        }
        sumKeyPtr[outIndex * hP + inIndex] = sumKey;
    }
}

void MNNQuantAttentionValueReference(int8_t* dst, const float* source, float* valueSum, int32_t* params) {
    // float   value src : [kvSeq,kvNumHead,headDim]
    // int8_t  value dest: [updiv(maxLength,flashAttentionBlockKv),
    // updiv(headDim,hp),updiv(flashAttentionBlockKv,lp),hp,lp] float   value sum:
    // [updiv(maxLength,flashAttentionBlockKv), roundup(headDim,hp)]
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
                float bias = fmaf(scale, 128.f, dMin);
                scalePtr[0] = scale;
                biasPtr[0] = bias;
            }
        }
    }

    // copy the scale&bias to each blockKv
    //                                    pastLength == 0: First time prefill
    // (seqLen + pastLength) % flashAttentionBlockKv == 0: Open a new blockKv
    if (pastLength == 0 || (pastLength % flashAttentionBlockKv) == 0) {
        int32_t d0 = UP_DIV(maxLength, flashAttentionBlockKv);
        int32_t d1 = UP_DIV(headDim, hP);
        for (int k = 0; k < d0; ++k) {
            for (int r = 0; r < d1; ++r) {
                float* scalePtr = (float*)(dst + k * packedStride0 + r * packedStride1 + weightStride1);
                float* biasPtr = scalePtr + hP;
                memcpy(scalePtr, dst + r * packedStride1 + weightStride1, hP * sizeof(float));
                memcpy(biasPtr, dst + r * packedStride1 + weightStride1 + hP * sizeof(float), hP * sizeof(float));
            }
        }
    }

    for (int d = 0; d < headDim; ++d) {
        // dst address
        int idxBase = (d / hP) * packedStride1 + (d % hP) * lP;
        int8_t* dstBase = dst + idxBase;
        float* scaleBase = (float*)(dst + (d / hP) * packedStride1 + weightStride1) + (d % hP);
        float* biasBase = scaleBase + hP;
        float* sumBase = valueSum + (d / hP) * hP + (d % hP);

        float qscale = scaleBase[0] < 1e-6 ? 0 : 1.0f / scaleBase[0];
        float qbias = scaleBase[0] < 1e-6 ? 0 : (-biasBase[0] / scaleBase[0]);
        // quant
        for (int s = 0; s < seqLen; ++s) {
            int kvSeqIndx = s + pastLength;
            int idxInner = (kvSeqIndx / flashAttentionBlockKv) * packedStride0 +
                           (kvSeqIndx % flashAttentionBlockKv) / lP * weightStride2 +
                           (kvSeqIndx % flashAttentionBlockKv) % lP;
            float xf = sourceFp32[s * srcStride0 + d + kvHeadIdx * headDim];
            int8_t xq = ALIMAX(ALIMIN(127, static_cast<int32_t>(roundf(fmaf(xf, qscale, qbias)))), -128);
            dstBase[idxInner] = xq;

            // sum
            int idxSum = (kvSeqIndx / flashAttentionBlockKv) * ROUND_UP(headDim, hP);
            sumBase[idxSum] += fmaf((float)xq, scaleBase[0], biasBase[0]);
        }
    }
}
static bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float));
}
int main() {
    size_t cases = 0;
    unsigned rng = 1977;
    for (int dim : {3, 16, 17, 32, 64})
        for (int seq : {1, 3, 17})
            for (int blocks : {1, 2, 4}) {
                if (blocks > dim)
                    continue;
                for (int lp : {1, 4, 16})
                    for (int hp : {4, 8})
                        for (int past : {0, 5, 16})
                            for (int head : {0, 2}) {
                                std::vector<float> src(seq * 3 * dim);
                                for (auto& f : src) {
                                    rng = rng * 1664525u + 1013904223u;
                                    f = float(int(rng % 10001) - 5000) / 997.0f;
                                }
                                int32_t p[10] = {3, seq, dim, blocks, 8, lp, hp, past, head, 16};
                                int stride = ROUND_UP(UP_DIV(dim, blocks), lp) * hp + 8 * hp;
                                size_t bytes = UP_DIV(past + seq, hp) * blocks * stride;
                                std::vector<int8_t> dst(bytes + 64, 53), ref(dst);
                                std::vector<float> maxs(dim), refs;
                                for (int d = 0; d < dim; ++d)
                                    maxs[d] = 0.031f * d;
                                refs = maxs;
                                std::vector<float> sums(ROUND_UP(past + seq, hp) + 8, 0.125f), refSums(sums);
                                MNNQuantAttentionKeyReference(ref.data(), src.data(), refSums.data(), refs.data(), p);
                                // The scalar zero-range formula divides 0/0 then converts NaN to int.
                                // Check the RVV path's explicit -128 representation instead of relying on that
                                // undefined conversion.
                                for (int s = 0; s < seq; ++s) {
                                    float lo = src[(s * 3 + head) * dim] - refs[0], hi = lo;
                                    for (int d = 1; d < dim; ++d) {
                                        float x = src[(s * 3 + head) * dim + d] - refs[d];
                                        lo = std::min(lo, x);
                                        hi = std::max(hi, x);
                                    }
                                    if (hi == lo)
                                        for (int b = 0; b < blocks; ++b)
                                            for (int d = 0; d < UP_DIV(dim, blocks) && b * UP_DIV(dim, blocks) + d < dim; ++d) {
                                                size_t off = (past + s) / hp * blocks * stride + b * stride +
                                                             d / lp * lp * hp + (past + s) % hp * lp + d % lp;
                                                ref[off] = -128;
                                            }
                                }
                                MNNQuantAttentionKey_RVV(dst.data(), src.data(), sums.data(), maxs.data(), p);
                                if (dst != ref || !same(sums, refSums) || !same(maxs, refs)) {
                                    std::printf(
                                        "key fail dim=%d seq=%d blocks=%d lp=%d hp=%d past=%d head=%d packed=%d "
                                        "sums=%d max=%d\n",
                                        dim, seq, blocks, lp, hp, past, head, dst == ref, same(sums, refSums),
                                        same(maxs, refs));
                                    for (size_t z = 0; z < sums.size(); ++z)
                                        if (std::memcmp(&sums[z], &refSums[z], 4))
                                            std::printf("sum[%zu] actual=%.9g ref=%.9g\n", z, sums[z], refSums[z]);
                                    for (size_t z = 0; z < dst.size(); ++z)
                                        if (dst[z] != ref[z])
                                            std::printf("byte[%zu] actual=%d ref=%d\n", z, int(dst[z]), int(ref[z]));
                                    return 1;
                                }
                                ++cases;
                            }
            }
    for (int dim : {1, 3, 16, 17, 33})
        for (int seq : {1, 3, 17, 33})
            for (int lp : {1, 4, 16})
                for (int hp : {4, 8})
                    for (int past : {0, 5, 16, 31})
                        for (int head : {0, 2})
                            for (int pattern : {0, 1, 2}) {
                                int flash = 16, length = ROUND_UP(past + seq, flash);
                                int32_t p[10] = {3, seq, dim, 1, length, lp, hp, past, head, flash};
                                std::vector<float> src(seq * 3 * dim);
                                for (auto& f : src) {
                                    rng = rng * 1664525u + 1013904223u;
                                    f = pattern == 0 ? float(int(rng % 10001) - 5000) / 997.0f
                                                     : (pattern == 1 ? 0.375f : float(int(rng % 257) - 128) + 0.5f);
                                }
                                int ws = UP_DIV(flash, lp) * lp * hp, ps = ws + 8 * hp, ps0 = UP_DIV(dim, hp) * ps;
                                std::vector<int8_t> dst(UP_DIV(length, flash) * ps0 + 64, 0);
                                for (int b = 0; b < UP_DIV(length, flash); ++b)
                                    for (int r = 0; r < UP_DIV(dim, hp); ++r) {
                                        float* scale = reinterpret_cast<float*>(dst.data() + b * ps0 + r * ps + ws);
                                        for (int j = 0; j < hp; ++j) {
                                            scale[j] = 1.0f;
                                            scale[hp + j] = 0.0f;
                                        }
                                    }
                                auto ref = dst;
                                std::vector<float> sums(UP_DIV(length, flash) * ROUND_UP(dim, hp) + 8, 0.125f),
                                    refSums(sums);
                                MNNQuantAttentionValueReference(ref.data(), src.data(), refSums.data(), p);
                                MNNQuantAttentionValue_RVV(dst.data(), src.data(), sums.data(), p);
                                if (dst != ref || !same(sums, refSums)) {
                                    std::printf(
                                        "value fail dim=%d seq=%d lp=%d hp=%d past=%d head=%d pattern=%d packed=%d "
                                        "sums=%d\n",
                                        dim, seq, lp, hp, past, head, pattern, dst == ref, same(sums, refSums));
                                    return 1;
                                }
                                ++cases;
                            }
    std::printf("KV quantization: %zu packed-output/metadata/sum comparisons passed\n", cases);
    return 0;
}
#endif
