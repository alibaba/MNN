// Copyright © 2026, Alibaba Group Holding Limited

#include "MNNTorqTileFunctions.hpp"
#include <math.h>
#include "MNNTorqTileKernel.hpp"

void MNNGetMatMulPackMode_RVV(int* eP, int* lP, int* hP);
void MNNPackedMatMulRemainFP32_RVV_WithAStride(float* C, const float* A, const float* B, size_t eSize,
                                               const size_t* parameter, const float* postParameters, const float* bias,
                                               const float* k, const float* b, size_t aStride);

static void packedMatMul(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                         const float* postParameters, const float* bias, const float* k, const float* b,
                         size_t aStride) {
    const size_t depth = parameter[1];
    const size_t channels = parameter[2];
    if (eSize == 0 || channels == 0) {
        return;
    }
    // The first slice supports complete 8-row blocks and nonempty reductions.
    // Keep all other shapes on the existing RVV path, including its post-treatment.
    if (depth == 0 || eSize % 8 != 0) {
        MNNPackedMatMulRemainFP32_RVV_WithAStride(C, A, B, eSize, parameter, postParameters, bias, k, b, aStride);
        return;
    }

    // MNN A: [K, eP], B: [ceil(H/4), K, 4], C: [ceil(H/4), E, 4].
    // Each C4 block is D = A^T * B. TORQ-Tile leading dimensions are in
    // elements; MNN parameter[3] and parameter[5] are in bytes.
    const size_t cStride = parameter[3] / sizeof(float);
    const size_t bStride = depth * 4 + parameter[5] / sizeof(float);
    const size_t channelBlocks = (channels + 3) / 4;
    const bool clamp = postParameters != nullptr && !(postParameters[2] < -3.0e38f && postParameters[3] > 3.0e38f);
    for (size_t y = 0; y < channelBlocks; ++y) {
        auto dst = C + y * cStride;
        for (size_t x = 0; x < eSize; x += 8) {
            MNNTorqTileMatMul8x4(dst + x * 4, A + x, B + y * bStride, depth, aStride);
        }
        if (bias == nullptr && !clamp) {
            continue;
        }
        for (size_t x = 0; x < eSize; ++x) {
            for (size_t lane = 0; lane < 4; ++lane) {
                float value = dst[x * 4 + lane];
                if (bias != nullptr) {
                    value += bias[y * 4 + lane];
                }
                if (clamp) {
                    // Match RVV vfmax/vfmin's numeric-operand handling of NaNs.
                    value = fminf(fmaxf(value, postParameters[2]), postParameters[3]);
                }
                dst[x * 4 + lane] = value;
            }
        }
    }
}

void MNNPackedMatMulFP32_TorqTile(float* C, const float* A, const float* B, const size_t* parameter,
                                  const float* postParameters, const float* bias, const float* k, const float* b) {
    int eP, lP, hP;
    MNNGetMatMulPackMode_RVV(&eP, &lP, &hP);
    // Like the RVV full-tile entry, use the pack mode rather than parameter[0].
    packedMatMul(C, A, B, eP, parameter, postParameters, bias, k, b, eP);
}

void MNNPackedMatMulRemainFP32_TorqTile(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                        const float* postParameters, const float* bias, const float* k,
                                        const float* b) {
    packedMatMul(C, A, B, eSize, parameter, postParameters, bias, k, b, parameter[0] / sizeof(float));
}
