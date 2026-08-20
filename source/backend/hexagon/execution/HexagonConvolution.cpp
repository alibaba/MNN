//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "HexagonConvolution.hpp"
#include "HexagonSharedGather.hpp"
#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "backend/hexagon/htp-ops-lib/include/dsp/ops.h"
#include "htp_command.h"
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
namespace MNN {
static_assert(sizeof(ConvolutionCommon::Im2ColParameter) == sizeof(Im2ColParameter), "Im2ColParameter layout mismatch");

static bool useW8A16GemvI8() {
    static const bool enabled = []() {
        const char* value = std::getenv("MNN_HEXAGON_W8A16_GEMV_I8");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }();
    return enabled;
}

static void setHexagonIm2ColParameter(ConvolutionCommon::Im2ColParameter& param,
                                      const Convolution2DCommon* convCommon,
                                      Tensor* input, Tensor* output, int padX, int padY, int pack) {
    param.dilateX = convCommon->dilateX();
    param.dilateY = convCommon->dilateY();
    param.strideX = convCommon->strideX();
    param.strideY = convCommon->strideY();
    param.icDiv4 = UP_DIV(input->channel(), pack);
    param.kernelX = convCommon->kernelX();
    param.kernelY = convCommon->kernelY();
    param.padX = padX;
    param.padY = padY;

    param.ih = input->height();
    param.iw = input->width();
    param.oh = output->height();
    param.ow = output->width();
    param.srcZStep = input->stride(1) * pack * input->batch();
    param.srcYStep = input->stride(2) * pack;
    param.packCUnit = pack;
    param.destICStride = input->height() * param.srcYStep;
    param.ic = input->channel();
    param.icup4 = input->channel();

    if (param.iw == 1 && param.ow == 1 && param.oh > 1 && param.kernelX == 1 && param.padX == 0) {
        const int inputHeight = param.ih;
        param.ow = param.oh;
        param.oh = 1;
        param.padX = param.padY;
        param.padY = 0;
        param.strideX = param.strideY;
        param.strideY = 1;
        param.iw = inputHeight;
        param.ih = 1;
        param.dilateX = param.dilateY;
        param.dilateY = 1;
        param.kernelX = param.kernelY;
        param.kernelY = 1;
    }
}

static bool hasNonZeroBias(const float* bias, int size) {
    if (bias == nullptr || size <= 0) {
        return false;
    }
    for (int i = 0; i < size; ++i) {
        if (bias[i] != 0.0f) {
            return true;
        }
    }
    return false;
}

struct HexagonTileShape {
    int mp;
    int np;
    HexagonTileShape(int m = 1, int n = 1) : mp(m), np(n) {}
};

enum class Q4ScaleMode {
    None,
    PerOutput,
    BlockSymmetric,
    BlockAsymmetric,
};

static HexagonTileShape chooseIm2ColTileShape(int totalMp, int totalNp, int KAlign, int availSize,
                                              bool useInt8Staging = false, int scaleBlockNum = 1,
                                              int scaleAsymmetric = 0) {
    const int bytesPerMp = 64 * KAlign;
    // W8A16 stages the per-oy scale region of each oy-chunk into VTCM (fp16 64 +
    // int8 32 per KAligned unit), plus scaleBlockNum * scaleUnit * 2 scale bytes.
    const int scaleUnit = scaleAsymmetric ? 128 : (scaleBlockNum > 1 ? 64 : 0);
    const int scaleBytesPerNp = (useInt8Staging && (scaleBlockNum > 1 || scaleAsymmetric))
                                    ? scaleBlockNum * scaleUnit * 2
                                    : 0;
    const int bytesPerNp = useInt8Staging ? 96 * KAlign + scaleBytesPerNp : 64 * KAlign;
    int maxSum = availSize / std::max(bytesPerMp, bytesPerNp);
    maxSum = std::max(maxSum, 2);

    HexagonTileShape best;
    int64_t bestCost = INT64_MAX;
    int bestChunkPairs = INT_MAX;
    int bestTileArea = 0;
    const int maxMp = std::min(totalMp, maxSum - 1);
    for (int candMp = 1; candMp <= maxMp; ++candMp) {
        if ((int64_t)candMp * bytesPerMp >= availSize) {
            continue;
        }
        const int maxNp = std::min(totalNp, (int)((availSize - (int64_t)candMp * bytesPerMp) / bytesPerNp));
        for (int candNp = 1; candNp <= maxNp; ++candNp) {
            const int oxChunks = UP_DIV(totalMp, candMp);
            const int oyChunks = UP_DIV(totalNp, candNp);
            const int64_t activationOuterCost = (int64_t)totalMp + (int64_t)oxChunks * totalNp;
            const int64_t weightOuterCost = (int64_t)oyChunks * totalMp + (int64_t)totalNp;
            const int64_t cost = std::min(activationOuterCost, weightOuterCost);
            const int chunkPairs = oxChunks * oyChunks;
            const int tileArea = candMp * candNp;
            if (cost < bestCost ||
                (cost == bestCost && chunkPairs < bestChunkPairs) ||
                (cost == bestCost && chunkPairs == bestChunkPairs && tileArea > bestTileArea)) {
                bestCost = cost;
                bestChunkPairs = chunkPairs;
                bestTileArea = tileArea;
                best = {candMp, candNp};
            }
        }
    }
    return best;
}

static HexagonTileShape chooseQ4BlockPrefillTileShape(HexagonTileShape base, int totalMp, int totalNp,
                                                      int KAlign, int vtcmSize);
static int limitQ4BlockDecodeNp(int currentNp, int totalNp, int KAlign, int scaleBlockNum, bool asymmetric,
                                int vtcmSize);
static HexagonTileShape chooseQ4PerOutputPrefillTileShape(HexagonTileShape base, int totalMp, int totalNp,
                                                          int KAlign, int vtcmSize);

static HexagonTileShape chooseDirectTileShape(int totalMp, int totalNp, int KAlign, int availSize,
                                              int vtcmSize, Q4ScaleMode q4ScaleMode, int scaleBlockNum) {
    const bool useInt4 = q4ScaleMode != Q4ScaleMode::None;
    int maxSum = useInt4 ? availSize / (64 * KAlign + 64 + 16 * KAlign) : availSize / (64 * KAlign);
    maxSum = std::max(maxSum, 3); // at least 1 mp (takes 2) and 1 np (takes 1)

    HexagonTileShape tile;
    tile.mp = std::min(totalMp, std::max(1, maxSum / 3));
    const int remainSize = availSize - 64 * KAlign * 2 * tile.mp;
    int maxNp = useInt4 ? remainSize / (64 + 64 * KAlign + 16 * KAlign + 2048)
                        : remainSize / (64 * KAlign);
    tile.np = std::min(totalNp, std::max(1, maxNp));
    if (tile.np + 2 * tile.mp < maxSum && tile.mp < totalMp) {
        tile.mp = std::min(totalMp, (maxSum - tile.np) / 2);
    }
    if (q4ScaleMode == Q4ScaleMode::BlockSymmetric || q4ScaleMode == Q4ScaleMode::BlockAsymmetric) {
        // Both block modes now share the M=1 kernels, so they share the tiling too: asymmetric only
        // widens the packed-scale plane, which limitQ4BlockDecodeNp accounts for.
        if (totalMp > 1) {
            return chooseQ4BlockPrefillTileShape(tile, totalMp, totalNp, KAlign, vtcmSize);
        }
        tile.np = limitQ4BlockDecodeNp(tile.np, totalNp, KAlign, scaleBlockNum,
                                       q4ScaleMode == Q4ScaleMode::BlockAsymmetric, vtcmSize);
    } else if (q4ScaleMode == Q4ScaleMode::PerOutput && tile.np > 1 && (tile.np & 1)) {
        --tile.np;
    }
    if (q4ScaleMode == Q4ScaleMode::PerOutput) {
        if (totalMp > 1) {
            return chooseQ4PerOutputPrefillTileShape(tile, totalMp, totalNp, KAlign, vtcmSize);
        }
        const size_t safeVtcmSize = vtcmSize > 16 * 1024 ? (size_t)vtcmSize - 16 * 1024 : (size_t)vtcmSize;
        auto footprint = [&](int mp, int np, bool asyncOutputStore) {
            const int activationBuffers = mp >= totalMp ? 1 : 2;
            const int outputBuffers = asyncOutputStore ? ((np > 1 && (np & 1) == 0) ? 4 : 2) : 1;
            const int scaleBuffers = asyncOutputStore ? 2 : 1;
            return (size_t)np * 64 * KAlign +                  // fp16 weight
                   (size_t)np * 16 * KAlign +                  // int4 weight
                   (size_t)activationBuffers * mp * 64 * KAlign +
                   (size_t)outputBuffers * np * 1024 * sizeof(int16_t) +
                   (size_t)np * 256 +                          // hmx scales
                   (size_t)scaleBuffers * (np * 64 + 64);
        };
        while (tile.np > 1 && footprint(tile.mp, tile.np, true) > safeVtcmSize) {
            tile.np -= tile.np > 2 ? 2 : 1;
        }
    }
    return tile;
}

static HexagonTileShape chooseQ4PerOutputPrefillTileShape(HexagonTileShape base, int totalMp, int totalNp,
                                                          int KAlign, int vtcmSize) {
    const size_t safeVtcmSize = vtcmSize > 16 * 1024 ? (size_t)vtcmSize - 16 * 1024 : (size_t)vtcmSize;
    auto footprint = [&](int mp, int np) {
        const int activationBuffers = mp >= totalMp ? 1 : 2;
        const int outputBuffers = (np > 1 && (np & 1) == 0) ? 4 : 2;
        return (size_t)np * 64 * KAlign +                 // fp16 weight
               (size_t)np * 16 * KAlign +                 // int4 weight
               (size_t)activationBuffers * mp * 64 * KAlign +
               (size_t)outputBuffers * np * 1024 * sizeof(int16_t) +
               (size_t)np * 256 +                         // hmx scales
               (size_t)2 * (np * 64 + 64);                 // double-buffered output scales
    };

    HexagonTileShape best = base;
    int bestReuseActivation = best.mp >= totalMp ? 1 : 0;
    int64_t bestCost = INT64_MAX;
    int bestChunkPairs = INT_MAX;
    int bestTileArea = 0;
    for (int candMp = 1; candMp <= totalMp; ++candMp) {
        for (int candNp = 1; candNp <= totalNp; ++candNp) {
            if (candNp > 1 && (candNp & 1)) {
                continue;
            }
            if (footprint(candMp, candNp) > safeVtcmSize) {
                continue;
            }
            const int oxChunks = UP_DIV(totalMp, candMp);
            const int oyChunks = UP_DIV(totalNp, candNp);
            const int reuseActivation = candMp >= totalMp ? 1 : 0;
            const int64_t activationCost = reuseActivation ? totalMp : (int64_t)oyChunks * totalMp;
            const int64_t cost = activationCost * 8 + (int64_t)oxChunks * oyChunks;
            const int chunkPairs = oxChunks * oyChunks;
            const int tileArea = candMp * candNp;
            if (reuseActivation > bestReuseActivation ||
                (reuseActivation == bestReuseActivation &&
                 (cost < bestCost ||
                  (cost == bestCost && chunkPairs < bestChunkPairs) ||
                  (cost == bestCost && chunkPairs == bestChunkPairs && tileArea > bestTileArea)))) {
                bestReuseActivation = reuseActivation;
                bestCost = cost;
                bestChunkPairs = chunkPairs;
                bestTileArea = tileArea;
                best = {candMp, candNp};
            }
        }
    }
    return best;
}

static HexagonTileShape chooseQ4BlockPrefillTileShape(HexagonTileShape base, int totalMp, int totalNp,
                                                      int KAlign, int vtcmSize) {
    const size_t fixedBytes = 4 * 1024 + 256;
    const size_t safeVtcmSize = vtcmSize > (int)fixedBytes ? (size_t)vtcmSize - fixedBytes : 0;
    const size_t activationBytesPerMp = (size_t)64 * KAlign;
    const size_t bytesPerNp = (size_t)64 * KAlign + (size_t)16 * KAlign + 2048 + 384;

    HexagonTileShape best = base;
    int bestReuseActivation = best.mp >= totalMp ? 1 : 0;
    int64_t bestCost = INT64_MAX;
    int bestChunkPairs = INT_MAX;
    int bestTileArea = 0;
    for (int candMp = 1; candMp <= totalMp; ++candMp) {
        const int activationBuffers = candMp >= totalMp ? 1 : 2;
        const size_t activationBytes = (size_t)activationBuffers * candMp * activationBytesPerMp;
        if (activationBytes >= safeVtcmSize) {
            continue;
        }
        const int maxCandNp = std::min(totalNp, (int)((safeVtcmSize - activationBytes) / bytesPerNp));
        for (int candNp = 1; candNp <= maxCandNp; ++candNp) {
            const int oxChunks = UP_DIV(totalMp, candMp);
            const int oyChunks = UP_DIV(totalNp, candNp);
            const int reuseActivation = candMp >= totalMp ? 1 : 0;
            const int64_t activationCost = reuseActivation ? totalMp : (int64_t)oyChunks * totalMp;
            const int64_t cost = activationCost * 8 + (int64_t)oxChunks * oyChunks;
            const int chunkPairs = oxChunks * oyChunks;
            const int tileArea = candMp * candNp;
            if (reuseActivation > bestReuseActivation ||
                (reuseActivation == bestReuseActivation &&
                 (cost < bestCost ||
                  (cost == bestCost && chunkPairs < bestChunkPairs) ||
                  (cost == bestCost && chunkPairs == bestChunkPairs && tileArea > bestTileArea)))) {
                bestReuseActivation = reuseActivation;
                bestCost = cost;
                bestChunkPairs = chunkPairs;
                bestTileArea = tileArea;
                best = {candMp, candNp};
            }
        }
    }
    return best;
}

static int limitQ4BlockDecodeNp(int currentNp, int totalNp, int KAlign, int scaleBlockNum, bool asymmetric,
                                int vtcmSize) {
    const int scaleOutputPasses = UP_DIV(scaleBlockNum, 32);
    const int outputPartitions = scaleOutputPasses > 1 ? scaleOutputPasses : 1;
    const int scalePartitions = scaleOutputPasses > 1 ? 2 : 1;
    const size_t topReservedBytes = 16 * 1024;
    const size_t safeVtcmSize = vtcmSize > (int)topReservedBytes ? (size_t)vtcmSize - topReservedBytes : (size_t)vtcmSize;
    const size_t kp = UP_DIV(KAlign, 32);
    const size_t fixedBytes = (size_t)64 * KAlign + kp * 128 + 256;
    const size_t bytesPerNp = (size_t)64 * KAlign + (size_t)16 * KAlign + (size_t)outputPartitions * 2048 +
                              // Asymmetric packed scale entries carry a qbias next to every scale, so
                              // that plane is twice as wide on the DSP too. This must stay in lockstep
                              // with the vtcm_seq_alloc for vtcm_packed_scales in
                              // matmul_q4block_fp16_mle32.c, which does not bounds check.
                              (size_t)scalePartitions * 2048 * (asymmetric ? 2 : 1) + 128;
    if (safeVtcmSize <= fixedBytes || bytesPerNp == 0) {
        return currentNp;
    }
    int safeNp = (int)((safeVtcmSize - fixedBytes) / bytesPerNp);
    if (safeNp > 1 && (safeNp & 1)) {
        --safeNp;
    }
    return std::min(currentNp, std::min(totalNp, std::max(1, safeNp)));
}

static bool reorderInt4WeightForHmx(uint8_t* dst, size_t dstBytes, const uint8_t* rawInt4Data, const float* rawAlphaData,
                                    int rawAlphaSize, int ic, int oc, int scaleBlockNum, bool scaleAsymmetric,
                                    void(*fp32tofp16)(const float*, int16_t*, size_t)) {
    const int icP = UP_DIV(ic, 32);
    const int ocP = UP_DIV(oc, 32);
    const int icBytes = UP_DIV(ic, 2);
    const bool aligned = (ic % 32 == 0) && (oc % 32 == 0);
    uint8_t* dstWeight = dst;
    int16_t* dstScale = reinterpret_cast<int16_t*>(dst + (size_t)icP * ocP * 32 * 16);
    const int scaleBlocks = std::max(scaleBlockNum, 1);
    const bool dequantInWeight = scaleBlocks > 1;
    const int alphaUnit = scaleAsymmetric ? 2 : 1;
    const int scaleUnit = dequantInWeight ? (scaleAsymmetric ? 128 : 64) : 32;
    // The packed plane is emitted for asymmetric too, so the M=1 HMX kernel can serve both. Its
    // entries are the same width as the plain plane's (scaleUnit), i.e. 128 fp16 when asymmetric:
    // 64 lanes of paired scale followed by 64 lanes of paired qbias.
    const int packedScaleBlocks = dequantInWeight ? UP_DIV(scaleBlocks, 2) : 0;
    int16_t* dstPackedScale = dequantInWeight ? dstScale + (size_t)ocP * scaleBlocks * scaleUnit : nullptr;
    const size_t weightBytes = (size_t)icP * ocP * 32 * 16;
    const size_t scaleBytes = (size_t)ocP * scaleBlocks * scaleUnit * sizeof(int16_t);
    const size_t packedScaleBytes = (size_t)ocP * packedScaleBlocks * scaleUnit * sizeof(int16_t);
    const size_t neededBytes = weightBytes + scaleBytes + packedScaleBytes;

    if (neededBytes > dstBytes || (rawAlphaData != nullptr && rawAlphaSize < oc * scaleBlocks * alphaUnit)) {
        MNN_PRINT("[MNN::Hexagon][int4] invalid q4block reorder bounds: ic=%d oc=%d icP=%d ocP=%d scaleBlocks=%d alpha=%d need=%zu dst=%zu weight=%zu scale=%zu packed=%zu\n",
                  ic, oc, icP, ocP, scaleBlocks, rawAlphaSize, neededBytes, dstBytes, weightBytes, scaleBytes, packedScaleBytes);
        return false;
    }
    if (!aligned) {
        ::memset(dst, 0, neededBytes);
    }

    alignas(128) uint8_t local[32 * 32];
#if !(defined(__ARM_NEON) || defined(__ARM_NEON__))
    alignas(128) uint8_t shuffled[32 * 32];
#endif
    for (int y = 0; y < ocP; ++y) {
        for (int x = 0; x < icP; ++x) {
            if (!aligned) {
                ::memset(local, 8, sizeof(local));
            }
            const int yCount = std::min(32, oc - y * 32);
            const int xCount = std::min(16, icBytes - x * 16);
            for (int yi = 0; yi < yCount; ++yi) {
                const uint8_t* src = rawInt4Data + (size_t)(y * 32 + yi) * icBytes + x * 16;
                for (int xi = 0; xi < xCount; ++xi) {
                    const uint8_t val = src[xi];
                    local[2 * xi * 32 + 2 * yi] = val >> 4;
                    local[2 * xi * 32 + 2 * yi + 1] = val & 0x0f;
                }
            }

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            uint8_t* dstChunk = dstWeight + (size_t)(y * icP + x) * 32 * 16;
            for (int q = 0; q < 4; ++q) {
                const uint8_t* lowSrc = local + q * 256;
                const uint8_t* highSrc = lowSrc + 128;
                uint8_t* dstQ = dstChunk + q * 128;
                for (int i = 0; i < 64; i += 16) {
                    const uint8x16_t low0 = vld1q_u8(lowSrc + i);
                    const uint8x16_t low1 = vld1q_u8(lowSrc + 64 + i);
                    const uint8x16_t high0 = vld1q_u8(highSrc + i);
                    const uint8x16_t high1 = vld1q_u8(highSrc + 64 + i);
                    const uint8x16x2_t lowZip = vzipq_u8(low0, low1);
                    const uint8x16x2_t highZip = vzipq_u8(high0, high1);
                    vst1q_u8(dstQ + 2 * i, vorrq_u8(lowZip.val[0], vshlq_n_u8(highZip.val[0], 4)));
                    vst1q_u8(dstQ + 2 * i + 16, vorrq_u8(lowZip.val[1], vshlq_n_u8(highZip.val[1], 4)));
                }
            }
#else
            for (int q = 0; q < 8; ++q) {
                const uint8_t* src = local + q * 128;
                uint8_t* dstChunk = shuffled + q * 128;
                for (int i = 0; i < 64; ++i) {
                    dstChunk[2 * i] = src[i];
                    dstChunk[2 * i + 1] = src[64 + i];
                }
            }

            uint8_t* dstChunk = dstWeight + (size_t)(y * icP + x) * 32 * 16;
            for (int q = 0; q < 4; ++q) {
                const uint8_t* low = shuffled + q * 256;
                const uint8_t* high = low + 128;
                for (int i = 0; i < 128; ++i) {
                    dstChunk[q * 128 + i] = (low[i] & 0x0f) | ((high[i] & 0x0f) << 4);
                }
            }
#endif
        }
    }

    if (rawAlphaData != nullptr && oc > 0 && !dequantInWeight) {
        fp32tofp16(rawAlphaData, dstScale, oc);
        if ((ocP * 32) > oc) {
            ::memset(dstScale + oc, 0, (size_t)(ocP * 32 - oc) * sizeof(int16_t));
        }
    } else if (rawAlphaData != nullptr && oc > 0) {
        const int blockUnit = 64;
        std::vector<float> scaleTile((size_t)scaleBlocks * scaleUnit);
        for (int y = 0; y < ocP; ++y) {
            for (int k = 0; k < scaleBlocks; ++k) {
                float* dstFloat = scaleTile.data() + (size_t)k * scaleUnit;
                for (int yi = 0; yi < 32; ++yi) {
                    const int o = y * 32 + yi;
                    float scale = 0.0f;
                    float offset = 0.0f;
                    if (o < oc) {
                        const size_t scaleIndex = (size_t)o * scaleBlocks + k;
                        if (scaleAsymmetric) {
                            offset = rawAlphaData[2 * scaleIndex];
                            scale = rawAlphaData[2 * scaleIndex + 1];
                        } else {
                            scale = rawAlphaData[scaleIndex];
                        }
                    }
                    dstFloat[2 * yi] = scale;
                    dstFloat[2 * yi + 1] = scale;
                    if (scaleAsymmetric) {
                        dstFloat[blockUnit + 2 * yi] = offset;
                        dstFloat[blockUnit + 2 * yi + 1] = offset;
                    }
                }
            }
            int16_t* dstScaleTile = dstScale + (size_t)y * scaleBlocks * scaleUnit;
            fp32tofp16(scaleTile.data(), dstScaleTile, (size_t)scaleBlocks * scaleUnit);
            // Packed plane (M=1 HMX kernel): block k lands in even lanes, block k+1 in odd lanes,
            // matching how prepare_q4block_m1_scale_rows_cached splits a block pair across the
            // halves of a 32-bit lane. qbias repeats that pairing in the second half of the entry.
            // An odd tail leaves its partner at 0, which the activation-sum builder mirrors.
            int16_t* dstPackedScaleTile = dstPackedScale + (size_t)y * packedScaleBlocks * scaleUnit;
            for (int k = 0; k < scaleBlocks; k += 2) {
                int16_t* dstInt = dstPackedScaleTile + (size_t)(k / 2) * scaleUnit;
                const int16_t* entry0 = dstScaleTile + (size_t)k * scaleUnit;
                const int16_t* entry1 = k + 1 < scaleBlocks ? dstScaleTile + (size_t)(k + 1) * scaleUnit : nullptr;
                for (int yi = 0; yi < 32; ++yi) {
                    dstInt[2 * yi] = entry0[2 * yi];
                    dstInt[2 * yi + 1] = entry1 ? entry1[2 * yi] : 0;
                    if (scaleAsymmetric) {
                        dstInt[blockUnit + 2 * yi] = entry0[blockUnit + 2 * yi];
                        dstInt[blockUnit + 2 * yi + 1] = entry1 ? entry1[blockUnit + 2 * yi] : 0;
                    }
                }
            }
        }
    } else if ((ocP * 32) > oc) {
        ::memset(dstScale + oc, 0, (size_t)(ocP * 32 - oc) * sizeof(int16_t));
    }
    return true;
}

static bool reorderInt8SymWeightForHmx(uint8_t* dst, size_t dstBytes, const int8_t* rawInt8Data,
                                       const float* rawAlphaData, int rawAlphaSize,
                                       int ic, int oc, int kernelX, int kernelY, int scaleBlockNum,
                                       bool scaleAsymmetric,
                                       void(*fp32tofp16)(const float*, int16_t*, size_t)) {
    constexpr int icPack = 32;
    constexpr int ocPack = 32;
    const int icP = UP_DIV(ic, icPack);
    const int ocP = UP_DIV(oc, ocPack);
    const int kp = kernelY * kernelX * icP;
    constexpr int packs = icPack * ocPack;
    const size_t weightBytes = (size_t)ocP * kp * packs;
    const int scaleBlocks = std::max(scaleBlockNum, 1);
    const int alphaUnit = scaleAsymmetric ? 2 : 1;
    const int scaleUnit = scaleAsymmetric ? 128 : (scaleBlocks > 1 ? 64 : 32);
    const size_t scaleBytes = (size_t)ocP * scaleBlocks * scaleUnit * sizeof(int16_t);
    const size_t neededBytes = weightBytes + scaleBytes;
    if (dst == nullptr || rawInt8Data == nullptr || rawAlphaData == nullptr ||
        rawAlphaSize < oc * scaleBlocks * alphaUnit || neededBytes > dstBytes) {
        MNN_PRINT("[MNN::Hexagon][int8] invalid w8 reorder bounds: ic=%d oc=%d alpha=%d need=%zu dst=%zu\n",
                  ic, oc, rawAlphaSize, neededBytes, dstBytes);
        return false;
    }
    ::memset(dst, 0, neededBytes);
    int8_t* dstWeight = reinterpret_cast<int8_t*>(dst);
    int16_t* dstScale = reinterpret_cast<int16_t*>(dst + weightBytes);
    if (scaleBlocks == 1 && !scaleAsymmetric) {
        fp32tofp16(rawAlphaData, dstScale, oc);
    } else {
        std::vector<float> scaleTile((size_t)scaleBlocks * scaleUnit);
        for (int oz = 0; oz < ocP; ++oz) {
            for (int block = 0; block < scaleBlocks; ++block) {
                float* dstBlock = scaleTile.data() + (size_t)block * scaleUnit;
                for (int oy = 0; oy < ocPack; ++oy) {
                    const int o = oz * ocPack + oy;
                    float scale = 0.0f;
                    float offset = 0.0f;
                    if (o < oc) {
                        const size_t scaleIndex = (size_t)o * scaleBlocks + block;
                        if (scaleAsymmetric) {
                            offset = rawAlphaData[2 * scaleIndex];
                            scale = rawAlphaData[2 * scaleIndex + 1];
                        } else {
                            scale = rawAlphaData[scaleIndex];
                        }
                    }
                    dstBlock[2 * oy] = scale;
                    dstBlock[2 * oy + 1] = scale;
                    if (scaleAsymmetric) {
                        dstBlock[64 + 2 * oy] = offset;
                        dstBlock[64 + 2 * oy + 1] = offset;
                    }
                }
            }
            fp32tofp16(scaleTile.data(), dstScale + (size_t)oz * scaleBlocks * scaleUnit,
                       (size_t)scaleBlocks * scaleUnit);
        }
    }

    for (int oz = 0; oz < ocP; ++oz) {
        for (int kk = 0; kk < kp; ++kk) {
            const int kernelIndex = kk / icP;
            const int iz = kk % icP;
            const int ky = kernelIndex / kernelX;
            const int kx = kernelIndex % kernelX;
            const size_t blockBase = ((size_t)oz * kp + kk) * packs;
            for (int oy = 0; oy < ocPack; ++oy) {
                const int o = oz * ocPack + oy;
                if (o >= oc) {
                    continue;
                }
                for (int ix = 0; ix < icPack; ++ix) {
                    const int i = iz * icPack + ix;
                    if (i >= ic) {
                        continue;
                    }
                    const size_t srcIndex = (((size_t)o * ic + i) * kernelY + ky) * kernelX + kx;
                    const int ixPair = ix / 2;
                    const int ixRem = ix & 1;
                    const int lane = oy * 2 + ixRem;
                    const size_t dstIndex = blockBase + (size_t)(ixPair / 2) * 128 + lane * 2 + (ixPair & 1);
                    dstWeight[dstIndex] = rawInt8Data[srcIndex];
                }
            }
        }
    }
    return true;
}

static bool reorderInt4WeightForVrmpyGemv(uint8_t* dst, size_t dstBytes, const uint8_t* rawInt4Data,
                                          const float* rawAlphaData, int ic, int oc, int nblk, bool asymmetric) {
    if ((ic % 32) != 0 || (oc % 32) != 0 || nblk <= 0) {
        return false;
    }
    const int icP = ic / 32;
    const int ocP = oc / 32;
    const int icBytes = UP_DIV(ic, 2);
    const int alphaUnit = asymmetric ? 2 : 1;
    // Scale entry per (oc-tile, block) is 32 fp32, followed by 32 fp32 qbias when asymmetric, so the
    // GEMV's single scale DMA per oc-tile still covers both.
    const int entryUnit = asymmetric ? 2 : 1;
    const size_t weightBytes = (size_t)icP * ocP * 512;
    const size_t scaleBytes = (size_t)ocP * nblk * 32 * entryUnit * sizeof(float);
    if (weightBytes + scaleBytes > dstBytes) {
        return false;
    }
    uint8_t* dstWeight = dst;
    float* dstScale = reinterpret_cast<float*>(dst + weightBytes);
    auto rawNibble = [&](int o, int k) -> uint8_t {
        const uint8_t byte = rawInt4Data[(size_t)o * icBytes + (k >> 1)];
        return (k & 1) ? (byte & 0x0f) : (uint8_t)(byte >> 4);
    };
    for (int y = 0; y < ocP; ++y) {
        for (int x = 0; x < icP; ++x) {
            uint8_t* tile = dstWeight + (size_t)(y * icP + x) * 512;
            for (int ocIn = 0; ocIn < 32; ++ocIn) {
                const int o = y * 32 + ocIn;
                for (int g = 0; g < 8; ++g) {
                    for (int p = 0; p < 2; ++p) {
                        const int kLo = x * 32 + 4 * g + 2 * p;
                        const int kHi = kLo + 1;
                        tile[g * 64 + ocIn * 2 + p] =
                            (uint8_t)(rawNibble(o, kLo) | (rawNibble(o, kHi) << 4));
                    }
                }
            }
        }
    }
    for (int y = 0; y < ocP; ++y) {
        for (int b = 0; b < nblk; ++b) {
            float* entry = dstScale + ((size_t)y * nblk + b) * 32 * entryUnit;
            for (int ocIn = 0; ocIn < 32; ++ocIn) {
                const int o = y * 32 + ocIn;
                const size_t src = ((size_t)o * nblk + b) * alphaUnit;
                entry[ocIn] = asymmetric ? rawAlphaData[src + 1] : rawAlphaData[src];
                if (asymmetric) {
                    entry[32 + ocIn] = rawAlphaData[src];
                }
            }
        }
    }
    return true;
}

// fp32 per-oc-tile block scales for the M=1 W8A16 GEMV: layout contract of
// matmul_w8a16_gemv_i8.c — sw[(oy*nblk + blk)*32 + ocIn] = alpha[(oy*32+ocIn)*nblk + blk].
// The int8 weights themselves are consumed from the existing HMX int8Weight blob
// (tile (oy,kx) at (oy*kp+kx)*1024, byte (g*128 + ocIn*4 + p) = w(ocIn,
// k = kx*32+4g+perm[p]), perm={0,2,1,3}), so no second weight copy is needed.
static bool reorderInt8ScaleForGemv(uint8_t* dst, size_t dstBytes, const float* rawAlphaData,
                                    int oc, int nblk) {
    if (nblk <= 0 || (oc % 32) != 0) {
        return false;
    }
    const int ocP = oc / 32;
    const size_t scaleBytes = (size_t)ocP * nblk * 32 * sizeof(float);
    if (scaleBytes > dstBytes) {
        return false;
    }
    ::memset(dst, 0, scaleBytes);  // padded oc (> real oc) get 0 scale
    float* dstScale = reinterpret_cast<float*>(dst);
    for (int y = 0; y < ocP; ++y) {
        for (int b = 0; b < nblk; ++b) {
            for (int ocIn = 0; ocIn < 32; ++ocIn) {
                const int o = y * 32 + ocIn;
                if (o >= oc) {
                    continue;
                }
                dstScale[((size_t)y * nblk + b) * 32 + ocIn] = rawAlphaData[(size_t)o * nblk + b];
            }
        }
    }
    return true;
}

static void reorderFp16WeightForHmx(int16_t* dst, const int16_t* src, int ic, int oc, int kernelX, int kernelY) {
    constexpr int icPack = 32;
    constexpr int ocPack = 32;
    const int icP = UP_DIV(ic, icPack);
    const int ocP = UP_DIV(oc, ocPack);
    const int kp = kernelY * kernelX * icP;
    constexpr int packs = icPack * ocPack;
    const size_t reorderedSize = (size_t)ocP * kp * packs;
    if (icP * icPack != ic || ocP * ocPack != oc) {
        ::memset(dst, 0, reorderedSize * sizeof(int16_t));
    }
    for (int oz = 0; oz < ocP; ++oz) {
        for (int kk = 0; kk < kp; ++kk) {
            const int kernelIndex = kk / icP;
            const int iz = kk % icP;
            const int ky = kernelIndex / kernelX;
            const int kx = kernelIndex % kernelX;
            const size_t blockBase = ((size_t)oz * kp + kk) * packs;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            if (kernelX == 1 && kernelY == 1 && oz * ocPack + ocPack <= oc && iz * icPack + icPack <= ic) {
                auto transpose8x8 = [](int16x8_t r0, int16x8_t r1, int16x8_t r2, int16x8_t r3,
                                        int16x8_t r4, int16x8_t r5, int16x8_t r6, int16x8_t r7,
                                        int16x8_t* c0, int16x8_t* c1, int16x8_t* c2, int16x8_t* c3,
                                        int16x8_t* c4, int16x8_t* c5, int16x8_t* c6, int16x8_t* c7) {
                    const int16x8x2_t t0 = vtrnq_s16(r0, r1);
                    const int16x8x2_t t1 = vtrnq_s16(r2, r3);
                    const int16x8x2_t t2 = vtrnq_s16(r4, r5);
                    const int16x8x2_t t3 = vtrnq_s16(r6, r7);
                    const int32x4x2_t u0 = vtrnq_s32(vreinterpretq_s32_s16(t0.val[0]), vreinterpretq_s32_s16(t1.val[0]));
                    const int32x4x2_t u1 = vtrnq_s32(vreinterpretq_s32_s16(t0.val[1]), vreinterpretq_s32_s16(t1.val[1]));
                    const int32x4x2_t u2 = vtrnq_s32(vreinterpretq_s32_s16(t2.val[0]), vreinterpretq_s32_s16(t3.val[0]));
                    const int32x4x2_t u3 = vtrnq_s32(vreinterpretq_s32_s16(t2.val[1]), vreinterpretq_s32_s16(t3.val[1]));
                    const int64x2_t a0 = vreinterpretq_s64_s32(u0.val[0]);
                    const int64x2_t a1 = vreinterpretq_s64_s32(u1.val[0]);
                    const int64x2_t a2 = vreinterpretq_s64_s32(u0.val[1]);
                    const int64x2_t a3 = vreinterpretq_s64_s32(u1.val[1]);
                    const int64x2_t b0 = vreinterpretq_s64_s32(u2.val[0]);
                    const int64x2_t b1 = vreinterpretq_s64_s32(u3.val[0]);
                    const int64x2_t b2 = vreinterpretq_s64_s32(u2.val[1]);
                    const int64x2_t b3 = vreinterpretq_s64_s32(u3.val[1]);
                    *c0 = vreinterpretq_s16_s64(vcombine_s64(vget_low_s64(a0), vget_low_s64(b0)));
                    *c1 = vreinterpretq_s16_s64(vcombine_s64(vget_low_s64(a1), vget_low_s64(b1)));
                    *c2 = vreinterpretq_s16_s64(vcombine_s64(vget_low_s64(a2), vget_low_s64(b2)));
                    *c3 = vreinterpretq_s16_s64(vcombine_s64(vget_low_s64(a3), vget_low_s64(b3)));
                    *c4 = vreinterpretq_s16_s64(vcombine_s64(vget_high_s64(a0), vget_high_s64(b0)));
                    *c5 = vreinterpretq_s16_s64(vcombine_s64(vget_high_s64(a1), vget_high_s64(b1)));
                    *c6 = vreinterpretq_s16_s64(vcombine_s64(vget_high_s64(a2), vget_high_s64(b2)));
                    *c7 = vreinterpretq_s16_s64(vcombine_s64(vget_high_s64(a3), vget_high_s64(b3)));
                };
                const int oBase = oz * ocPack;
                const int iBase = iz * icPack;
                int16_t* blockDst = dst + blockBase;
                for (int oyBase = 0; oyBase < ocPack; oyBase += 8) {
                    for (int ixBase = 0; ixBase < icPack; ixBase += 8) {
                        const int16_t* srcBase = src + (size_t)(oBase + oyBase) * ic + iBase + ixBase;
                        const int16x8_t r0 = vld1q_s16(srcBase + (size_t)0 * ic);
                        const int16x8_t r1 = vld1q_s16(srcBase + (size_t)1 * ic);
                        const int16x8_t r2 = vld1q_s16(srcBase + (size_t)2 * ic);
                        const int16x8_t r3 = vld1q_s16(srcBase + (size_t)3 * ic);
                        const int16x8_t r4 = vld1q_s16(srcBase + (size_t)4 * ic);
                        const int16x8_t r5 = vld1q_s16(srcBase + (size_t)5 * ic);
                        const int16x8_t r6 = vld1q_s16(srcBase + (size_t)6 * ic);
                        const int16x8_t r7 = vld1q_s16(srcBase + (size_t)7 * ic);
                        int16x8_t c0, c1, c2, c3, c4, c5, c6, c7;
                        transpose8x8(r0, r1, r2, r3, r4, r5, r6, r7, &c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7);
                        int16x8x2_t p01 = {c0, c1};
                        int16x8x2_t p23 = {c2, c3};
                        int16x8x2_t p45 = {c4, c5};
                        int16x8x2_t p67 = {c6, c7};
                        vst2q_s16(blockDst + (size_t)(ixBase / 2 + 0) * 64 + oyBase * 2, p01);
                        vst2q_s16(blockDst + (size_t)(ixBase / 2 + 1) * 64 + oyBase * 2, p23);
                        vst2q_s16(blockDst + (size_t)(ixBase / 2 + 2) * 64 + oyBase * 2, p45);
                        vst2q_s16(blockDst + (size_t)(ixBase / 2 + 3) * 64 + oyBase * 2, p67);
                    }
                }
                continue;
            }
#endif
            for (int oy = 0; oy < ocPack; ++oy) {
                const int o = oz * ocPack + oy;
                if (o >= oc) {
                    continue;
                }
                for (int ix = 0; ix < icPack; ++ix) {
                    const int i = iz * icPack + ix;
                    if (i >= ic) {
                        continue;
                    }
                    const size_t srcIndex = (((size_t)o * ic + i) * kernelY + ky) * kernelX + kx;
                    const int ixPair = ix / 2;
                    const int ixRem = ix & 1;
                    const size_t dstIndex = blockBase + (size_t)ixPair * 64 + oy * 2 + ixRem;
                    dst[dstIndex] = src[srcIndex];
                }
            }
        }
    }
}

HexagonConvolution::Resource::~Resource() {
    if (weight.first != nullptr) {
        allocator->free(weight);
    }
    if (bias.first != nullptr) {
        allocator->free(bias);
    }
    if (int4Weight.first != nullptr) {
        allocator->free(int4Weight);
    }
    if (gatherInt4Weight.first != nullptr) {
        allocator->free(gatherInt4Weight);
    }
    if (int8Weight.first != nullptr) {
        allocator->free(int8Weight);
    }
    if (gemvI8Weight.first != nullptr) {
        allocator->free(gemvI8Weight);
    }
    if (gemvW8Scale.first != nullptr) {
        allocator->free(gemvW8Scale);
    }
}
HexagonConvolution::HexagonConvolution(Backend *backend, std::shared_ptr<Resource> res, const Op* op) : HexagonExecution(backend) {
    mResource = res;
    mOp = op;
    if (op != nullptr) {
        auto conv2d = op->main_as_Convolution2D();
        if (conv2d != nullptr && conv2d->common() != nullptr) {
            auto common = conv2d->common();
            mKernelY = common->kernelY();
            mKernelX = common->kernelX();
            mStrideY = common->strideY();
            mStrideX = common->strideX();
            mDilateY = common->dilateY();
            mDilateX = common->dilateX();
            mRelu = common->relu() ? 1 : 0;
            mRelu6 = common->relu6() ? 1 : 0;
            mUseIm2Col = (mResource == nullptr || !mResource->useInt4W4A16);
        }
    }
}

ErrorCode HexagonConvolution::onBuildCmd(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                         std::vector<HexagonCommand>& dst) {
    const auto runtime = static_cast<const HexagonRuntime*>(backend()->getRuntime());
    int vtcmSize = runtime->info().vtcmSize;
    if (vtcmSize <= 0) {
        vtcmSize = 4 * 1024 * 1024; // Default 4MB if not available
    }

    int batch = outputs[0]->length(0);
    int oc = outputs[0]->length(1);
    int oh = outputs[0]->dimensions() > 2 ? outputs[0]->length(2) : 1;
    int ow = outputs[0]->dimensions() > 3 ? outputs[0]->length(3) : 1;
    int area = batch * oh * ow;
    int ih = inputs[0]->dimensions() > 2 ? inputs[0]->length(2) : 1;
    int iw = inputs[0]->dimensions() > 3 ? inputs[0]->length(3) : 1;
    int ic = inputs[0]->length(1);

    int M = area;
    int icP = UP_DIV(ic, 32);
    int K = mUseIm2Col ? (mKernelY * mKernelX * icP * 32) : ic;
    int N = oc;
    int KAlign = UP_DIV(K, 32) * 32;

    HmxIm2ColConvParam im2colParams{};
    bool useConv1x1Direct = false;
    if (mUseIm2Col) {
        auto conv2d = mOp->main_as_Convolution2D();
        auto common = conv2d->common();
        auto pads = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], common);
        ::memset(&mParam, 0, sizeof(mParam));
        setHexagonIm2ColParameter(mParam, common, inputs[0], outputs[0], pads.first, pads.second, 64);
        useConv1x1Direct = common->kernelX() == 1 && common->kernelY() == 1 &&
                           common->strideX() == 1 && common->strideY() == 1 &&
                           common->dilateX() == 1 && common->dilateY() == 1 &&
                           pads.first == 0 && pads.second == 0;
        mParam.kernelCountUnit = common->kernelX() * common->kernelY() * UP_DIV(ic, 32);
        mParam.ic = UP_DIV(ic, 32) * 32;
        mParam.icup4 = UP_DIV(ic, 32) * 32;
        ::memcpy(&im2colParams.im2col, &mParam, sizeof(mParam));
        im2colParams.oc = oc;
    }

    int total_mp = UP_DIV(M, 32);
    int total_np = UP_DIV(N, 32);

    // Im2Col VTCM tiles:
    //   T = 32 * 32 * kp * sizeof(fp16) = 64 * KAlign bytes
    //   (mMp + mNp) * T + fixed output/scale workspace <= vtcmSize
    //
    // Variable fill traffic, in units of T:
    //   oxChunks = ceil(total_mp / mMp), oyChunks = ceil(total_np / mNp)
    //   activation-outer order: total_mp + oxChunks * total_np
    //   weight-outer order:     oyChunks * total_mp + total_np
    // HMX tile reads and output stores are essentially fixed for a convolution, so choose the
    // tile shape that minimizes min(activation-outer, weight-outer).
    // For int4
    // (mNp + 2 * mMp + mNp / 4) * 64 * K + other_vtcm_overhead <= vtcmSize
    // We reserve some space for other structures (approx 8KB)
    const int avail_size = vtcmSize - 4 * 1024 - 256;
    Q4ScaleMode q4ScaleMode = Q4ScaleMode::None;
    if (!mUseIm2Col && mResource != nullptr && mResource->useInt4W4A16) {
        if (mResource->int4ScaleBlockNum > 1) {
            q4ScaleMode = mResource->int4ScaleAsymmetric ? Q4ScaleMode::BlockAsymmetric
                                                         : Q4ScaleMode::BlockSymmetric;
        } else {
            q4ScaleMode = Q4ScaleMode::PerOutput;
        }
    }
    const bool useInt8Staging = mUseIm2Col && mResource != nullptr && mResource->useInt8W8A16;
    HexagonTileShape tile = mUseIm2Col
        ? chooseIm2ColTileShape(total_mp, total_np, KAlign, avail_size, useInt8Staging,
                                mResource && mResource->useInt8W8A16 ? mResource->int8ScaleBlockNum : 1,
                                mResource && mResource->useInt8W8A16 && mResource->int8ScaleAsymmetric ? 1 : 0)
                                       : chooseDirectTileShape(total_mp, total_np, KAlign, avail_size,
                                                               vtcmSize, q4ScaleMode,
                                                               mResource ? mResource->int4ScaleBlockNum : 1);
    mMp = tile.mp;
    mNp = tile.np;
    mKp = UP_DIV(K, 32);
    if (mUseIm2Col) {
        im2colParams.mp = mMp;
        im2colParams.np = mNp;
        im2colParams.relu = mRelu;
        im2colParams.relu6 = mRelu6;
        im2colParams.batch = batch;
        im2colParams.outputBytes = (int32_t)static_cast<HexagonBackend*>(backend())->getSize(outputs[0]);
        im2colParams.scaleBlockNum = mResource && mResource->useInt8W8A16 ? mResource->int8ScaleBlockNum : 1;
        im2colParams.scaleAsymmetric =
            mResource && mResource->useInt8W8A16 && mResource->int8ScaleAsymmetric ? 1 : 0;
    }
//    FUNC_PRINT(vtcmSize);
//    FUNC_PRINT(mMp);
//    FUNC_PRINT(mNp);
//    FUNC_PRINT(maxNp);

    auto input = HexagonBackend::getDevicePtr(inputs[0]);
    auto output = HexagonBackend::getDevicePtr(outputs[0]);
    std::pair<int, int> bias = {-1, 0};
    if (mResource->hasBias) {
        bias = HexagonBackend::getDevicePtr(mResource->bias);
    }

    auto hex_backend = static_cast<HexagonBackend*>(backend());
    int ocP = UP_DIV(oc, 32);
    std::vector<std::pair<int, int>> outputFds = {output};

    if (area == 1 && mResource->useGemvW8a16 && mResource->gemvW8Scale.first != nullptr &&
        mResource->useInt8W8A16 && !mResource->int8ScaleAsymmetric && mKernelX == 1 && mKernelY == 1 &&
        mRelu == 0 && mRelu6 == 0 && (ic % 64 == 0) && (oc % 32 == 0) &&
        mResource->int8ScaleBlockNum == (ic / 64)) {
        // Specialized M=1 W8A16 decode GEMV (INT8 symmetric, block size 64).
        // Weights are the existing HMX-layout int8Weight (no second copy).
        auto weight = HexagonBackend::getDevicePtr(mResource->int8Weight);
        auto scale = HexagonBackend::getDevicePtr(mResource->gemvW8Scale);
        const int gemvK = icP * 32;  // == ic when ic%32==0
        int params[] = {1, gemvK, ocP * 32, 0, 0, 1, 1, UP_DIV(gemvK, 32), mResource->int8ScaleBlockNum, 0};
        std::vector<std::pair<int, int>> inputFds = {input, weight, scale, bias};
        dst.emplace_back();
        dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_MATMUL_W8A16_GEMV_I8, params,
                         sizeof(params), inputFds, outputFds, inputs, outputs);
    } else if (mUseIm2Col) {
        auto weight = mResource->useInt8W8A16 ? HexagonBackend::getDevicePtr(mResource->int8Weight)
                                              : HexagonBackend::getDevicePtr(mResource->weight);
        std::vector<std::pair<int, int>> inputFds = {input, weight, bias};
        dst.emplace_back();
        const auto opType = mResource->useInt8W8A16 ? DSP_OP_MATMUL_W8A16_BLOCK_FP16
                                                    : (useConv1x1Direct ? DSP_OP_CONV1X1_DIRECT_FP16 : DSP_OP_IM2COL_CONVOLUTION_FP16);
        dst.back().build(static_cast<HexagonBackend*>(backend()), opType, &im2colParams, sizeof(im2colParams),
                         inputFds,  outputFds,  inputs, outputs);
    } else if (area == 1 && mResource->useGemvI8 && mResource->gemvI8Weight.first != nullptr) {
        auto weight = HexagonBackend::getDevicePtr(mResource->gemvI8Weight);
        int params[] = {area, icP * 32, ocP * 32, mResource->int4WeightType,    mResource->int4LayoutType,
                        mMp,  mNp,      mKp,      mResource->int4ScaleBlockNum, mResource->int4ScaleAsymmetric ? 1 : 0};
        std::vector<std::pair<int, int>> inputFds = {input, weight, bias};
        dst.emplace_back();
        dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_MATMUL_Q4A16_GEMV_I8, params,
                         sizeof(params), inputFds, outputFds, inputs, outputs);
    } else if (mResource->useInt4W4A16) {
        // Kernel don't need treat not aligned ic / oc
        const bool pathA =
            mResource->int4ScaleBlockNum > 1 && mResource->useGemvI8 && mResource->gemvI8Weight.first != nullptr;
        auto weight = pathA ? HexagonBackend::getDevicePtr(mResource->gemvI8Weight)
                            : HexagonBackend::getDevicePtr(mResource->int4Weight);
        int params[] = {area, icP * 32, ocP * 32, mResource->int4WeightType, mResource->int4LayoutType,
                        mMp, mNp, mKp, mResource->int4ScaleBlockNum,
                        mResource->int4ScaleAsymmetric ? 1 : 0, pathA ? 1 : 0};
        std::vector<std::pair<int, int>> inputFds = {input, weight, bias};
        const auto opType = mResource->int4ScaleBlockNum > 1 ? DSP_OP_MATMUL_Q4A16_BLOCK_FP16 : DSP_OP_MATMUL_Q4A16_FP16;
        dst.emplace_back();
        dst.back().build(static_cast<HexagonBackend*>(backend()), opType, params, sizeof(params),
                         inputFds,  outputFds,  inputs, outputs);
    }

    return NO_ERROR;
}
bool HexagonConvolution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    if (op != nullptr && op->type() == OpType_GatherV2) {
        if (mResource == nullptr) {
            return false;
        }
        if ((!mResource->useInt4W4A16 && !mResource->useInt8W8A16 && mResource->weight.first == nullptr) ||
            (mResource->useInt8W8A16 && mResource->int8Weight.first == nullptr) ||
            (mResource->useInt4W4A16 && mResource->int4Weight.first == nullptr)) {
            return false;
        }
        *dst = new HexagonSharedGather(bn, mResource);
        return true;
    }
    auto exe = new HexagonConvolution(bn, mResource, op);
    exe->mParam = mParam;
    *dst = exe;
    return true;
}
HexagonConvolution* HexagonConvolution::create(Backend *backend, const Op* op) {
    auto conv2d = op->main_as_Convolution2D();
    if (conv2d == nullptr || conv2d->common() == nullptr) {
        return nullptr;
    }
    auto common = conv2d->common();
    int ic = common->inputCount();
    int oc = common->outputCount();
    const bool fastWay = common->kernelY() == 1 && common->kernelX() == 1 && common->strideX() == 1 &&
                         common->strideY() == 1 && common->dilateX() == 1 && common->dilateY() == 1;
    const int ocPack = 32;
    const int icPack = 32;

    const float* originWeight = nullptr;
    int originWeightSize = 0;
    const float* originBias = nullptr;
    int originBiasSize = 0;
    if (conv2d->bias() != nullptr) {
        originBias = conv2d->bias()->data();
        originBiasSize = conv2d->bias()->size();
    }
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;

    bool useInt4W4A16 = false;
    bool useInt8W8A16 = false;
    int int4WeightType = 0;
    int int4LayoutType = 1;
    if (fastWay && nullptr != conv2d->quanParameter()) {
        auto quan = conv2d->quanParameter();
        auto ext = conv2d->external();
        if (quan->type() == 1 && quan->aMaxOrBits() == 4 && ext != nullptr && ext->size() > 0) {
            useInt4W4A16 = true;
        }
    }
    bool useIm2Col = !useInt4W4A16;
    if (nullptr != conv2d->quanParameter()) {
        if (useInt4W4A16) {
            quanCommon = ConvolutionCommon::load(op, backend, false, true, nullptr);
            originWeight = nullptr;
            originWeightSize = quanCommon->weight.size();
        } else {
            quanCommon = ConvolutionCommon::load(op, backend, false, true, nullptr);
            if (fastWay && quanCommon != nullptr && quanCommon->weight.get() != nullptr &&
                quanCommon->alpha.get() != nullptr && quanCommon->alpha.size() >= oc &&
                conv2d->quanParameter()->index() == nullptr) {
                useInt8W8A16 = true;
                originWeight = nullptr;
                originWeightSize = quanCommon->weight.size();
            } else {
                quanCommon = ConvolutionCommon::load(op, backend, true, false, nullptr);
                originWeight = quanCommon->weightFloat.get();
                originWeightSize = quanCommon->weightFloat.size();
            }
        }
    } else {
        originWeight = conv2d->weight()->data();
        originWeightSize = conv2d->weight()->size();
    }

    const int kernelSize = common->kernelY() * common->kernelX();
    if (ic == 0) {
        if (useInt4W4A16) {
            ic = (originWeightSize * 2) / oc;
        } else if (useInt8W8A16) {
            ic = originWeightSize / (oc * kernelSize);
        } else {
            ic = originWeightSize / (oc * kernelSize);
        }
    }
    if (ic <= 0 || oc <= 0) {
        return nullptr;
    }
    auto icP = UP_DIV(ic, icPack);
    auto ocP = UP_DIV(oc, ocPack);
    auto packs = icPack * ocPack;
    int int4ScaleBlockNum = 1;
    int int8ScaleBlockNum = 1;
    if (useInt4W4A16 && quanCommon && quanCommon->alpha.get() != nullptr) {
        const int alphaSize = quanCommon->alpha.size();
        const int alphaUnit = quanCommon->asymmetric ? 2 : 1;
        if (alphaSize >= oc * alphaUnit && alphaSize % (oc * alphaUnit) == 0) {
            int4ScaleBlockNum = alphaSize / (oc * alphaUnit);
        }
        if (int4ScaleBlockNum <= 0 || icP % int4ScaleBlockNum != 0 ||
            (quanCommon->asymmetric && int4ScaleBlockNum <= 1)) {
            useInt4W4A16 = false;
            useIm2Col = true;
            quanCommon = ConvolutionCommon::load(op, backend, true, false, nullptr);
            originWeight = quanCommon->weightFloat.get();
            originWeightSize = quanCommon->weightFloat.size();
            int4ScaleBlockNum = 1;
        }
    }
    if (useInt8W8A16 && quanCommon && quanCommon->alpha.get() != nullptr) {
        const int alphaSize = quanCommon->alpha.size();
        const int alphaUnit = quanCommon->asymmetric ? 2 : 1;
        if (alphaSize >= oc * alphaUnit && alphaSize % (oc * alphaUnit) == 0) {
            int8ScaleBlockNum = alphaSize / (oc * alphaUnit);
        }
        if (int8ScaleBlockNum <= 0 || kernelSize != 1 || icP % int8ScaleBlockNum != 0) {
            useInt8W8A16 = false;
            quanCommon = ConvolutionCommon::load(op, backend, true, false, nullptr);
            originWeight = quanCommon->weightFloat.get();
            originWeightSize = quanCommon->weightFloat.size();
            int8ScaleBlockNum = 1;
        }
    }
    auto weightIC = useIm2Col ? (kernelSize * icP * icPack) : ic;
    auto weightICP = UP_DIV(weightIC, icPack);
    const size_t im2colBlockedWeightSize = (size_t)ocP * icP * common->kernelY() * common->kernelX() * ocPack * icPack;
    // Quantized weights are small enough to stay mapped for the runtime lifetime. Full FP16
    // models can exceed the CDSP mapping window, so their reordered weights use the lazy
    // allocator and are mapped only for the command group that consumes them.
    auto bufferAlloc = static_cast<HexagonBackend*>(backend)->getAllocator(
        useInt4W4A16 || useInt8W8A16 ? 2 : 3);
    std::shared_ptr<Resource> res(new Resource);
    res->allocator = bufferAlloc;
    res->hasBias = hasNonZeroBias(originBias, std::min(oc, originBiasSize));
    if (res->hasBias) {
        const int biasSize = ocP * ocPack * (int)sizeof(int16_t) + 64;
        res->bias = bufferAlloc->alloc((size_t)biasSize);
        auto biasPtr = HexagonBackend::getPtr(res->bias);
        ::memset(biasPtr, 0, (size_t)biasSize);
        HexagonBackend::fp32ToFp16(originBias, (int16_t*)biasPtr, std::min(oc, originBiasSize));
        static_cast<HexagonBackend*>(backend)->markHostInput(res->bias, biasSize);
    }

    res->gatherInputChannels = ic;
    res->gatherOutputChannels = oc;

    bool int4Success = false;
    bool int8Success = false;
    if (useInt4W4A16) {
        const bool dequantInWeight = int4ScaleBlockNum > 1;
        const bool scaleAsymmetric = quanCommon->asymmetric;
        const int scaleUnit = dequantInWeight ? (scaleAsymmetric ? 128 : 64) : 32;
        const size_t packedScaleSize =
            dequantInWeight ? (size_t)ocP * UP_DIV(int4ScaleBlockNum, 2) * scaleUnit * sizeof(int16_t) : 0;
        const size_t int4WeightSize = (size_t)icP * ocP * 32 * 16 +
                                      (size_t)ocP * int4ScaleBlockNum * scaleUnit * sizeof(int16_t) +
                                      packedScaleSize;
        const uint8_t* rawInt4Data = reinterpret_cast<const uint8_t*>(quanCommon->weight.get());
        const float* rawAlphaData = quanCommon->alpha.get();
        const size_t rawInt4Size = quanCommon->weight.size();
        const size_t gatherInt4Size = ((size_t)(ic + 1) / 2) * oc;
        if (rawInt4Size < gatherInt4Size || rawAlphaData == nullptr) {
            return nullptr;
        }
        res->int4WeightType = int4WeightType;
        res->int4LayoutType = int4LayoutType;
        res->int4ScaleBlockNum = int4ScaleBlockNum;
        res->int4ScaleAsymmetric = scaleAsymmetric;
        const int hvxArch = static_cast<const HexagonRuntime*>(backend->getRuntime())->info().hvxArch;
        const bool gemvShapeOk = dequantInWeight && (ic % 32 == 0) && (oc % 32 == 0) && (ic % 64 == 0) &&
                                 ((ic / 32) % int4ScaleBlockNum == 0);
        if (hvxArch >= 81 && gemvShapeOk) {
            const size_t gemvWeightBytes = (size_t)icP * ocP * 512;
            // Asymmetric entries are 32 fp32 scale + 32 fp32 qbias, so one scale DMA covers both.
            const size_t gemvScaleBytes =
                (size_t)ocP * int4ScaleBlockNum * 32 * (scaleAsymmetric ? 2 : 1) * sizeof(float);
            const size_t gemvBytes = gemvWeightBytes + gemvScaleBytes;
            res->gemvI8Weight = bufferAlloc->alloc(gemvBytes);
            if (res->gemvI8Weight.first != nullptr &&
                reorderInt4WeightForVrmpyGemv(HexagonBackend::getPtr(res->gemvI8Weight), gemvBytes, rawInt4Data,
                                              rawAlphaData, ic, oc, int4ScaleBlockNum, scaleAsymmetric)) {
                res->useGemvI8 = true;
                static_cast<HexagonBackend*>(backend)->markHostInput(res->gemvI8Weight, (int)gemvBytes);
            } else if (res->gemvI8Weight.first != nullptr) {
                bufferAlloc->free(res->gemvI8Weight);
                res->gemvI8Weight = MemChunk();
            }
        }
        if (!res->useGemvI8) {
            res->int4Weight = bufferAlloc->alloc(int4WeightSize);
            if (res->int4Weight.first != nullptr &&
                reorderInt4WeightForHmx(HexagonBackend::getPtr(res->int4Weight), int4WeightSize, rawInt4Data,
                                        rawAlphaData, quanCommon->alpha.size(), ic, oc, int4ScaleBlockNum,
                                        scaleAsymmetric, HexagonBackend::fp32ToFp16)) {
                static_cast<HexagonBackend*>(backend)->markHostInput(res->int4Weight, (int)int4WeightSize);
            } else if (res->int4Weight.first != nullptr) {
                bufferAlloc->free(res->int4Weight);
                res->int4Weight = MemChunk();
            }
        }
        if (res->int4Weight.first != nullptr || res->useGemvI8) {
            int4Success = true;
            res->useInt4W4A16 = true;
        }
    }

    if (useInt8W8A16) {
        const size_t expectedWeightSize = (size_t)oc * ic * kernelSize;
        const size_t int8WeightBytes = im2colBlockedWeightSize;
        const bool int8ScaleAsymmetric = quanCommon->asymmetric;
        const int int8ScaleUnit = int8ScaleAsymmetric ? 128 : (int8ScaleBlockNum > 1 ? 64 : 32);
        const size_t int8ScaleBytes = (size_t)ocP * int8ScaleBlockNum * int8ScaleUnit * sizeof(int16_t);
        const size_t int8BufferSize = int8WeightBytes + int8ScaleBytes;
        if (quanCommon == nullptr || quanCommon->weight.get() == nullptr || quanCommon->alpha.get() == nullptr ||
            (size_t)quanCommon->weight.size() < expectedWeightSize || quanCommon->alpha.size() < oc) {
            return nullptr;
        }
        // This M=1 fast path dynamically quantizes FP16 activation to symmetric
        // INT8. It is enabled by default; set MNN_HEXAGON_W8A16_GEMV_I8=0 to
        // retain the exact W8A16 path for accuracy comparison or fallback.
        const bool gemvW8ShapeOk = useW8A16GemvI8() && !int8ScaleAsymmetric && (ic % 64 == 0) &&
                                   (oc % 32 == 0) && int8ScaleBlockNum > 0 && int8ScaleBlockNum == (ic / 64);
        res->int8Weight = bufferAlloc->alloc(int8BufferSize);
        if (res->int8Weight.first == nullptr) {
            return nullptr;
        }
        if (!reorderInt8SymWeightForHmx(reinterpret_cast<uint8_t*>(HexagonBackend::getPtr(res->int8Weight)),
                                        int8BufferSize, quanCommon->weight.get(), quanCommon->alpha.get(),
                                        quanCommon->alpha.size(), ic, oc,
                                        common->kernelX(), common->kernelY(), int8ScaleBlockNum,
                                        int8ScaleAsymmetric,
                                        HexagonBackend::fp32ToFp16)) {
            return nullptr;
        }
        res->useInt8W8A16 = true;
        res->int8ScaleBlockNum = int8ScaleBlockNum;
        res->int8ScaleAsymmetric = int8ScaleAsymmetric;
        int8Success = true;
        static_cast<HexagonBackend*>(backend)->markHostInput(res->int8Weight, (int)int8BufferSize);

        if (gemvW8ShapeOk) {
            const size_t gemvW8ScaleBytes = (size_t)ocP * int8ScaleBlockNum * 32 * sizeof(float);
            res->gemvW8Scale = bufferAlloc->alloc(gemvW8ScaleBytes);
            if (res->gemvW8Scale.first != nullptr &&
                reorderInt8ScaleForGemv(reinterpret_cast<uint8_t*>(HexagonBackend::getPtr(res->gemvW8Scale)),
                                        gemvW8ScaleBytes, quanCommon->alpha.get(), oc, int8ScaleBlockNum)) {
                res->useGemvW8a16 = true;
                static_cast<HexagonBackend*>(backend)->markHostInput(res->gemvW8Scale, (int)gemvW8ScaleBytes);
            } else if (res->gemvW8Scale.first != nullptr) {
                bufferAlloc->free(res->gemvW8Scale);
                res->gemvW8Scale = MemChunk();
            }
        }
    }

    if (!int4Success && !int8Success) {
        const size_t expectedWeightSize = (size_t)oc * ic * (useIm2Col ? kernelSize : 1);
        const size_t reorderedWeightSize = im2colBlockedWeightSize;
        if (originWeight == nullptr || (size_t)originWeightSize < expectedWeightSize) {
            return nullptr;
        }
        res->weight = bufferAlloc->alloc(reorderedWeightSize * sizeof(int16_t));
        if (res->weight.first == nullptr) {
            return nullptr;
        }
        std::vector<int16_t> tempWeight(expectedWeightSize);
        HexagonBackend::fp32ToFp16(originWeight, tempWeight.data(), tempWeight.size());
        reorderFp16WeightForHmx((int16_t*)HexagonBackend::getPtr(res->weight), tempWeight.data(), ic, oc,
                                useIm2Col ? common->kernelX() : 1,
                                useIm2Col ? common->kernelY() : 1);
        static_cast<HexagonBackend*>(backend)->markHostInput(res->weight, (int)(reorderedWeightSize * sizeof(int16_t)));
    }

    return new HexagonConvolution(backend, res, op);
}

} // namespace MNN
