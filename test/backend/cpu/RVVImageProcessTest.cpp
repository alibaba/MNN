// Copyright © 2026, Alibaba Group Holding Limited
#if defined(MNN_BUILD_STATIC_LIBS) && defined(__riscv)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "MNNTestSuite.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ImageProcessFunction.hpp"
#include "backend/cpu/riscv/rvv/MNNSamplerC4Bilinear_RVV.hpp"

// MNNSamplerC4Bilinear_RVV only exists in the MNNRVV object library, so the
// direct symbol reference lives under MNN_TEST_RVV_ENABLED. An MNN_USE_RVV=OFF
// build still compiles and links; it just reports the scalar registration.
#if MNN_TEST_RVV_ENABLED
void MNNSamplerC4Bilinear_RVV(const unsigned char*, unsigned char*, MNN::CV::Point*, size_t, size_t, size_t, size_t,
                              size_t, size_t);

static float clampCoordinate(float value, float maxValue) {
    return std::max(0.0f, std::min(value, maxValue));
}

static void bilinearReference(const unsigned char* source, unsigned char* dest, const MNN::CV::Point* points,
                              size_t sta, size_t count, size_t iw, size_t ih, size_t yStride) {
    float x = points[0].fX;
    float y = points[0].fY;
    for (size_t i = 0; i < count; ++i) {
        const float cx = clampCoordinate(x, (float)(iw - 1));
        const float cy = clampCoordinate(y, (float)(ih - 1));
        const int x0 = (int)cx;
        const int y0 = (int)cy;
        const int x1 = std::min((int)ceilf(cx), (int)iw - 1);
        const int y1 = std::min((int)ceilf(cy), (int)ih - 1);
        const float xF = cx - (float)x0;
        const float yF = cy - (float)y0;
        for (size_t c = 0; c < 4; ++c) {
            const unsigned char c00 = source[(size_t)y0 * yStride + 4 * (size_t)x0 + c];
            const unsigned char c01 = source[(size_t)y0 * yStride + 4 * (size_t)x1 + c];
            const unsigned char c10 = source[(size_t)y1 * yStride + 4 * (size_t)x0 + c];
            const unsigned char c11 = source[(size_t)y1 * yStride + 4 * (size_t)x1 + c];
            // The c10 term of the generic body is `yF * (1.0 - xF) * c10`, where
            // `1.0 - xF` is a double; the RVV kernel evaluates all four weights in
            // FP32. This oracle copies the original arithmetic verbatim, so it
            // describes the semantics MNN shipped rather than the new kernel -
            // which is exactly why the comparison below allows one grey level.
            float value =
                (1.0f - xF) * (1.0f - yF) * c00 + xF * (1.0f - yF) * c01 + yF * (1.0 - xF) * c10 + xF * yF * c11;
            value = std::min(std::max(value, 0.0f), 255.0f);
            dest[4 * (sta + i) + c] = static_cast<unsigned char>(roundf(value));
        }
        x += points[1].fX;
        y += points[1].fY;
    }
}

// Drive the function table, not the _RVV symbol. ImageProcessUtils::choose()
// hands out coreFunctions->MNNSamplerC4Bilinear and the pipeline calls through
// that pointer, so going through the same slot is what actually exercises the
// registration. Calling MNNSamplerC4Bilinear_RVV() directly would pass even if
// the slot still pointed at the scalar kernel.
//
// `sourceOverride` replaces the generated pattern, which the boundary case below
// uses to place exact neighbour values at the sampled position.
static bool bilinearCase(size_t count, size_t sta, MNN::CV::Point start, MNN::CV::Point delta,
                         const std::vector<unsigned char>* sourceOverride = nullptr, size_t sourceOffset = 0) {
    const size_t iw = 19, ih = 11, yStride = iw * 4 + 12;
    const size_t sourceSize = yStride * ih;
    const size_t wordCount = (sourceSize + sourceOffset + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    std::vector<uint32_t> sourceStorage(wordCount, 0);
    unsigned char* source = reinterpret_cast<unsigned char*>(sourceStorage.data()) + sourceOffset;
    if (sourceOverride != nullptr) {
        if (sourceOverride->size() != sourceSize) {
            MNN_ERROR("RVV image bilinear source override has size=%zu, expected=%zu\n", sourceOverride->size(),
                      sourceSize);
            return false;
        }
        std::copy(sourceOverride->begin(), sourceOverride->end(), source);
    } else {
        for (size_t y = 0; y < ih; ++y) {
            for (size_t x = 0; x < iw; ++x) {
                for (size_t c = 0; c < 4; ++c) {
                    source[y * yStride + 4 * x + c] = static_cast<unsigned char>((y * 61 + x * 29 + c * 47) & 255);
                }
            }
        }
    }
    MNN::CV::Point points[2] = {start, delta};
    const size_t outputSize = 4 * (sta + count) + 8;
    std::vector<unsigned char> actual(outputSize, 0x5a), expected(actual);
    bilinearReference(source, expected.data(), points, sta, count, iw, ih, yStride);
    MNN::MNNGetCoreFunctions()->MNNSamplerC4Bilinear(source, actual.data(), points, sta, count, outputSize, iw, ih,
                                                     yStride);
    // The kernel is FP32 throughout while the generic body keeps `1.0 - xF` for
    // the c10 term in double, so an input sitting exactly on a rounding boundary
    // can legitimately land one grey level lower here. Allow that single step and
    // no more: a kernel that drifted further, or reordered the weights, still has
    // to fail. The margin after the written pixels must stay byte-exact, because
    // a write past the requested range is a real bug and must not be masked by
    // the tolerance.
    const size_t pixelBytes = 4 * (sta + count);
    for (size_t i = 0; i < pixelBytes; ++i) {
        const int diff = static_cast<int>(actual[i]) - static_cast<int>(expected[i]);
        if (diff < -1 || diff > 1) {
            MNN_ERROR(
                "RVV image bilinear mismatch count=%zu sta=%zu start=(%g,%g) delta=(%g,%g) byte=%zu got=%d want=%d\n",
                count, sta, start.fX, start.fY, delta.fX, delta.fY, i, static_cast<int>(actual[i]),
                static_cast<int>(expected[i]));
            return false;
        }
    }
    for (size_t i = pixelBytes; i < outputSize; ++i) {
        if (actual[i] != expected[i]) {
            MNN_ERROR(
                "RVV image bilinear wrote past the requested range count=%zu sta=%zu byte=%zu got=0x%02x want=0x%02x\n",
                count, sta, i, static_cast<int>(actual[i]), static_cast<int>(expected[i]));
            return false;
        }
    }
    return true;
}

// Rounding boundary: with xF=0.3701782822608948, yF=0.26382631063461304 and
// neighbours (c00,c01,c10,c11) = (193,161,211,52), the generic body's double
// intermediate puts the c10 term on 173.5 exactly, so roundf() gives 174, while
// the FP32 weights evaluate 173.4999847 and round to 173. Pinned here so the
// one-level tolerance above is exercised rather than merely declared.
static bool bilinearRoundingBoundaryCase() {
    const size_t iw = 19, ih = 11, yStride = iw * 4 + 12;
    const unsigned char neighbours[4] = {193, 161, 211, 52}; // c00, c01, c10, c11
    std::vector<unsigned char> source(yStride * ih, 0);
    for (size_t c = 0; c < 4; ++c) {
        source[0 * yStride + 4 * 0 + c] = neighbours[0];
        source[0 * yStride + 4 * 1 + c] = neighbours[1];
        source[1 * yStride + 4 * 0 + c] = neighbours[2];
        source[1 * yStride + 4 * 1 + c] = neighbours[3];
    }
    return bilinearCase(1, 0, {0.3701782822608948f, 0.26382631063461304f}, {0.0f, 0.0f}, &source);
}

static bool bilinearUnalignedSourceCase() {
    alignas(uint32_t) unsigned char source[sizeof(uint32_t) + 1] = {};
    if (MNN::CV::RVV::canUseC4BilinearIndexedLoad(source + 1, 1, 1, sizeof(uint32_t))) {
        MNN_ERROR("RVV image bilinear accepted the unaligned source used by the fallback test\n");
        return false;
    }
    return bilinearCase(17, 2, {0.5f, 0.5f}, {0.5f, 0.25f}, nullptr, 1);
}

static bool bilinearAddressingBoundaryCase() {
    alignas(uint32_t) unsigned char source[sizeof(uint32_t)] = {};
    const size_t maxOffset = static_cast<size_t>(UINT32_MAX);

    if (!MNN::CV::RVV::canUseC4BilinearIndexedLoad(source, 2, 2, maxOffset - 7)) {
        MNN_ERROR("RVV image bilinear rejected a valid indexed offset\n");
        return false;
    }
    if (MNN::CV::RVV::canUseC4BilinearIndexedLoad(source, 2, 2, maxOffset - 3)) {
        MNN_ERROR("RVV image bilinear accepted an overflowing row-plus-column offset\n");
        return false;
    }
    if (MNN::CV::RVV::canUseC4BilinearIndexedLoad(source, maxOffset / 4 + 2, 1, 4)) {
        MNN_ERROR("RVV image bilinear accepted an overflowing column offset\n");
        return false;
    }
    if (MNN::CV::RVV::canUseC4BilinearIndexedLoad(source + 1, 2, 2, 8)) {
        MNN_ERROR("RVV image bilinear accepted an unaligned source pointer\n");
        return false;
    }
    if (MNN::CV::RVV::canUseC4BilinearIndexedLoad(source, 2, 2, 6)) {
        MNN_ERROR("RVV image bilinear accepted an unaligned row stride\n");
        return false;
    }
    if (MNN::CV::RVV::canUseC4BilinearIndexedLoad(source, 0, 2, 8) ||
        MNN::CV::RVV::canUseC4BilinearIndexedLoad(source, 2, 0, 8)) {
        MNN_ERROR("RVV image bilinear accepted an empty image\n");
        return false;
    }
    return true;
}
#endif // MNN_TEST_RVV_ENABLED

class RVVImageProcessTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        auto core = MNN::MNNGetCoreFunctions();
        if (!core) {
            MNN_ERROR("RVV image-process test requires an initialized CPU backend\n");
            return false;
        }
#if MNN_TEST_RVV_ENABLED
        // The sampler is reached through `coreFunctions->`, so an unregistered
        // slot is a real bug rather than a missing one: the pipeline would keep
        // calling the scalar kernel.
        const bool bilinearRegistered = (core->MNNSamplerC4Bilinear == MNNSamplerC4Bilinear_RVV);
        if (core->supportRVV != bilinearRegistered) {
            MNN_ERROR("Unexpected RVV image-process sampler registration (supportRVV=%d)\n",
                      static_cast<int>(core->supportRVV));
            return false;
        }
        if (!core->supportRVV) {
            MNN_PRINT("RVV image-process: skipped, runtime reports supportRVV=0\n");
            return true;
        }
        const size_t lengths[] = {0, 1, 2, 3, 4, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129};
        size_t cases = 0;
        for (size_t count : lengths) {
            if (!bilinearCase(count, 3, {-2.25f, 13.0f}, {0.8125f, -0.4375f}) ||
                !bilinearCase(count, 1, {18.0f, 10.0f}, {-0.53125f, -0.28125f}) ||
                !bilinearCase(count, 2, {0.5f, 0.5f}, {0.5f, 0.25f})) {
                return false;
            }
            cases += 3;
        }
        // The reviewer's rounding-boundary inputs: the generic double intermediate
        // lands on 173.5, the FP32 kernel on 173.4999847.
        if (!bilinearRoundingBoundaryCase() || !bilinearUnalignedSourceCase() || !bilinearAddressingBoundaryCase()) {
            return false;
        }
        cases += 3;
        MNN_PRINT("RVV image-process: %zu cases passed (supportRVV=%d)\n", cases, static_cast<int>(core->supportRVV));
#else
        if (core->MNNSamplerC4Bilinear == nullptr) {
            MNN_ERROR("Unexpected scalar image-process function registration\n");
            return false;
        }
#endif
        return true;
    }
};

MNNTestSuiteRegister(RVVImageProcessTest, "backend/cpu/rvv/image_process");

#else

#include "MNNTestSuite.h"

class RVVImageProcessTest : public MNNTestCase {
public:
    virtual bool run(int precision) { return true; }
};

MNNTestSuiteRegister(RVVImageProcessTest, "backend/cpu/rvv/image_process");

#endif
