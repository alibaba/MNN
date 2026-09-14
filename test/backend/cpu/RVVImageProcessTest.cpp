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
            // The generic body keeps `1.0 - xF` as a double; this oracle copies
            // that arithmetic verbatim so the comparison is against the original
            // scalar semantics rather than against the new kernel.
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
static bool bilinearCase(size_t count, size_t sta, MNN::CV::Point start, MNN::CV::Point delta) {
    const size_t iw = 19, ih = 11, yStride = iw * 4 + 12;
    std::vector<unsigned char> source(yStride * ih, 0);
    for (size_t y = 0; y < ih; ++y) {
        for (size_t x = 0; x < iw; ++x) {
            for (size_t c = 0; c < 4; ++c) {
                source[y * yStride + 4 * x + c] = static_cast<unsigned char>((y * 61 + x * 29 + c * 47) & 255);
            }
        }
    }
    MNN::CV::Point points[2] = {start, delta};
    const size_t outputSize = 4 * (sta + count) + 8;
    std::vector<unsigned char> actual(outputSize, 0x5a), expected(actual);
    bilinearReference(source.data(), expected.data(), points, sta, count, iw, ih, yStride);
    MNN::MNNGetCoreFunctions()->MNNSamplerC4Bilinear(source.data(), actual.data(), points, sta, count, outputSize, iw,
                                                     ih, yStride);
    if (actual != expected) {
        MNN_ERROR("RVV image bilinear mismatch count=%zu sta=%zu start=(%g,%g) delta=(%g,%g)\n", count, sta, start.fX,
                  start.fY, delta.fX, delta.fY);
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
