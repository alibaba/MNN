// Direct-kernel regression test for the RVV reduction (CountMaxMinValue) and
// Int8 ReLU kernels. See RVVReductionReluTest.md.
//
// This file does not register its own test case: the assertions live in
// MNNTestRVVReductionReluFunctions(), which test/op/ReLUTest.cpp calls from the
// already-registered "op/relu" case. That keeps the checks on the default
// run_test.out path instead of hiding them behind a standalone main().
//
// The RVV kernels only exist in the MNNRVV object library. Both the direct
// kernel calls and the function-table comparison therefore live under
// MNN_TEST_RVV_ENABLED, so an MNN_USE_RVV=OFF build keeps compiling and linking
// (it simply skips the RVV-specific checks instead of emitting undefined
// references to MNNCountMaxMinValue_RVV / MNNReluInt8_RVV).
#include <cstdio>
#include <cstring>
#include <vector>
#include "MNNTestSuite.h"
#include "backend/cpu/compute/CommonOptFunction.h"

#if defined(MNN_BUILD_STATIC_LIBS) && defined(__riscv) && MNN_TEST_RVV_ENABLED
void MNNCountMaxMinValue_RVV(const float*, float*, float*, size_t);
// The vector kernel has its own symbol; the generic MNNReluInt8 declared by the
// shared header stays reserved for targets without vector support.
void MNNReluInt8_RVV(int8_t*, const int8_t*, size_t, ssize_t);
#endif

#if defined(MNN_BUILD_STATIC_LIBS) && defined(__riscv)
static float fromBits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static bool sameBits(float a, float b) {
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}

#if MNN_TEST_RVV_ENABLED
// The non-NEON comparison order from CommonOptFunction.cpp.
static void countReference(const float* src, float* minValue, float* maxValue, size_t size) {
    if (size == 0) {
        *minValue = *maxValue = 0.0f;
        return;
    }
    float minResult = src[0], maxResult = src[0];
    for (size_t i = 1; i < size; ++i) {
        if (maxResult < src[i]) {
            maxResult = src[i];
        }
        if (minResult > src[i]) {
            minResult = src[i];
        }
    }
    *minValue = minResult;
    *maxValue = maxResult;
}
#endif
#endif

bool MNNTestRVVReductionReluFunctions() {
#if defined(MNN_BUILD_STATIC_LIBS) && defined(__riscv)
    auto core = MNN::MNNGetCoreFunctions();
    if (!core) {
        MNN_ERROR("RVV reduction/ReLU test requires an initialized CPU backend\n");
        return false;
    }
#if MNN_TEST_RVV_ENABLED
    const bool countDispatchMatches = core->supportRVV ? core->MNNCountMaxMinValue == MNNCountMaxMinValue_RVV
                                                       : core->MNNCountMaxMinValue != MNNCountMaxMinValue_RVV;
    const bool reluDispatchMatches = core->supportRVV ? core->MNNReluInt8 == MNNReluInt8_RVV
                                                      : core->MNNReluInt8 == MNNReluInt8;
    if (!countDispatchMatches || !reluDispatchMatches) {
        MNN_ERROR("RVV reduction/ReLU dispatch mismatch: supportRVV=%d\n", static_cast<int>(core->supportRVV));
        return false;
    }
    if (!core->supportRVV) {
        MNN_PRINT("RVV reduction/ReLU: dispatch passed; numerical checks skipped, runtime reports supportRVV=0\n");
        return true;
    }
    size_t reductions = 0, relus = 0;
    const uint32_t specials[] = {0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321,
                                 0x00000000, 0x80000000, 0x00000001, 0x80000001};
    for (size_t lengthIndex = 0; lengthIndex < 259; ++lengthIndex) {
        const size_t size = lengthIndex <= 257 ? lengthIndex : 4099;
        for (int pattern = 0; pattern < 14; ++pattern) {
            std::vector<float> source(size + 2, 999.0f);
            uint32_t random = 17;
            for (size_t i = 0; i < size; ++i) {
                random = random * 1664525u + 1013904223u;
                float value = fromBits(random);
                if (pattern < 8) {
                    value = fromBits(specials[pattern]);
                } else if (pattern == 8 || pattern == 9) {
                    value = fromBits((i + pattern) % 2 ? 0x80000000u : 0u);
                } else if (pattern == 10) {
                    value = i == 0 ? fromBits(0x7fc12345) : static_cast<float>(i);
                } else if (pattern == 11) {
                    value = i == size / 2 ? fromBits(0xffc54321) : static_cast<float>(i) - 127.0f;
                } else if (pattern == 12) {
                    value = i < size / 2 ? 7.0f : fromBits(i % 2 ? 0x80000000u : 0u);
                }
                source[i + 1] = value;
            }
            float expectedMin, expectedMax;
            float minValue = 123.0f, maxValue = 456.0f;
            countReference(source.data() + 1, &expectedMin, &expectedMax, size);
            MNNCountMaxMinValue_RVV(source.data() + 1, &minValue, &maxValue, size);
            if (!sameBits(minValue, expectedMin) || !sameBits(maxValue, expectedMax)) {
                MNN_ERROR("RVV reduction mismatch size=%zu pattern=%d\n", size, pattern);
                return false;
            }
            ++reductions;
        }
    }
    const size_t lengths[] = {0, 1, 2, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 513};
    for (size_t size : lengths) {
        for (ssize_t zeroPoint = -256; zeroPoint <= 255; ++zeroPoint) {
            for (int seed = 0; seed < 8; ++seed) {
                std::vector<int8_t> source(size + 2, 99), expected(size + 2, 99), output(size + 2, 99);
                for (size_t i = 0; i < size; ++i) {
                    source[i + 1] = static_cast<int8_t>((i * 37 + seed * 31) % 256 - 128);
                    expected[i + 1] = source[i + 1] < zeroPoint ? static_cast<int8_t>(zeroPoint) : source[i + 1];
                }
                MNNReluInt8_RVV(output.data() + 1, source.data() + 1, size, zeroPoint);
                if (output != expected) {
                    MNN_ERROR("RVV ReLU mismatch size=%zu zeroPoint=%zd seed=%d\n", size, zeroPoint, seed);
                    return false;
                }
                MNNReluInt8_RVV(source.data() + 1, source.data() + 1, size, zeroPoint);
                if (source != expected) {
                    MNN_ERROR("RVV in-place ReLU mismatch size=%zu zeroPoint=%zd seed=%d\n", size, zeroPoint, seed);
                    return false;
                }
                relus += 2;
            }
        }
    }
    MNN_PRINT("RVV reduction/ReLU: reductions=%zu ReLU=%zu (including in-place and guards), RVV=%d\n", reductions,
              relus, static_cast<int>(core->supportRVV));
#else
    if (core->MNNCountMaxMinValue == nullptr || core->MNNReluInt8 != MNNReluInt8) {
        MNN_ERROR("Unexpected scalar reduction/ReLU function registration\n");
        return false;
    }
    MNN_PRINT("RVV reduction/ReLU: scalar dispatch passed, MNN_USE_RVV=OFF (RVV=%d)\n",
              static_cast<int>(core->supportRVV));
#endif
#endif
    return true;
}