#if defined(MNN_BUILD_STATIC_LIBS) && defined(__riscv) && defined(MNN_USE_RVV)

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "MNNTestSuite.h"
#include "backend/cpu/compute/CommonOptFunction.h"

void MNNPackInt8C2_RVV(float*, const float*, size_t, size_t, int*);
void MNNPackCUnit_RVV(float*, const float*, size_t, size_t, int*);
void MNNUnpackCUnit_RVV(float*, const float*, size_t, size_t, int*);
void MNNTranspose32Bit_RVV(int32_t*, const int32_t*, int32_t*);
void MNNTranspose16Bit_RVV(int16_t*, const int16_t*, int32_t*);

template <typename T>
static bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

static bool packC2Case(int area, int depth, int padding, bool exactSource = false) {
    int offset[2] = {area + padding, area + 2 * padding};
    // The last channel need not reserve another full stride. Exercise an odd
    // channel tail with no addressable data beyond its last valid element.
    const size_t srcSize = exactSource && area > 0 && depth > 0 ? 4 + static_cast<size_t>(depth - 1) * offset[0] + area
                                                                : static_cast<size_t>(depth + 1) * offset[0] + 16;
    const size_t dstSize = static_cast<size_t>((depth + 1) / 2 + 1) * offset[1] * 2 + 16;
    std::vector<float> src(srcSize), expected(dstSize, 123.0f), direct = expected, dispatched = expected;
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<float>(static_cast<int>(i * 17 % 101) - 50);
    }
    for (int c = 0; c < (depth + 1) / 2 * 2; ++c) {
        for (int x = 0; x < area; ++x) {
            expected[4 + (c / 2 * offset[1] + x) * 2 + c % 2] = c < depth ? src[4 + c * offset[0] + x] : 0.0f;
        }
    }
    MNNPackInt8C2_RVV(direct.data() + 4, src.data() + 4, area, depth, offset);
    MNNPackInt8C2(dispatched.data() + 4, src.data() + 4, area, depth, offset);
    if (!same(direct, expected) || !same(dispatched, expected)) {
        MNN_ERROR("RVV C2 pack mismatch: area=%d depth=%d padding=%d\n", area, depth, padding);
        return false;
    }
    return true;
}

static bool packC2SignedStrideCase(bool reverseSrc, bool reverseDst) {
    const int area = 8;
    const int depth = 5;
    const int srcStep = area + 3;
    const int dstStep = area + 5;
    const int groups = (depth + 1) / 2;
    int offset[2] = {reverseSrc ? -srcStep : srcStep, reverseDst ? -dstStep : dstStep};
    const size_t srcBase = 4 + (reverseSrc ? (depth - 1) * srcStep : 0);
    const size_t dstBase = 4 + (reverseDst ? (groups - 1) * dstStep * 2 : 0);
    const size_t srcSize = 4 + (depth - 1) * srcStep + area;
    const size_t dstSize = 4 + (groups - 1) * dstStep * 2 + area * 2;
    std::vector<float> src(srcSize), expected(dstSize, 123.0f), direct = expected, dispatched = expected;
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<float>(static_cast<int>(i * 17 % 101) - 50);
    }
    for (int c = 0; c < groups * 2; ++c) {
        for (int x = 0; x < area; ++x) {
            const ptrdiff_t dstIndex = static_cast<ptrdiff_t>(dstBase) + (c / 2) * offset[1] * 2 + x * 2 + c % 2;
            expected[static_cast<size_t>(dstIndex)] =
                c < depth ? src[static_cast<size_t>(static_cast<ptrdiff_t>(srcBase) + c * offset[0] + x)] : 0.0f;
        }
    }
    MNNPackInt8C2_RVV(direct.data() + dstBase, src.data() + srcBase, area, depth, offset);
    MNNPackInt8C2(dispatched.data() + dstBase, src.data() + srcBase, area, depth, offset);
    if (!same(direct, expected) || !same(dispatched, expected)) {
        MNN_ERROR("RVV C2 signed stride mismatch: reverseSrc=%d reverseDst=%d\n", reverseSrc, reverseDst);
        return false;
    }
    return true;
}

static bool packC4Case(int area, int depth, int padding) {
    int offset[2] = {area + padding, area + 2 * padding};
    const int rounded = (depth + 3) / 4 * 4;
    const size_t srcSize = static_cast<size_t>(depth + 1) * offset[0] + 16;
    const size_t dstSize = static_cast<size_t>(rounded / 4 + 1) * offset[1] * 4 + 16;
    std::vector<float> src(srcSize), expected(dstSize, 123.0f), direct = expected, dispatched = expected;
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<float>(static_cast<int>(i * 17 % 101) - 50);
    }
    for (int c = 0; c < rounded; ++c) {
        for (int x = 0; x < area; ++x) {
            expected[4 + (c / 4 * offset[1] + x) * 4 + c % 4] = c < depth ? src[4 + c * offset[0] + x] : 0.0f;
        }
    }
    MNNPackCUnit_RVV(direct.data() + 4, src.data() + 4, area, depth, offset);
    MNNPackC4(dispatched.data() + 4, src.data() + 4, area, depth, offset);
    if (!same(direct, expected) || !same(dispatched, expected)) {
        MNN_ERROR("RVV C4 pack mismatch: area=%d depth=%d padding=%d\n", area, depth, padding);
        return false;
    }
    return true;
}

static bool unpackC4Case(int area, int depth, int padding) {
    int offset[2] = {area + padding, area + 2 * padding};
    const int rounded = (depth + 3) / 4 * 4;
    const size_t srcSize = static_cast<size_t>(rounded / 4 + 1) * offset[0] * 4 + 16;
    const size_t dstSize = static_cast<size_t>(depth + 1) * offset[1] + 16;
    std::vector<float> src(srcSize), expected(dstSize, 123.0f), direct = expected, dispatched = expected;
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<float>(static_cast<int>(i * 17 % 101) - 50);
    }
    for (int c = 0; c < depth; ++c) {
        for (int x = 0; x < area; ++x) {
            expected[4 + c * offset[1] + x] = src[4 + (c / 4 * offset[0] + x) * 4 + c % 4];
        }
    }
    MNNUnpackCUnit_RVV(direct.data() + 4, src.data() + 4, area, depth, offset);
    MNNUnpackC4(dispatched.data() + 4, src.data() + 4, area, depth, offset);
    if (!same(direct, expected) || !same(dispatched, expected)) {
        MNN_ERROR("RVV C4 unpack mismatch: area=%d depth=%d padding=%d\n", area, depth, padding);
        return false;
    }
    return true;
}

template <typename T>
static bool transposeCase(int width, int height, int padding, void (*direct)(T*, const T*, int32_t*),
                          void (*dispatched)(T*, const T*, int32_t*)) {
    int32_t dim[4] = {width, height, height + padding, width + padding};
    const size_t srcSize = static_cast<size_t>(width + 1) * dim[2] + 16;
    const size_t dstSize = static_cast<size_t>(height + 1) * dim[3] + 16;
    std::vector<T> src(srcSize), expected(dstSize, T(123)), actual = expected, viaEntry = expected;
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<T>(static_cast<int>(i * 37 % 1009) - 504);
    }
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            expected[4 + i * dim[3] + j] = src[4 + i + j * dim[2]];
        }
    }
    direct(actual.data() + 4, src.data() + 4, dim);
    dispatched(viaEntry.data() + 4, src.data() + 4, dim);
    if (!same(actual, expected) || !same(viaEntry, expected)) {
        MNN_ERROR("RVV transpose mismatch: bytes=%zu width=%d height=%d padding=%d\n", sizeof(T), width, height,
                  padding);
        return false;
    }
    return true;
}

#if defined(__linux__)
// Keep the first or last addressable element against an inaccessible page.
// Allocate only the span reached by the operation, including any in-span gaps.
template <typename T>
class GuardedArray {
public:
    GuardedArray(size_t count, bool atEnd) {
        const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0) {
            return;
        }
        const size_t pageBytes = static_cast<size_t>(page);
        const size_t valueBytes = (count == 0 ? 1 : count) * sizeof(T);
        mDataBytes = ((valueBytes + pageBytes - 1) / pageBytes) * pageBytes;
        mMapBytes = mDataBytes + 2 * pageBytes;
        mMap = mmap(nullptr, mMapBytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mMap == MAP_FAILED) {
            mMap = nullptr;
            return;
        }
        mDataPages = static_cast<char*>(mMap) + pageBytes;
        if (mprotect(mDataPages, mDataBytes, PROT_READ | PROT_WRITE) != 0) {
            munmap(mMap, mMapBytes);
            mMap = nullptr;
            return;
        }
        mValues = reinterpret_cast<T*>(static_cast<char*>(mDataPages) + (atEnd ? mDataBytes - valueBytes : 0));
    }

    ~GuardedArray() {
        if (mMap != nullptr) {
            munmap(mMap, mMapBytes);
        }
    }

    GuardedArray(const GuardedArray&) = delete;
    GuardedArray& operator=(const GuardedArray&) = delete;

    T* data() const { return mValues; }
    bool makeReadOnly() const { return mprotect(mDataPages, mDataBytes, PROT_READ) == 0; }

private:
    void* mMap = nullptr;
    void* mDataPages = nullptr;
    size_t mMapBytes = 0;
    size_t mDataBytes = 0;
    T* mValues = nullptr;
};

template <typename T>
static bool matchesGuarded(const T* actual, const std::vector<T>& expected, const char* name, int a, int b, int padding,
                           bool atEnd) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            MNN_ERROR("RVV %s guard mismatch: a=%d b=%d padding=%d atEnd=%d index=%zu\n", name, a, b, padding,
                      static_cast<int>(atEnd), i);
            return false;
        }
    }
    return true;
}

static bool guardedPackCase(int area, int depth, int padding, bool atEnd, bool c2, bool reverseSrc = false,
                            bool reverseDst = false) {
    if (area == 0 && depth != 0) {
        return true;
    }
    const int pack = c2 ? 2 : 4;
    const int srcStep = area + padding;
    const int dstStep = area + 2 * padding;
    const int groups = (depth + pack - 1) / pack;
    int offset[2] = {reverseSrc ? -srcStep : srcStep, reverseDst ? -dstStep : dstStep};
    const size_t srcBase = reverseSrc && depth ? static_cast<size_t>(depth - 1) * srcStep : 0;
    const size_t dstBase = reverseDst && groups ? static_cast<size_t>(groups - 1) * dstStep * pack : 0;
    const size_t srcSize = depth && area ? static_cast<size_t>(depth - 1) * srcStep + area : 1;
    const size_t dstSize = groups && area ? static_cast<size_t>(groups - 1) * dstStep * pack + area * pack : 1;
    const float untouched = 1234.5f;
    GuardedArray<float> src(srcSize, atEnd), dst(dstSize, atEnd);
    if (src.data() == nullptr || dst.data() == nullptr) {
        MNN_ERROR("RVV pack guard allocation failed\n");
        return false;
    }
    for (size_t i = 0; i < srcSize; ++i) {
        src.data()[i] = static_cast<float>(static_cast<int>(i * 17 % 101) - 50);
    }
    std::vector<float> expected(dstSize, untouched);
    for (int c = 0; c < groups * pack; ++c) {
        for (int x = 0; x < area; ++x) {
            const ptrdiff_t dstIndex =
                static_cast<ptrdiff_t>(dstBase) + (c / pack) * offset[1] * pack + x * pack + c % pack;
            const ptrdiff_t srcIndex = static_cast<ptrdiff_t>(srcBase) + c * offset[0] + x;
            expected[static_cast<size_t>(dstIndex)] = c < depth ? src.data()[static_cast<size_t>(srcIndex)] : 0.0f;
        }
    }
    if (!src.makeReadOnly()) {
        MNN_ERROR("RVV pack source protection failed\n");
        return false;
    }
    for (int entry = 0; entry < 2; ++entry) {
        for (size_t i = 0; i < dstSize; ++i) {
            dst.data()[i] = untouched;
        }
        if (c2) {
            (entry == 0 ? MNNPackInt8C2_RVV : MNNPackInt8C2)(dst.data() + dstBase, src.data() + srcBase, area, depth,
                                                             offset);
        } else {
            (entry == 0 ? MNNPackCUnit_RVV : MNNPackC4)(dst.data() + dstBase, src.data() + srcBase, area, depth,
                                                        offset);
        }
        if (!matchesGuarded(dst.data(), expected, c2 ? "C2 pack" : "C4 pack", area, depth, padding, atEnd)) {
            return false;
        }
    }
    return true;
}

static bool guardedUnpackCase(int area, int depth, int padding, bool atEnd) {
    if (area == 0 && depth != 0) {
        return true;
    }
    int offset[2] = {area + padding, area + 2 * padding};
    const size_t srcSize =
        depth && area ? static_cast<size_t>((depth - 1) / 4) * offset[0] * 4 + (area - 1) * 4 + (depth - 1) % 4 + 1 : 1;
    const size_t dstSize = depth && area ? static_cast<size_t>(depth - 1) * offset[1] + area : 1;
    const float untouched = 1234.5f;
    GuardedArray<float> src(srcSize, atEnd), dst(dstSize, atEnd);
    if (src.data() == nullptr || dst.data() == nullptr) {
        MNN_ERROR("RVV unpack guard allocation failed\n");
        return false;
    }
    for (size_t i = 0; i < srcSize; ++i) {
        src.data()[i] = static_cast<float>(static_cast<int>(i * 17 % 101) - 50);
    }
    std::vector<float> expected(dstSize, untouched);
    for (int c = 0; c < depth; ++c) {
        for (int x = 0; x < area; ++x) {
            expected[c * offset[1] + x] = src.data()[(c / 4 * offset[0] + x) * 4 + c % 4];
        }
    }
    if (!src.makeReadOnly()) {
        MNN_ERROR("RVV unpack source protection failed\n");
        return false;
    }
    for (int entry = 0; entry < 2; ++entry) {
        for (size_t i = 0; i < dstSize; ++i) {
            dst.data()[i] = untouched;
        }
        (entry == 0 ? MNNUnpackCUnit_RVV : MNNUnpackC4)(dst.data(), src.data(), area, depth, offset);
        if (!matchesGuarded(dst.data(), expected, "C4 unpack", area, depth, padding, atEnd)) {
            return false;
        }
    }
    return true;
}

template <typename T>
static bool guardedTransposeCase(int width, int height, int padding, bool atEnd, void (*direct)(T*, const T*, int32_t*),
                                 void (*dispatched)(T*, const T*, int32_t*)) {
    int32_t dim[4] = {width, height, height + padding, width + padding};
    const size_t srcSize = width && height ? static_cast<size_t>(width - 1) * dim[2] + height : 1;
    const size_t dstSize = width && height ? static_cast<size_t>(height - 1) * dim[3] + width : 1;
    const T untouched = static_cast<T>(1234);
    GuardedArray<T> src(srcSize, atEnd), dst(dstSize, atEnd);
    if (src.data() == nullptr || dst.data() == nullptr) {
        MNN_ERROR("RVV transpose guard allocation failed\n");
        return false;
    }
    for (size_t i = 0; i < srcSize; ++i) {
        src.data()[i] = static_cast<T>(static_cast<int>(i * 37 % 1009) - 504);
    }
    std::vector<T> expected(dstSize, untouched);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            expected[i * dim[3] + j] = src.data()[i + j * dim[2]];
        }
    }
    if (!src.makeReadOnly()) {
        MNN_ERROR("RVV transpose source protection failed\n");
        return false;
    }
    for (int entry = 0; entry < 2; ++entry) {
        for (size_t i = 0; i < dstSize; ++i) {
            dst.data()[i] = untouched;
        }
        (entry == 0 ? direct : dispatched)(dst.data(), src.data(), dim);
        if (!matchesGuarded(dst.data(), expected, sizeof(T) == 2 ? "transpose16" : "transpose32", width, height,
                            padding, atEnd)) {
            return false;
        }
    }
    return true;
}

static bool checkGuardedBoundaries() {
    size_t cases = 0;
    for (bool atEnd : {false, true}) {
        for (int padding : {0, 3}) {
            for (int area : {0, 1, 7, 8, 9, 31, 32, 33}) {
                for (int depth : {0, 1, 3, 4, 5}) {
                    if (area == 0 && depth != 0) {
                        continue;
                    }
                    if (!guardedPackCase(area, depth, padding, atEnd, false) ||
                        !guardedUnpackCase(area, depth, padding, atEnd) ||
                        !guardedPackCase(area, depth, padding, atEnd, true)) {
                        return false;
                    }
                    cases += 3;
                }
            }
            for (int width : {0, 1, 3, 4, 7, 16, 17, 31, 32, 33, 63, 64, 65}) {
                for (int height : {0, 1, 3}) {
                    if (width == 0 && height > 1) {
                        continue;
                    }
                    if (!guardedTransposeCase<int32_t>(width, height, padding, atEnd, MNNTranspose32Bit_RVV,
                                                       MNNTranspose32Bit) ||
                        !guardedTransposeCase<int16_t>(width, height, padding, atEnd, MNNTranspose16Bit_RVV,
                                                       MNNTranspose16Bit)) {
                        return false;
                    }
                    cases += 2;
                }
            }
        }
        for (int mode : {1, 2, 3}) {
            for (int area : {1, 7, 8, 9, 31, 32, 33}) {
                if (!guardedPackCase(area, 5, 3, atEnd, true, (mode & 1) != 0, (mode & 2) != 0)) {
                    return false;
                }
                ++cases;
            }
        }
    }
    MNN_PRINT("RVV memory access guard pages: %zu boundary cases passed\n", cases);
    return true;
}
#endif

class RVVMemoryAccessTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        auto core = MNN::MNNGetCoreFunctions();
        if (core == nullptr || !core->supportRVV) {
            MNN_ERROR("RVV memory-access test requires the RVV CPU path\n");
            return false;
        }
        size_t cases = 0;
        for (int padding : {0, 3}) {
            for (int area : {0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 65}) {
                for (int depth : {0, 1, 2, 3, 4, 5, 16, 17}) {
                    if (!packC2Case(area, depth, padding) || !packC4Case(area, depth, padding) ||
                        !unpackC4Case(area, depth, padding)) {
                        return false;
                    }
                    cases += 3;
                }
            }
            if (padding == 3) {
                if (!packC2Case(8, 5, padding, true)) {
                    return false;
                }
                ++cases;
            }
            for (int width : {0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65}) {
                for (int height : {0, 1, 3, 8, 65}) {
                    if (!transposeCase<int32_t>(width, height, padding, MNNTranspose32Bit_RVV, MNNTranspose32Bit) ||
                        !transposeCase<int16_t>(width, height, padding, MNNTranspose16Bit_RVV, MNNTranspose16Bit)) {
                        return false;
                    }
                    cases += 2;
                }
            }
        }
        for (int mode : {1, 2, 3}) {
            if (!packC2SignedStrideCase((mode & 1) != 0, (mode & 2) != 0)) {
                return false;
            }
            ++cases;
        }
#if defined(__linux__)
        if (!checkGuardedBoundaries()) {
            return false;
        }
#endif
        MNN_PRINT("RVV memory access: %zu C2/C4/transpose comparisons passed\n", cases);
        return true;
    }
};

MNNTestSuiteRegister(RVVMemoryAccessTest, "backend/cpu/rvv/memory_access");

#endif
