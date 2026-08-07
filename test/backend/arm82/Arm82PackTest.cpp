#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <MNN/MNNDefine.h>
#include "MNNTestSuite.h"
#include "backend/arm82/compute/Arm82Pack.hpp"

namespace {

void referencePack(uint16_t* destBase, const uint16_t* sourceBase, int e, int l, int eDest, size_t srcRowStride) {
    constexpr int pack = 8;
    constexpr int lP = 2;
    const size_t srcColBlockStride = static_cast<size_t>(e) * pack;
    const size_t dstColBlockStride = static_cast<size_t>(eDest) * lP;
    for (int y = 0; y < e; ++y) {
        const int yR = y % eDest;
        for (int x = 0; x < l; ++x) {
            const int xR = x % pack;
            const int xC = x / pack;
            destBase[(x / lP) * dstColBlockStride + yR * lP + (x % lP)] =
                sourceBase[xC * srcColBlockStride + static_cast<size_t>(y) * srcRowStride + xR];
        }
    }
}

class Arm82PackSmallChannelTest : public MNNTestCase {
public:
    bool run(int precision) override {
        const int eValues[] = {1, 2, 7, 8, 15, 16, 17, 31};
        const int eDestValues[] = {1, 4, 8, 16};
        const int offsets[] = {1, 2};
        constexpr uint16_t sentinel = 0x7b7b;

        for (int l = 1; l < 8; ++l) {
            for (int e : eValues) {
                for (int eDest : eDestValues) {
                    const int eOffsets[] = {0, eDest / 2, eDest - 1};
                    for (int eOffset : eOffsets) {
                        for (int offset : offsets) {
                            const size_t srcRowStride = 8 * offset;
                            std::vector<uint16_t> source(static_cast<size_t>(e) * srcRowStride);
                            for (size_t i = 0; i < source.size(); ++i) {
                                source[i] = static_cast<uint16_t>(i * 13 + l * 17 + e * 19 + 1);
                            }

                            const size_t destBaseOffset = static_cast<size_t>(eOffset) * 2;
                            const size_t destSize = destBaseOffset + static_cast<size_t>(4) * eDest * 2 + 16;
                            std::vector<uint16_t> expected(destSize, sentinel);
                            std::vector<uint16_t> actual(destSize, sentinel);

                            referencePack(expected.data() + destBaseOffset, source.data(), e, l, eDest, srcRowStride);
                            MNN::Arm82::packSmallChannelForMatMulA(actual.data() + destBaseOffset, source.data(), e, l,
                                                                   eDest, srcRowStride);

                            if (actual != expected) {
                                const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
                                MNN_ERROR(
                                    "Arm82 small-channel pack mismatch: l=%d e=%d eDest=%d eOffset=%d "
                                    "offset=%d index=%d\n",
                                    l, e, eDest, eOffset, offset, static_cast<int>(mismatch.first - actual.begin()));
                                return false;
                            }
                        }
                    }
                }
            }
        }
        return true;
    }
};

} // namespace

MNNTestSuiteRegister(Arm82PackSmallChannelTest, "backend/arm82/sme2_pack_small_channel");
