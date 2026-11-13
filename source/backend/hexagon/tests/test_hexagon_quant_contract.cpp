#include "HexagonQuantization.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

#define CHECK(condition)                                                                            \
    do {                                                                                            \
        if (!(condition)) {                                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);                        \
            return false;                                                                           \
        }                                                                                           \
    } while (0)

bool testRequantizeTiesToEven() {
    const auto half = MNN::makeRequantMultiplier(0.5);
    CHECK(half.multiplier > 0);
    CHECK(MNN::requantizeReference(1, half) == 0);
    CHECK(MNN::requantizeReference(3, half) == 2);
    CHECK(MNN::requantizeReference(-1, half) == 0);
    CHECK(MNN::requantizeReference(-3, half) == -2);

    const auto withZero = MNN::makeRequantMultiplier(0.5, 5, -10, 10);
    CHECK(MNN::requantizeReference(2, withZero) == 6);
    CHECK(MNN::requantizeReference(1000, withZero) == 10);
    CHECK(MNN::requantizeReference(-1000, withZero) == -10);
    return true;
}

bool testRequantizeRejectsInvalidScale() {
    CHECK(MNN::makeRequantMultiplier(0.0).multiplier == 0);
    CHECK(MNN::makeRequantMultiplier(-1.0).multiplier == 0);
    CHECK(MNN::makeRequantMultiplier(
              std::numeric_limits<double>::infinity()).multiplier == 0);
    CHECK(MNN::makeRequantMultiplier(
              std::numeric_limits<double>::quiet_NaN()).multiplier == 0);
    return true;
}

bool testWeightPacking() {
    const int8_t weight[] = {1, 2, 3, 4};
    std::vector<int8_t> packed;
    CHECK(MNN::w8a8Conv1x1PackedWeightBytes(2, 2) == 1024);
    CHECK(MNN::packW8A8Conv1x1Weight(packed, weight, 4, 2, 2));
    CHECK(packed.size() == 1024);
    CHECK(packed[0] == 1);
    CHECK(packed[2] == 2);
    CHECK(packed[4] == 3);
    CHECK(packed[6] == 4);
    CHECK(packed[1] == 0);
    CHECK(packed[1023] == 0);

    CHECK(!MNN::packW8A8Conv1x1Weight(packed, nullptr, 4, 2, 2));
    CHECK(!MNN::packW8A8Conv1x1Weight(packed, weight, 3, 2, 2));
    CHECK(MNN::w8a8Conv1x1PackedWeightBytes(0, 2) == 0);
    return true;
}

bool testPostParameters() {
    const float bias[] = {0.25f, -0.5f};
    const float weightScale[] = {0.125f, 0.25f};
    std::vector<int32_t> biasInt32;
    std::vector<DSPRequantParam> requant;
    CHECK(MNN::buildW8A8Conv1x1PostParams(
        biasInt32, requant, bias, 2, weightScale, 2, 2,
        0.25f, 0.5f, false, false));
    CHECK(biasInt32.size() == 32);
    CHECK(requant.size() == 32);
    CHECK(biasInt32[0] == 8);
    CHECK(biasInt32[1] == -8);
    CHECK(requant[0].outputZero == 0);
    CHECK(requant[0].clampMin == -127);
    CHECK(requant[0].clampMax == 127);
    CHECK(MNN::requantizeReference(16, requant[0]) == 1);
    CHECK(MNN::requantizeReference(16, requant[1]) == 2);

    CHECK(MNN::buildW8A8Conv1x1PostParams(
        biasInt32, requant, nullptr, 0, weightScale, 2, 2,
        0.25f, 0.5f, false, true));
    CHECK(requant[0].clampMin == 0);
    CHECK(requant[0].clampMax == 12);

    const float invalidScale[] = {0.125f, 0.0f};
    CHECK(!MNN::buildW8A8Conv1x1PostParams(
        biasInt32, requant, bias, 2, invalidScale, 2, 2,
        0.25f, 0.5f, false, false));
    const float overflowBias[] = {std::numeric_limits<float>::max(), 0.0f};
    CHECK(!MNN::buildW8A8Conv1x1PostParams(
        biasInt32, requant, overflowBias, 2, weightScale, 2, 2,
        0.25f, 0.5f, false, false));
    return true;
}

bool testPackedActivationConversion() {
    const float source[] = {
        -0.5f, 0.5f,
        1.0f, -1.0f,
        0.25f, 4.0f,
    };
    std::vector<int8_t> packed(2 * 32, 99);
    CHECK(MNN::packFloatToInt8Nc(
        source, packed.data(), 1, 3, 1, 2, 32, 1,
        0.25f, 3, -128, 127));
    CHECK(packed[0] == 1);
    CHECK(packed[32] == 5);
    CHECK(packed[1] == 7);
    CHECK(packed[33] == -1);
    CHECK(packed[2] == 4);
    CHECK(packed[34] == 19);
    CHECK(packed[3] == 0);

    std::vector<float> unpacked(6, 0.0f);
    CHECK(MNN::unpackInt8NcToFloat(
        packed.data(), unpacked.data(), 1, 3, 1, 2, 32, 1, 0.25f, 3));
    for (size_t i = 0; i < unpacked.size(); ++i) {
        CHECK(unpacked[i] == source[i]);
    }

    const float nonFinite[] = {
        1.0f, 2.0f, std::numeric_limits<float>::quiet_NaN(), 4.0f,
    };
    std::fill(packed.begin(), packed.end(), static_cast<int8_t>(0x5a));
    const auto original = packed;
    CHECK(!MNN::packFloatToInt8Nc(
        nonFinite, packed.data(), 1, 4, 1, 1, 32, 1,
        0.25f, 0, -128, 127));
    CHECK(packed == original);

    const float nc4Source[] = {
        1.0f, 2.0f, 3.0f, 0.0f,
        4.0f, 5.0f, 6.0f, 0.0f,
    };
    std::vector<int8_t> twoBatchPacked(2 * 32, 0);
    CHECK(MNN::packFloatToInt8Nc(
        nc4Source, twoBatchPacked.data(), 2, 3, 1, 1, 32, 4,
        1.0f, 0, -128, 127));
    CHECK(twoBatchPacked[0] == 1);
    CHECK(twoBatchPacked[1] == 2);
    CHECK(twoBatchPacked[2] == 3);
    CHECK(twoBatchPacked[32] == 4);
    CHECK(twoBatchPacked[33] == 5);
    CHECK(twoBatchPacked[34] == 6);
    return true;
}

} // namespace

int main() {
    if (!testRequantizeTiesToEven() ||
        !testRequantizeRejectsInvalidScale() ||
        !testWeightPacking() ||
        !testPostParameters() ||
        !testPackedActivationConversion()) {
        return 1;
    }
    std::printf("PASS: Hexagon W8A8 quantization contract\n");
    return 0;
}
