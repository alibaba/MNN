#include "dsp/int8_cnn.h"
#include "../../backend/HexagonQuantization.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

bool testScalarRequantization() {
    const double scales[] = {0.03125, 0.125, 0.5, 1.0, 2.0, 17.0};
    const int32_t values[] = {
        std::numeric_limits<int32_t>::min(), -1000001, -129, -3, -1,
        0, 1, 3, 127, 1000001, std::numeric_limits<int32_t>::max(),
    };
    for (double scale : scales) {
        const auto param = MNN::makeRequantMultiplier(scale, 0, -127, 127);
        CHECK(param.multiplier > 0);
        for (int32_t value : values) {
            uint8_t actual = 0;
            CHECK(htp_ops_requantize_i8(
                      &actual, &value, 1, &param) == AEE_SUCCESS);
            CHECK(static_cast<int8_t>(actual) ==
                  MNN::requantizeReference(value, param));
        }
    }
    return true;
}

bool testPerChannelRequantization() {
    int32_t source[32];
    DSPRequantParam params[32];
    uint8_t actual[32] = {};
    for (int lane = 0; lane < 32; ++lane) {
        source[lane] = (lane - 16) * 12345 + (lane & 1);
        params[lane] = MNN::makeRequantMultiplier(
            std::ldexp(0.75, -(lane % 5)), 0, -127, 127);
    }
    CHECK(htp_ops_requantize_i8_per_channel_32(
              actual, source, params) == AEE_SUCCESS);
    for (int lane = 0; lane < 32; ++lane) {
        CHECK(static_cast<int8_t>(actual[lane]) ==
              MNN::requantizeReference(source[lane], params[lane]));
    }
    return true;
}

bool testQuantizeDequantize() {
    const uint16_t tinyHalf[] = {0x0001, 0x0002, 0x03ff, 0x8001, 0x83ff};
    uint8_t tinyQuantized[5] = {};
    CHECK(htp_ops_quantize_i8(
              tinyQuantized, reinterpret_cast<const uint8_t*>(tinyHalf),
              5, 2, std::ldexp(1.0f, -24), 0) == AEE_SUCCESS);
    const int8_t tinyExpected[] = {1, 2, 127, -1, -127};
    CHECK(std::memcmp(tinyQuantized, tinyExpected, sizeof(tinyExpected)) == 0);

    const float source[] = {
        -100.0f, -0.625f, -0.375f, -0.125f,
        0.125f, 0.375f, 0.625f, 100.0f,
    };
    uint8_t quantized[8] = {};
    CHECK(htp_ops_quantize_i8(
              quantized, reinterpret_cast<const uint8_t*>(source), 8,
              sizeof(float), 0.25f, 0) == AEE_SUCCESS);
    const int8_t expected[] = {-127, -2, -2, 0, 0, 2, 2, 127};
    CHECK(std::memcmp(quantized, expected, sizeof(expected)) == 0);

    uint8_t dequantized[16] = {};
    uint8_t reference[16] = {};
    CHECK(htp_ops_dequantize_i8(
              dequantized, quantized, 8, 2, 0.25f, 0) == AEE_SUCCESS);
    CHECK(dequantize_i8_reference(
              reference, quantized, 8, 2, 0.25f, 0) == AEE_SUCCESS);
    CHECK(std::memcmp(dequantized, reference, sizeof(reference)) == 0);

    const float invalid = std::numeric_limits<float>::quiet_NaN();
    CHECK(htp_ops_quantize_i8(
              quantized, reinterpret_cast<const uint8_t*>(&invalid), 1,
              sizeof(float), 0.25f, 0) == AEE_EBADPARM);
    return true;
}

} // namespace

int main() {
    if (!testScalarRequantization() ||
        !testPerChannelRequantization() ||
        !testQuantizeDequantize()) {
        return 1;
    }
    std::printf("PASS: Hexagon INT8 requantization contract\n");
    return 0;
}
