#include "dsp/int8_cnn.h"

#include <cstring>
#include <limits>

#ifdef __hexagon__
#include "dsp/hvx_utils.h"
#include <hexagon_protos.h>
#include <hexagon_types.h>
#endif

namespace {

bool floatIsFinite(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000u) != 0x7f800000u;
}

float halfToFloat(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffu;
            bits = sign |
                   (static_cast<uint32_t>(127 - 14 - shift) << 23) |
                   (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127u - 15u) << 23) |
               (mantissa << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t floatToHalf(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        return static_cast<uint16_t>(
            sign | 0x7c00u | (mantissa ? 0x0200u : 0u));
    }
    int32_t halfExponent =
        static_cast<int32_t>(exponent) - 127 + 15;
    if (halfExponent <= 0) {
        if (halfExponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        const uint32_t valueMantissa = mantissa | 0x800000u;
        const uint32_t shift =
            static_cast<uint32_t>(14 - halfExponent);
        uint32_t halfMantissa = valueMantissa >> shift;
        const uint32_t remainder =
            valueMantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway ||
            (remainder == halfway && (halfMantissa & 1u))) {
            ++halfMantissa;
        }
        return static_cast<uint16_t>(sign | halfMantissa);
    }
    if (halfExponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    uint32_t halfMantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u ||
        (remainder == 0x1000u && (halfMantissa & 1u))) {
        ++halfMantissa;
        if (halfMantissa == 0x400u) {
            halfMantissa = 0;
            if (++halfExponent >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }
    }
    return static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(halfExponent) << 10) |
        halfMantissa);
}

bool validCommon(const void* dst, const void* src, int32_t size,
                 float scale, int32_t zero) {
    return size >= 0 && floatIsFinite(scale) && scale > 0.0f &&
           zero == 0 &&
           (size == 0 || (dst != nullptr && src != nullptr));
}

int32_t roundToNearestEven(float value) {
    const int32_t base = static_cast<int32_t>(value);
    const float remainder = value - static_cast<float>(base);
    if (remainder > 0.5f ||
        (remainder == 0.5f && (base & 1) != 0)) {
        return base + 1;
    }
    if (remainder < -0.5f ||
        (remainder == -0.5f && (base & 1) != 0)) {
        return base - 1;
    }
    return base;
}

uint64_t roundUnsignedRightShift(uint64_t value, int64_t shift) {
    if (shift <= 0) {
        return value;
    }
    if (shift > 63) {
        return 0;
    }
    const uint64_t quotient = value >> shift;
    const uint64_t mask = (uint64_t{1} << shift) - 1u;
    const uint64_t remainder = value & mask;
    const uint64_t half = uint64_t{1} << (shift - 1);
    return remainder > half ||
                   (remainder == half && (quotient & 1u) != 0u)
               ? quotient + 1u
               : quotient;
}

int32_t requantizeScalar(int32_t value,
                         const DSPRequantParam& param) {
    if (value == 0 || param.multiplier <= 0) {
        return param.outputZero < param.clampMin
                   ? param.clampMin
                   : (param.outputZero > param.clampMax
                          ? param.clampMax
                          : param.outputZero);
    }
    const bool negative = value < 0;
    const uint64_t magnitude =
        negative ? static_cast<uint64_t>(-static_cast<int64_t>(value))
                 : static_cast<uint64_t>(value);
    const uint64_t product =
        magnitude * static_cast<uint32_t>(param.multiplier);
    const int64_t binaryShift =
        static_cast<int64_t>(param.shift) - 31;
    uint64_t scaledMagnitude = 0;
    if (binaryShift >= 0) {
        if (binaryShift >= 63 ||
            product >
                (static_cast<uint64_t>(
                     std::numeric_limits<int64_t>::max()) >>
                 binaryShift)) {
            return negative ? param.clampMin : param.clampMax;
        }
        scaledMagnitude = product << binaryShift;
    } else {
        scaledMagnitude =
            roundUnsignedRightShift(product, -binaryShift);
    }
    if (scaledMagnitude >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return negative ? param.clampMin : param.clampMax;
    }
    const int64_t scaled =
        negative ? -static_cast<int64_t>(scaledMagnitude)
                 : static_cast<int64_t>(scaledMagnitude);
    const int64_t withZero = scaled + param.outputZero;
    return withZero < param.clampMin
               ? param.clampMin
               : (withZero > param.clampMax
                      ? param.clampMax
                      : static_cast<int32_t>(withZero));
}

float loadFloat(const uint8_t* src, int32_t index, int32_t bytes) {
    if (bytes == 4) {
        float value = 0.0f;
        std::memcpy(&value, src + static_cast<size_t>(index) * 4,
                    sizeof(value));
        return value;
    }
    uint16_t value = 0;
    std::memcpy(&value, src + static_cast<size_t>(index) * 2,
                sizeof(value));
    return halfToFloat(value);
}

#ifdef __hexagon__

bool fp16AllFiniteHvx(const uint8_t* src, int32_t size) {
    const int32_t vectorEnd = size & ~127;
    const HVX_Vector exponentMask = Q6_Vh_vsplat_R(0x7c00);
    for (int32_t i = 0; i < vectorEnd; i += 128) {
        HVX_Vector lo = Q6_V_vand_VV(
            vmemu(src + static_cast<size_t>(i) * 2), exponentMask);
        HVX_Vector hi = Q6_V_vand_VV(
            vmemu(src + static_cast<size_t>(i) * 2 + 128),
            exponentMask);
        HVX_Vector maximum = Q6_Vuh_vmax_VuhVuh(lo, hi);
        maximum = Q6_Vuh_vmax_VuhVuh(
            maximum, Q6_V_vror_VR(maximum, 64));
        maximum = Q6_Vuh_vmax_VuhVuh(
            maximum, Q6_V_vror_VR(maximum, 32));
        maximum = Q6_Vuh_vmax_VuhVuh(
            maximum, Q6_V_vror_VR(maximum, 16));
        maximum = Q6_Vuh_vmax_VuhVuh(
            maximum, Q6_V_vror_VR(maximum, 8));
        maximum = Q6_Vuh_vmax_VuhVuh(
            maximum, Q6_V_vror_VR(maximum, 4));
        maximum = Q6_Vuh_vmax_VuhVuh(
            maximum, Q6_V_vror_VR(maximum, 2));
        uint16_t lanes[64] __attribute__((aligned(128)));
        vmem(lanes) = maximum;
        if (lanes[0] == 0x7c00u) {
            return false;
        }
    }
    for (int32_t i = vectorEnd; i < size; ++i) {
        uint16_t value = 0;
        std::memcpy(
            &value, src + static_cast<size_t>(i) * 2, sizeof(value));
        if ((value & 0x7c00u) == 0x7c00u) {
            return false;
        }
    }
    return true;
}

bool safeFp16Reciprocal(float scale, uint16_t* reciprocalBits) {
    const float reciprocal = 1.0f / scale;
    if (!floatIsFinite(reciprocal) || reciprocal <= 0.0f) {
        return false;
    }
    const uint16_t half = floatToHalf(reciprocal);
    const uint16_t exponent = half & 0x7c00u;
    if (exponent == 0 || exponent == 0x7c00u) {
        return false;
    }
    const float rounded = halfToFloat(half);
    const float error =
        rounded > reciprocal ? rounded - reciprocal : reciprocal - rounded;
    if (error > reciprocal * (1.0f / 1024.0f)) {
        return false;
    }
    *reciprocalBits = half;
    return true;
}

AEEResult quantizeFp16Hvx(uint8_t* dst, const uint8_t* src,
                          int32_t size, float scale) {
    uint16_t reciprocalBits = 0;
    if (!safeFp16Reciprocal(scale, &reciprocalBits)) {
        return quantize_i8_reference(dst, src, size, 2, scale, 0);
    }
    if (!fp16AllFiniteHvx(src, size)) {
        return AEE_EBADPARM;
    }

    const int32_t vectorEnd = size & ~127;
    const HVX_Vector reciprocal = Q6_Vh_vsplat_R(reciprocalBits);
    const HVX_Vector clampMin = Q6_Vb_vsplat_R(-127);
    for (int32_t i = 0; i < vectorEnd; i += 128) {
        HVX_Vector lo =
            vmemu(src + static_cast<size_t>(i) * 2);
        HVX_Vector hi =
            vmemu(src + static_cast<size_t>(i) * 2 + 128);
        lo = Q6_Vhf_equals_Vqf16(
            Q6_Vqf16_vmpy_VhfVhf(lo, reciprocal));
        hi = Q6_Vhf_equals_Vqf16(
            Q6_Vqf16_vmpy_VhfVhf(hi, reciprocal));
        lo = Q6_Vh_equals_Vhf(lo);
        hi = Q6_Vh_equals_Vhf(hi);
        const HVX_Vector packed =
            Q6_Vb_vpack_VhVh_sat(hi, lo);
        vmemu(dst + i) = Q6_Vb_vmax_VbVb(packed, clampMin);
    }
    return quantize_i8_reference(
        dst + vectorEnd,
        src + static_cast<size_t>(vectorEnd) * 2,
        size - vectorEnd, 2, scale, 0);
}

#endif

} // namespace

extern "C" {

AEEResult quantize_i8_reference(uint8_t* dst, const uint8_t* src,
                                int32_t size, int32_t srcBytes, float scale,
                                int32_t zero) {
    if (!validCommon(dst, src, size, scale, zero) ||
        (srcBytes != 2 && srcBytes != 4)) {
        return AEE_EBADPARM;
    }
    for (int32_t i = 0; i < size; ++i) {
        const float value = loadFloat(src, i, srcBytes);
        if (!floatIsFinite(value)) {
            return AEE_EBADPARM;
        }
        const float scaled = value / scale;
        const int32_t quantized =
            scaled <= -127.0f
                ? -127
                : (scaled >= 127.0f ? 127
                                    : roundToNearestEven(scaled));
        dst[i] = static_cast<uint8_t>(
            static_cast<int8_t>(quantized));
    }
    return AEE_SUCCESS;
}

AEEResult dequantize_i8_reference(uint8_t* dst, const uint8_t* src,
                                  int32_t size, int32_t dstBytes,
                                  float scale, int32_t zero) {
    if (!validCommon(dst, src, size, scale, zero) || dstBytes != 2 ||
        scale > 65504.0f / 127.0f) {
        return AEE_EBADPARM;
    }
    for (int32_t i = 0; i < size; ++i) {
        const float value =
            static_cast<float>(static_cast<int8_t>(src[i])) * scale;
        const uint16_t half = floatToHalf(value);
        std::memcpy(dst + static_cast<size_t>(i) * 2, &half,
                    sizeof(half));
    }
    return AEE_SUCCESS;
}

AEEResult requantize_i8_reference(uint8_t* dst, const int32_t* src,
                                  int32_t size,
                                  const DSPRequantParam* params) {
    if (size < 0 || params == nullptr ||
        (size > 0 && (dst == nullptr || src == nullptr)) ||
        params->clampMin < -127 || params->clampMax > 127 ||
        params->clampMin > params->clampMax) {
        return AEE_EBADPARM;
    }
    for (int32_t i = 0; i < size; ++i) {
        dst[i] = static_cast<uint8_t>(static_cast<int8_t>(
            requantizeScalar(src[i], *params)));
    }
    return AEE_SUCCESS;
}

AEEResult htp_ops_quantize_i8(uint8_t* dst, const uint8_t* src,
                              int32_t size, int32_t srcBytes, float scale,
                              int32_t zero) {
#ifdef __hexagon__
    if (!validCommon(dst, src, size, scale, zero) ||
        (srcBytes != 2 && srcBytes != 4)) {
        return AEE_EBADPARM;
    }
    if (srcBytes == 2) {
        return quantizeFp16Hvx(dst, src, size, scale);
    }
#endif
    return quantize_i8_reference(
        dst, src, size, srcBytes, scale, zero);
}

AEEResult htp_ops_quantize_i8_nc(
    uint8_t* dst, const uint8_t* src, int32_t size, int32_t srcBytes,
    float scale, int32_t zero, int32_t batch, int32_t channels,
    int32_t area, int32_t pack) {
    if (batch <= 0 || channels <= 0 || area <= 0 || pack <= 0 ||
        static_cast<int64_t>((channels + pack - 1) / pack) * batch *
                area * pack !=
            size) {
        return AEE_EBADPARM;
    }
    const AEEResult status = htp_ops_quantize_i8(
        dst, src, size, srcBytes, scale, zero);
    if (status != AEE_SUCCESS || channels % pack == 0) {
        return status;
    }
    const int channelPacks = (channels + pack - 1) / pack;
    const int validChannels = channels - (channelPacks - 1) * pack;
    const size_t lastPackOffset =
        static_cast<size_t>(channelPacks - 1) * batch * area * pack;
    for (int n = 0; n < batch; ++n) {
        for (int a = 0; a < area; ++a) {
            const size_t offset =
                lastPackOffset +
                (static_cast<size_t>(n) * area + a) * pack +
                validChannels;
            std::memset(dst + offset, 0,
                        static_cast<size_t>(pack - validChannels));
        }
    }
    return AEE_SUCCESS;
}

AEEResult htp_ops_dequantize_i8(uint8_t* dst, const uint8_t* src,
                                int32_t size, int32_t dstBytes,
                                float scale, int32_t zero) {
    return dequantize_i8_reference(
        dst, src, size, dstBytes, scale, zero);
}

AEEResult htp_ops_requantize_i8(uint8_t* dst, const int32_t* src,
                                int32_t size,
                                const DSPRequantParam* params) {
    return requantize_i8_reference(dst, src, size, params);
}

} // extern "C"
