//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(cppcoreguidelines-avoid-do-while,cppcoreguidelines-pro-type-vararg,cert-err33-c)
//
//   * cppcoreguidelines-avoid-do-while: do-while is necessary for macros.
//   * cppcoreguidelines-pro-type-vararg: use of variadic arguments in fprintf is expected.
//   * cert-err33-c: checking the output of fflush and fprintf is not necessary for error reporting.
#define KAI_ERROR(msg)                                        \
    do {                                                      \
        fflush(stdout);                                       \
        fprintf(stderr, "%s:%d %s", __FILE__, __LINE__, msg); \
        exit(EXIT_FAILURE);                                   \
    } while (0)

#define KAI_ASSERT_MSG(cond, msg) \
    do {                          \
        if (!(cond)) {            \
            KAI_ERROR(msg);       \
        }                         \
    } while (0)

// NOLINTEND(cppcoreguidelines-avoid-do-while,cppcoreguidelines-pro-type-vararg,cert-err33-c)

#define KAI_ASSERT(cond) KAI_ASSERT_MSG(cond, #cond)

#define KAI_ASSERT_IF_MSG(precond, cond, msg) KAI_ASSERT_MSG(!(precond) || (cond), msg)
#define KAI_ASSERT_IF(precond, cond) KAI_ASSERT_IF_MSG(precond, cond, #precond " |-> " #cond)

#define KAI_ASSUME_MSG KAI_ASSERT_MSG
#define KAI_ASSUME KAI_ASSERT
#define KAI_ASSUME_IF_MSG KAI_ASSERT_IF_MSG
#define KAI_ASSUME_IF KAI_ASSERT_IF

#define KAI_UNUSED(x) (void)(x)
#define KAI_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define KAI_MAX(a, b) (((a) > (b)) ? (a) : (b))

/// KleidiAI data types
/// Format: <byte 3>(reserved)|<byte 2>(num-bytes)|<byte 1>(type)|<byte 0>(variant-type)
enum kai_datatype {
    kai_dt_unknown = 0x0000,
    kai_dt_f32 = 0x0411,
    kai_dt_f16 = 0x0212,
    kai_dt_bf16 = 0x0213,
    kai_dt_int32 = 0x0421,
    kai_dt_int16 = 0x0222,
    kai_dt_int8 = 0x0124,
    kai_dt_uint32 = 0x0431,
    kai_dt_uint16 = 0x0232,
    kai_dt_uint8 = 0x0134,
    kai_dt_bool = 0x0441
};

/// Gets number of bytes for a given data type
/// @param[in] dt KleidiAI data type
///
/// @return the numbers of bytes for the data type
inline static size_t kai_get_datatype_size_in_bytes(enum kai_datatype dt) {
    return (size_t)(dt >> 8);
}

/// Converts a scalar f16 value to f32
/// @param[in] f16 The f16 value
///
/// @return the f32 value
inline static float kai_cast_f32_f16(uint16_t f16) {
#if defined(__ARM_NEON)
    __fp16 f32 = 0;
    memcpy(&f32, &f16, sizeof(uint16_t));
    return (float)f32;
#endif
}

/// Converts a scalar bf16 value to f32
/// @param[in] bf16 The f16 value
///
/// @return the f32 value
inline static float kai_cast_f32_bf16(uint16_t bf16) {
    const uint32_t i32 = (bf16 << 16);
    float f32;
    memcpy(&f32, &i32, sizeof(i32));
    return f32;
}

/// Converts a f32 value to bf16
/// @param[in] f32 The f32 value
///
/// @return the bf16 value
inline static uint16_t kai_cast_bf16_f32(float f32) {
    uint16_t bf16 = 0;
#ifdef __ARM_FEATURE_BF16
    __asm__ __volatile__("bfcvt %h[output], %s[input]" : [output] "=w"(bf16) : [input] "w"(f32));
#else
    const uint32_t* i32 = (uint32_t*)(&f32);
    bf16 = (*i32 >> 16);
#endif
    return bf16;
}

/// Converts a scalar f32 value to f16
/// @param[in] f32 The f32 value
///
/// @return the f16 value
inline static uint16_t kai_cast_f16_f32(float f32) {
#if defined(__ARM_NEON)
    uint16_t f16 = 0;
    __fp16 tmp = f32;
    memcpy(&f16, &tmp, sizeof(uint16_t));
    return f16;
#endif
}

inline static size_t kai_roundup(size_t a, size_t b) {
    return ((a + b - 1) / b) * b;
}

#ifdef __ARM_FEATURE_SVE

/// Gets the SME vector length for 8-bit elements.
inline static uint64_t kai_get_sme_vector_length_u8(void) {
    uint64_t res = 0;

    __asm__ __volatile__(
        ".inst 0xd503477f  // SMSTART ZA\n"
        "cntb %0\n"
        ".inst 0xd503467f  // SMSTOP\n"
        : "=r"(res)
        :
        : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16",
          "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");

    return res;
}

/// Gets the SME vector length for 16-bit elements.
inline static uint64_t kai_get_sme_vector_length_u16(void) {
    uint64_t res = 0;

    __asm__ __volatile__(
        ".inst 0xd503477f  // SMSTART ZA\n"
        "cnth %0\n"
        ".inst 0xd503467f  // SMSTOP\n"
        : "=r"(res)
        :
        : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16",
          "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");

    return res;
}

/// Gets the SME vector length for 32-bit elements.
inline static uint64_t kai_get_sme_vector_length_u32(void) {
    uint64_t res = 0;

    __asm__ __volatile__(
        ".inst 0xd503477f  // SMSTART ZA\n"
        "cntw %0\n"
        ".inst 0xd503467f  // SMSTOP\n"
        : "=r"(res)
        :
        : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16",
          "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");

    return res;
}

#endif  // __ARM_FEATURE_SVE

/// Extends the sign bit of int 4-bit value (stored in int8_t variable)
/// @param[in] value The 4-bit int value
///
/// @return the int8_t value with sign extended
inline static int8_t kai_ext_sign_i8_i4(int8_t value) {
    return (value ^ 0x8) - 8;
}

#ifdef __cplusplus
}
#endif
