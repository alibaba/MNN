//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(cppcoreguidelines-avoid-do-while,cppcoreguidelines-pro-type-vararg,cert-err33-c)
//
//   * cppcoreguidelines-avoid-do-while: do-while is necessary for macros.
//   * cppcoreguidelines-pro-type-vararg: use of variadic arguments in fprintf is expected.
//   * cert-err33-c: checking the output of fflush and fprintf is not necessary for error reporting.
#define KAI_ERROR(msg)              \
    do {                            \
        fflush(stdout);             \
        fprintf(stderr, "%s", msg); \
        exit(EXIT_FAILURE);         \
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
    Unknown = 0x0000,
    F32 = 0x0411,
    F16 = 0x0212,
    Bf16 = 0x0213,
    Int32 = 0x0421,
    Int16 = 0x0222,
    Int8 = 0x0124,
    Uint32 = 0x0431,
    Uint16 = 0x0232,
    Uint8 = 0x0134,
    Bool = 0x0441
};

/// Gets number of bytes for a given data type
/// @param[in] dt KleidiAI data type
///
/// @return the numbers of bytes for the data type
inline static size_t kai_num_bytes_datatype(enum kai_datatype dt) {
    return (size_t)(dt >> 8);
}

/// Converts a scalar f16 value to f32
/// @param[in] f16 The f16 value
///
/// @return the f32 value
inline static float kai_f16_to_f32(uint16_t f16) {
#if defined(__ARM_NEON)
    __fp16 f32 = 0;
    memcpy(&f32, &f16, sizeof(uint16_t));
    return (float)f32;
#endif
}

/// Converts a scalar f32 value to f16
/// @param[in] f32 The f32 value
///
/// @return the f16 value
inline static uint16_t kai_f32_to_f16(float f32) {
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

#ifdef __cplusplus
}
#endif
