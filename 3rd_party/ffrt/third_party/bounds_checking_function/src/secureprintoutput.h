/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2014-2021. All rights reserved.
 * Licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * Description: Define macro, enum, data struct, and declare internal used function
 *              prototype, which is used by output.inl, secureprintoutput_w.c and
 *              secureprintoutput_a.c.
 * Create: 2014-02-25
 */

#ifndef SECUREPRINTOUTPUT_H_E950DA2C_902F_4B15_BECD_948E99090D9C
#define SECUREPRINTOUTPUT_H_E950DA2C_902F_4B15_BECD_948E99090D9C
#include "securecutil.h"

/* Shield compilation alerts about using sprintf without format attribute to format float value. */
#ifndef SECUREC_HANDLE_WFORMAT
#define SECUREC_HANDLE_WFORMAT 1
#endif

#if defined(__clang__)
#if SECUREC_HANDLE_WFORMAT && defined(__GNUC__) && ((__GNUC__ >= 5) || \
    (defined(__GNUC_MINOR__) && (__GNUC__ == 4 && __GNUC_MINOR__ >= 2)))
#define SECUREC_MASK_WFORMAT_WARNING  _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wformat-nonliteral\"")
#define SECUREC_END_MASK_WFORMAT_WARNING  _Pragma("GCC diagnostic pop")
#else
#define SECUREC_MASK_WFORMAT_WARNING
#define SECUREC_END_MASK_WFORMAT_WARNING
#endif
#else
#if SECUREC_HANDLE_WFORMAT && defined(__GNUC__) && ((__GNUC__ >= 5 ) || \
    (defined(__GNUC_MINOR__) && (__GNUC__ == 4 && __GNUC_MINOR__ > 7)))
#define SECUREC_MASK_WFORMAT_WARNING  _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wformat-nonliteral\"") \
    _Pragma("GCC diagnostic ignored \"-Wmissing-format-attribute\"") \
    _Pragma("GCC diagnostic ignored \"-Wsuggest-attribute=format\"")
#define SECUREC_END_MASK_WFORMAT_WARNING  _Pragma("GCC diagnostic pop")
#else
#define SECUREC_MASK_WFORMAT_WARNING
#define SECUREC_END_MASK_WFORMAT_WARNING
#endif
#endif

#define SECUREC_MASK_VSPRINTF_WARNING  SECUREC_MASK_WFORMAT_WARNING \
    SECUREC_MASK_MSVC_CRT_WARNING

#define SECUREC_END_MASK_VSPRINTF_WARNING  SECUREC_END_MASK_WFORMAT_WARNING \
    SECUREC_END_MASK_MSVC_CRT_WARNING

/*
 * Flag definitions.
 * Using macros instead of enumerations is because some of the enumerated types under the compiler are 16bit.
 */
#define SECUREC_FLAG_SIGN           0x00001U
#define SECUREC_FLAG_SIGN_SPACE     0x00002U
#define SECUREC_FLAG_LEFT           0x00004U
#define SECUREC_FLAG_LEADZERO       0x00008U
#define SECUREC_FLAG_LONG           0x00010U
#define SECUREC_FLAG_SHORT          0x00020U
#define SECUREC_FLAG_SIGNED         0x00040U
#define SECUREC_FLAG_ALTERNATE      0x00080U
#define SECUREC_FLAG_NEGATIVE       0x00100U
#define SECUREC_FLAG_FORCE_OCTAL    0x00200U
#define SECUREC_FLAG_LONG_DOUBLE    0x00400U
#define SECUREC_FLAG_WIDECHAR       0x00800U
#define SECUREC_FLAG_LONGLONG       0x01000U
#define SECUREC_FLAG_CHAR           0x02000U
#define SECUREC_FLAG_POINTER        0x04000U
#define SECUREC_FLAG_I64            0x08000U
#define SECUREC_FLAG_PTRDIFF        0x10000U
#define SECUREC_FLAG_SIZE           0x20000U
#ifdef  SECUREC_COMPATIBLE_LINUX_FORMAT
#define SECUREC_FLAG_INTMAX         0x40000U
#endif

/* State definitions. Identify the status of the current format */
typedef enum {
    STAT_NORMAL,
    STAT_PERCENT,
    STAT_FLAG,
    STAT_WIDTH,
    STAT_DOT,
    STAT_PRECIS,
    STAT_SIZE,
    STAT_TYPE,
    STAT_INVALID
} SecFmtState;

#ifndef SECUREC_BUFFER_SIZE
#if SECUREC_IN_KERNEL
#define SECUREC_BUFFER_SIZE    32
#elif defined(SECUREC_STACK_SIZE_LESS_THAN_1K)
/*
 * SECUREC BUFFER SIZE Can not be less than 23
 * The length of the octal representation of 64-bit integers with zero lead
 */
#define SECUREC_BUFFER_SIZE    256
#else
#define SECUREC_BUFFER_SIZE    512
#endif
#endif
#if SECUREC_BUFFER_SIZE < 23
#error SECUREC_BUFFER_SIZE Can not be less than 23
#endif
/* Buffer size for wchar, use 4 to make the compiler aligns as 8 bytes as possible */
#define SECUREC_WCHAR_BUFFER_SIZE 4

#define SECUREC_MAX_PRECISION  SECUREC_BUFFER_SIZE
/* Max. # bytes in multibyte char,see MB_LEN_MAX */
#define SECUREC_MB_LEN 16
/* The return value of the internal function, which is returned when truncated */
#define SECUREC_PRINTF_TRUNCATE (-2)

#define SECUREC_VSPRINTF_PARAM_ERROR(format, strDest, destMax, maxLimit) \
    ((format) == NULL || (strDest) == NULL || (destMax) == 0 || (destMax) > (maxLimit))

#define SECUREC_VSPRINTF_CLEAR_DEST(strDest, destMax, maxLimit) do { \
    if ((strDest) != NULL && (destMax) > 0 && (destMax) <= (maxLimit)) { \
        *(strDest) = '\0'; \
    } \
} SECUREC_WHILE_ZERO

#ifdef SECUREC_COMPATIBLE_WIN_FORMAT
#define SECUREC_VSNPRINTF_PARAM_ERROR(format, strDest, destMax, count, maxLimit) \
    (((format) == NULL || (strDest) == NULL || (destMax) == 0 || (destMax) > (maxLimit)) || \
    ((count) > (SECUREC_STRING_MAX_LEN - 1) && (count) != (size_t)(-1)))

#else
#define SECUREC_VSNPRINTF_PARAM_ERROR(format, strDest, destMax, count, maxLimit) \
    (((format) == NULL || (strDest) == NULL || (destMax) == 0 || (destMax) > (maxLimit)) || \
    ((count) > (SECUREC_STRING_MAX_LEN - 1)))
#endif

#ifdef __cplusplus
extern "C" {
#endif
#ifdef SECUREC_FOR_WCHAR
int SecVswprintfImpl(wchar_t *string, size_t count, const wchar_t *format, va_list argList);
#else
int SecVsnprintfImpl(char *string, size_t count, const char *format, va_list argList);
#endif
#ifdef __cplusplus
}
#endif

#endif

