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
 * Description: memset_s function
 * Create: 2014-02-25
 */
/*
 * [Standardize-exceptions] Use unsafe function: Portability
 * [reason] Use unsafe function to implement security function to maintain platform compatibility.
 *          And sufficient input validation is performed before calling
 */

#include "securecutil.h"

#define SECUREC_MEMSET_PARAM_OK(dest, destMax, count) (SECUREC_LIKELY((destMax) <= SECUREC_MEM_MAX_LEN && \
    (dest) != NULL && (count) <= (destMax)))

#if SECUREC_WITH_PERFORMANCE_ADDONS

/* Use union to clear strict-aliasing warning */
typedef union {
    SecStrBuf32 buf32;
    SecStrBuf31 buf31;
    SecStrBuf30 buf30;
    SecStrBuf29 buf29;
    SecStrBuf28 buf28;
    SecStrBuf27 buf27;
    SecStrBuf26 buf26;
    SecStrBuf25 buf25;
    SecStrBuf24 buf24;
    SecStrBuf23 buf23;
    SecStrBuf22 buf22;
    SecStrBuf21 buf21;
    SecStrBuf20 buf20;
    SecStrBuf19 buf19;
    SecStrBuf18 buf18;
    SecStrBuf17 buf17;
    SecStrBuf16 buf16;
    SecStrBuf15 buf15;
    SecStrBuf14 buf14;
    SecStrBuf13 buf13;
    SecStrBuf12 buf12;
    SecStrBuf11 buf11;
    SecStrBuf10 buf10;
    SecStrBuf9 buf9;
    SecStrBuf8 buf8;
    SecStrBuf7 buf7;
    SecStrBuf6 buf6;
    SecStrBuf5 buf5;
    SecStrBuf4 buf4;
    SecStrBuf3 buf3;
    SecStrBuf2 buf2;
} SecStrBuf32Union;
/* C standard initializes the first member of the consortium. */
static const SecStrBuf32 g_allZero = {{
    0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U,
    0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U,
    0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U,
    0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U
}};
static const SecStrBuf32 g_allFF = {{
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
}};

/* Clear conversion warning strict aliasing" */
SECUREC_INLINE const SecStrBuf32Union *SecStrictAliasingCast(const SecStrBuf32 *buf)
{
    return (const SecStrBuf32Union *)buf;
}

#ifndef SECUREC_MEMSET_THRESHOLD_SIZE
#define SECUREC_MEMSET_THRESHOLD_SIZE 32UL
#endif

#define SECUREC_UNALIGNED_SET(dest, c, count) do { \
    unsigned char *pDest_ = (unsigned char *)(dest); \
    switch (count) { \
        case 32: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 31: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 30: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 29: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 28: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 27: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 26: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 25: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 24: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 23: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 22: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 21: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 20: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 19: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 18: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 17: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 16: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 15: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 14: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 13: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 12: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 11: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 10: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 9: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 8: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 7: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 6: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 5: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 4: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 3: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 2: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        case 1: \
            *(pDest_++) = (unsigned char)(c); \
            /* fall-through */ /* FALLTHRU */ \
        default: \
            /* Do nothing */ \
            break; \
    } \
} SECUREC_WHILE_ZERO

#define SECUREC_SET_VALUE_BY_STRUCT(dest, dataName, n) do { \
    *(SecStrBuf##n *)(dest) = *(const SecStrBuf##n *)(&((SecStrictAliasingCast(&(dataName)))->buf##n)); \
} SECUREC_WHILE_ZERO

#define SECUREC_ALIGNED_SET_OPT_ZERO_FF(dest, c, count) do { \
    switch (c) { \
        case 0: \
            switch (count) { \
                case 1: \
                    *(unsigned char *)(dest) = (unsigned char)0; \
                    break; \
                case 2: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 2); \
                    break; \
                case 3: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 3); \
                    break; \
                case 4: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 4); \
                    break; \
                case 5: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 5); \
                    break; \
                case 6: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 6); \
                    break; \
                case 7: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 7); \
                    break; \
                case 8: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 8); \
                    break; \
                case 9: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 9); \
                    break; \
                case 10: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 10); \
                    break; \
                case 11: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 11); \
                    break; \
                case 12: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 12); \
                    break; \
                case 13: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 13); \
                    break; \
                case 14: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 14); \
                    break; \
                case 15: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 15); \
                    break; \
                case 16: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 16); \
                    break; \
                case 17: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 17); \
                    break; \
                case 18: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 18); \
                    break; \
                case 19: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 19); \
                    break; \
                case 20: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 20); \
                    break; \
                case 21: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 21); \
                    break; \
                case 22: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 22); \
                    break; \
                case 23: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 23); \
                    break; \
                case 24: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 24); \
                    break; \
                case 25: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 25); \
                    break; \
                case 26: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 26); \
                    break; \
                case 27: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 27); \
                    break; \
                case 28: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 28); \
                    break; \
                case 29: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 29); \
                    break; \
                case 30: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 30); \
                    break; \
                case 31: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 31); \
                    break; \
                case 32: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allZero, 32); \
                    break; \
                default: \
                    /* Do nothing */ \
                    break; \
            } \
            break; \
        case 0xFF: \
            switch (count) { \
                case 1: \
                    *(unsigned char *)(dest) = (unsigned char)0xffU; \
                    break; \
                case 2: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 2); \
                    break; \
                case 3: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 3); \
                    break; \
                case 4: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 4); \
                    break; \
                case 5: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 5); \
                    break; \
                case 6: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 6); \
                    break; \
                case 7: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 7); \
                    break; \
                case 8: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 8); \
                    break; \
                case 9: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 9); \
                    break; \
                case 10: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 10); \
                    break; \
                case 11: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 11); \
                    break; \
                case 12: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 12); \
                    break; \
                case 13: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 13); \
                    break; \
                case 14: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 14); \
                    break; \
                case 15: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 15); \
                    break; \
                case 16: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 16); \
                    break; \
                case 17: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 17); \
                    break; \
                case 18: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 18); \
                    break; \
                case 19: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 19); \
                    break; \
                case 20: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 20); \
                    break; \
                case 21: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 21); \
                    break; \
                case 22: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 22); \
                    break; \
                case 23: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 23); \
                    break; \
                case 24: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 24); \
                    break; \
                case 25: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 25); \
                    break; \
                case 26: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 26); \
                    break; \
                case 27: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 27); \
                    break; \
                case 28: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 28); \
                    break; \
                case 29: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 29); \
                    break; \
                case 30: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 30); \
                    break; \
                case 31: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 31); \
                    break; \
                case 32: \
                    SECUREC_SET_VALUE_BY_STRUCT((dest), g_allFF, 32); \
                    break; \
                default: \
                    /* Do nothing */ \
                    break; \
            } \
            break; \
        default: \
            SECUREC_UNALIGNED_SET((dest), (c), (count)); \
            break; \
    } /* END switch */ \
} SECUREC_WHILE_ZERO

#define SECUREC_SMALL_MEM_SET(dest, c, count) do { \
    if (SECUREC_ADDR_ALIGNED_8((dest))) { \
        SECUREC_ALIGNED_SET_OPT_ZERO_FF((dest), (c), (count)); \
    } else { \
        SECUREC_UNALIGNED_SET((dest), (c), (count)); \
    } \
} SECUREC_WHILE_ZERO

/*
 * Performance optimization
 */
#define SECUREC_MEMSET_OPT(dest, c, count) do { \
    if ((count) > SECUREC_MEMSET_THRESHOLD_SIZE) { \
        SECUREC_MEMSET_PREVENT_DSE((dest), (c), (count)); \
    } else { \
        SECUREC_SMALL_MEM_SET((dest), (c), (count)); \
    } \
} SECUREC_WHILE_ZERO
#endif

/*
 * Handling errors
 */
SECUREC_INLINE errno_t SecMemsetError(void *dest, size_t destMax, int c)
{
    /* Check destMax is 0 compatible with _sp macro */
    if (destMax == 0 || destMax > SECUREC_MEM_MAX_LEN) {
        SECUREC_ERROR_INVALID_RANGE("memset_s");
        return ERANGE;
    }
    if (dest == NULL) {
        SECUREC_ERROR_INVALID_PARAMTER("memset_s");
        return EINVAL;
    }
    SECUREC_MEMSET_PREVENT_DSE(dest, c, destMax); /* Set entire buffer to value c */
    SECUREC_ERROR_INVALID_RANGE("memset_s");
    return ERANGE_AND_RESET;
}

/*
 * <FUNCTION DESCRIPTION>
 *    The memset_s function copies the value of c (converted to an unsigned char)
 *     into each of the first count characters of the object pointed to by dest.
 *
 * <INPUT PARAMETERS>
 *    dest                Pointer to destination.
 *    destMax             The size of the buffer.
 *    c                   Character to set.
 *    count               Number of characters.
 *
 * <OUTPUT PARAMETERS>
 *    dest buffer         is updated.
 *
 * <RETURN VALUE>
 *    EOK                 Success
 *    EINVAL              dest == NULL and destMax != 0 and destMax <= SECUREC_MEM_MAX_LEN
 *    ERANGE              destMax > SECUREC_MEM_MAX_LEN or (destMax is 0 and count > destMax)
 *    ERANGE_AND_RESET    count > destMax and destMax != 0 and destMax <= SECUREC_MEM_MAX_LEN and dest != NULL
 *
 *    if return ERANGE_AND_RESET then fill dest to c ,fill length is destMax
 */
errno_t memset_s(void *dest, size_t destMax, int c, size_t count)
{
    if (SECUREC_MEMSET_PARAM_OK(dest, destMax, count)) {
        SECUREC_MEMSET_PREVENT_DSE(dest, c, count);
        return EOK;
    }
    /* Meet some runtime violation, return error code */
    return SecMemsetError(dest, destMax, c);
}

#if SECUREC_EXPORT_KERNEL_SYMBOL
EXPORT_SYMBOL(memset_s);
#endif

#if SECUREC_WITH_PERFORMANCE_ADDONS
/*
 * Performance optimization
 */
errno_t memset_sOptAsm(void *dest, size_t destMax, int c, size_t count)
{
    if (SECUREC_MEMSET_PARAM_OK(dest, destMax, count)) {
        SECUREC_MEMSET_OPT(dest, c, count);
        return EOK;
    }
    /* Meet some runtime violation, return error code */
    return SecMemsetError(dest, destMax, c);
}

/*
 * Performance optimization, trim judgement on "destMax <= SECUREC_MEM_MAX_LEN"
 */
errno_t memset_sOptTc(void *dest, size_t destMax, int c, size_t count)
{
    if (SECUREC_LIKELY(count <= destMax && dest != NULL)) {
        SECUREC_MEMSET_OPT(dest, c, count);
        return EOK;
    }
    /* Meet some runtime violation, return error code */
    return SecMemsetError(dest, destMax, c);
}
#endif

