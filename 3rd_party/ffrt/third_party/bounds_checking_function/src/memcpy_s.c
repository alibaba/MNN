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
 * Description: memcpy_s function
 * Create: 2014-02-25
 */
/*
 * [Standardize-exceptions] Use unsafe function: Portability
 * [reason] Use unsafe function to implement security function to maintain platform compatibility.
 *          And sufficient input validation is performed before calling
 */

#include "securecutil.h"

#if SECUREC_WITH_PERFORMANCE_ADDONS
#ifndef SECUREC_MEMCOPY_THRESHOLD_SIZE
#define SECUREC_MEMCOPY_THRESHOLD_SIZE 64UL
#endif

#define SECUREC_SMALL_MEM_COPY(dest, src, count) do { \
    if (SECUREC_ADDR_ALIGNED_8(dest) && SECUREC_ADDR_ALIGNED_8(src)) { \
        /* Use struct assignment */ \
        switch (count) { \
            case 1: \
                *(unsigned char *)(dest) = *(const unsigned char *)(src); \
                break; \
            case 2: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 2); \
                break; \
            case 3: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 3); \
                break; \
            case 4: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 4); \
                break; \
            case 5: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 5); \
                break; \
            case 6: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 6); \
                break; \
            case 7: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 7); \
                break; \
            case 8: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 8); \
                break; \
            case 9: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 9); \
                break; \
            case 10: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 10); \
                break; \
            case 11: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 11); \
                break; \
            case 12: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 12); \
                break; \
            case 13: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 13); \
                break; \
            case 14: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 14); \
                break; \
            case 15: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 15); \
                break; \
            case 16: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 16); \
                break; \
            case 17: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 17); \
                break; \
            case 18: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 18); \
                break; \
            case 19: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 19); \
                break; \
            case 20: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 20); \
                break; \
            case 21: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 21); \
                break; \
            case 22: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 22); \
                break; \
            case 23: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 23); \
                break; \
            case 24: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 24); \
                break; \
            case 25: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 25); \
                break; \
            case 26: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 26); \
                break; \
            case 27: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 27); \
                break; \
            case 28: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 28); \
                break; \
            case 29: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 29); \
                break; \
            case 30: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 30); \
                break; \
            case 31: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 31); \
                break; \
            case 32: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 32); \
                break; \
            case 33: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 33); \
                break; \
            case 34: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 34); \
                break; \
            case 35: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 35); \
                break; \
            case 36: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 36); \
                break; \
            case 37: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 37); \
                break; \
            case 38: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 38); \
                break; \
            case 39: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 39); \
                break; \
            case 40: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 40); \
                break; \
            case 41: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 41); \
                break; \
            case 42: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 42); \
                break; \
            case 43: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 43); \
                break; \
            case 44: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 44); \
                break; \
            case 45: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 45); \
                break; \
            case 46: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 46); \
                break; \
            case 47: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 47); \
                break; \
            case 48: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 48); \
                break; \
            case 49: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 49); \
                break; \
            case 50: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 50); \
                break; \
            case 51: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 51); \
                break; \
            case 52: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 52); \
                break; \
            case 53: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 53); \
                break; \
            case 54: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 54); \
                break; \
            case 55: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 55); \
                break; \
            case 56: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 56); \
                break; \
            case 57: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 57); \
                break; \
            case 58: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 58); \
                break; \
            case 59: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 59); \
                break; \
            case 60: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 60); \
                break; \
            case 61: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 61); \
                break; \
            case 62: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 62); \
                break; \
            case 63: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 63); \
                break; \
            case 64: \
                SECUREC_COPY_VALUE_BY_STRUCT((dest), (src), 64); \
                break; \
            default: \
                /* Do nothing */ \
                break; \
        } /* END switch */ \
    } else { \
        unsigned char *tmpDest_ = (unsigned char *)(dest); \
        const unsigned char *tmpSrc_ = (const unsigned char *)(src); \
        switch (count) { \
            case 64: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 63: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 62: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 61: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 60: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 59: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 58: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 57: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 56: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 55: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 54: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 53: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 52: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 51: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 50: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 49: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 48: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 47: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 46: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 45: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 44: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 43: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 42: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 41: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 40: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 39: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 38: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 37: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 36: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 35: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 34: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 33: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 32: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 31: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 30: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 29: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 28: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 27: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 26: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 25: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 24: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 23: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 22: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 21: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 20: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 19: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 18: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 17: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 16: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 15: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 14: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 13: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 12: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 11: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 10: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 9: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 8: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 7: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 6: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 5: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 4: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 3: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 2: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 1: \
                *(tmpDest_++) = *(tmpSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            default: \
                /* Do nothing */ \
                break; \
        } \
    } \
} SECUREC_WHILE_ZERO

/*
 * Performance optimization
 */
#define SECUREC_MEMCPY_OPT(dest, src, count) do { \
    if ((count) > SECUREC_MEMCOPY_THRESHOLD_SIZE) { \
        SECUREC_MEMCPY_WARP_OPT((dest), (src), (count)); \
    } else { \
        SECUREC_SMALL_MEM_COPY((dest), (src), (count)); \
    } \
} SECUREC_WHILE_ZERO
#endif

/*
 * Handling errors
 */
SECUREC_INLINE errno_t SecMemcpyError(void *dest, size_t destMax, const void *src, size_t count)
{
    if (destMax == 0 || destMax > SECUREC_MEM_MAX_LEN) {
        SECUREC_ERROR_INVALID_RANGE("memcpy_s");
        return ERANGE;
    }
    if (dest == NULL || src == NULL) {
        SECUREC_ERROR_INVALID_PARAMTER("memcpy_s");
        if (dest != NULL) {
            (void)SECUREC_MEMSET_FUNC_OPT(dest, 0, destMax);
            return EINVAL_AND_RESET;
        }
        return EINVAL;
    }
    if (count > destMax) {
        (void)SECUREC_MEMSET_FUNC_OPT(dest, 0, destMax);
        SECUREC_ERROR_INVALID_RANGE("memcpy_s");
        return ERANGE_AND_RESET;
    }
    if (SECUREC_MEMORY_IS_OVERLAP(dest, src, count)) {
        (void)SECUREC_MEMSET_FUNC_OPT(dest, 0, destMax);
        SECUREC_ERROR_BUFFER_OVERLAP("memcpy_s");
        return EOVERLAP_AND_RESET;
    }
    /* Count is 0 or dest equal src also ret EOK */
    return EOK;
}

#if defined(SECUREC_COMPATIBLE_WIN_FORMAT)
    /*
     * The fread API in windows will call memcpy_s and pass 0xffffffff to destMax.
     * To avoid the failure of fread, we don't check desMax limit.
     */
#define SECUREC_MEMCPY_PARAM_OK(dest, destMax, src, count) (SECUREC_LIKELY((count) <= (destMax) && \
    (dest) != NULL && (src) != NULL && \
    (count) > 0 && SECUREC_MEMORY_NO_OVERLAP((dest), (src), (count))))
#else
#define SECUREC_MEMCPY_PARAM_OK(dest, destMax, src, count) (SECUREC_LIKELY((count) <= (destMax) && \
    (dest) != NULL && (src) != NULL && (destMax) <= SECUREC_MEM_MAX_LEN && \
    (count) > 0 && SECUREC_MEMORY_NO_OVERLAP((dest), (src), (count))))
#endif

/*
 * <FUNCTION DESCRIPTION>
 *    The memcpy_s function copies n characters from the object pointed to by src into the object pointed to by dest
 *
 * <INPUT PARAMETERS>
 *    dest                      Destination buffer.
 *    destMax                   Size of the destination buffer.
 *    src                       Buffer to copy from.
 *    count                     Number of characters to copy
 *
 * <OUTPUT PARAMETERS>
 *    dest buffer               is updated.
 *
 * <RETURN VALUE>
 *    EOK                      Success
 *    EINVAL                   dest is  NULL and destMax != 0 and destMax <= SECUREC_MEM_MAX_LEN
 *    EINVAL_AND_RESET         dest != NULL and src is NULL and destMax != 0 and destMax <= SECUREC_MEM_MAX_LEN
 *    ERANGE                   destMax > SECUREC_MEM_MAX_LEN or destMax is 0
 *    ERANGE_AND_RESET         count > destMax and destMax != 0 and destMax <= SECUREC_MEM_MAX_LEN
 *                             and dest  !=  NULL  and src != NULL
 *    EOVERLAP_AND_RESET       dest buffer and source buffer are overlapped and
 *                             count <= destMax destMax != 0 and destMax <= SECUREC_MEM_MAX_LEN and dest  !=  NULL
 *                             and src != NULL  and dest != src
 *
 *    if an error occurred, dest will be filled with 0.
 *    If the source and destination overlap, the behavior of memcpy_s is undefined.
 *    Use memmove_s to handle overlapping regions.
 */
errno_t memcpy_s(void *dest, size_t destMax, const void *src, size_t count)
{
    if (SECUREC_MEMCPY_PARAM_OK(dest, destMax, src, count)) {
        SECUREC_MEMCPY_WARP_OPT(dest, src, count);
        return EOK;
    }
    /* Meet some runtime violation, return error code */
    return SecMemcpyError(dest, destMax, src, count);
}

#if SECUREC_EXPORT_KERNEL_SYMBOL
EXPORT_SYMBOL(memcpy_s);
#endif

#if SECUREC_WITH_PERFORMANCE_ADDONS
/*
 * Performance optimization
 */
errno_t memcpy_sOptAsm(void *dest, size_t destMax, const void *src, size_t count)
{
    if (SECUREC_MEMCPY_PARAM_OK(dest, destMax, src, count)) {
        SECUREC_MEMCPY_OPT(dest, src, count);
        return EOK;
    }
    /* Meet some runtime violation, return error code */
    return SecMemcpyError(dest, destMax, src, count);
}

/* Trim judgement on "destMax <= SECUREC_MEM_MAX_LEN" */
errno_t memcpy_sOptTc(void *dest, size_t destMax, const void *src, size_t count)
{
    if (SECUREC_LIKELY(count <= destMax && dest != NULL && src != NULL && \
                       count > 0 && SECUREC_MEMORY_NO_OVERLAP((dest), (src), (count)))) {
        SECUREC_MEMCPY_OPT(dest, src, count);
        return EOK;
    }
    /* Meet some runtime violation, return error code */
    return SecMemcpyError(dest, destMax, src, count);
}
#endif

