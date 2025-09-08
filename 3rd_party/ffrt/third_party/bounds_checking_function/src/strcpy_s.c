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
 * Description: strcpy_s  function
 * Create: 2014-02-25
 */
/*
 * [Standardize-exceptions] Use unsafe function: Performance-sensitive
 * [reason] Always used in the performance critical path,
 *          and sufficient input validation is performed before calling
 */

#include "securecutil.h"

#ifndef SECUREC_STRCPY_WITH_PERFORMANCE
#define SECUREC_STRCPY_WITH_PERFORMANCE 1
#endif

#define SECUREC_STRCPY_PARAM_OK(strDest, destMax, strSrc) ((destMax) > 0 && \
    (destMax) <= SECUREC_STRING_MAX_LEN && (strDest) != NULL && (strSrc) != NULL && (strDest) != (strSrc))

#if (!SECUREC_IN_KERNEL) && SECUREC_STRCPY_WITH_PERFORMANCE
#ifndef SECUREC_STRCOPY_THRESHOLD_SIZE
#define SECUREC_STRCOPY_THRESHOLD_SIZE   32UL
#endif
/* The purpose of converting to void is to clean up the alarm */
#define SECUREC_SMALL_STR_COPY(strDest, strSrc, lenWithTerm) do { \
    if (SECUREC_ADDR_ALIGNED_8(strDest) && SECUREC_ADDR_ALIGNED_8(strSrc)) { \
        /* Use struct assignment */ \
        switch (lenWithTerm) { \
            case 1: \
                *(strDest) = *(strSrc); \
                break; \
            case 2: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 2); \
                break; \
            case 3: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 3); \
                break; \
            case 4: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 4); \
                break; \
            case 5: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 5); \
                break; \
            case 6: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 6); \
                break; \
            case 7: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 7); \
                break; \
            case 8: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 8); \
                break; \
            case 9: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 9); \
                break; \
            case 10: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 10); \
                break; \
            case 11: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 11); \
                break; \
            case 12: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 12); \
                break; \
            case 13: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 13); \
                break; \
            case 14: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 14); \
                break; \
            case 15: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 15); \
                break; \
            case 16: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 16); \
                break; \
            case 17: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 17); \
                break; \
            case 18: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 18); \
                break; \
            case 19: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 19); \
                break; \
            case 20: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 20); \
                break; \
            case 21: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 21); \
                break; \
            case 22: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 22); \
                break; \
            case 23: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 23); \
                break; \
            case 24: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 24); \
                break; \
            case 25: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 25); \
                break; \
            case 26: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 26); \
                break; \
            case 27: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 27); \
                break; \
            case 28: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 28); \
                break; \
            case 29: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 29); \
                break; \
            case 30: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 30); \
                break; \
            case 31: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 31); \
                break; \
            case 32: \
                SECUREC_COPY_VALUE_BY_STRUCT((strDest), (strSrc), 32); \
                break; \
            default: \
                /* Do nothing */ \
                break; \
        } /* END switch */ \
    } else { \
        char *tmpStrDest_ = (char *)(strDest); \
        const char *tmpStrSrc_ = (const char *)(strSrc); \
        switch (lenWithTerm) { \
            case 32: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 31: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 30: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 29: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 28: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 27: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 26: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 25: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 24: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 23: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 22: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 21: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 20: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 19: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 18: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 17: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 16: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 15: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 14: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 13: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 12: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 11: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 10: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 9: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 8: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 7: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 6: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 5: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 4: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 3: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 2: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            case 1: \
                *(tmpStrDest_++) = *(tmpStrSrc_++); \
                /* fall-through */ /* FALLTHRU */ \
            default: \
                /* Do nothing */ \
                break; \
        } \
    } \
} SECUREC_WHILE_ZERO
#endif

#if SECUREC_IN_KERNEL || (!SECUREC_STRCPY_WITH_PERFORMANCE)
#define SECUREC_STRCPY_OPT(dest, src, lenWithTerm) SECUREC_MEMCPY_WARP_OPT((dest), (src), (lenWithTerm))
#else
/*
 * Performance optimization. lenWithTerm  include '\0'
 */
#define SECUREC_STRCPY_OPT(dest, src, lenWithTerm) do { \
    if ((lenWithTerm) > SECUREC_STRCOPY_THRESHOLD_SIZE) { \
        SECUREC_MEMCPY_WARP_OPT((dest), (src), (lenWithTerm)); \
    } else { \
        SECUREC_SMALL_STR_COPY((dest), (src), (lenWithTerm)); \
    } \
} SECUREC_WHILE_ZERO
#endif

/*
 * Check Src Range
 */
SECUREC_INLINE errno_t CheckSrcRange(char *strDest, size_t destMax, const char *strSrc)
{
    size_t tmpDestMax = destMax;
    const char *tmpSrc = strSrc;
    /* Use destMax as boundary checker and destMax must be greater than zero */
    while (*tmpSrc != '\0' && tmpDestMax > 0) {
        ++tmpSrc;
        --tmpDestMax;
    }
    if (tmpDestMax == 0) {
        strDest[0] = '\0';
        SECUREC_ERROR_INVALID_RANGE("strcpy_s");
        return ERANGE_AND_RESET;
    }
    return EOK;
}

/*
 * Handling errors
 */
errno_t strcpy_error(char *strDest, size_t destMax, const char *strSrc)
{
    if (destMax == 0 || destMax > SECUREC_STRING_MAX_LEN) {
        SECUREC_ERROR_INVALID_RANGE("strcpy_s");
        return ERANGE;
    }
    if (strDest == NULL || strSrc == NULL) {
        SECUREC_ERROR_INVALID_PARAMTER("strcpy_s");
        if (strDest != NULL) {
            strDest[0] = '\0';
            return EINVAL_AND_RESET;
        }
        return EINVAL;
    }
    return CheckSrcRange(strDest, destMax, strSrc);
}

/*
 * <FUNCTION DESCRIPTION>
 *    The strcpy_s function copies the string pointed to  strSrc
 *          (including the terminating null character) into the array pointed to by strDest
 *    The destination string must be large enough to hold the source string,
 *    including the terminating null character. strcpy_s will return EOVERLAP_AND_RESET
 *    if the source and destination strings overlap.
 *
 * <INPUT PARAMETERS>
 *    strDest                          Location of destination string buffer
 *    destMax                        Size of the destination string buffer.
 *    strSrc                            Null-terminated source string buffer.
 *
 * <OUTPUT PARAMETERS>
 *    strDest                         is updated.
 *
 * <RETURN VALUE>
 *    EOK                               Success
 *    EINVAL                          strDest is  NULL and destMax != 0 and destMax <= SECUREC_STRING_MAX_LEN
 *    EINVAL_AND_RESET       strDest !=  NULL and strSrc is NULL and destMax != 0 and destMax <= SECUREC_STRING_MAX_LEN
 *    ERANGE                         destMax is 0 and destMax > SECUREC_STRING_MAX_LEN
 *    ERANGE_AND_RESET      strDest have not enough space  and all other parameters are valid  and not overlap
 *    EOVERLAP_AND_RESET   dest buffer and source buffer are overlapped and all  parameters are valid
 *
 *    If there is a runtime-constraint violation, strDest[0] will be set to the '\0' when strDest and destMax valid
 */
errno_t strcpy_s(char *strDest, size_t destMax, const char *strSrc)
{
    if (SECUREC_STRCPY_PARAM_OK(strDest, destMax, strSrc)) {
        size_t srcStrLen;
        SECUREC_CALC_STR_LEN(strSrc, destMax, &srcStrLen);
        ++srcStrLen; /* The length include '\0' */

        if (srcStrLen <= destMax) {
            /* Use mem overlap check include '\0' */
            if (SECUREC_MEMORY_NO_OVERLAP(strDest, strSrc, srcStrLen)) {
                /* Performance optimization srcStrLen include '\0' */
                SECUREC_STRCPY_OPT(strDest, strSrc, srcStrLen);
                return EOK;
            } else {
                strDest[0] = '\0';
                SECUREC_ERROR_BUFFER_OVERLAP("strcpy_s");
                return EOVERLAP_AND_RESET;
            }
        }
    }
    return strcpy_error(strDest, destMax, strSrc);
}

#if SECUREC_EXPORT_KERNEL_SYMBOL
EXPORT_SYMBOL(strcpy_s);
#endif

