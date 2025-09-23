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
 * Description: wcsncpy_s  function
 * Create: 2014-02-25
 */

#include "securecutil.h"

SECUREC_INLINE errno_t SecDoCpyLimitW(wchar_t *strDest, size_t destMax, const wchar_t *strSrc, size_t count)
{
    size_t srcStrLen;
    if (count < destMax) {
        SECUREC_CALC_WSTR_LEN(strSrc, count, &srcStrLen);
    } else {
        SECUREC_CALC_WSTR_LEN(strSrc, destMax, &srcStrLen);
    }
    if (srcStrLen == destMax) {
        strDest[0] = L'\0';
        SECUREC_ERROR_INVALID_RANGE("wcsncpy_s");
        return ERANGE_AND_RESET;
    }
    if (strDest == strSrc) {
        return EOK;
    }
    if (SECUREC_STRING_NO_OVERLAP(strDest, strSrc, srcStrLen)) {
        /* Performance optimization srcStrLen not include '\0' */
        SECUREC_MEMCPY_WARP_OPT(strDest, strSrc, srcStrLen * sizeof(wchar_t));
        *(strDest + srcStrLen) = L'\0';
        return EOK;
    } else {
        strDest[0] = L'\0';
        SECUREC_ERROR_BUFFER_OVERLAP("wcsncpy_s");
        return EOVERLAP_AND_RESET;
    }
}

/*
 * <FUNCTION DESCRIPTION>
 *    The wcsncpy_s function copies not more than n successive wide characters
 *     (not including the terminating null wide character)
 *     from the array pointed to by strSrc to the array pointed to by strDest
 *
 * <INPUT PARAMETERS>
 *    strDest             Destination string.
 *    destMax             The size of the destination string, in characters.
 *    strSrc              Source string.
 *    count                Number of characters to be copied.
 *
 * <OUTPUT PARAMETERS>
 *    strDest              is updated
 *
 * <RETURN VALUE>
 *    EOK                  Success
 *    EINVAL               strDest is  NULL and destMax != 0 and destMax <= SECUREC_WCHAR_STRING_MAX_LEN
 *    EINVAL_AND_RESET     strDest != NULL and strSrc is NULL and destMax != 0
 *                         and destMax <= SECUREC_WCHAR_STRING_MAX_LEN
 *    ERANGE               destMax > SECUREC_WCHAR_STRING_MAX_LEN or destMax is 0
 *    ERANGE_AND_RESET     count > SECUREC_WCHAR_STRING_MAX_LEN or
 *                         (destMax <= length of strSrc and destMax <= count and strDest != strSrc
 *                          and strDest != NULL and strSrc != NULL and destMax != 0 and
 *                          destMax <= SECUREC_WCHAR_STRING_MAX_LEN and not overlap)
 *    EOVERLAP_AND_RESET     dest buffer and source buffer are overlapped and  all  parameters are valid
 *
 *
 *    If there is a runtime-constraint violation, strDest[0] will be set to the '\0' when strDest and destMax valid
 */
errno_t wcsncpy_s(wchar_t *strDest, size_t destMax, const wchar_t *strSrc, size_t count)
{
    if (destMax == 0 || destMax > SECUREC_WCHAR_STRING_MAX_LEN) {
        SECUREC_ERROR_INVALID_RANGE("wcsncpy_s");
        return ERANGE;
    }
    if (strDest == NULL || strSrc == NULL) {
        SECUREC_ERROR_INVALID_PARAMTER("wcsncpy_s");
        if (strDest != NULL) {
            strDest[0] = L'\0';
            return EINVAL_AND_RESET;
        }
        return EINVAL;
    }
    if (count > SECUREC_WCHAR_STRING_MAX_LEN) {
#ifdef SECUREC_COMPATIBLE_WIN_FORMAT
        if (count == (size_t)(-1)) {
            return SecDoCpyLimitW(strDest, destMax, strSrc, destMax - 1);
        }
#endif
        strDest[0] = L'\0';      /* Clear dest string */
        SECUREC_ERROR_INVALID_RANGE("wcsncpy_s");
        return ERANGE_AND_RESET;
    }

    if (count == 0) {
        strDest[0] = L'\0';
        return EOK;
    }

    return SecDoCpyLimitW(strDest, destMax, strSrc, count);
}

