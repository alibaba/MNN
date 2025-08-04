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
 * Description: wcscpy_s  function
 * Create: 2014-02-25
 */

#include "securecutil.h"

SECUREC_INLINE errno_t SecDoCpyW(wchar_t *strDest, size_t destMax, const wchar_t *strSrc)
{
    size_t srcStrLen;
    SECUREC_CALC_WSTR_LEN(strSrc, destMax, &srcStrLen);

    if (srcStrLen == destMax) {
        strDest[0] = L'\0';
        SECUREC_ERROR_INVALID_RANGE("wcscpy_s");
        return ERANGE_AND_RESET;
    }
    if (strDest == strSrc) {
        return EOK;
    }

    if (SECUREC_STRING_NO_OVERLAP(strDest, strSrc, srcStrLen)) {
        /* Performance optimization, srcStrLen is single character length  include '\0' */
        SECUREC_MEMCPY_WARP_OPT(strDest, strSrc, (srcStrLen + 1) * sizeof(wchar_t));
        return EOK;
    } else {
        strDest[0] = L'\0';
        SECUREC_ERROR_BUFFER_OVERLAP("wcscpy_s");
        return EOVERLAP_AND_RESET;
    }
}

/*
 * <FUNCTION DESCRIPTION>
 *   The wcscpy_s function copies the wide string pointed to by strSrc
 *   (including the terminating null wide character) into the array pointed to by strDest

 * <INPUT PARAMETERS>
 *    strDest               Destination string buffer
 *    destMax               Size of the destination string buffer.
 *    strSrc                Null-terminated source string buffer.
 *
 * <OUTPUT PARAMETERS>
 *    strDest               is updated.
 *
 * <RETURN VALUE>
 *    EOK                   Success
 *    EINVAL                strDest is  NULL and destMax != 0 and destMax <= SECUREC_WCHAR_STRING_MAX_LEN
 *    EINVAL_AND_RESET      strDest != NULL and strSrc is NULL and destMax != 0
 *                          and destMax <= SECUREC_WCHAR_STRING_MAX_LEN
 *    ERANGE                destMax > SECUREC_WCHAR_STRING_MAX_LEN or destMax is 0
 *    ERANGE_AND_RESET      destMax <= length of strSrc and strDest != strSrc
 *                          and strDest != NULL and strSrc != NULL and destMax != 0
 *                          and destMax <= SECUREC_WCHAR_STRING_MAX_LEN and not overlap
 *    EOVERLAP_AND_RESET    dest buffer and source buffer are overlapped and destMax != 0
 *                          and destMax <= SECUREC_WCHAR_STRING_MAX_LEN
 *                          and strDest != NULL and strSrc !=NULL and strDest != strSrc
 *
 *    If there is a runtime-constraint violation, strDest[0] will be set to the '\0' when strDest and destMax valid
 */
errno_t wcscpy_s(wchar_t *strDest, size_t destMax, const wchar_t *strSrc)
{
    if (destMax == 0 || destMax > SECUREC_WCHAR_STRING_MAX_LEN) {
        SECUREC_ERROR_INVALID_RANGE("wcscpy_s");
        return ERANGE;
    }
    if (strDest == NULL || strSrc == NULL) {
        SECUREC_ERROR_INVALID_PARAMTER("wcscpy_s");
        if (strDest != NULL) {
            strDest[0] = L'\0';
            return EINVAL_AND_RESET;
        }
        return EINVAL;
    }
    return SecDoCpyW(strDest, destMax, strSrc);
}

