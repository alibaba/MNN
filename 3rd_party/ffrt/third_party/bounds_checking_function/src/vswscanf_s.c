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
 * Description: vswscanf_s  function
 * Create: 2014-02-25
 */

#ifndef SECUREC_FOR_WCHAR
#define SECUREC_FOR_WCHAR
#endif

#include "secinput.h"

SECUREC_INLINE size_t SecWcslen(const wchar_t *s)
{
    const wchar_t *end = s;
    while (*end != L'\0') {
        ++end;
    }
    return ((size_t)((end - s)));
}

/*
 * <FUNCTION DESCRIPTION>
 *    The  vswscanf_s  function  is  the  wide-character  equivalent  of the vsscanf_s function
 *    The vsscanf_s function reads data from buffer into the location given by
 *    each argument. Every argument must be a pointer to a variable with a type
 *    that corresponds to a type specifier in format.
 *    The format argument controls the interpretation of the input fields and
 *    has the same form and function as the format argument for the scanf function.
 *    If copying takes place between strings that overlap, the behavior is undefined.
 *
 * <INPUT PARAMETERS>
 *    buffer                Stored data
 *    format                Format control string, see Format Specifications.
 *    argList               pointer to list of arguments
 *
 * <OUTPUT PARAMETERS>
 *    argList               the converted value stored in user assigned address
 *
 * <RETURN VALUE>
 *    Each of these functions returns the number of fields successfully converted
 *    and assigned; the return value does not include fields that were read but
 *    not assigned. A return value of 0 indicates that no fields were assigned.
 *    return -1 if an error occurs.
 */
int vswscanf_s(const wchar_t *buffer, const wchar_t *format, va_list argList)
{
    size_t count; /* If initialization causes  e838 */
    SecFileStream fStr;
    int retVal;

    /* Validation section */
    if (buffer == NULL || format == NULL) {
        SECUREC_ERROR_INVALID_PARAMTER("vswscanf_s");
        return SECUREC_SCANF_EINVAL;
    }
    count = SecWcslen(buffer);
    if (count == 0 || count > SECUREC_WCHAR_STRING_MAX_LEN) {
        SecClearDestBufW(buffer, format, argList);
        SECUREC_ERROR_INVALID_PARAMTER("vswscanf_s");
        return SECUREC_SCANF_EINVAL;
    }
    SECUREC_FILE_STREAM_FROM_STRING(&fStr, (const char *)buffer, count * sizeof(wchar_t));
    retVal = SecInputSW(&fStr, format, argList);
    if (retVal < 0) {
        SECUREC_ERROR_INVALID_PARAMTER("vswscanf_s");
        return SECUREC_SCANF_EINVAL;
    }
    return retVal;
}

