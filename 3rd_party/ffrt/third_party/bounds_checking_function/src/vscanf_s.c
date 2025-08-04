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
 * Description: vscanf_s  function
 * Create: 2014-02-25
 */

#include "secinput.h"

/*
 * <FUNCTION DESCRIPTION>
 *     The vscanf_s function is equivalent to scanf_s, with the variable argument list replaced by argList,
 *     The vscanf_s function reads data from the standard input stream stdin and
 *    writes the data into the location that's given by argument. Each argument
 *    must be a pointer to a variable of a type that corresponds to a type specifier
 *    in format. If copying occurs between strings that overlap, the behavior is
 *    undefined.
 *
 * <INPUT PARAMETERS>
 *    format                Format control string.
 *    argList               pointer to list of arguments
 *
 * <OUTPUT PARAMETERS>
 *    argList               the converted value stored in user assigned address
 *
 * <RETURN VALUE>
 *    Returns the number of fields successfully converted and assigned;
 *    the return value does not include fields that were read but not assigned.
 *    A return value of 0 indicates that no fields were assigned.
 *    return -1 if an error occurs.
 */
int vscanf_s(const char *format, va_list argList)
{
    int retVal;                 /* If initialization causes  e838 */
    SecFileStream fStr;
    SECUREC_FILE_STREAM_FROM_STDIN(&fStr);
    /*
     * The "va_list" has different definition on different platform, so we can't use argList == NULL
     * To determine it's invalid. If you has fixed platform, you can check some fields to validate it,
     * such as "argList == NULL" or argList.xxx != NULL or *(size_t *)&argList != 0.
     */
#if SECUREC_ENABLE_SCANF_FILE
    if (format == NULL || fStr.pf == NULL) {
#else
    if (format == NULL) {
#endif
        SECUREC_ERROR_INVALID_PARAMTER("vscanf_s");
        return SECUREC_SCANF_EINVAL;
    }

#if SECUREC_ENABLE_SCANF_FILE
    SECUREC_LOCK_STDIN(0, fStr.pf);
#endif
    retVal = SecInputS(&fStr, format, argList);
#if SECUREC_ENABLE_SCANF_FILE
    SECUREC_UNLOCK_STDIN(0, fStr.pf);
#endif
    if (retVal < 0) {
        SECUREC_ERROR_INVALID_PARAMTER("vscanf_s");
        return SECUREC_SCANF_EINVAL;
    }
    return retVal;
}

