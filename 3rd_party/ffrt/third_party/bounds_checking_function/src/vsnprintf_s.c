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
 * Description: vsnprintf_s  function
 * Create: 2014-02-25
 */

#include "secureprintoutput.h"

#if SECUREC_ENABLE_VSNPRINTF
/*
 * <FUNCTION DESCRIPTION>
 *    The vsnprintf_s function is equivalent to the vsnprintf function
 *     except for the parameter destMax/count and the explicit runtime-constraints violation
 *    The vsnprintf_s function takes a pointer to an argument list, then formats
 *    and writes up to count characters of the given data to the memory pointed
 *    to by strDest and appends a terminating null.
 *
 * <INPUT PARAMETERS>
 *    strDest                  Storage location for the output.
 *    destMax                The size of the strDest for output.
 *    count                    Maximum number of character to write(not including
 *                                the terminating NULL)
 *    format                   Format-control string.
 *    argList                     pointer to list of arguments.
 *
 * <OUTPUT PARAMETERS>
 *    strDest                is updated
 *
 * <RETURN VALUE>
 *    return  the number of characters written, not including the terminating null
 *    return -1 if an  error occurs.
 *    return -1 if count < destMax and the output string  has been truncated
 *
 * If there is a runtime-constraint violation, strDest[0] will be set to the '\0' when strDest and destMax valid
 */
int vsnprintf_s(char *strDest, size_t destMax, size_t count, const char *format, va_list argList)
{
    int retVal;

    if (SECUREC_VSNPRINTF_PARAM_ERROR(format, strDest, destMax, count, SECUREC_STRING_MAX_LEN)) {
        SECUREC_VSPRINTF_CLEAR_DEST(strDest, destMax, SECUREC_STRING_MAX_LEN);
        SECUREC_ERROR_INVALID_PARAMTER("vsnprintf_s");
        return -1;
    }

    if (destMax > count) {
        retVal = SecVsnprintfImpl(strDest, count + 1, format, argList);
        if (retVal == SECUREC_PRINTF_TRUNCATE) {  /* To keep dest buffer not destroyed 2014.2.18 */
            /* The string has been truncated, return  -1 */
            return -1;          /* To skip error handler,  return strlen(strDest) or -1 */
        }
    } else {
        retVal = SecVsnprintfImpl(strDest, destMax, format, argList);
#ifdef SECUREC_COMPATIBLE_WIN_FORMAT
        if (retVal == SECUREC_PRINTF_TRUNCATE && count == (size_t)(-1)) {
            return -1;
        }
#endif
    }

    if (retVal < 0) {
        strDest[0] = '\0';      /* Empty the dest strDest */
        if (retVal == SECUREC_PRINTF_TRUNCATE) {
            /* Buffer too small */
            SECUREC_ERROR_INVALID_RANGE("vsnprintf_s");
        }
        SECUREC_ERROR_INVALID_PARAMTER("vsnprintf_s");
        return -1;
    }

    return retVal;
}
#if SECUREC_EXPORT_KERNEL_SYMBOL
EXPORT_SYMBOL(vsnprintf_s);
#endif
#endif

#if SECUREC_SNPRINTF_TRUNCATED
/*
 * <FUNCTION DESCRIPTION>
 *    The vsnprintf_truncated_s function is equivalent to the vsnprintf function
 *     except for the parameter destMax/count and the explicit runtime-constraints violation
 *    The vsnprintf_truncated_s function takes a pointer to an argument list, then formats
 *    and writes up to count characters of the given data to the memory pointed
 *    to by strDest and appends a terminating null.
 *
 * <INPUT PARAMETERS>
 *    strDest                  Storage location for the output.
 *    destMax                The size of the strDest for output.
 *                                the terminating NULL)
 *    format                   Format-control string.
 *    argList                     pointer to list of arguments.
 *
 * <OUTPUT PARAMETERS>
 *    strDest                is updated
 *
 * <RETURN VALUE>
 *    return  the number of characters written, not including the terminating null
 *    return -1 if an  error occurs.
 *    return destMax-1 if output string  has been truncated
 *
 * If there is a runtime-constraint violation, strDest[0] will be set to the '\0' when strDest and destMax valid
 */
int vsnprintf_truncated_s(char *strDest, size_t destMax, const char *format, va_list argList)
{
    int retVal;

    if (SECUREC_VSPRINTF_PARAM_ERROR(format, strDest, destMax, SECUREC_STRING_MAX_LEN)) {
        SECUREC_VSPRINTF_CLEAR_DEST(strDest, destMax, SECUREC_STRING_MAX_LEN);
        SECUREC_ERROR_INVALID_PARAMTER("vsnprintf_truncated_s");
        return -1;
    }

    retVal = SecVsnprintfImpl(strDest, destMax, format, argList);
    if (retVal < 0) {
        if (retVal == SECUREC_PRINTF_TRUNCATE) {
            return (int)(destMax - 1);  /* To skip error handler,  return strlen(strDest) */
        }
        strDest[0] = '\0';      /* Empty the dest strDest */
        SECUREC_ERROR_INVALID_PARAMTER("vsnprintf_truncated_s");
        return -1;
    }

    return retVal;
}
#if SECUREC_EXPORT_KERNEL_SYMBOL
EXPORT_SYMBOL(vsnprintf_truncated_s);
#endif
#endif

