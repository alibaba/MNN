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
 * Description: wmemmove_s  function
 * Create: 2014-02-25
 */
/*
 * [Standardize-exceptions] Use unsafe function: Portability
 * [reason] Use unsafe function to implement security function to maintain platform compatibility.
 *          And sufficient input validation is performed before calling
 */

#include "securecutil.h"

/*
 * <FUNCTION DESCRIPTION>
 *   The wmemmove_s function copies n successive wide characters from the object pointed
 *   to by src into the object pointed to by dest.
 *
 * <INPUT PARAMETERS>
 *    dest                     Destination buffer.
 *    destMax                  Size of the destination buffer.
 *    src                      Source object.
 *    count                    Number of bytes or character to copy.
 *
 * <OUTPUT PARAMETERS>
 *    dest                     is updated.
 *
 * <RETURN VALUE>
 *    EOK                      Success
 *    EINVAL                   dest is  NULL and destMax != 0 and count <= destMax
 *                             and destMax <= SECUREC_WCHAR_MEM_MAX_LEN
 *    EINVAL_AND_RESET         dest != NULL and src is NULL and destMax != 0
 *                             and destMax <= SECUREC_WCHAR_MEM_MAX_LEN and count <= destMax
 *    ERANGE                   destMax > SECUREC_WCHAR_MEM_MAX_LEN or destMax is 0 or
 *                             (count > destMax and dest is  NULL and destMax != 0
 *                             and destMax <= SECUREC_WCHAR_MEM_MAX_LEN)
 *    ERANGE_AND_RESET        count > destMax and dest  !=  NULL and destMax != 0
 *                             and destMax <= SECUREC_WCHAR_MEM_MAX_LEN
 *
 *
 *     If an error occurred, dest will  be filled with 0 when dest and destMax valid.
 *     If some regions of the source area and the destination overlap, wmemmove_s
 *     ensures that the original source bytes in the overlapping region are copied
 *     before being overwritten
 */
errno_t wmemmove_s(wchar_t *dest, size_t destMax, const wchar_t *src, size_t count)
{
    if (destMax == 0 || destMax > SECUREC_WCHAR_MEM_MAX_LEN) {
        SECUREC_ERROR_INVALID_PARAMTER("wmemmove_s");
        return ERANGE;
    }
    if (count > destMax) {
        SECUREC_ERROR_INVALID_PARAMTER("wmemmove_s");
        if (dest != NULL) {
            (void)SECUREC_MEMSET_FUNC_OPT(dest, 0, destMax * sizeof(wchar_t));
            return ERANGE_AND_RESET;
        }
        return ERANGE;
    }
    return memmove_s(dest, destMax * sizeof(wchar_t), src, count * sizeof(wchar_t));
}

