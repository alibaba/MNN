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
 * Description: By defining data type for ANSI string and including "input.inl",
 *              this file generates real underlying function used by scanf family API.
 * Create: 2014-02-25
 */

#define SECUREC_FORMAT_OUTPUT_INPUT 1
#ifdef SECUREC_FOR_WCHAR
#undef SECUREC_FOR_WCHAR
#endif

#include "secinput.h"

#include "input.inl"

SECUREC_INLINE int SecIsDigit(SecInt ch)
{
    /* SecInt to unsigned char clear  571, use bit mask to clear negative return of ch */
    return isdigit((int)((unsigned int)(unsigned char)(ch) & 0xffU));
}
SECUREC_INLINE int SecIsXdigit(SecInt ch)
{
    return isxdigit((int)((unsigned int)(unsigned char)(ch) & 0xffU));
}
SECUREC_INLINE int SecIsSpace(SecInt ch)
{
    return isspace((int)((unsigned int)(unsigned char)(ch) & 0xffU));
}

