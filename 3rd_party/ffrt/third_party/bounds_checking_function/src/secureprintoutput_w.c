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
 * Description: By defining corresponding macro for UNICODE string and including "output.inl",
 *              this file generates real underlying function used by printf family API.
 * Create: 2014-02-25
 */

/* If some platforms don't have wchar.h, don't include it */
#if !(defined(SECUREC_VXWORKS_PLATFORM))
/* If there is no macro above, it will cause compiling alarm */
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
#ifndef _CRTIMP_ALTERNATIVE
#define _CRTIMP_ALTERNATIVE     /* Comment microsoft *_s function */
#endif
#ifndef __STDC_WANT_SECURE_LIB__
#define __STDC_WANT_SECURE_LIB__ 0
#endif
#endif
#include <wchar.h>
#endif

/* fix redefined */
#undef SECUREC_ENABLE_WCHAR_FUNC
/* Disable wchar func to clear vs warning */
#define SECUREC_ENABLE_WCHAR_FUNC   0
#define SECUREC_FORMAT_OUTPUT_INPUT 1

#ifndef SECUREC_FOR_WCHAR
#define SECUREC_FOR_WCHAR
#endif

#include "secureprintoutput.h"

#include "output.inl"

