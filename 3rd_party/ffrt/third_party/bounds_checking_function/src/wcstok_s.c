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
 * Description: wcstok_s  function
 * Create: 2014-02-25
 */

#include "securecutil.h"

SECUREC_INLINE int SecIsInDelimitW(wchar_t ch, const wchar_t *strDelimit)
{
    const wchar_t *ctl = strDelimit;
    while (*ctl != L'\0' && *ctl != ch) {
        ++ctl;
    }
    return (int)(*ctl != L'\0');
}

/*
 * Find beginning of token (skip over leading delimiters).
 * Note that there is no token if this loop sets string to point to the terminal null.
 */
SECUREC_INLINE wchar_t *SecFindBeginW(wchar_t *strToken, const wchar_t *strDelimit)
{
    wchar_t *token = strToken;
    while (*token != L'\0') {
        if (SecIsInDelimitW(*token, strDelimit) != 0) {
            ++token;
            continue;
        }
        /* Don't find any delimiter in string header, break the loop */
        break;
    }
    return token;
}

/*
 * Find the end of the token. If it is not the end of the string, put a null there.
 */
SECUREC_INLINE wchar_t *SecFindRestW(wchar_t *strToken, const wchar_t *strDelimit)
{
    wchar_t *token = strToken;
    while (*token != L'\0') {
        if (SecIsInDelimitW(*token, strDelimit) != 0) {
            /* Find a delimiter, set string terminator */
            *token = L'\0';
            ++token;
            break;
        }
        ++token;
    }
    return token;
}

/*
 * Update Token wide character  function
 */
SECUREC_INLINE wchar_t *SecUpdateTokenW(wchar_t *strToken, const wchar_t *strDelimit, wchar_t **context)
{
    /* Point to updated position.  Record string position for next search in the context */
    *context = SecFindRestW(strToken, strDelimit);
    /* Determine if a token has been found */
    if (*context == strToken) {
        return NULL;
    }
    return strToken;
}

/*
 * <NAME>
 *    wcstok_s
 *
 *
 * <FUNCTION DESCRIPTION>
 *   The  wcstok_s  function  is  the  wide-character  equivalent  of the strtok_s function
 *
 * <INPUT PARAMETERS>
 *    strToken               String containing token or tokens.
 *    strDelimit             Set of delimiter characters.
 *    context                Used to store position information between calls to
 *                               wcstok_s.
 *
 * <OUTPUT PARAMETERS>
 *    context               is updated
 * <RETURN VALUE>
 *   The  wcstok_s  function  is  the  wide-character  equivalent  of the strtok_s function
 */
wchar_t *wcstok_s(wchar_t *strToken, const wchar_t *strDelimit, wchar_t **context)
{
    wchar_t *orgToken = strToken;
    /* Validation section */
    if (context == NULL || strDelimit == NULL) {
        return NULL;
    }
    if (orgToken == NULL && *context == NULL) {
        return NULL;
    }
    /* If string==NULL, continue with previous string */
    if (orgToken == NULL) {
        orgToken = *context;
    }
    orgToken = SecFindBeginW(orgToken, strDelimit);
    return SecUpdateTokenW(orgToken, strDelimit, context);
}

