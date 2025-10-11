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
 * Description: gets_s function
 * Create: 2014-02-25
 */

#include "securecutil.h"

/*
 * The parameter size is buffer size in byte
 */
SECUREC_INLINE void SecTrimCRLF(char *buffer, size_t size)
{
    size_t len = strlen(buffer);
    --len; /* Unsigned integer wrapping is accepted and is checked afterwards */
    while (len < size && (buffer[len] == '\r' || buffer[len] == '\n')) {
        buffer[len] = '\0';
        --len; /* Unsigned integer wrapping is accepted and is checked next loop */
    }
}

/*
 * <FUNCTION DESCRIPTION>
 *    The gets_s function reads at most one less than the number of characters
 *    specified by destMax from the std input stream, into the array pointed to by buffer
 *    The line consists of all characters up to and including
 *    the first newline character ('\n'). gets_s then replaces the newline
 *    character with a null character ('\0') before returning the line.
 *    If the first character read is the end-of-file character, a null character
 *    is stored at the beginning of buffer and NULL is returned.
 *
 * <INPUT PARAMETERS>
 *    buffer                         Storage location for input string.
 *    destMax                        The size of the buffer.
 *
 * <OUTPUT PARAMETERS>
 *    buffer                         is updated
 *
 * <RETURN VALUE>
 *    buffer                         Successful operation
 *    NULL                           Improper parameter or read fail
 */
char *gets_s(char *buffer, size_t destMax)
{
#ifdef SECUREC_COMPATIBLE_WIN_FORMAT
    size_t bufferSize = ((destMax == (size_t)(-1)) ? SECUREC_STRING_MAX_LEN : destMax);
#else
    size_t bufferSize = destMax;
#endif

    if (buffer == NULL || bufferSize == 0 || bufferSize > SECUREC_STRING_MAX_LEN) {
        SECUREC_ERROR_INVALID_PARAMTER("gets_s");
        return NULL;
    }

    if (fgets(buffer, (int)bufferSize, SECUREC_STREAM_STDIN) != NULL) {
        SecTrimCRLF(buffer, bufferSize);
        return buffer;
    }

    return NULL;
}

