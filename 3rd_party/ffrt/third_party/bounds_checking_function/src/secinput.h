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
 * Description: Define macro, data struct, and declare function prototype,
 *              which is used by input.inl, secureinput_a.c and secureinput_w.c.
 * Create: 2014-02-25
 */

#ifndef SEC_INPUT_H_E950DA2C_902F_4B15_BECD_948E99090D9C
#define SEC_INPUT_H_E950DA2C_902F_4B15_BECD_948E99090D9C
#include "securecutil.h"

#define SECUREC_SCANF_EINVAL             (-1)
#define SECUREC_SCANF_ERROR_PARA         (-2)

/* For internal stream flag */
#define SECUREC_MEM_STR_FLAG             0x01U
#define SECUREC_FILE_STREAM_FLAG         0x02U
#define SECUREC_PIPE_STREAM_FLAG         0x04U
#define SECUREC_LOAD_FILE_TO_MEM_FLAG    0x08U

#define SECUREC_UCS_BOM_HEADER_SIZE      2U
#define SECUREC_UCS_BOM_HEADER_BE_1ST    0xfeU
#define SECUREC_UCS_BOM_HEADER_BE_2ST    0xffU
#define SECUREC_UCS_BOM_HEADER_LE_1ST    0xffU
#define SECUREC_UCS_BOM_HEADER_LE_2ST    0xfeU
#define SECUREC_UTF8_BOM_HEADER_SIZE     3U
#define SECUREC_UTF8_BOM_HEADER_1ST      0xefU
#define SECUREC_UTF8_BOM_HEADER_2ND      0xbbU
#define SECUREC_UTF8_BOM_HEADER_3RD      0xbfU
#define SECUREC_UTF8_LEAD_1ST            0xe0U
#define SECUREC_UTF8_LEAD_2ND            0x80U

#define SECUREC_BEGIN_WITH_UCS_BOM(s, len) ((len) == SECUREC_UCS_BOM_HEADER_SIZE && \
    (((unsigned char)((s)[0]) == SECUREC_UCS_BOM_HEADER_LE_1ST && \
    (unsigned char)((s)[1]) == SECUREC_UCS_BOM_HEADER_LE_2ST) || \
    ((unsigned char)((s)[0]) == SECUREC_UCS_BOM_HEADER_BE_1ST && \
    (unsigned char)((s)[1]) == SECUREC_UCS_BOM_HEADER_BE_2ST)))

#define SECUREC_BEGIN_WITH_UTF8_BOM(s, len) ((len) == SECUREC_UTF8_BOM_HEADER_SIZE && \
    (unsigned char)((s)[0]) == SECUREC_UTF8_BOM_HEADER_1ST && \
    (unsigned char)((s)[1]) == SECUREC_UTF8_BOM_HEADER_2ND && \
    (unsigned char)((s)[2]) == SECUREC_UTF8_BOM_HEADER_3RD)

#ifdef SECUREC_FOR_WCHAR
#define SECUREC_BOM_HEADER_SIZE SECUREC_UCS_BOM_HEADER_SIZE
#define SECUREC_BEGIN_WITH_BOM(s, len) SECUREC_BEGIN_WITH_UCS_BOM((s), (len))
#else
#define SECUREC_BOM_HEADER_SIZE SECUREC_UTF8_BOM_HEADER_SIZE
#define SECUREC_BEGIN_WITH_BOM(s, len) SECUREC_BEGIN_WITH_UTF8_BOM((s), (len))
#endif

typedef struct {
    unsigned int flag;          /* Mark the properties of input stream */
    char *base;                 /* The pointer to the header of buffered string */
    const char *cur;            /* The pointer to next read position */
    size_t count;               /* The size of buffered string in bytes */
#if SECUREC_ENABLE_SCANF_FILE
    FILE *pf;                   /* The file pointer */
    size_t fileRealRead;
    long oriFilePos;            /* The original position of file offset when fscanf is called */
#endif
} SecFileStream;

#if SECUREC_ENABLE_SCANF_FILE
#define SECUREC_FILE_STREAM_INIT_FILE(stream, fp) do { \
    (stream)->pf = (fp); \
    (stream)->fileRealRead = 0; \
    (stream)->oriFilePos = 0; \
} SECUREC_WHILE_ZERO
#else
/* Disable file */
#define SECUREC_FILE_STREAM_INIT_FILE(stream, fp)
#endif

/* This initialization for eliminating redundant initialization. */
#define SECUREC_FILE_STREAM_FROM_STRING(stream, buf, cnt) do { \
    (stream)->flag = SECUREC_MEM_STR_FLAG; \
    (stream)->base = NULL; \
    (stream)->cur = (buf); \
    (stream)->count = (cnt); \
    SECUREC_FILE_STREAM_INIT_FILE((stream), NULL); \
} SECUREC_WHILE_ZERO

/* This initialization for eliminating redundant initialization. */
#define SECUREC_FILE_STREAM_FROM_FILE(stream, fp) do { \
    (stream)->flag = SECUREC_FILE_STREAM_FLAG; \
    (stream)->base = NULL; \
    (stream)->cur = NULL; \
    (stream)->count = 0; \
    SECUREC_FILE_STREAM_INIT_FILE((stream), (fp)); \
} SECUREC_WHILE_ZERO

/* This initialization for eliminating redundant initialization. */
#define SECUREC_FILE_STREAM_FROM_STDIN(stream) do { \
    (stream)->flag = SECUREC_PIPE_STREAM_FLAG; \
    (stream)->base = NULL; \
    (stream)->cur = NULL; \
    (stream)->count = 0; \
    SECUREC_FILE_STREAM_INIT_FILE((stream), SECUREC_STREAM_STDIN); \
} SECUREC_WHILE_ZERO

#ifdef __cplusplus
extern "C" {
#endif
int SecInputS(SecFileStream *stream, const char *cFormat, va_list argList);
void SecClearDestBuf(const char *buffer, const char *format, va_list argList);
#ifdef SECUREC_FOR_WCHAR
int SecInputSW(SecFileStream *stream, const wchar_t *cFormat, va_list argList);
void SecClearDestBufW(const wchar_t *buffer, const wchar_t *format, va_list argList);
#endif

/* 20150105 For software and hardware decoupling,such as UMG */
#ifdef SECUREC_SYSAPI4VXWORKS
#ifdef feof
#undef feof
#endif
extern int feof(FILE *stream);
#endif

#if defined(SECUREC_SYSAPI4VXWORKS) || defined(SECUREC_CTYPE_MACRO_ADAPT)
#ifndef isspace
#define isspace(c) (((c) == ' ') || ((c) == '\t') || ((c) == '\r') || ((c) == '\n'))
#endif
#ifndef iswspace
#define iswspace(c) (((c) == L' ') || ((c) == L'\t') || ((c) == L'\r') || ((c) == L'\n'))
#endif
#ifndef isascii
#define isascii(c) (((unsigned char)(c)) <= 0x7f)
#endif
#ifndef isupper
#define isupper(c) ((c) >= 'A' && (c) <= 'Z')
#endif
#ifndef islower
#define islower(c) ((c) >= 'a' && (c) <= 'z')
#endif
#ifndef isalpha
#define isalpha(c) (isupper(c) || (islower(c)))
#endif
#ifndef isdigit
#define isdigit(c) ((c) >= '0' && (c) <= '9')
#endif
#ifndef isxupper
#define isxupper(c) ((c) >= 'A' && (c) <= 'F')
#endif
#ifndef isxlower
#define isxlower(c) ((c) >= 'a' && (c) <= 'f')
#endif
#ifndef isxdigit
#define isxdigit(c) (isdigit(c) || isxupper(c) || isxlower(c))
#endif
#endif

#ifdef __cplusplus
}
#endif
/* Reserved file operation macro interface, s is FILE *, i is fileno zero. */
#ifndef SECUREC_LOCK_FILE
#define SECUREC_LOCK_FILE(s)
#endif

#ifndef SECUREC_UNLOCK_FILE
#define SECUREC_UNLOCK_FILE(s)
#endif

#ifndef SECUREC_LOCK_STDIN
#define SECUREC_LOCK_STDIN(i, s)
#endif

#ifndef SECUREC_UNLOCK_STDIN
#define SECUREC_UNLOCK_STDIN(i, s)
#endif
#endif

