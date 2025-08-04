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
 * Description: Define internal used macro and data type. The marco of SECUREC_ON_64BITS
 *              will be determined in this header file, which is a switch for part
 *              of code. Some macro are used to suppress warning by MS compiler.
 * Create: 2014-02-25
 * Notes: User can change the value of SECUREC_STRING_MAX_LEN and SECUREC_MEM_MAX_LEN
 *        macro to meet their special need, but The maximum value should not exceed 2G.
 */
/*
 * [Standardize-exceptions]: Performance-sensitive
 * [reason]: Strict parameter verification has been done before use
 */

#ifndef SECURECTYPE_H_A7BBB686_AADA_451B_B9F9_44DACDAE18A7
#define SECURECTYPE_H_A7BBB686_AADA_451B_B9F9_44DACDAE18A7

#ifndef SECUREC_USING_STD_SECURE_LIB
#if defined(_MSC_VER) && _MSC_VER >= 1400
#if defined(__STDC_WANT_SECURE_LIB__) && (!__STDC_WANT_SECURE_LIB__)
/* Security functions have been provided since vs2005, default use of system library functions */
#define SECUREC_USING_STD_SECURE_LIB    0
#else
#define SECUREC_USING_STD_SECURE_LIB    1
#endif
#else
#define SECUREC_USING_STD_SECURE_LIB    0
#endif
#endif

/* Compatibility with older Secure C versions, shielding VC symbol redefinition warning */
#if defined(_MSC_VER) && (_MSC_VER >= 1400) && (!SECUREC_USING_STD_SECURE_LIB)
#ifndef SECUREC_DISABLE_CRT_FUNC
#define SECUREC_DISABLE_CRT_FUNC        1
#endif
#ifndef SECUREC_DISABLE_CRT_IMP
#define SECUREC_DISABLE_CRT_IMP         1
#endif
#else /*  MSC VER */
#ifndef SECUREC_DISABLE_CRT_FUNC
#define SECUREC_DISABLE_CRT_FUNC        0
#endif
#ifndef SECUREC_DISABLE_CRT_IMP
#define SECUREC_DISABLE_CRT_IMP         0
#endif
#endif

#if SECUREC_DISABLE_CRT_FUNC
#ifdef __STDC_WANT_SECURE_LIB__
#undef __STDC_WANT_SECURE_LIB__
#endif
#define __STDC_WANT_SECURE_LIB__        0
#endif

#if SECUREC_DISABLE_CRT_IMP
#ifdef _CRTIMP_ALTERNATIVE
#undef _CRTIMP_ALTERNATIVE
#endif
#define _CRTIMP_ALTERNATIVE     /* Comment Microsoft *_s function */
#endif

/* Compile in kernel under macro control */
#ifndef SECUREC_IN_KERNEL
#ifdef __KERNEL__
#define SECUREC_IN_KERNEL               1
#else
#define SECUREC_IN_KERNEL               0
#endif
#endif

/* make kernel symbols of functions available to loadable modules */
#ifndef SECUREC_EXPORT_KERNEL_SYMBOL
#if SECUREC_IN_KERNEL
#define SECUREC_EXPORT_KERNEL_SYMBOL    1
#else
#define SECUREC_EXPORT_KERNEL_SYMBOL    0
#endif
#endif

#if SECUREC_IN_KERNEL
#ifndef SECUREC_ENABLE_SCANF_FILE
#define SECUREC_ENABLE_SCANF_FILE       0
#endif
#ifndef SECUREC_ENABLE_WCHAR_FUNC
#define SECUREC_ENABLE_WCHAR_FUNC       0
#endif
#else /* SECUREC_IN_KERNEL */
#ifndef SECUREC_ENABLE_SCANF_FILE
#define SECUREC_ENABLE_SCANF_FILE       1
#endif
#ifndef SECUREC_ENABLE_WCHAR_FUNC
#define SECUREC_ENABLE_WCHAR_FUNC       1
#endif
#endif

/* Default secure function declaration, default declarations for non-standard functions */
#ifndef SECUREC_SNPRINTF_TRUNCATED
#define SECUREC_SNPRINTF_TRUNCATED      1
#endif

#if SECUREC_USING_STD_SECURE_LIB
#if defined(_MSC_VER) && _MSC_VER >= 1400
/* Declare secure functions that are not available in the VS compiler */
#ifndef SECUREC_ENABLE_MEMSET
#define SECUREC_ENABLE_MEMSET           1
#endif
/* VS 2005 have vsnprintf_s function */
#ifndef SECUREC_ENABLE_VSNPRINTF
#define SECUREC_ENABLE_VSNPRINTF        0
#endif
#ifndef SECUREC_ENABLE_SNPRINTF
/* VS 2005 have vsnprintf_s function Adapt the snprintf_s of the security function */
#define snprintf_s _snprintf_s
#define SECUREC_ENABLE_SNPRINTF         0
#endif
/* Before VS 2010 do not have v functions */
#if _MSC_VER <= 1600 || defined(SECUREC_FOR_V_SCANFS)
#ifndef SECUREC_ENABLE_VFSCANF
#define SECUREC_ENABLE_VFSCANF          1
#endif
#ifndef SECUREC_ENABLE_VSCANF
#define SECUREC_ENABLE_VSCANF           1
#endif
#ifndef SECUREC_ENABLE_VSSCANF
#define SECUREC_ENABLE_VSSCANF          1
#endif
#endif

#else /* MSC VER */
#ifndef SECUREC_ENABLE_MEMSET
#define SECUREC_ENABLE_MEMSET           0
#endif
#ifndef SECUREC_ENABLE_SNPRINTF
#define SECUREC_ENABLE_SNPRINTF         0
#endif
#ifndef SECUREC_ENABLE_VSNPRINTF
#define SECUREC_ENABLE_VSNPRINTF        0
#endif
#endif

#ifndef SECUREC_ENABLE_MEMMOVE
#define SECUREC_ENABLE_MEMMOVE          0
#endif
#ifndef SECUREC_ENABLE_MEMCPY
#define SECUREC_ENABLE_MEMCPY           0
#endif
#ifndef SECUREC_ENABLE_STRCPY
#define SECUREC_ENABLE_STRCPY           0
#endif
#ifndef SECUREC_ENABLE_STRNCPY
#define SECUREC_ENABLE_STRNCPY          0
#endif
#ifndef SECUREC_ENABLE_STRCAT
#define SECUREC_ENABLE_STRCAT           0
#endif
#ifndef SECUREC_ENABLE_STRNCAT
#define SECUREC_ENABLE_STRNCAT          0
#endif
#ifndef SECUREC_ENABLE_SPRINTF
#define SECUREC_ENABLE_SPRINTF          0
#endif
#ifndef SECUREC_ENABLE_VSPRINTF
#define SECUREC_ENABLE_VSPRINTF          0
#endif
#ifndef SECUREC_ENABLE_SSCANF
#define SECUREC_ENABLE_SSCANF           0
#endif
#ifndef SECUREC_ENABLE_VSSCANF
#define SECUREC_ENABLE_VSSCANF          0
#endif
#ifndef SECUREC_ENABLE_SCANF
#define SECUREC_ENABLE_SCANF            0
#endif
#ifndef SECUREC_ENABLE_VSCANF
#define SECUREC_ENABLE_VSCANF           0
#endif

#ifndef SECUREC_ENABLE_FSCANF
#define SECUREC_ENABLE_FSCANF           0
#endif
#ifndef SECUREC_ENABLE_VFSCANF
#define SECUREC_ENABLE_VFSCANF          0
#endif
#ifndef SECUREC_ENABLE_STRTOK
#define SECUREC_ENABLE_STRTOK           0
#endif
#ifndef SECUREC_ENABLE_GETS
#define SECUREC_ENABLE_GETS             0
#endif

#else /* SECUREC USE STD SECURE LIB */

#ifndef SECUREC_ENABLE_MEMSET
#define SECUREC_ENABLE_MEMSET           1
#endif
#ifndef SECUREC_ENABLE_MEMMOVE
#define SECUREC_ENABLE_MEMMOVE          1
#endif
#ifndef SECUREC_ENABLE_MEMCPY
#define SECUREC_ENABLE_MEMCPY           1
#endif
#ifndef SECUREC_ENABLE_STRCPY
#define SECUREC_ENABLE_STRCPY           1
#endif
#ifndef SECUREC_ENABLE_STRNCPY
#define SECUREC_ENABLE_STRNCPY          1
#endif
#ifndef SECUREC_ENABLE_STRCAT
#define SECUREC_ENABLE_STRCAT           1
#endif
#ifndef SECUREC_ENABLE_STRNCAT
#define SECUREC_ENABLE_STRNCAT          1
#endif
#ifndef SECUREC_ENABLE_SPRINTF
#define SECUREC_ENABLE_SPRINTF          1
#endif
#ifndef SECUREC_ENABLE_VSPRINTF
#define SECUREC_ENABLE_VSPRINTF          1
#endif
#ifndef SECUREC_ENABLE_SNPRINTF
#define SECUREC_ENABLE_SNPRINTF         1
#endif
#ifndef SECUREC_ENABLE_VSNPRINTF
#define SECUREC_ENABLE_VSNPRINTF        1
#endif
#ifndef SECUREC_ENABLE_SSCANF
#define SECUREC_ENABLE_SSCANF           1
#endif
#ifndef SECUREC_ENABLE_VSSCANF
#define SECUREC_ENABLE_VSSCANF          1
#endif
#ifndef SECUREC_ENABLE_SCANF
#if SECUREC_ENABLE_SCANF_FILE
#define SECUREC_ENABLE_SCANF            1
#else
#define SECUREC_ENABLE_SCANF            0
#endif
#endif
#ifndef SECUREC_ENABLE_VSCANF
#if SECUREC_ENABLE_SCANF_FILE
#define SECUREC_ENABLE_VSCANF           1
#else
#define SECUREC_ENABLE_VSCANF           0
#endif
#endif

#ifndef SECUREC_ENABLE_FSCANF
#if SECUREC_ENABLE_SCANF_FILE
#define SECUREC_ENABLE_FSCANF           1
#else
#define SECUREC_ENABLE_FSCANF           0
#endif
#endif
#ifndef SECUREC_ENABLE_VFSCANF
#if SECUREC_ENABLE_SCANF_FILE
#define SECUREC_ENABLE_VFSCANF          1
#else
#define SECUREC_ENABLE_VFSCANF          0
#endif
#endif

#ifndef SECUREC_ENABLE_STRTOK
#define SECUREC_ENABLE_STRTOK           1
#endif
#ifndef SECUREC_ENABLE_GETS
#define SECUREC_ENABLE_GETS             1
#endif
#endif /* SECUREC_USE_STD_SECURE_LIB */

#if !SECUREC_ENABLE_SCANF_FILE
#if SECUREC_ENABLE_FSCANF
#undef SECUREC_ENABLE_FSCANF
#define SECUREC_ENABLE_FSCANF           0
#endif
#if SECUREC_ENABLE_VFSCANF
#undef SECUREC_ENABLE_VFSCANF
#define SECUREC_ENABLE_VFSCANF          0
#endif
#if SECUREC_ENABLE_SCANF
#undef SECUREC_ENABLE_SCANF
#define SECUREC_ENABLE_SCANF            0
#endif
#if SECUREC_ENABLE_FSCANF
#undef SECUREC_ENABLE_FSCANF
#define SECUREC_ENABLE_FSCANF           0
#endif

#endif

#if SECUREC_IN_KERNEL
#include <linux/kernel.h>
#include <linux/module.h>
#else
#ifndef SECUREC_HAVE_STDIO_H
#define SECUREC_HAVE_STDIO_H 1
#endif
#ifndef SECUREC_HAVE_STRING_H
#define SECUREC_HAVE_STRING_H 1
#endif
#ifndef SECUREC_HAVE_STDLIB_H
#define SECUREC_HAVE_STDLIB_H 1
#endif
#if SECUREC_HAVE_STDIO_H
#include <stdio.h>
#endif
#if SECUREC_HAVE_STRING_H
#include <string.h>
#endif
#if SECUREC_HAVE_STDLIB_H
#include <stdlib.h>
#endif
#endif

/*
 * If you need high performance, enable the SECUREC_WITH_PERFORMANCE_ADDONS macro, default is enable.
 * The macro is automatically closed on the windows platform and linux kernel
 */
#ifndef SECUREC_WITH_PERFORMANCE_ADDONS
#if SECUREC_IN_KERNEL
#define SECUREC_WITH_PERFORMANCE_ADDONS 0
#else
#define SECUREC_WITH_PERFORMANCE_ADDONS 1
#endif
#endif

/* If enable SECUREC_COMPATIBLE_WIN_FORMAT, the output format will be compatible to Windows. */
#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)) && !defined(SECUREC_COMPATIBLE_LINUX_FORMAT)
#ifndef SECUREC_COMPATIBLE_WIN_FORMAT
#define SECUREC_COMPATIBLE_WIN_FORMAT
#endif
#endif

#if defined(SECUREC_COMPATIBLE_WIN_FORMAT)
/* On windows platform, can't use optimized function for there is no __builtin_constant_p like function */
/* If need optimized macro, can define this: define __builtin_constant_p(x) 0 */
#ifdef SECUREC_WITH_PERFORMANCE_ADDONS
#undef SECUREC_WITH_PERFORMANCE_ADDONS
#define SECUREC_WITH_PERFORMANCE_ADDONS 0
#endif
#endif

#if defined(__VXWORKS__) || defined(__vxworks) || defined(__VXWORKS) || defined(_VXWORKS_PLATFORM_)  || \
    defined(SECUREC_VXWORKS_VERSION_5_4)
#ifndef SECUREC_VXWORKS_PLATFORM
#define SECUREC_VXWORKS_PLATFORM
#endif
#endif

/* If enable SECUREC_COMPATIBLE_LINUX_FORMAT, the output format will be compatible to Linux. */
#if !defined(SECUREC_COMPATIBLE_WIN_FORMAT) && !defined(SECUREC_VXWORKS_PLATFORM)
#ifndef SECUREC_COMPATIBLE_LINUX_FORMAT
#define SECUREC_COMPATIBLE_LINUX_FORMAT
#endif
#endif

#ifdef SECUREC_COMPATIBLE_LINUX_FORMAT
#ifndef SECUREC_HAVE_STDDEF_H
#define SECUREC_HAVE_STDDEF_H 1
#endif
/* Some system may no stddef.h */
#if SECUREC_HAVE_STDDEF_H
#if !SECUREC_IN_KERNEL
#include <stddef.h>
#endif
#endif
#endif

/*
 * Add  the -DSECUREC_SUPPORT_FORMAT_WARNING=1  compiler option to supoort  -Wformat=2.
 * Default does not check the format is that the same data type in the actual code.
 * In the product is different in the original data type definition of VxWorks and Linux.
 */
#ifndef SECUREC_SUPPORT_FORMAT_WARNING
#define SECUREC_SUPPORT_FORMAT_WARNING 0
#endif

#if SECUREC_SUPPORT_FORMAT_WARNING
#define SECUREC_ATTRIBUTE(x, y)  __attribute__((format(printf, (x), (y))))
#else
#define SECUREC_ATTRIBUTE(x, y)
#endif

/*
 * Add the -DSECUREC_SUPPORT_BUILTIN_EXPECT=0 compiler option, if compiler can not support __builtin_expect.
 */
#ifndef SECUREC_SUPPORT_BUILTIN_EXPECT
#define SECUREC_SUPPORT_BUILTIN_EXPECT 1
#endif

#if SECUREC_SUPPORT_BUILTIN_EXPECT && defined(__GNUC__) && ((__GNUC__ > 3) || \
    (defined(__GNUC_MINOR__) && (__GNUC__ == 3 && __GNUC_MINOR__ > 3)))
/*
 * This is a built-in function that can be used without a declaration, if warning for declaration not found occurred,
 * you can add -DSECUREC_NEED_BUILTIN_EXPECT_DECLARE to compiler options
 */
#ifdef SECUREC_NEED_BUILTIN_EXPECT_DECLARE
long __builtin_expect(long exp, long c);
#endif

#define SECUREC_LIKELY(x) __builtin_expect(!!(x), 1)
#define SECUREC_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define SECUREC_LIKELY(x) (x)
#define SECUREC_UNLIKELY(x) (x)
#endif

/* Define the max length of the string */
#ifndef SECUREC_STRING_MAX_LEN
#define SECUREC_STRING_MAX_LEN 0x7fffffffUL
#endif
#define SECUREC_WCHAR_STRING_MAX_LEN (SECUREC_STRING_MAX_LEN / sizeof(wchar_t))

/* Add SECUREC_MEM_MAX_LEN for memcpy and memmove */
#ifndef SECUREC_MEM_MAX_LEN
#define SECUREC_MEM_MAX_LEN 0x7fffffffUL
#endif
#define SECUREC_WCHAR_MEM_MAX_LEN (SECUREC_MEM_MAX_LEN / sizeof(wchar_t))

#if SECUREC_STRING_MAX_LEN > 0x7fffffffUL
#error "max string is 2G"
#endif

#if (defined(__GNUC__) && defined(__SIZEOF_POINTER__))
#if (__SIZEOF_POINTER__ != 4) && (__SIZEOF_POINTER__ != 8)
#error "unsupported system"
#endif
#endif

#if defined(_WIN64) || defined(WIN64) || defined(__LP64__) || defined(_LP64)
#define SECUREC_ON_64BITS
#endif

#if (!defined(SECUREC_ON_64BITS) && defined(__GNUC__) && defined(__SIZEOF_POINTER__))
#if __SIZEOF_POINTER__ == 8
#define SECUREC_ON_64BITS
#endif
#endif

#if defined(__SVR4) || defined(__svr4__)
#define SECUREC_ON_SOLARIS
#endif

#if (defined(__hpux) || defined(_AIX) || defined(SECUREC_ON_SOLARIS))
#define SECUREC_ON_UNIX
#endif

/*
 * Codes should run under the macro SECUREC_COMPATIBLE_LINUX_FORMAT in unknown system on default,
 * and strtold.
 * The function strtold is referenced first at ISO9899:1999(C99), and some old compilers can
 * not support these functions. Here provides a macro to open these functions:
 * SECUREC_SUPPORT_STRTOLD  -- If defined, strtold will be used
 */
#ifndef SECUREC_SUPPORT_STRTOLD
#define SECUREC_SUPPORT_STRTOLD 0
#if (defined(SECUREC_COMPATIBLE_LINUX_FORMAT))
#if defined(__USE_ISOC99)  || \
    (defined(_AIX) && defined(_ISOC99_SOURCE)) || \
    (defined(__hpux) && defined(__ia64)) || \
    (defined(SECUREC_ON_SOLARIS) && (!defined(_STRICT_STDC) && !defined(__XOPEN_OR_POSIX)) || \
    defined(_STDC_C99) || defined(__EXTENSIONS__))
#undef  SECUREC_SUPPORT_STRTOLD
#define SECUREC_SUPPORT_STRTOLD 1
#endif
#endif
#if ((defined(SECUREC_WRLINUX_BELOW4) || defined(_WRLINUX_BELOW4_)))
#undef  SECUREC_SUPPORT_STRTOLD
#define SECUREC_SUPPORT_STRTOLD 0
#endif
#endif

#if SECUREC_WITH_PERFORMANCE_ADDONS

#ifndef SECUREC_TWO_MIN
#define SECUREC_TWO_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

/* For strncpy_s performance optimization */
#define SECUREC_STRNCPY_SM(dest, destMax, src, count) \
    (((void *)(dest) != NULL && (const void *)(src) != NULL && (size_t)(destMax) > 0 && \
    (((unsigned long long)(destMax) & (unsigned long long)(-2)) < SECUREC_STRING_MAX_LEN) && \
    (SECUREC_TWO_MIN((size_t)(count), strlen(src)) + 1) <= (size_t)(destMax)) ? \
    (((size_t)(count) < strlen(src)) ? (memcpy((dest), (src), (count)), *((char *)(dest) + (count)) = '\0', EOK) : \
    (memcpy((dest), (src), strlen(src) + 1), EOK)) : (strncpy_error((dest), (destMax), (src), (count))))

#define SECUREC_STRCPY_SM(dest, destMax, src) \
    (((void *)(dest) != NULL && (const void *)(src) != NULL && (size_t)(destMax) > 0 && \
    (((unsigned long long)(destMax) & (unsigned long long)(-2)) < SECUREC_STRING_MAX_LEN) && \
    (strlen(src) + 1) <= (size_t)(destMax)) ? (memcpy((dest), (src), strlen(src) + 1), EOK) : \
    (strcpy_error((dest), (destMax), (src))))

/* For strcat_s performance optimization */
#if defined(__GNUC__)
#define SECUREC_STRCAT_SM(dest, destMax, src) ({ \
    int catRet_ = EOK; \
    if ((void *)(dest) != NULL && (const void *)(src) != NULL && (size_t)(destMax) > 0 && \
        (((unsigned long long)(destMax) & (unsigned long long)(-2)) < SECUREC_STRING_MAX_LEN)) { \
        char *catTmpDst_ = (char *)(dest); \
        size_t catRestSize_ = (destMax); \
        while (catRestSize_ > 0 && *catTmpDst_ != '\0') { \
            ++catTmpDst_; \
            --catRestSize_; \
        } \
        if (catRestSize_ == 0) { \
            catRet_ = EINVAL; \
        } else if ((strlen(src) + 1) <= catRestSize_) { \
            memcpy(catTmpDst_, (src), strlen(src) + 1); \
            catRet_ = EOK; \
        } else { \
            catRet_ = ERANGE; \
        } \
        if (catRet_ != EOK) { \
            catRet_ = strcat_s((dest), (destMax), (src)); \
        } \
    } else { \
        catRet_ = strcat_s((dest), (destMax), (src)); \
    } \
    catRet_; \
})
#else
#define SECUREC_STRCAT_SM(dest, destMax, src) strcat_s((dest), (destMax), (src))
#endif

/* For strncat_s performance optimization */
#if defined(__GNUC__)
#define SECUREC_STRNCAT_SM(dest, destMax, src, count) ({ \
    int ncatRet_ = EOK; \
    if ((void *)(dest) != NULL && (const void *)(src) != NULL && (size_t)(destMax) > 0 && \
        (((unsigned long long)(destMax) & (unsigned long long)(-2)) < SECUREC_STRING_MAX_LEN)  && \
        (((unsigned long long)(count) & (unsigned long long)(-2)) < SECUREC_STRING_MAX_LEN)) { \
        char *ncatTmpDest_ = (char *)(dest); \
        size_t ncatRestSize_ = (size_t)(destMax); \
        while (ncatRestSize_ > 0 && *ncatTmpDest_ != '\0') { \
            ++ncatTmpDest_; \
            --ncatRestSize_; \
        } \
        if (ncatRestSize_ == 0) { \
            ncatRet_ = EINVAL; \
        } else if ((SECUREC_TWO_MIN((count), strlen(src)) + 1) <= ncatRestSize_) { \
            if ((size_t)(count) < strlen(src)) { \
                memcpy(ncatTmpDest_, (src), (count)); \
                *(ncatTmpDest_ + (count)) = '\0'; \
            } else { \
                memcpy(ncatTmpDest_, (src), strlen(src) + 1); \
            } \
        } else { \
            ncatRet_ = ERANGE; \
        } \
        if (ncatRet_ != EOK) { \
            ncatRet_ = strncat_s((dest), (destMax), (src), (count)); \
        } \
    } else { \
        ncatRet_ = strncat_s((dest), (destMax), (src), (count)); \
    } \
    ncatRet_; \
})
#else
#define SECUREC_STRNCAT_SM(dest, destMax, src, count) strncat_s((dest), (destMax), (src), (count))
#endif

/* This macro do not check buffer overlap by default */
#define  SECUREC_MEMCPY_SM(dest, destMax, src, count) \
    (!(((size_t)(destMax) == 0) || \
        (((unsigned long long)(destMax) & (unsigned long long)(-2)) > SECUREC_MEM_MAX_LEN) || \
        ((size_t)(count) > (size_t)(destMax)) || ((void *)(dest)) == NULL || ((const void *)(src) == NULL)) ? \
        (memcpy((dest), (src), (count)), EOK) : \
        (memcpy_s((dest), (destMax), (src), (count))))

#define  SECUREC_MEMSET_SM(dest, destMax, c, count) \
    (!((((unsigned long long)(destMax) & (unsigned long long)(-2)) > SECUREC_MEM_MAX_LEN) || \
        ((void *)(dest) == NULL) || ((size_t)(count) > (size_t)(destMax))) ? \
        (memset((dest), (c), (count)), EOK) : \
        (memset_s((dest), (destMax), (c), (count))))

#endif
#endif

