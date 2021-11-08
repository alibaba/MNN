//
//  GLDebug.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLDEBUG_H
#define GLDEBUG_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "core/Macro.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEBUG_ON

#ifdef DEBUG_ON
#ifdef GL_BUILD_FOR_ANDROID
#include <android/log.h>
#define GPPRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MNNJNI", format, ##__VA_ARGS__)
#else
#define GPPRINT(format, ...) printf(format, ##__VA_ARGS__)
#endif

#define CHECK_POINTER(x)          \
    {                             \
        if (NULL == x) {          \
            FUNC_PRINT_ALL(x, p); \
            break;                \
        }                         \
    }

#ifndef GL_BUILD_FOR_ANDROID
#define GLASSERT(x) assert(x)
#else
#define GLASSERT(x)               \
    {                             \
        bool result = (x);        \
        if (!(result))            \
            FUNC_PRINT((result)); \
    }
#endif
#else

#define FUNC_PRINT(x)
#define FUNC_PRINT_ALL(x, type)
#define CHECK_POINTER(x)

#endif

#define GPASSERT(x) GLASSERT(x)
    
#ifdef OPEN_GL_CHECK_ERROR
#define OPENGL_CHECK_ERROR              \
    {                                   \
        GLenum error = glGetError();    \
        if (GL_NO_ERROR != error){       \
        MNN_PRINT("File = %s Line = %d Func=%s\n", __FILE__, __LINE__, __FUNCTION__); \
        FUNC_PRINT_ALL(error, 0x);  }\
        GLASSERT(GL_NO_ERROR == error); \
    }
#else
#define OPENGL_CHECK_ERROR
#endif
    
#define OPENGL_HAS_ERROR GL_NO_ERROR != glGetError()

void dump_stack();
#ifdef __cplusplus
}
#endif

#endif
