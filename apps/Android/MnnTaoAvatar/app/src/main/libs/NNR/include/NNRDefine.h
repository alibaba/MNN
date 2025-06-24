//
//  NNRDefine.h
//  NNR
//
//  Created by NNR on 2018/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NNRDefine_h
#define NNRDefine_h

#include <assert.h>
#include <stdio.h>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define NNR_BUILD_FOR_IOS
#endif
#endif

#ifdef NNR_USE_LOGCAT
#include <android/log.h>
#define NNR_ERROR(format, ...) __android_log_print(ANDROID_LOG_ERROR, "NNRJNI", format, ##__VA_ARGS__)
#define NNR_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "NNRJNI", format, ##__VA_ARGS__)
#else
#define NNR_PRINT(format, ...) printf(format, ##__VA_ARGS__)
#define NNR_ERROR(format, ...) printf(format, ##__VA_ARGS__)
#define NNR_WARNING(format, ...) printf(format, ##__VA_ARGS__)
#endif

#define NNR_PRINT_WITH_FUNC(format, ...) printf(format", FILE: %s, LINE: %d\n", ##__VA_ARGS__, __FILE__, __LINE__)
#define NNR_ERROR_WITH_FUNC(format, ...) printf(format", FILE: %s, LINE: %d\n", ##__VA_ARGS__, __FILE__, __LINE__)

#ifdef NNR_RUNTIME_DEBUG
#define NNR_ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            NNR_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
            assert(res);                                         \
        }                                                        \
    }
#else
#define NNR_ASSERT(x)
#endif

#define NNR_FUNC_PRINT(x) NNR_PRINT(#x "=%d in %s, %d \n", x, __func__, __LINE__);
#define NNR_FUNC_PRINT_ALL(x, type) NNR_PRINT(#x "=" #type " %" #type " in %s, %d \n", x, __func__, __LINE__);

#define NNR_CHECK(success, log) \
if(!(success)){ \
NNR_ERROR("Check failed: %s ==> %s\n", #success, #log); \
}

#if defined(_MSC_VER)
#if defined(BUILDING_NNR_DLL)
#define NNR_PUBLIC __declspec(dllexport)
#elif defined(USING_NNR_DLL)
#define NNR_PUBLIC __declspec(dllimport)
#else
#define NNR_PUBLIC
#endif
#else
#define NNR_PUBLIC __attribute__((visibility("default")))
#endif

// For Python Generate
#define NNR_PYTHON_FUNC NNR_PUBLIC
#define NNR_PYTHON_CLASS(name) NNR_PUBLIC name
#define NNR_PYTHON_CLS_FUNC(name)
#define NNR_PYTHON_NAMESPACE(name) namespace name

//#define DUMP_SHADER_SOURCE
#endif /* NNRDefine_h */
