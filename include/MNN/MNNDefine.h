//
//  MNNDefine.h
//  MNN
//
//  Created by MNN on 2018/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNDefine_h
#define MNNDefine_h

#include <assert.h>
#include <stdio.h>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define MNN_BUILD_FOR_IOS
#endif
#endif

#ifdef MNN_USE_LOGCAT
#include <android/log.h>
#define MNN_ERROR(format, ...) __android_log_print(ANDROID_LOG_ERROR, "MNNJNI", format, ##__VA_ARGS__)
#define MNN_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MNNJNI", format, ##__VA_ARGS__)
#elif defined MNN_BUILD_FOR_IOS
// on iOS, stderr prints to XCode debug area and syslog prints Console. You need both.
#include <syslog.h>
#define MNN_PRINT(format, ...) syslog(LOG_WARNING, format, ##__VA_ARGS__); fprintf(stderr, format, ##__VA_ARGS__)
#define MNN_ERROR(format, ...) syslog(LOG_WARNING, format, ##__VA_ARGS__); fprintf(stderr, format, ##__VA_ARGS__)
#else
#define MNN_PRINT(format, ...) printf(format, ##__VA_ARGS__)
#define MNN_ERROR(format, ...) printf(format, ##__VA_ARGS__)
#endif

#ifdef DEBUG
#define MNN_ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            MNN_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
            assert(res);                                         \
        }                                                        \
    }
#else
#define MNN_ASSERT(x)
#endif

#define FUNC_PRINT(x) MNN_PRINT(#x "=%d in %s, %d \n", x, __func__, __LINE__);
#define FUNC_PRINT_ALL(x, type) MNN_PRINT(#x "=" #type " %" #type " in %s, %d \n", x, __func__, __LINE__);

#define MNN_CHECK(success, log) \
if(!(success)){ \
MNN_ERROR("Check failed: %s ==> %s\n", #success, #log); \
}

#if defined(_MSC_VER)
#if defined(BUILDING_MNN_DLL)
#define MNN_PUBLIC __declspec(dllexport)
#elif defined(USING_MNN_DLL)
#define MNN_PUBLIC __declspec(dllimport)
#else
#define MNN_PUBLIC
#endif
#else
#define MNN_PUBLIC __attribute__((visibility("default")))
#endif
#define STR_IMP(x) #x
#define STR(x) STR_IMP(x)
#define MNN_VERSION_MAJOR 2
#define MNN_VERSION_MINOR 8
#define MNN_VERSION_PATCH 1
#define MNN_VERSION STR(MNN_VERSION_MAJOR) "." STR(MNN_VERSION_MINOR) "." STR(MNN_VERSION_PATCH)
#endif /* MNNDefine_h */
