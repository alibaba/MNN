//
// Created by ruoyi.sjd on 2024/12/25.
//

#pragma once

#include <android/log.h>
#define LOG_TAG "MH_DEBUG"
#define LOG_VERBOSE 0
#if LOG_VERBOSE
#define MH_LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)
#else
#define MH_LOGV(...)
#endif
#define MH_DEBUG(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define MH_ERROR(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)