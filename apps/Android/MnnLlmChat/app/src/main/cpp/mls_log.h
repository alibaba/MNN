//
// Created by ruoyi.sjd on 2024/12/25.
//

#pragma once
#include <android/log.h>
#define LOG_TAG_DEBUG "MNN_DEBUG"
#define LOG_TAG_ERROR "MNN_ERROR"
#define MNN_DEBUG(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG_DEBUG, __VA_ARGS__)
#ifndef MNN_ERROR
#define MNN_ERROR(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG_ERROR, __VA_ARGS__)
#endif
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG_DEBUG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG_ERROR, __VA_ARGS__)
