//
// Created by ruoyi.sjd on 2024/12/25.
//

#pragma once
#include <android/log.h>
#define LOG_TAG "MNN_DEBUG"
#define MNN_DEBUG(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)