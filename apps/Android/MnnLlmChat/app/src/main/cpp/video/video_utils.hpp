#ifndef VIDEO_UTILS_HPP
#define VIDEO_UTILS_HPP

#include <android/log.h>

#ifndef VIDEO_DEBUG_ENABLED
#define VIDEO_DEBUG_ENABLED 0
#endif

#if VIDEO_DEBUG_ENABLED
    #define VIDEO_LOGV(tag, ...) __android_log_print(ANDROID_LOG_VERBOSE, tag, __VA_ARGS__)
    #define VIDEO_LOGD(tag, ...) __android_log_print(ANDROID_LOG_DEBUG, tag, __VA_ARGS__)
    #define VIDEO_LOGI(tag, ...) __android_log_print(ANDROID_LOG_INFO, tag, __VA_ARGS__)
    #define VIDEO_LOGW(tag, ...) __android_log_print(ANDROID_LOG_WARN, tag, __VA_ARGS__)
    #define VIDEO_LOGE(tag, ...) __android_log_print(ANDROID_LOG_ERROR, tag, __VA_ARGS__)
    #define VIDEO_LOGF(tag, ...) __android_log_print(ANDROID_LOG_FATAL, tag, __VA_ARGS__)
#else
    #define VIDEO_LOGV(tag, ...) ((void)0)
    #define VIDEO_LOGD(tag, ...) ((void)0)
    #define VIDEO_LOGI(tag, ...) ((void)0)
    #define VIDEO_LOGW(tag, ...) ((void)0)
    #define VIDEO_LOGE(tag, ...) ((void)0)
    #define VIDEO_LOGF(tag, ...) ((void)0)
#endif

#endif // VIDEO_UTILS_HPP
