/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __LOG_BASE_H__
#define __LOG_BASE_H__

#include <stdio.h>

void LogErr(const char* format, ...);
void LogWarn(const char* format, ...);
void LogInfo(const char* format, ...);
void LogDebug(const char* format, ...);

int GetFFRTLogLevel(void);

#define FFRT_LOG(level, format, ...) \
    do { \
        if ((level) > GetFFRTLogLevel()) \
            break; \
        if (level == FFRT_LOG_ERROR) { \
            LogErr("%u:%s:%d " format "\n", GetLogId(), __func__, __LINE__, ##__VA_ARGS__); \
        } else if (level == FFRT_LOG_WARN) { \
            LogWarn("%u:%s:%d " format "\n", GetLogId(), __func__, __LINE__, ##__VA_ARGS__); \
        } else if (level == FFRT_LOG_INFO) { \
            LogInfo("%u:%s:%d " format "\n", GetLogId(), __func__, __LINE__, ##__VA_ARGS__); \
        } else if (level == FFRT_LOG_DEBUG) { \
            LogDebug("%u:%s:%d " format "\n", GetLogId(), __func__, __LINE__, ##__VA_ARGS__); \
        } else { \
            printf("Log Level is Invalid!"); \
        } \
    } while (0)

#endif // __LOG_BASE_H__