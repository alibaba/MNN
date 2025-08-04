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

#include "dfx/log/log_base.h"
#include <string>
#include <cstdarg>
#include <iostream>
#include <securec.h>
#include <chrono>
#include "internal_inc/osal.h"

static const int g_logBufferSize = 2048;

static std::string GetCurrentTime(void)
{
    const int startYear = 1900;
    auto now = std::chrono::system_clock::now();
    auto curMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
    auto sectime = std::chrono::duration_cast<std::chrono::seconds>(curMs);
    auto milltime = curMs % 1000;
    std::time_t timet = sectime.count();
    struct tm curtime;
    localtime_r(&timet, &curtime);

    auto year = std::to_string(curtime.tm_year + startYear);
    auto mon = std::to_string(curtime.tm_mon + 1);
    auto day = std::to_string(curtime.tm_mday);
    auto hour = std::to_string(curtime.tm_hour);
    auto min = std::to_string(curtime.tm_min);
    auto sec = std::to_string(curtime.tm_sec);
    auto ms = std::to_string(milltime.count());

    return year + "-" + mon + "-" + day + " " + hour + ":" + min + ":" + sec + "." + ms;
}

static void LogOutput(const char* level, const char* log)
{
    std::string pid, tid, strBuf;
    pid = std::to_string(GetPid());
    tid = std::to_string(syscall(SYS_gettid));

    strBuf = GetCurrentTime() + "  ";
    strBuf += pid + "  " + tid + " ";
    strBuf += level;
    strBuf += " ffrt : ";
    strBuf += log;

    std::cout << strBuf;
}

void LogErr(const char* fmt, ...)
{
    char errLog[g_logBufferSize];
    va_list arg;
    va_start(arg, fmt);
    int ret = vsnprintf_s(errLog, sizeof(errLog), sizeof(errLog) - 1, fmt, arg);
    va_end(arg);
    if (ret < 0) {
        return;
    }
    LogOutput("E", errLog);
}

void LogWarn(const char* fmt, ...)
{
    char warnLog[g_logBufferSize];
    va_list arg;
    va_start(arg, fmt);
    int ret = vsnprintf_s(warnLog, sizeof(warnLog), sizeof(warnLog) - 1, fmt, arg);
    va_end(arg);
    if (ret < 0) {
        return;
    }
    LogOutput("W", warnLog);
}

void LogInfo(const char* fmt, ...)
{
    char infoLog[g_logBufferSize];
    va_list arg;
    va_start(arg, fmt);
    int ret = vsnprintf_s(infoLog, sizeof(infoLog), sizeof(infoLog) - 1, fmt, arg);
    va_end(arg);
    if (ret < 0) {
        return;
    }
    LogOutput("I", infoLog);
}

void LogDebug(const char* fmt, ...)
{
    char debugLog[g_logBufferSize];
    va_list arg;
    va_start(arg, fmt);
    int ret = vsnprintf_s(debugLog, sizeof(debugLog), sizeof(debugLog) - 1, fmt, arg);
    va_end(arg);
    if (ret < 0) {
        return;
    }
    LogOutput("D", debugLog);
}