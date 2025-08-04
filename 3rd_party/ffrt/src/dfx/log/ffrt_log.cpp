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

#ifdef OHOS_STANDARD_SYSTEM
#include "faultloggerd_client.h"
#endif
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <atomic>
#include "dfx/log/ffrt_log_api.h"

#ifdef FFRT_SEND_EVENT
#include <securec.h>
#include "hisysevent.h"
#endif
#include "internal_inc/osal.h"
#include "util/white_list.h"

static int g_ffrtLogLevel = FFRT_LOG_DEBUG;
static std::atomic<unsigned int> g_ffrtLogId(0);
static bool g_whiteListFlag = false;
namespace {
    constexpr int LOG_BUFFER_SIZE = 2048;
}

unsigned int GetLogId(void)
{
    return ++g_ffrtLogId;
}

bool IsInWhitelist(void)
{
    return g_whiteListFlag;
}

int GetFFRTLogLevel(void)
{
    return g_ffrtLogLevel;
}

static void SetLogLevel(void)
{
    std::string envLogStr = GetEnv("FFRT_LOG_LEVEL");
    if (envLogStr.size() != 0) {
        int level = std::stoi(envLogStr);
        if (level < FFRT_LOG_LEVEL_MAX && level >= FFRT_LOG_ERROR) {
            g_ffrtLogLevel = level;
            return;
        }
    }
}

void InitWhiteListFlag(void)
{
    g_whiteListFlag = WhiteList::GetInstance().IsEnabled("log_ctr", true);
}

static __attribute__((constructor)) void LogInit(void)
{
    SetLogLevel();
    InitWhiteListFlag();
}

#ifdef FFRT_SEND_EVENT
void ReportSysEvent(const char* format, ...)
{
    char buffer[LOG_BUFFER_SIZE] = {0};
    va_list args;
    va_start(args, format);
    int ret = vsnprintf_s(buffer, LOG_BUFFER_SIZE, LOG_BUFFER_SIZE - 1, format, args);
    va_end(args);
    if (ret < 0) {
        return;
    }
    std::string msg = buffer;
    HiSysEventWrite(OHOS::HiviewDFX::HiSysEvent::Domain::FFRT, "TASK_TIMEOUT",
        OHOS::HiviewDFX::HiSysEvent::EventType::FAULT, "MSG", msg);
}
#endif // FFRT_SEND_EVENT