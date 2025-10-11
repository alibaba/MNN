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
#ifdef FFRT_SEND_EVENT
#include "sysevent.h"
#include <dlfcn.h>
#include <mutex>
#include "hisysevent.h"
#include "tm/task_base.h"
#include "dfx/log/ffrt_log_api.h"
#include "internal_inc/osal.h"
#include "util/ffrt_facade.h"

namespace {
constexpr long long lONG_TASK_TIME_LIMIT = 1; // 1s
constexpr long long lONG_TASK_REPORT_FRE = 30 * 60; // 30min == 30 * 60s
constexpr uint64_t FISRT_CALL_TIME_LIMIT = 5 * 60; // 5min == 5 * 60s
constexpr const char* BEGETUTIL_LIB_PATH = "libbegetutil.z.so";
}

namespace ffrt {
std::mutex mtx;
class LibbegetutilAdapter {
public:
    LibbegetutilAdapter()
    {
        if (!Load()) {
            FFRT_LOGD("Failed to load the library in constructor.");
        }
    }

    ~LibbegetutilAdapter()
    {
        if (!UnLoad()) {
            FFRT_LOGD("Failed to unload the library in destructor.");
        }
    }

    static LibbegetutilAdapter* Instance()
    {
        static LibbegetutilAdapter instance;
        return &instance;
    }

    using GetParameterType = int (*)(const char *, const char *, char *, uint32_t);
    GetParameterType GetParameter = nullptr;

private:
    bool Load()
    {
        if (handle != nullptr) {
            FFRT_LOGD("handle exits");
            return true;
        }

        handle = dlopen(BEGETUTIL_LIB_PATH, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
        if (handle == nullptr) {
            FFRT_LOGE("load so[%s] fail", BEGETUTIL_LIB_PATH);
            return false;
        }

        GetParameter = reinterpret_cast<GetParameterType>(dlsym(handle, "GetParameter"));
        if (GetParameter == nullptr) {
            FFRT_LOGE("load func from %s failed", BEGETUTIL_LIB_PATH);
            return false;
        }
        return true;
    }

    bool UnLoad()
    {
        if (handle != nullptr) {
            if (dlclose(handle) != 0) {
                FFRT_LOGE("Failed to close the handle.");
                return false;
            }
            handle = nullptr;
            return true;
        }
        return true;
    }

    void* handle = nullptr;
};

bool IsBeta()
{
    LibbegetutilAdapter* adapter = LibbegetutilAdapter::Instance();
    constexpr int versionTypeLen = 32;
    char retValue[versionTypeLen] = {0};
    if (adapter->GetParameter != nullptr) {
        int ret = adapter->GetParameter("const.logsystem.versiontype", "false", retValue, versionTypeLen);
        if (ret > 0) {
            int result = strcmp(retValue, "beta");
            if (result == 0) {
                return true;
            }
        }
    }
    return false;
}

void TaskBlockInfoReport(const long long passed, const std::string& task_label, int qos, uint64_t freq)
{
    static std::once_flag firstCallFlag;
    if (unlikely(passed > lONG_TASK_TIME_LIMIT)) {
        uint64_t now = TimeStamp();
        {
            std::lock_guard<std::mutex> lock(mtx);
            static uint64_t firstCallTime = 0;
            static uint64_t lastEventTime = 0;
            std::call_once(firstCallFlag, [&]() {
                firstCallTime = now;
                lastEventTime = now;
            });
            uint64_t diff = now - firstCallTime;
            if ((diff / freq) > FISRT_CALL_TIME_LIMIT) {
                if (now >= lastEventTime && unlikely(((now - lastEventTime) / freq) > lONG_TASK_REPORT_FRE)) {
                    std::string eventName = "TASK_TIMEOUT";
                    std::string buffer = "task:" + task_label + ", passed: "
                        + std::to_string(passed) + " s, qos:" + std::to_string(qos);
                    HiSysEventWrite(OHOS::HiviewDFX::HiSysEvent::Domain::FFRT,
                                    eventName, OHOS::HiviewDFX::HiSysEvent::EventType::FAULT,
                                    "SENARIO", "Long_Task", "PROCESS_NAME", std::string(GetCurrentProcessName()),
                                    "MSG", buffer);
                    lastEventTime = now;
                }
            }
        }
    }
}

void TaskTimeoutReport(std::stringstream& ss, const std::string& processName, const std::string& senarioName)
{
    std::string msg = ss.str();
    std::string eventName = "TASK_TIMEOUT";
    time_t cur_time = time(nullptr);
    std::string sendMsg = std::string((ctime(&cur_time) == nullptr) ? "" : ctime(&cur_time)) + "\n" + msg + "\n";
    HiSysEventWrite(OHOS::HiviewDFX::HiSysEvent::Domain::FFRT, eventName,
        OHOS::HiviewDFX::HiSysEvent::EventType::FAULT, "SENARIO", senarioName,
        "PROCESS_NAME", processName, "MSG", sendMsg);
}

void TrafficOverloadReport(std::stringstream& ss, const std::string& senarioName)
{
    std::string msg = ss.str();
    std::string eventName = "TASK_TIMEOUT";
    time_t cur_time = time(nullptr);
    std::string sendMsg = std::string((ctime(&cur_time) == nullptr) ? "" : ctime(&cur_time)) + "\n" + msg + "\n";
    HiSysEventWrite(OHOS::HiviewDFX::HiSysEvent::Domain::FFRT, eventName,
        OHOS::HiviewDFX::HiSysEvent::EventType::FAULT, "SENARIO", senarioName,
        "PROCESS_NAME", GetCurrentProcessName(), "MSG", sendMsg);
}

void WorkerEscapeReport(const std::string& processName, int qos, size_t totalNum)
{
    time_t cur_time = time(nullptr);
    size_t near_gid = TaskBase::GetLastGid();
    std::string msg = "report time: " + std::string((ctime(&cur_time) == nullptr) ? "" : ctime(&cur_time)) + "\n"
                    + ", qos: " + std::to_string(qos)
                    + ", worker num: " + std::to_string(totalNum)
                    + ", near gid:" + std::to_string((near_gid > 0) ? near_gid - 1 : 0);
    std::string eventName = "TASK_TIMEOUT";
    HiSysEventWrite(OHOS::HiviewDFX::HiSysEvent::Domain::FFRT, eventName,
        OHOS::HiviewDFX::HiSysEvent::EventType::FAULT, "SENARIO", "Trigger_Escape",
        "PROCESS_NAME", processName, "MSG", msg);
    FFRT_LOGW("Process: %s trigger escape. %s", processName.c_str(), msg.c_str());
}
}
#endif