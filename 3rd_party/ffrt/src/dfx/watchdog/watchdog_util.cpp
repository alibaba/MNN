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
#include "dfx/watchdog/watchdog_util.h"
#include <sstream>
#include <algorithm>
#include <map>
#include "sync/sync.h"
#ifdef FFRT_OH_WATCHDOG_ENABLE
#include "c/ffrt_dump.h"
#endif
#include "dfx/log/ffrt_log_api.h"
#include "util/ffrt_facade.h"
#include "util/slab.h"
namespace {
constexpr uint64_t VALID_TIMEOUT_MIN = 10000;
constexpr uint64_t VALID_TIMEOUT_MAX = 30000;
constexpr uint32_t CONVERT_TIME_UNIT = 1000;
constexpr int SEND_COUNT_MIN = 1;
constexpr int SEND_COUNT_MAX = 3;
}

namespace ffrt {
    static std::map<uint64_t, int> taskStatusMap;
    static std::mutex lock;


    bool IsValidTimeout(uint64_t gid, uint64_t timeout_us)
    {
        // us convert to ms
        uint64_t timeout_ms = timeout_us / CONVERT_TIME_UNIT;
        // 当前有效的并行任务timeout时间范围是10-30s
        if (timeout_ms >= VALID_TIMEOUT_MIN && timeout_ms <= VALID_TIMEOUT_MAX) {
            FFRT_LOGI("task gid=%llu with timeout [%llu ms] is valid", gid, timeout_ms);
            return true;
        } else if (timeout_ms > 0) {
            FFRT_LOGE("task gid=%llu with timeout [%llu ms] is invalid", gid, timeout_ms);
        }
        return false;
    }

    void AddTaskToWatchdog(uint64_t gid)
    {
        std::lock_guard<decltype(lock)> l(lock);
        taskStatusMap.insert(std::make_pair(gid, SEND_COUNT_MIN));
    }

    void RemoveTaskFromWatchdog(uint64_t gid)
    {
        std::lock_guard<decltype(lock)> l(lock);
        taskStatusMap.erase(gid);
    }

    bool SendTimeoutWatchdog(uint64_t gid, uint64_t timeout, uint64_t delay)
    {
#ifdef FFRT_OH_WATCHDOG_ENABLE
        // us convert to ms
        uint64_t timeout_ms = timeout / CONVERT_TIME_UNIT;
        FFRT_LOGI("start to set watchdog for task gid=%llu with timeout [%llu ms] ", gid, timeout_ms);
        auto now = std::chrono::steady_clock::now();
        WaitUntilEntry* we = new (SimpleAllocator<WaitUntilEntry>::AllocMem()) WaitUntilEntry();
        // set dealyedworker callback
        we->cb = ([gid, timeout_ms](WaitEntry* we) {
            bool taskFinished = true;
            {
                std::lock_guard<decltype(lock)> l(lock);
                if (taskStatusMap.count(gid) > 0) {
                    int sendCount = taskStatusMap[gid];
                    if (sendCount > SEND_COUNT_MAX) {
                        FFRT_LOGE("parallel task gid=%llu send watchdog delaywork failed, the count more than %d times",
                            gid, SEND_COUNT_MAX);
                        SimpleAllocator<WaitUntilEntry>::FreeMem(static_cast<WaitUntilEntry*>(we));
                        return;
                    }
                    taskStatusMap[gid] = (++sendCount);
                    taskFinished = false;
                }
            }

            if (!taskFinished) {
                RunTimeOutCallback(gid, timeout_ms);
                if (!SendTimeoutWatchdog(gid, timeout_ms * CONVERT_TIME_UNIT, 0)) {
                    FFRT_LOGE("parallel task gid=%llu send next watchdog delaywork failed", gid);
                    SimpleAllocator<WaitUntilEntry>::FreeMem(static_cast<WaitUntilEntry*>(we));
                    return;
                };
            } else {
                FFRT_LOGI("task gid=%llu has finished", gid);
            }
            SimpleAllocator<WaitUntilEntry>::FreeMem(static_cast<WaitUntilEntry*>(we));
        });
        // set dealyedworker wakeup time
        std::chrono::microseconds timeoutTime(timeout);
        std::chrono::microseconds delayTime(delay);
        we->tp = (now + timeoutTime + delayTime);
        if (!DelayedWakeup(we->tp, we, we->cb, true)) {
            SimpleAllocator<WaitUntilEntry>::FreeMem(we);
            FFRT_LOGE("failed to set watchdog for task gid=%llu with timeout [%llu ms] ", gid, timeout_ms);
            return false;
        }
#endif
        return true;
    }

    void RunTimeOutCallback(uint64_t gid, uint64_t timeout)
    {
#ifdef FFRT_OH_WATCHDOG_ENABLE
        std::stringstream ss;
        ss << "parallel task gid=" << gid << " execution time exceeds " << timeout << " ms";
        std::string msg = ss.str();
        FFRT_LOGE("%s", msg.c_str());

        if (ffrt_task_timeout_get_cb()) {
            FFRTFacade::GetDWInstance().SubmitAsyncTask([gid, msg] {
                ffrt_task_timeout_cb func = ffrt_task_timeout_get_cb();
                if (func) {
                    func(gid, msg.c_str(), msg.size());
                }
            });
        }
#endif
    }
}