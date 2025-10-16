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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <map>
#include <functional>
#include "sync/sync.h"

#include "sched/execute_ctx.h"
#include "eu/co_routine.h"
#include "internal_inc/osal.h"
#include "internal_inc/types.h"
#include "dfx/log/ffrt_log_api.h"
#include "dfx/trace/ffrt_trace.h"
#include "tm/cpu_task.h"
#include "cpp/sleep.h"

constexpr uint64_t MAX_US_COUNT = static_cast<uint64_t>(std::chrono::microseconds::max().count());

namespace ffrt {
namespace this_task {
void SleepUntilImpl(const TimePoint& to)
{
    auto task = ExecuteCtx::Cur()->task;
    if (task == nullptr || task->Block() == BlockType::BLOCK_THREAD) {
        std::this_thread::sleep_until(to);
        if (task) {
            task->Wake();
        }
        return;
    }
    // be careful about local-var use-after-free here
    static std::function<void(WaitEntry*)> cb ([](WaitEntry* we) {
        CoRoutineFactory::CoWakeFunc(static_cast<CoTask*>(we->task), CoWakeType::NO_TIMEOUT_WAKE);
    });
    FFRT_BLOCK_TRACER(ExecuteCtx::Cur()->task->gid, slp);
    CoWait([&](CoTask* task) -> bool {
        return DelayedWakeup(to, &task->we, cb);
    });
}
}
} // namespace ffrt

#ifdef __cplusplus
extern "C" {
#endif

API_ATTRIBUTE((visibility("default")))
void ffrt_yield()
{
    auto curTask = ffrt::ExecuteCtx::Cur()->task;
    if (curTask == nullptr || curTask->Block() == ffrt::BlockType::BLOCK_THREAD) {
        std::this_thread::yield();
        if (curTask) {
            curTask->Wake();
        }
        return;
    }
    FFRT_BLOCK_TRACER(curTask->gid, yld);
    CoWait([](ffrt::CoTask* task) -> bool {
        CoRoutineFactory::CoWakeFunc(task, CoWakeType::NO_TIMEOUT_WAKE);
        return true;
    });
}

API_ATTRIBUTE((visibility("default")))
int ffrt_usleep(uint64_t usec)
{
    if (usec > MAX_US_COUNT) {
        FFRT_LOGW("usec exceeds maximum allowed value %llu us. Clamping to %llu us.", usec, MAX_US_COUNT);
        usec = MAX_US_COUNT;
    }

    auto now = std::chrono::steady_clock::now();
    auto nowUs = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    auto sleepUs = std::chrono::microseconds(usec);
    std::chrono::steady_clock::time_point to;
    if (sleepUs.count() > (std::chrono::microseconds::max().count() - nowUs.count())) {
        to = std::chrono::steady_clock::time_point::max();
    } else {
        to = now + sleepUs;
    }

    ffrt::this_task::SleepUntilImpl(to);
    return ffrt_success;
}

#ifdef __cplusplus
}
#endif