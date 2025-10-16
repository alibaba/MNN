/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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

#ifndef FFRT_TIMER_MANAGER_HPP
#define FFRT_TIMER_MANAGER_HPP

#include <array>
#include <functional>
#include <unordered_map>
#include "internal_inc/osal.h"
#include "internal_inc/non_copyable.h"
#include "dfx/log/ffrt_log_api.h"
#include "cpp/queue.h"
#include "sync/sync.h"
#include "sched/execute_ctx.h"
#include "tm/task_base.h"
#ifdef FFRT_ENABLE_HITRACE_CHAIN
#include "dfx/trace/ffrt_trace_chain.h"
#endif

namespace ffrt {
/** un-repeat timer: not_executed -> executintg -> executed -> ereased **/
/** repeat timer: not_executed -> executintg -> executed -> executing -> executed ... **/
enum class TimerState {
    NOT_EXECUTED, // the timer has not expired (in the initialization state).
    EXECUTING,    // The timer has expired and is executing the callback.
    EXECUTED,     // The timer has expired and the callback has been executed completely.
    INVALID
};

struct TimerData {
    TimerData(void *dataVal, std::function<void(void *)> cbVal, bool repeat, int qos, uint64_t timeout)
        : data(dataVal), cb(cbVal), repeat(repeat), qos(qos), timeout(timeout)
    {
        if (cb != nullptr) {
#ifdef FFRT_ENABLE_HITRACE_CHAIN
            if (TraceChainAdapter::Instance().HiTraceChainGetId().valid == HITRACE_ID_VALID) {
                traceId = TraceChainAdapter::Instance().HiTraceChainCreateSpan();
            };
#endif
        }
    }

    void* data;
    std::function<void(void*)> cb;
    bool repeat;
    int qos;
    uint64_t timeout;
    int handle;
    TimerState state {TimerState::NOT_EXECUTED};
    HiTraceIdStruct traceId;
};

class TimerManager : private NonCopyable {
public:
    ~TimerManager();
    static TimerManager& Instance();

    ffrt_timer_t RegisterTimer(int qos, uint64_t timeout, void* data, ffrt_timer_cb cb, bool repeat = false) noexcept;
    int UnregisterTimer(ffrt_timer_t handle) noexcept;
    ffrt_timer_query_t GetTimerStatus(ffrt_timer_t handle) noexcept;

private:
    TimerManager();

    void InitWorkQueAndCb(int qos);
    void RegisterTimerImpl(std::shared_ptr<TimerData> data);

    mutable spin_mutex timerMutex_;
    ffrt_timer_t timerHandle_ { -1 };
    bool teardown { false };
    std::unordered_map<int, std::shared_ptr<TimerData>> timerMap_; // valid timer data manage
    std::array<uint64_t, QoS::MaxNum()> workQueDeps; // deps to ensure callbacks execute in order
    std::array<std::function<void(WaitEntry*)>, QoS::MaxNum()> workCb; // timeout cb for submit timer cb to queue
};
}
#endif
