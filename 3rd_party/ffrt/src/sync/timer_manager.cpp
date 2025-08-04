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

#include <mutex>
#include <chrono>
#include <iostream>
#include "sync/timer_manager.h"
#include "dfx/log/ffrt_log_api.h"

constexpr uint64_t MAX_TIMER_MS_COUNT = 1000ULL * 100 * 60 * 60 * 24 * 365; // 100year
namespace ffrt {
TimerManager& TimerManager::Instance()
{
    static TimerManager ins;
    return ins;
}

TimerManager::TimerManager()
{
    for (int i = 0; i < QoS::MaxNum(); ++i) {
        InitWorkQueAndCb(qos(i));
    }
}

TimerManager::~TimerManager()
{
    std::lock_guard lock(timerMutex_);
    teardown = true;
}

void TimerManager::InitWorkQueAndCb(int qos)
{
    workCb[qos] = [this, qos](WaitEntry* we) {
        {
            std::lock_guard lock(timerMutex_);
            if (teardown) {
                return;
            }
        }

        int handle = (int)reinterpret_cast<uint64_t>(we);
        submit([this, handle]() {
            std::unique_lock timerLock(timerMutex_);
            if (teardown) {
                return;
            }

            auto it = timerMap_.find(handle);
            if (it == timerMap_.end()) {
                // timer unregistered
                return;
            }

            // execute timer
            std::shared_ptr<TimerData> timerMapValue = it->second;
            timerMapValue->state = TimerState::EXECUTING;
            if (timerMapValue->cb != nullptr) {
                timerLock.unlock();
#ifdef FFRT_ENABLE_HITRACE_CHAIN
            if (timerMapValue->traceId.valid == HITRACE_ID_VALID) {
                TraceChainAdapter::Instance().HiTraceChainRestoreId(&timerMapValue->traceId);
            }
#endif
                timerMapValue->cb(timerMapValue->data);
#ifdef FFRT_ENABLE_HITRACE_CHAIN
            if (timerMapValue->traceId.valid == HITRACE_ID_VALID) {
                TraceChainAdapter::Instance().HiTraceChainClearId();
            }
#endif
                timerLock.lock();
            }
            timerMapValue->state = TimerState::EXECUTED;

            if (timerMapValue->repeat) {
                // re-register timer data
                RegisterTimerImpl(timerMapValue);
            } else {
                // delete timer data
                timerMap_.erase(it);
            }
        },
            {}, {&workQueDeps[qos]}, ffrt::task_attr().qos(qos));
    };
}

ffrt_timer_t TimerManager::RegisterTimer(int qos, uint64_t timeout, void* data, ffrt_timer_cb cb, bool repeat) noexcept
{
    std::lock_guard lock(timerMutex_);
    if (teardown) {
        return -1;
    }

    if (timeout > MAX_TIMER_MS_COUNT) {
        FFRT_LOGW("timeout exceeds maximum allowed value %llu ms. Clamping to %llu ms.", timeout, MAX_TIMER_MS_COUNT);
        timeout = MAX_TIMER_MS_COUNT;
    }
    std::shared_ptr<TimerData> timerMapValue = std::make_shared<TimerData>(data, cb, repeat, qos, timeout);
    timerMapValue->handle = ++timerHandle_;
    timerMapValue->state = TimerState::NOT_EXECUTED;
    timerMap_.emplace(timerHandle_, timerMapValue);

    RegisterTimerImpl(timerMapValue);
    return timerHandle_;
}

void TimerManager::RegisterTimerImpl(std::shared_ptr<TimerData> data)
{
    TimePoint absoluteTime = std::chrono::steady_clock::now() + std::chrono::milliseconds(data->timeout);
    if (!DelayedWakeup(absoluteTime, reinterpret_cast<WaitEntry*>(data->handle), workCb[data->qos], true)) {
        FFRT_LOGW("timer start failed, process may be exiting now");
    }
}

int TimerManager::UnregisterTimer(ffrt_timer_t handle) noexcept
{
    std::unique_lock timerLock(timerMutex_);
    if (teardown) {
        return -1;
    }

    if (handle > timerHandle_ || handle <= -1) { // invalid handle
        return -1;
    }

    auto it = timerMap_.find(handle);
    if (it == timerMap_.end()) {
        return 0;
    }

    if (it->second->state == TimerState::NOT_EXECUTED || it->second->state == TimerState::EXECUTED) {
        // timer not executed or executed, delete timer data
        timerMap_.erase(it);
        return 0;
    }
    if (it->second->state == TimerState::EXECUTING) {
        // timer executing, spin wait it done
        while (it->second->state == TimerState::EXECUTING) {
            timerLock.unlock();
            std::this_thread::yield();
            timerLock.lock();
            it = timerMap_.find(handle);
            if (it == timerMap_.end()) {
                // timer already erased
                return 0;
            }
        }
        // executed, delete timer data
        timerMap_.erase(it);
        return 0;
    }
    // timer already erased
    return 0;
}

ffrt_timer_query_t TimerManager::GetTimerStatus(ffrt_timer_t handle) noexcept
{
    std::unique_lock timerLock(timerMutex_);
    if (teardown) {
        return ffrt_timer_notfound;
    }

    if (handle > timerHandle_) { // invalid handle
        return ffrt_timer_notfound;
    }

    auto it = timerMap_.find(handle);
    if (it == timerMap_.end()) {
        return ffrt_timer_executed;
    }

    if (it->second->state == TimerState::NOT_EXECUTED) {
        // timer has not been executed
        return ffrt_timer_not_executed;
    }
    if (it->second->state == TimerState::EXECUTING || it->second->state == TimerState::EXECUTED) {
        // timer executing or has been executed (don't spin wait executing)
        // timer executing, spin wait it done
        while (it->second->state == TimerState::EXECUTING) {
            timerLock.unlock();
            std::this_thread::yield();
            timerLock.lock();
            it = timerMap_.find(handle);
            if (it == timerMap_.end()) {
                // timer already erased
                break;
            }
        }
        return ffrt_timer_executed;
    }
    // timer has been executed or unregistered
    return ffrt_timer_executed;
}
}
