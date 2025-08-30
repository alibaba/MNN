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
#include "queue/queue_monitor.h"
#include "queue/queue_handler.h"
#include "dfx/log/ffrt_log_api.h"
#include "util/slab.h"
#include "sync/sync.h"
#include "c/ffrt_dump.h"
#include "c/queue.h"
#include "internal_inc/osal.h"
#include "util/ffrt_facade.h"
#include "util/time_format.h"

namespace {
constexpr uint32_t US_PER_MS = 1000;
constexpr uint64_t ALLOW_ACC_ERROR_US = 10 * US_PER_MS; // 10ms
constexpr uint64_t MIN_TIMEOUT_THRESHOLD_US = 1000 * US_PER_MS; // 1s
constexpr uint32_t MAX_RECORD_LIMIT = 64;
constexpr uint32_t INITIAL_RECORD_LIMIT = 16;
}

namespace ffrt {
QueueMonitor::QueueMonitor()
{
    FFRT_LOGI("QueueMonitor ctor enter");
    we_ = new (SimpleAllocator<WaitUntilEntry>::AllocMem()) WaitUntilEntry();
    timeoutUs_ = ffrt_task_timeout_get_threshold() * US_PER_MS;
    FFRT_LOGI("queue monitor ctor leave, watchdog timeout of %llu us", timeoutUs_.load());
}

QueueMonitor::~QueueMonitor()
{
    FFRT_LOGI("destruction of QueueMonitor");
    DelayedRemove(we_->tp, we_);
    SimpleAllocator<WaitUntilEntry>::FreeMem(we_);
}

QueueMonitor& QueueMonitor::GetInstance()
{
    static QueueMonitor instance;
    return instance;
}

void QueueMonitor::RegisterQueue(QueueHandler* queue)
{
    std::lock_guard lock(infoMutex_);
    queuesInfo_.push_back(queue);
    FFRT_LOGD("queue [%s] register in QueueMonitor", queue->GetName().c_str());
}

void QueueMonitor::DeregisterQueue(QueueHandler* queue)
{
    std::lock_guard lock(infoMutex_);
    auto it = std::find(queuesInfo_.begin(), queuesInfo_.end(), queue);
    if (it != queuesInfo_.end()) {
        queuesInfo_.erase(it);
    }
}

void QueueMonitor::UpdateQueueInfo()
{
    if (suspendAlarm_.exchange(false)) {
        uint64_t alarmTime = static_cast<uint64_t>(std::chrono::time_point_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now()).time_since_epoch().count()) + timeoutUs_;
        SetAlarm(alarmTime);
    }
}

void QueueMonitor::SetAlarm(uint64_t steadyUs)
{
    we_->tp = std::chrono::steady_clock::time_point() + std::chrono::microseconds(steadyUs);
    we_->cb = ([this](WaitEntry* we) { ScheduleAlarm(); });

    // generally does not fail
    if (!DelayedWakeup(we_->tp, we_, we_->cb, true)) {
        FFRT_LOGW("failed to set delayedworker");
    }
}

void QueueMonitor::ScheduleAlarm()
{
    uint64_t nextTaskStart = UINT64_MAX;
    CheckTimeout(nextTaskStart);
    FFRT_LOGD("queue monitor checked, going next");
    // 所有队列都没有任务，暂停定时器
    if (nextTaskStart == UINT64_MAX) {
        suspendAlarm_.exchange(true);
        return;
    }

    SetAlarm(nextTaskStart + timeoutUs_);
}

void QueueMonitor::CheckTimeout(uint64_t& nextTaskStart)
{
    // 未来ALLOW_ACC_ERROR_US可能超时的任务，一起上报
    uint64_t now = TimeStampCntvct();
    uint64_t minStart = now - ((timeoutUs_ - ALLOW_ACC_ERROR_US));
    std::vector<std::pair<std::pair<std::vector<uint64_t>, uint64_t>, std::stringstream>> curTaskTimeInfoVec;

    {
        std::shared_lock lock(infoMutex_);
        for (auto& queueInfo : queuesInfo_) {
            auto curTaskTimeStamp = queueInfo->EvaluateTaskTimeout(minStart, timeoutUs_,
            timeoutMSG_);
            curTaskTimeInfoVec.emplace_back(std::make_pair(curTaskTimeStamp, timeoutMSG_.str()));
        }
    }

    {
        std::unique_lock lock(infoMutex_);
        for (auto& curTaskTimeInfo : curTaskTimeInfoVec) {
            // first为gid，second为下次触发超时的时间
            for (size_t i = 0; i < curTaskTimeInfo.first.first.size(); i++) {
                if (curTaskTimeInfo.first.second < UINT64_MAX && curTaskTimeInfo.first.first[i] != 0) {
                    ReportEventTimeout(curTaskTimeInfo.first.first[i], curTaskTimeInfo.second);
                    if (taskTimeoutInfo_.size() > MAX_RECORD_LIMIT) {
                        taskTimeoutInfo_.erase(taskTimeoutInfo_.begin());
                    }
                    taskTimeoutInfo_.emplace_back(std::make_pair(now, curTaskTimeInfo.second.str()));
                }

                if (curTaskTimeInfo.first.second < nextTaskStart) {
                    nextTaskStart = curTaskTimeInfo.first.second;
                }
            }
        }
    }
}

void QueueMonitor::ReportEventTimeout(uint64_t curGid, const std::stringstream& ss)
{
    std::string ssStr = ss.str();
    if (ffrt_task_timeout_get_cb()) {
        FFRTFacade::GetDWInstance().SubmitAsyncTask([curGid, ssStr] {
            ffrt_task_timeout_cb func = ffrt_task_timeout_get_cb();
            if (func) {
                func(curGid, ssStr.c_str(), ssStr.size());
            }
        });
    }
}

std::string QueueMonitor::DumpQueueTimeoutInfo()
{
    std::shared_lock<std::shared_mutex> lock(infoMutex_);
    std::stringstream ss;
    if (taskTimeoutInfo_.size() != 0) {
        for (auto it = taskTimeoutInfo_.rbegin(); it != taskTimeoutInfo_.rend(); ++it) {
            auto& record = *it;
            ss << "{" << FormatDateString4SteadyClock(record.first) << ", " << record.second << "} \n";
        }
    } else {
        ss << "Queue Timeout info Empty";
    }
    return ss.str();
}

void QueueMonitor::UpdateTimeoutUs()
{
    timeoutUs_ = ffrt_task_timeout_get_threshold() * US_PER_MS;
}
} // namespace ffrt
