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
#ifndef FFRT_QUEUE_MONITOR_H
#define FFRT_QUEUE_MONITOR_H

#include <vector>
#include <sstream>
#include <shared_mutex>
#include "util/slab.h"
#include "sched/execute_ctx.h"
#include "tm/queue_task.h"

namespace ffrt {
class QueueHandler;
class QueueMonitor {
public:
    static QueueMonitor &GetInstance();
    void RegisterQueue(QueueHandler* queue);
    void DeregisterQueue(QueueHandler* queue);
    void UpdateQueueInfo();
    std::string DumpQueueTimeoutInfo();

private:
    QueueMonitor();
    ~QueueMonitor();
    QueueMonitor(const QueueMonitor &) = delete;
    QueueMonitor(QueueMonitor &&) = delete;
    QueueMonitor &operator=(const QueueMonitor &) = delete;
    QueueMonitor &operator=(QueueMonitor &&) = delete;

    void SetAlarm(uint64_t steadyUs);
    void ScheduleAlarm();
    void CheckTimeout(uint64_t& nextTaskStart);
    void ReportEventTimeout(uint64_t curGid, const std::stringstream& ss);
    void UpdateTimeoutUs();

    WaitUntilEntry* we_ = nullptr;
    std::atomic<uint64_t> timeoutUs_ = 0;
    std::stringstream timeoutMSG_;
    std::shared_mutex infoMutex_;
    std::atomic_bool suspendAlarm_ = {true};
    std::vector<QueueHandler*> queuesInfo_;
    std::vector<std::pair<uint64_t, std::string>> taskTimeoutInfo_;
};
} // namespace ffrt

#endif // FFRT_QUEUE_MONITOR_H
