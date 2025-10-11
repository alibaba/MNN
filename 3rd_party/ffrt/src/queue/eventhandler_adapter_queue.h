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
#ifndef FFRT_EVENTHANDLER_ADAPTER_QUEUE_H
#define FFRT_EVENTHANDLER_ADAPTER_QUEUE_H

#include <vector>
#include <numeric>
#include "tm/queue_task.h"
#include "queue/base_queue.h"
#include "eventhandler_interactive_queue.h"

namespace ffrt {
struct HistoryTask {
    int32_t senderKernelThreadId_{0};
    std::string taskName_{0};
    uint64_t sendTime_{0};
    uint64_t handleTime_{0};
    uint64_t beginTime_{0};
    uint64_t triggerTime_{0};
    uint64_t completeTime_{0};

    HistoryTask()
    {
        beginTime_ = std::numeric_limits<uint64_t>::max();
    }

    HistoryTask(uint64_t beginTime, QueueTask* task)
    {
        beginTime_ = beginTime;
        senderKernelThreadId_ = task->fromTid;
        sendTime_ = task->GetUptime() - task->GetDelay();
        taskName_ = task->label;
        handleTime_ = task->GetUptime();
    }
};

class EventHandlerAdapterQueue : public EventHandlerInteractiveQueue {
public:
    explicit EventHandlerAdapterQueue();
    ~EventHandlerAdapterQueue() override;

    int Push(QueueTask* task) override;
    QueueTask* Pull() override;

    bool GetActiveStatus() override
    {
        std::lock_guard lock(mutex_);
        return isActiveState_.load();
    }

    int GetQueueType() const override
    {
        return ffrt_queue_eventhandler_adapter;
    }

    uint64_t GetMapSize() override
    {
        std::unique_lock lock(mutex_);
        return std::accumulate(std::begin(whenMapVec_), std::end(whenMapVec_), 0u,
            [] (uint64_t size, const std::multimap<uint64_t, QueueTask*>& whenMap) { return size + whenMap.size(); });
    }

    void Stop() override;
    bool HasTask(const char* name) override;
    int Remove() override;
    int Remove(const char* name) override;
    int Remove(const QueueTask* task) override;
    uint64_t GetDueTaskCount() override;

    bool IsIdle();
    int Dump(const char* tag, char* buf, uint32_t len, bool historyInfo = true);
    int DumpSize(ffrt_inner_queue_priority_t priority);

    void SetCurrentRunningTask(QueueTask* task);
    void PushHistoryTask(QueueTask* task, uint64_t triggerTime, uint64_t completeTime);

private:
    HistoryTask currentRunningTask_;
    std::vector<HistoryTask> historyTasks_;
    std::atomic_uint8_t historyTaskIndex_ {0};
    std::vector<int> pulledTaskCount_;
    std::multimap<uint64_t, QueueTask*> whenMapVec_[5];
};

std::unique_ptr<BaseQueue> CreateEventHandlerAdapterQueue(const ffrt_queue_attr_t* attr);
} // namespace ffrt

#endif // FFRT_EVENTHANDLER_ADAPTER_QUEUE_H
