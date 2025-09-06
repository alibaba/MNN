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

#include "eventhandler_adapter_queue.h"
#include <securec.h>
#include <sstream>
#include "dfx/log/ffrt_log_api.h"
#include "util/time_format.h"

namespace {
constexpr int MAX_DUMP_SIZE = 500;
constexpr uint8_t HISTORY_TASK_NUM_POWER = 32;

void DumpRunningTaskInfo(const char* tag, const ffrt::HistoryTask& currentRunningTask, std::ostringstream& oss)
{
    oss << tag << " Current Running: ";
    if (currentRunningTask.beginTime_ == std::numeric_limits<uint64_t>::max()) {
        oss << "{}";
    } else {
        oss << "start at " << ffrt::FormatDateString4SteadyClock(currentRunningTask.beginTime_) << ", ";
        oss << "Event { ";
        oss << "send thread = " << currentRunningTask.senderKernelThreadId_;
        oss << ", send time = " << ffrt::FormatDateString4SteadyClock(currentRunningTask.sendTime_);
        oss << ", handle time = " << ffrt::FormatDateString4SteadyClock(currentRunningTask.handleTime_);
        oss << ", task name = " << currentRunningTask.taskName_;
        oss << " }\n";
    }
}

void DumpHistoryTaskInfo(const char* tag, const std::vector<ffrt::HistoryTask>& historyTasks, std::ostringstream& oss)
{
    oss << tag << " History event queue information:\n";
    for (uint8_t i = 0; i < HISTORY_TASK_NUM_POWER; i++) {
        auto historyTask = historyTasks[i];
        if (historyTask.senderKernelThreadId_ == 0) {
            continue;
        }

        oss << tag << " No. " << (i + 1) << " : Event { ";
        oss << "send thread = " << historyTask.senderKernelThreadId_;
        oss << ", send time = " << ffrt::FormatDateString4SteadyClock(historyTask.sendTime_);
        oss << ", handle time = " << ffrt::FormatDateString4SteadyClock(historyTask.handleTime_);
        oss << ", trigger time = " << ffrt::FormatDateString4SteadyClock(historyTask.triggerTime_);
        oss << ", complete time = " << ffrt::FormatDateString4SteadyClock(historyTask.completeTime_);
        oss << ", task name = " << historyTask.taskName_;
        oss << " }\n";
    }
}

void DumpUnexecutedTaskInfo(const char* tag,
    const std::multimap<uint64_t, ffrt::QueueTask*>* whenMapVec, std::ostringstream& oss)
{
    static std::pair<ffrt_inner_queue_priority_t, std::string> priorityPairArr[] = {
        {ffrt_inner_queue_priority_immediate, "Immediate"}, {ffrt_inner_queue_priority_high, "High"},
        {ffrt_inner_queue_priority_low, "Low"}, {ffrt_inner_queue_priority_idle, "Idle"},
        {ffrt_inner_queue_priority_vip, "Vip"}
    };
    uint32_t total = 0;
    uint32_t dumpSize = MAX_DUMP_SIZE;

    std::multimap<ffrt_inner_queue_priority_t, ffrt::QueueTask*> priorityMap;
    for (int idx = 0; idx <= ffrt_inner_queue_priority_idle; idx++) {
        for (auto it = whenMapVec[idx].begin(); it != whenMapVec[idx].end(); it++) {
            priorityMap.insert({static_cast<ffrt_inner_queue_priority_t>(idx), it->second});
        }
    }

    auto taskDumpFun = [&](int n, ffrt::QueueTask* task) {
        oss << tag << " No. " << n << " : Event { ";
        oss << "send thread = " << task->fromTid;
        oss << ", send time = " << ffrt::FormatDateString4SteadyClock(task->GetUptime() - task->GetDelay());
        oss << ", handle time = " << ffrt::FormatDateString4SteadyClock(task->GetUptime());
        oss << ", task name = " << task->label;
        oss << " }\n";
        dumpSize--;
    };

    for (auto pair : priorityPairArr) {
        auto range = priorityMap.equal_range(pair.first);
        oss << tag << " " << pair.second << " priority event queue information:\n";
        int n = 0;
        for (auto it = range.first; it != range.second; ++it) {
            total++;
            n++;
            if (dumpSize > 0) {
                taskDumpFun(n, it->second);
            }
        }
        oss << tag << " Total size of " << pair.second << " events : " << n << "\n";
    }
    oss << tag << " Total event size : " << total << "\n";
}

uint64_t GetMinMapTime(const std::multimap<uint64_t, ffrt::QueueTask*>* whenMapVec)
{
    uint64_t minTime = std::numeric_limits<uint64_t>::max();

    for (int idx = 0; idx <= ffrt_inner_queue_priority_idle; idx++) {
        if (!whenMapVec[idx].empty()) {
            auto it = whenMapVec[idx].begin();
            if (it->first < minTime) {
                minTime = it->first;
            }
        }
    }
    return minTime;
}

bool WhenMapVecEmpty(const std::multimap<uint64_t, ffrt::QueueTask*>* whenMapVec)
{
    for (int idx = 0; idx <= ffrt_inner_queue_priority_idle; idx++) {
        if (!whenMapVec[idx].empty()) {
            return false;
        }
    }
    return true;
}
}

namespace ffrt {
EventHandlerAdapterQueue::EventHandlerAdapterQueue() : EventHandlerInteractiveQueue()
{
    dequeFunc_ = QueueStrategy<QueueTask>::DequeSingleAgainstStarvation;
    historyTasks_ = std::vector<HistoryTask>(HISTORY_TASK_NUM_POWER);
    pulledTaskCount_ = std::vector<int>(ffrt_inner_queue_priority_idle + 1, 0);
}

EventHandlerAdapterQueue::~EventHandlerAdapterQueue()
{
    FFRT_LOGI("destruct eventhandler adapter queueId=%u leave", queueId_);
}

void EventHandlerAdapterQueue::Stop()
{
    std::lock_guard lock(mutex_);
    for (auto& currentMap : whenMapVec_) {
        BaseQueue::Stop(currentMap);
    }
    FFRT_LOGI("clear [queueId=%u] succ", queueId_);
}

int EventHandlerAdapterQueue::Push(QueueTask* task)
{
    std::lock_guard lock(mutex_);
    FFRT_COND_DO_ERR(isExit_, return FAILED, "cannot push task, [queueId=%u] is exiting", queueId_);

    int taskPriority = task->GetPriority();
    if (!isActiveState_.load()) {
        pulledTaskCount_[taskPriority]++;
        isActiveState_.store(true);
        return INACTIVE;
    }

    if (task->InsertHead()) {
        std::multimap<uint64_t, QueueTask*> tmpWhenMap {{0, task}};
        tmpWhenMap.insert(whenMapVec_[taskPriority].begin(), whenMapVec_[taskPriority].end());
        whenMapVec_[taskPriority].swap(tmpWhenMap);
    } else {
        whenMapVec_[taskPriority].insert({task->GetUptime(), task});
    }
    if (task == whenMapVec_[taskPriority].begin()->second) {
        cond_.notify_one();
    }

    return SUCC;
}

QueueTask* EventHandlerAdapterQueue::Pull()
{
    std::unique_lock lock(mutex_);
    // wait for delay task
    uint64_t now = GetNow();
    uint64_t minMaptime = GetMinMapTime(whenMapVec_);
    while (!WhenMapVecEmpty(whenMapVec_) && now < minMaptime && !isExit_) {
        uint64_t diff = minMaptime - now;
        FFRT_LOGD("[queueId=%u] stuck in %llu us wait", queueId_, diff);
        delayStatus_.store(true);
        cond_.wait_for(lock, std::chrono::microseconds(diff));
        delayStatus_.store(false);
        FFRT_LOGD("[queueId=%u] wakeup from wait", queueId_);
        now = GetNow();
        minMaptime = GetMinMapTime(whenMapVec_);
    }

    // abort dequeue in abnormal scenarios
    if (WhenMapVecEmpty(whenMapVec_)) {
        FFRT_LOGD("[queueId=%u] switch into inactive", queueId_);
        isActiveState_.store(false);
        return nullptr;
    }
    FFRT_COND_DO_ERR(isExit_, return nullptr, "cannot pull task, [queueId=%u] is exiting", queueId_);

    // dequeue due tasks in batch
    return dequeFunc_(queueId_, now, whenMapVec_, &pulledTaskCount_);
}

int EventHandlerAdapterQueue::Remove()
{
    std::lock_guard lock(mutex_);
    int count = 0;
    for (auto& currentMap : whenMapVec_) {
        count += BaseQueue::Remove(currentMap);
    }
    return count;
}

int EventHandlerAdapterQueue::Remove(const char* name)
{
    std::lock_guard lock(mutex_);
    int count = 0;
    for (auto& currentMap : whenMapVec_) {
        count += BaseQueue::Remove(name, currentMap);
    }
    return count;
}

int EventHandlerAdapterQueue::Remove(const QueueTask* task)
{
    std::lock_guard lock(mutex_);
    int count = 0;
    for (auto& currentMap : whenMapVec_) {
        count += BaseQueue::Remove(task, currentMap);
    }
    return count;
}

bool EventHandlerAdapterQueue::HasTask(const char* name)
{
    std::lock_guard lock(mutex_);
    for (auto& currentMap : whenMapVec_) {
        if (BaseQueue::HasTask(name, currentMap)) {
            return true;
        }
    }
    return false;
}

bool EventHandlerAdapterQueue::IsIdle()
{
    std::lock_guard lock(mutex_);

    for (int idx = 0; idx <= ffrt_queue_priority_idle; idx++) {
        if (!whenMapVec_[idx].empty()) {
            return false;
        }
    }
    return true;
}

uint64_t EventHandlerAdapterQueue::GetDueTaskCount()
{
    std::lock_guard lock(mutex_);
    uint64_t count = 0;
    for (auto& currentMap : whenMapVec_) {
        count += BaseQueue::GetDueTaskCount(currentMap);
    }
    if (count != 0) {
        FFRT_LOGI("qid = %llu Current eh queueDue Task Count %u", GetQueueId(), count);
    }
    return count;
}

int EventHandlerAdapterQueue::Dump(const char* tag, char* buf, uint32_t len, bool historyInfo)
{
    std::lock_guard lock(mutex_);
    std::ostringstream oss;
    if (historyInfo) {
        DumpRunningTaskInfo(tag, currentRunningTask_, oss);
        DumpHistoryTaskInfo(tag, historyTasks_, oss);
    }
    DumpUnexecutedTaskInfo(tag, whenMapVec_, oss);
    return snprintf_s(buf, len, len - 1, "%s", oss.str().c_str());
}

int EventHandlerAdapterQueue::DumpSize(ffrt_inner_queue_priority_t priority)
{
    std::lock_guard lock(mutex_);
    return static_cast<int>(whenMapVec_[priority].size());
}

void EventHandlerAdapterQueue::SetCurrentRunningTask(QueueTask* task)
{
    currentRunningTask_ = HistoryTask(GetNow(), task);
}

void EventHandlerAdapterQueue::PushHistoryTask(QueueTask* task, uint64_t triggerTime, uint64_t completeTime)
{
    HistoryTask historyTask;
    historyTask.senderKernelThreadId_ = task->fromTid;
    historyTask.taskName_ = task->label;
    historyTask.sendTime_ = task->GetUptime() - task->GetDelay();
    historyTask.handleTime_ = task->GetUptime();
    historyTask.triggerTime_ = triggerTime;
    historyTask.completeTime_ = completeTime;
    historyTasks_[historyTaskIndex_.fetch_add(1) & (HISTORY_TASK_NUM_POWER - 1)] = historyTask;
}

std::unique_ptr<BaseQueue> CreateEventHandlerAdapterQueue(const ffrt_queue_attr_t* attr)
{
    (void)attr;
    return std::make_unique<EventHandlerAdapterQueue>();
}
} // namespace ffrt
