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
#ifndef FFRT_QUEUE_HANDLER_H
#define FFRT_QUEUE_HANDLER_H

#include <atomic>
#include <memory>
#include <string>
#include <shared_mutex>
#include <unordered_map>
#include "c/queue.h"
#include "c/queue_ext.h"
#include "cpp/task.h"
#include "base_queue.h"
#include "sched/execute_ctx.h"
#include "traffic_record.h"
#include "tm/task_base.h"

namespace ffrt {
class QueueTask;
class SerialQueue;
class Loop;

class QueueHandler {
public:
    QueueHandler(const char* name, const ffrt_queue_attr_t* attr, const int type = ffrt_queue_serial);
    ~QueueHandler();

    void Cancel();
    void CancelAndWait();
    int Cancel(const char* name);
    int Cancel(QueueTask* task);
    void Dispatch(QueueTask* inTask);
    void Submit(QueueTask* task);
    void TransferTask(QueueTask* task);
    void TransferInitTask();

    std::string GetDfxInfo(int index) const;
    std::pair<std::vector<uint64_t>, uint64_t> EvaluateTaskTimeout(uint64_t timeoutThreshold, uint64_t timeoutUs,
        std::stringstream& ss);

    bool SetLoop(Loop* loop);
    bool ClearLoop();

    inline bool IsOnLoop() const
    {
        return queue_->IsOnLoop();
    }

    QueueTask* PickUpTask();

    inline bool IsValidForLoop()
    {
        return !isUsed_.load() && (queue_->GetQueueType() == ffrt_queue_concurrent
               || queue_->GetQueueType() == ffrt_queue_eventhandler_interactive);
    }

    inline std::string GetName()
    {
        return name_;
    }

    inline uint32_t GetQueueId()
    {
        FFRT_COND_DO_ERR((queue_ == nullptr), return 0, "queue construct failed");
        return queue_->GetQueueId();
    }

    inline uint32_t GetExecTaskId() const
    {
        return execTaskId_.load();
    }

    inline bool HasTask(const char* name)
    {
        FFRT_COND_DO_ERR((queue_ == nullptr), return false, "[queueId=%u] constructed failed", GetQueueId());
        return queue_->HasTask(name);
    }

    inline uint64_t GetTaskCnt()
    {
        FFRT_COND_DO_ERR((queue_ == nullptr), return false, "[queueId=%u] constructed failed", GetQueueId());
        return queue_->GetMapSize();
    }

    inline int WaitAll()
    {
        FFRT_COND_DO_ERR((queue_ == nullptr), return -1, "[queueId=%u] constructed failed", GetQueueId());
        return queue_->WaitAll();
    }

    inline uint64_t GetQueueDueCount()
    {
        FFRT_COND_DO_ERR((queue_ == nullptr), return 0, "[queueId=%u] constructed failed", GetQueueId());
        return queue_->GetDueTaskCount();
    }

    inline bool CheckDelayStatus()
    {
        return queue_->DelayStatus();
    }

    bool IsIdle();
    void SetEventHandler(void* eventHandler);
    void* GetEventHandler();

    int Dump(const char* tag, char* buf, uint32_t len, bool historyInfo = true);
    int DumpSize(ffrt_inner_queue_priority_t priority);

    inline const std::unique_ptr<BaseQueue>& GetQueue()
    {
        return queue_;
    }

    inline int GetType()
    {
        return queue_->GetQueueType();
    }

    inline bool GetMode()
    {
        return threadMode_;
    }

private:
    void Deliver();
    void SetTimeoutMonitor(QueueTask* task);
    void RemoveTimeoutMonitor(QueueTask* task);
    void RunTimeOutCallback(QueueTask* task);

    void ReportTimeout(const std::vector<std::pair<uint64_t, std::string>>& timeoutTaskInfo);
    bool ControlTimeoutFreq(uint64_t timeoutCnt);
    void CheckSchedDeadline();
    bool CheckExecutingTask();
    void SendSchedTimer(TimePoint delay);
    void AddSchedDeadline(QueueTask* task);
    void RemoveSchedDeadline(QueueTask* task);
    void ReportTaskTimeout(uint64_t timeoutUs, std::stringstream& ss, int index);
    uint64_t CheckTimeSchedule(uint64_t time, uint64_t timeoutUs);

    void SetCurTask(QueueTask* task);
    void UpdateCurTask(QueueTask* task);

    // queue info
    std::string name_;
    int qos_ = qos_default;
    std::unique_ptr<BaseQueue> queue_ = nullptr;
    std::atomic_bool isUsed_ = false;
    std::atomic_uint64_t execTaskId_ = 0;
    int maxConcurrency_ = 1;
    std::vector<QueueTask*> curTaskVec_;
    uint64_t desWaitCnt_ = 0;

    // for timeout watchdog
    uint64_t timeout_ = 0;
    std::vector<TimeoutTask> timeoutTaskVec_;
    std::atomic_int delayedCbCnt_ = {0};
    ffrt_function_header_t* timeoutCb_ = nullptr;
    TrafficRecord trafficRecord_;
    uint64_t trafficRecordInterval_ = DEFAULT_TRAFFIC_INTERVAL;

    ffrt::mutex mutex_;
    bool initSchedTimer_ = false;
    WaitUntilEntry* we_ = nullptr;
    std::unordered_map<QueueTask*, uint64_t> schedDeadline_;
    std::atomic_int deliverCnt_ = {0};
    bool threadMode_ = false;
};
} // namespace ffrt

#endif // FFRT_QUEUE_HANDLER_H
