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
#ifndef FFRT_QUEUE_TASK_H
#define FFRT_QUEUE_TASK_H

#include <atomic>
#include <regex>
#include "queue/queue_attr_private.h"
#include "queue/queue_handler.h"
#include "tm/task_factory.h"
#ifdef FFRT_ENABLE_HITRACE_CHAIN
#include "dfx/trace/ffrt_trace_chain.h"
#endif

#define GetQueueTaskByFuncStorageOffset(f)                                                                     \
    (reinterpret_cast<QueueTask*>(static_cast<uintptr_t>(static_cast<size_t>(reinterpret_cast<uintptr_t>(f)) - \
        (reinterpret_cast<size_t>(&((reinterpret_cast<QueueTask*>(0))->func_storage))))))

namespace ffrt {
class QueueTask : public CoTask {
public:
    explicit QueueTask(QueueHandler* handler, const task_attr_private* attr = nullptr, bool insertHead = false);
    ~QueueTask() override;

    void Wait();
    void Notify();
    void Destroy();

    uint32_t GetQueueId() const;

    inline uint64_t GetDelay() const
    {
        return delay_;
    }

    inline uint64_t GetUptime() const
    {
        return uptime_;
    }

    inline QueueHandler* GetHandler() const
    {
        return handler_;
    }

    inline bool GetFinishStatus() const
    {
        return isFinished_.load();
    }

    inline QueueTask* GetNextTask() const
    {
        return nextTask_;
    }

    inline void SetNextTask(QueueTask* task)
    {
        nextTask_ = task;
    }

    inline void SetPriority(const ffrt_queue_priority_t prio)
    {
        prio_ = prio;
    }

    inline ffrt_queue_priority_t GetPriority()
    {
        return prio_;
    }

    inline bool IsMatch(std::string name) const
    {
        std::string pattern = ".*_" + name + "_.*";
        return std::regex_match(label, std::regex(pattern));
    }

    inline bool InsertHead() const
    {
        return insertHead_;
    }

    inline uint64_t GetSchedTimeout() const
    {
        return schedTimeout_;
    }

    inline void SetMonitorTask(WaitUntilEntry* monitorWe)
    {
        monitorWe_ = monitorWe;
    }

    inline WaitUntilEntry* GetMonitorTask()
    {
        return monitorWe_;
    }

    inline void MonitorTaskStart()
    {
        isWeStart_.store(true);
    }

    inline bool IsMonitorTaskStart()
    {
        return isWeStart_;
    }
    int curTaskIdx = 0;

    void Prepare() override;
    void Ready() override;

    // dequeue means task has been pulled out from it's queue
    inline void Dequeue()
    {
        SetStatus(TaskStatus::DEQUEUED);
    }

    // pop means task has been popped from scheduler
    void Pop() override
    {
        SetStatus(TaskStatus::POPPED);
    }

    void Execute() override;

    BlockType Block() override
    {
        if (USE_COROUTINE && !threadMode_ && legacyCountNum <= 0 && (handler_ && !handler_->IsOnLoop())) {
            blockType = BlockType::BLOCK_COROUTINE;
            SetStatus(TaskStatus::COROUTINE_BLOCK);
        } else {
            blockType = BlockType::BLOCK_THREAD;
            SetStatus(TaskStatus::THREAD_BLOCK);
        }
        return blockType;
    }

    void Wake() override
    {
        SetStatus(TaskStatus::EXECUTING);
        blockType = BlockType::BLOCK_COROUTINE;
    }

    void Finish() override;

    void Cancel() override
    {
        FFRT_LOGD("cancel task[%llu] %s succ", gid, label.c_str());
        SetStatus(TaskStatus::CANCELED);
        Notify();
        Destroy();
    }

    void FreeMem() override;

    void SetQos(const QoS& newQos) override
    {
        qos_ = newQos;
    }

    BlockType GetBlockType() const override
    {
        return blockType;
    }

private:
    uint64_t uptime_;
    QueueHandler* handler_;
    bool insertHead_ = false;
    uint64_t delay_ = 0;
    uint64_t schedTimeout_ = 0;

    QueueTask* nextTask_ = nullptr;
    std::atomic_bool isFinished_ = {false};
    bool onWait_ = {false};
    std::atomic_bool isWeStart_ = {false};

    ffrt_queue_priority_t prio_ = ffrt_queue_priority_low;
    WaitUntilEntry* monitorWe_ = nullptr;
};
} // namespace ffrt

#endif // FFRT_QUEUE_TASK_H
