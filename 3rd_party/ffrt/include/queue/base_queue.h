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
#ifndef FFRT_BASE_QUEUE_H
#define FFRT_BASE_QUEUE_H

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <memory>
#include "c/queue.h"
#include "cpp/condition_variable.h"
#include "internal_inc/non_copyable.h"
#include "queue_strategy.h"

namespace ffrt {
class QueueTask;
class Loop;

enum QueueAction {
    INACTIVE = -1, // queue is nullptr or serial queue is empty
    SUCC,
    FAILED,
    CONCURRENT, // concurrency less than max concurrency
};

class BaseQueue : public NonCopyable {
public:
    BaseQueue();
    virtual ~BaseQueue() = default;

    virtual int Push(QueueTask* task) = 0;
    virtual QueueTask* Pull() = 0;
    virtual bool GetActiveStatus() = 0;
    virtual int GetQueueType() const = 0;
    virtual int Remove();
    virtual int Remove(const char* name);
    virtual int Remove(const QueueTask* task);
    virtual void Stop();
    virtual uint64_t GetDueTaskCount();

    virtual bool IsOnLoop()
    {
        return false;
    }

    virtual int WaitAll()
    {
        return -1;
    }

    virtual inline uint64_t GetMapSize()
    {
        std::lock_guard lock(mutex_);
        return whenMap_.size();
    }

    inline uint32_t GetQueueId() const
    {
        return queueId_;
    }

    inline bool DelayStatus()
    {
        return delayStatus_.load();
    }

    virtual bool HasTask(const char* name);
    virtual std::vector<QueueTask*> GetHeadTask();
    ffrt::mutex mutex_;
protected:
    inline uint64_t GetNow() const
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void Stop(std::multimap<uint64_t, QueueTask*>& whenMap);
    int Remove(std::multimap<uint64_t, QueueTask*>& whenMap);
    int Remove(const QueueTask* task, std::multimap<uint64_t, QueueTask*>& whenMap);
    int Remove(const char* name, std::multimap<uint64_t, QueueTask*>& whenMap);
    bool HasTask(const char* name, std::multimap<uint64_t, QueueTask*> whenMap);
    uint64_t GetDueTaskCount(std::multimap<uint64_t, QueueTask*>& whenMap);

    const uint32_t queueId_;
    std::atomic_bool delayStatus_ { false };
    bool isExit_ { false };
    std::atomic_bool isActiveState_ { false };
    std::multimap<uint64_t, QueueTask*> whenMap_;
    std::vector<QueueTask*> headTaskVec_;
    QueueStrategy<QueueTask>::DequeFunc dequeFunc_ { nullptr };

    ffrt::condition_variable cond_;
};

std::unique_ptr<BaseQueue> CreateQueue(int queueType, const ffrt_queue_attr_t* attr);
} // namespace ffrt

#endif // FFRT_BASE_QUEUE_H
