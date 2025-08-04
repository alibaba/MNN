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
#ifndef FFRT_CONCURRENT_QUEUE_H
#define FFRT_CONCURRENT_QUEUE_H

#include "queue/base_queue.h"

namespace ffrt {
class ConcurrentQueue : public BaseQueue {
public:
    explicit ConcurrentQueue(const int maxConcurrency = 1)
        : maxConcurrency_(maxConcurrency)
    {
        dequeFunc_ = QueueStrategy<QueueTask>::DequeSingleByPriority;
        headTaskVec_.resize(maxConcurrency_);
    }
    ~ConcurrentQueue() override;

    int Push(QueueTask* task) override;
    QueueTask* Pull() override;
    int Remove() override;
    int Remove(const char* name) override;
    int Remove(const QueueTask* task) override;
    void Stop() override;
    int WaitAll() override;
    std::vector<QueueTask*> GetHeadTask() override;

    bool GetActiveStatus() override
    {
        return concurrency_.load();
    }

    int GetQueueType() const override
    {
        return ffrt_queue_concurrent;
    }

    inline uint64_t GetMapSize() override
    {
        std::lock_guard lock(mutex_);
        return whenMap_.size() + waitingMap_.size();
    }

    bool SetLoop(Loop* loop);
    bool HasTask(const char* name) override;

    inline bool ClearLoop()
    {
        if (loop_ == nullptr) {
            return false;
        }

        loop_ = nullptr;
        return true;
    }

    bool IsOnLoop() override
    {
        return isOnLoop_.load();
    }

private:
    int PushDelayTaskToTimer(QueueTask* task);
    int PushAndCalConcurrency(QueueTask* task, ffrt_queue_priority_t taskPriority, std::unique_lock<ffrt::mutex>& lock);
    void Stop(std::multimap<uint64_t, QueueTask*>& whenMap);

    Loop* loop_ { nullptr };
    std::atomic_bool isOnLoop_ { false };

    int maxConcurrency_ {1};
    std::vector<QueueTask*> headTaskVec_;
    std::atomic_int concurrency_ {0};

    bool waitingAll_ = false;
    std::multimap<uint64_t, QueueTask*> waitingMap_;
    std::multimap<uint64_t, QueueTask*> whenMapVec_[4];
    std::vector<std::pair<uint64_t, QueueTask*>> allWhenmapTask;
};

std::unique_ptr<BaseQueue> CreateConcurrentQueue(const ffrt_queue_attr_t* attr);
} // namespace ffrt
#endif // FFRT_CONCURRENT_QUEUE_H
