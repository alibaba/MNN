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

#include "queue/serial_queue.h"
#include "dfx/log/ffrt_log_api.h"
#include "tm/queue_task.h"

namespace {
constexpr uint32_t MIN_OVERLOAD_INTERVAL = 16;
constexpr uint32_t MAX_OVERLOAD_INTERVAL = 128;
}
namespace ffrt {
SerialQueue::SerialQueue()
{
    dequeFunc_ = QueueStrategy<QueueTask>::DequeBatch;
    overloadThreshold_ = MIN_OVERLOAD_INTERVAL;
}

SerialQueue::~SerialQueue()
{
    FFRT_LOGI("destruct serial queueId=%u leave", queueId_);
}

int SerialQueue::Push(QueueTask* task)
{
    std::lock_guard lock(mutex_);
    FFRT_COND_DO_ERR(isExit_, return FAILED, "cannot push task, [queueId=%u] is exiting", queueId_);

    if (!isActiveState_.load()) {
        isActiveState_.store(true);
        return INACTIVE;
    }

    if (task->InsertHead() && !whenMap_.empty()) {
        FFRT_LOGD("head insert task=%u in [queueId=%u]", task->gid, queueId_);
        uint64_t headTime = (whenMap_.begin()->first > 0) ? whenMap_.begin()->first - 1 : 0;
        whenMap_.insert({std::min(headTime, task->GetUptime()), task});
    } else {
        whenMap_.insert({task->GetUptime(), task});
    }

    if (task == whenMap_.begin()->second) {
        cond_.notify_one();
    } else if ((whenMap_.begin()->second->GetDelay() > 0) && (GetNow() > whenMap_.begin()->first)) {
        FFRT_LOGI("push task notify cond_wait.");
        cond_.notify_one();
    }

    if (whenMap_.size() >= overloadThreshold_) {
        FFRT_LOGW("[queueId=%u] overload warning, size=%llu", queueId_, whenMap_.size());
        overloadThreshold_ += std::min(overloadThreshold_, MAX_OVERLOAD_INTERVAL);
    }

    return SUCC;
}

QueueTask* SerialQueue::Pull()
{
    std::unique_lock lock(mutex_);
    // wait for delay task
    uint64_t now = GetNow();
    while (!whenMap_.empty() && now < whenMap_.begin()->first && !isExit_) {
        uint64_t diff = whenMap_.begin()->first - now;
        FFRT_LOGD("[queueId=%u] stuck in %llu us wait", queueId_, diff);
        delayStatus_.store(true);
        cond_.wait_for(lock, std::chrono::microseconds(diff));
        delayStatus_.store(false);
        FFRT_LOGD("[queueId=%u] wakeup from wait", queueId_);
        now = GetNow();
    }

    // abort dequeue in abnormal scenarios
    if (whenMap_.empty()) {
        FFRT_LOGD("[queueId=%u] switch into inactive", queueId_);
        isActiveState_.store(false);
        return nullptr;
    }
    FFRT_COND_DO_ERR(isExit_, return nullptr, "cannot pull task, [queueId=%u] is exiting", queueId_);

    if (overloadThreshold_ > MAX_OVERLOAD_INTERVAL && whenMap_.size() < MAX_OVERLOAD_INTERVAL) {
        overloadThreshold_ = MAX_OVERLOAD_INTERVAL;
    }
    // dequeue due tasks in batch
    return dequeFunc_(queueId_, now, &whenMap_, nullptr);
}

std::unique_ptr<BaseQueue> CreateSerialQueue(const ffrt_queue_attr_t* attr)
{
    (void)attr;
    return std::make_unique<SerialQueue>();
}
} // namespace ffrt
