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

#include "concurrent_queue.h"
#include <climits>
#include "dfx/log/ffrt_log_api.h"
#include "tm/queue_task.h"
#include "eu/loop.h"

namespace {
uint64_t GetMinMapTime(const std::multimap<uint64_t, ffrt::QueueTask*>* whenMapVec)
{
    uint64_t minTime = std::numeric_limits<uint64_t>::max();

    for (int idx = 0; idx <= ffrt_queue_priority_idle; idx++) {
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
    for (int idx = 0; idx <= ffrt_queue_priority_idle; idx++) {
        if (!whenMapVec[idx].empty()) {
            return false;
        }
    }
    return true;
}
}

namespace ffrt {
static void DelayTaskCb(void* task)
{
    static_cast<QueueTask*>(task)->Execute();
}

ConcurrentQueue::~ConcurrentQueue()
{
    FFRT_LOGI("destruct concurrent queueId=%u leave", queueId_);
}

int ConcurrentQueue::Push(QueueTask* task)
{
    std::unique_lock lock(mutex_);
    FFRT_COND_DO_ERR(isExit_, return FAILED, "cannot push task, [queueId=%u] is exiting", queueId_);
    ffrt_queue_priority_t taskPriority = task->GetPriority();
    if (taskPriority > ffrt_queue_priority_idle) {
        task->SetPriority(ffrt_queue_priority_low);
        taskPriority = task->GetPriority();
    }

    if (waitingAll_) {
        waitingMap_.insert({task->GetUptime(), task});
        return SUCC;
    }

    if (loop_ != nullptr) {
        if (task->GetDelay() == 0) {
            whenMapVec_[taskPriority].insert({task->GetUptime(), task});
            loop_->WakeUp();
            return SUCC;
        }
        return PushDelayTaskToTimer(task);
    }
    return PushAndCalConcurrency(task, taskPriority, lock);
}

QueueTask* ConcurrentQueue::Pull()
{
    std::unique_lock lock(mutex_);
    // wait for delay task
    uint64_t now = GetNow();
    if (loop_ != nullptr) {
        if (!WhenMapVecEmpty(whenMapVec_) && now >= GetMinMapTime(whenMapVec_) && !isExit_) {
            return dequeFunc_(queueId_, now, whenMapVec_, nullptr);
        }
        return nullptr;
    }

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
        int oldValue = concurrency_.fetch_sub(1); // 取不到后继的task，当前这个task正式退出
        FFRT_LOGD("concurrency[%d] - 1 [queueId=%u] switch into inactive", oldValue, queueId_);
        if (oldValue == 1) {
            cond_.notify_all();
        }
        return nullptr;
    }
    FFRT_COND_DO_ERR(isExit_, return nullptr, "cannot pull task, [queueId=%u] is exiting", queueId_);

    // dequeue next expired task by priority
    return dequeFunc_(queueId_, now, whenMapVec_, nullptr);
}

int ConcurrentQueue::Remove()
{
    std::lock_guard lock(mutex_);
    uint64_t removeCount = 0;
    for (auto& currentMap : whenMapVec_) {
        removeCount += BaseQueue::Remove(currentMap);
    }
    return removeCount + BaseQueue::Remove(waitingMap_);
}

int ConcurrentQueue::Remove(const char* name)
{
    std::lock_guard lock(mutex_);
    uint64_t removeCount = 0;
    for (auto& currentMap : whenMapVec_) {
        removeCount += BaseQueue::Remove(name, currentMap);
    }
    return removeCount + BaseQueue::Remove(name, waitingMap_);
}

int ConcurrentQueue::Remove(const QueueTask* task)
{
    std::lock_guard lock(mutex_);
    for (auto& currentMap : whenMapVec_) {
        if (BaseQueue::Remove(task, currentMap) == SUCC) {
            return SUCC;
        }
    }
    return BaseQueue::Remove(task, waitingMap_);
}

void ConcurrentQueue::Stop()
{
    std::lock_guard lock(mutex_);
    isExit_ = true;

    for (int idx = 0; idx <= ffrt_queue_priority_idle; idx++) {
        for (auto it = whenMapVec_[idx].begin(); it != whenMapVec_[idx].end(); it++) {
            if (it->second) {
                it->second->Notify();
                it->second->Destroy();
            }
        }
        whenMapVec_[idx].clear();
    }
    Stop(waitingMap_);
    if (loop_ == nullptr) {
        cond_.notify_all();
    }

    FFRT_LOGI("clear [queueId=%u] succ", queueId_);
}

int ConcurrentQueue::WaitAll()
{
    std::unique_lock lock(mutex_);
    FFRT_COND_DO_ERR(isOnLoop_, return -1, "loop does not support");
    FFRT_COND_DO_ERR(isExit_, return -1, "cannot waiting task, [queueId=%u] is exiting", queueId_);

    if (waitingAll_) {
        return 1;
    }

    waitingAll_ = true;
    cond_.wait(lock, [this] { return concurrency_.load() == 0;}); // 是否需要加上whenMap_empty()

    if (waitingMap_.empty()) {
        waitingAll_ = false;
        return 0 ;
    }

    // resubmit tasks to the queue and wake up workers
    for (const auto& iter : waitingMap_) {
        QueueTask* task = iter.second;
        QueueHandler* handler = task->GetHandler();
        int ret = PushAndCalConcurrency(task, task->GetPriority(), lock);
        if (ret == CONCURRENT) {
            if (task->GetDelay() == 0) {
                handler->TransferTask(task);
            } else {
                handler->TransferInitTask();
            }
        }
    }

    waitingAll_ = false;
    waitingMap_.clear();
    return 0;
}

bool ConcurrentQueue::SetLoop(Loop* loop)
{
    if (loop == nullptr || loop_ != nullptr) {
        FFRT_SYSEVENT_LOGE("queueId %s should bind to loop invalid", queueId_);
        return false;
    }

    loop_ = loop;
    isOnLoop_.store(true);
    return true;
}

int ConcurrentQueue::PushDelayTaskToTimer(QueueTask* task)
{
    uint64_t delayMs = (task->GetDelay() - 1) / 1000 + 1;
    int timeout = delayMs > INT_MAX ? INT_MAX : delayMs;
    if (loop_->TimerStart(timeout, task, DelayTaskCb, false) < 0) {
        FFRT_SYSEVENT_LOGE("push delay queue task to timer fail");
        return FAILED;
    }
    return SUCC;
}

int ConcurrentQueue::PushAndCalConcurrency(QueueTask* task, ffrt_queue_priority_t taskPriority, std::unique_lock<ffrt::mutex>& lock)
{
    if (concurrency_.load() < maxConcurrency_) {
        int oldValue = concurrency_.fetch_add(1);
        FFRT_LOGD("task [gid=%llu] concurrency[%u] + 1 [queueId=%u]", task->gid, oldValue, queueId_);

        if (task->GetDelay() > 0) {
            whenMapVec_[taskPriority].insert({task->GetUptime(), task});
        }

        return CONCURRENT;
    }

    whenMapVec_[taskPriority].insert({task->GetUptime(), task});
    if (task == whenMapVec_[taskPriority].begin()->second) {
        lock.unlock();
        cond_.notify_all();
    }

    return SUCC;
}

void ConcurrentQueue::Stop(std::multimap<uint64_t, QueueTask*>& whenMap)
{
    for (auto it = whenMap.begin(); it != whenMap.end(); it++) {
        if (it->second) {
            it->second->Notify();
            it->second->Destroy();
        }
    }

    whenMap.clear();
}

std::unique_ptr<BaseQueue> CreateConcurrentQueue(const ffrt_queue_attr_t* attr)
{
    int maxConcurrency = ffrt_queue_attr_get_max_concurrency(attr) <= 0 ? 1 : ffrt_queue_attr_get_max_concurrency(attr);
    return std::make_unique<ConcurrentQueue>(maxConcurrency);
}

bool ConcurrentQueue::HasTask(const char* name)
{
    std::lock_guard lock(mutex_);
    for (auto& currentMap : whenMapVec_) {
        if (BaseQueue::HasTask(name, currentMap)) {
            return true;
        }
    }
    return false;
}

std::vector<QueueTask*> ConcurrentQueue::GetHeadTask()
{
    std::lock_guard lock(mutex_);
    if (WhenMapVecEmpty(whenMapVec_)) {
        return {};
    }

    allWhenmapTask.clear();

    for (int idx = 0; idx <= ffrt_queue_priority_idle; idx++) {
        if (!whenMapVec_[idx].empty()) {
            for (const auto& [upTime, qtask] : whenMapVec_[idx]) {
                allWhenmapTask.emplace_back(upTime, qtask);
            }
        }
    }

    std::sort(allWhenmapTask.begin(), allWhenmapTask.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.first < rhs.first;
                });

    for (size_t i = 0; i < maxConcurrency_; i++) {
        if (i < allWhenmapTask.size()) {
            headTaskVec_[i] = allWhenmapTask[i].second;
        } else {
            headTaskVec_[i] = nullptr;
        }
    }
    return headTaskVec_;
}
} // namespace ffrt
