/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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

#include "sched/task_scheduler.h"
#include <random>
#include "eu/execute_unit.h"
#include "util/ffrt_facade.h"

namespace {
const unsigned int LOCAL_QUEUE_SIZE = 128;
const int INSERT_GLOBAL_QUEUE_FREQ = 5;
const int GLOBAL_INTERVAL = 60;
const int STEAL_LOCAL_HALF = 2;
constexpr int UV_TASK_MAX_CONCURRENCY = 8;

void InsertTask(void *task)
{
    ffrt::TaskBase *baseTask = reinterpret_cast<ffrt::TaskBase *>(task);
    ffrt::FFRTFacade::GetSchedInstance()->GetScheduler(baseTask->qos_).PushTaskGlobal(baseTask);
}
} // namespace

namespace ffrt {
int PLACE_HOLDER = 0;
TaskBase *TaskScheduler::PopTask()
{
    TaskBase *task = nullptr;
    if (GetTaskSchedMode() == TaskSchedMode::LOCAL_TASK_SCHED_MODE) {
        task = PopTaskLocalOrPriority();
        if (task == nullptr) {
            StealTask();
            bool stealSuccees = false;
            if (GetLocalTaskCnt() > 0) {
                stealSuccees = true;
                task = reinterpret_cast<TaskBase *>(GetLocalQueue()->PopHead());
            }
            unsigned int *workerTickPtr = *GetWorkerTick();
            if (stealSuccees && workerTickPtr != nullptr) {
                *workerTickPtr = 1;
            }
        }
    } else if (GetTaskSchedMode() == TaskSchedMode::DEFAULT_TASK_SCHED_MODE) {
        task = PopTaskGlobal();
    }
    if (task) {
        task->Pop();
    }
    return task;
}

SpmcQueue *TaskScheduler::GetLocalQueue()
{
    thread_local static LocalQueue localQueue(qos, this->localQueues);
    return localQueue.localQueue;
}

void **TaskScheduler::GetPriorityTask()
{
    thread_local void *priorityTask{nullptr};
    return &priorityTask;
}

unsigned int **TaskScheduler::GetWorkerTick()
{
    thread_local unsigned int *workerTick{nullptr};
    return &workerTick;
}

int TaskScheduler::StealTask()
{
    std::lock_guard<std::mutex> lock(*GetMutex());
    if (GetStealingWorkers() > localQueues.size() / STEAL_LOCAL_HALF) {
        return 0;
    }

    AddStealingWorker();
    std::unordered_map<pid_t, SpmcQueue *>::iterator iter = localQueues.begin();
    while (iter != localQueues.end()) {
        SpmcQueue* queue = iter->second;
        unsigned int queueLen = queue->GetLength();
        if (queue != GetLocalQueue() && queueLen > 0) {
            unsigned int popLen = queue->PopHeadToAnotherQueue(*GetLocalQueue(), (queueLen + 1) / 2, InsertTask);
            SubStealingWorker();
            return popLen;
        }
        iter++;
    }
    SubStealingWorker();
    return 0;
}

void TaskScheduler::RemoveLocalQueue(SpmcQueue *localQueue)
{
    if (localQueue != nullptr) {
        localQueues.erase(syscall(SYS_gettid));
    }
}

uint64_t TaskScheduler::GetPriorityTaskCnt()
{
    if (*GetPriorityTask() != nullptr) {
        return 1;
    } else {
        return 0;
    }
}

bool TaskScheduler::PushTaskToPriorityStack(TaskBase *executorTask)
{
    if (*GetPriorityTask() == nullptr) {
        *GetPriorityTask() = reinterpret_cast<void *>(executorTask);
        return true;
    }
    return false;
}

void TaskScheduler::PushTaskLocalOrPriority(TaskBase *task)
{
    // in self-wakeup scenario, tasks are placed in local fifo to delay scheduling, implementing the yeild function
    bool selfWakeup = (ffrt::ExecuteCtx::Cur()->task == task);
    if (!selfWakeup) {
        if (PushTaskToPriorityStack(task)) {
            return;
        }

        if ((rand() % INSERT_GLOBAL_QUEUE_FREQ > 0)) {
            if (GetLocalQueue() != nullptr && GetLocalQueue()->PushTail(task) == 0) {
                ffrt::FFRTFacade::GetEUInstance().NotifyTask<TaskNotifyType::TASK_LOCAL>(task->qos_);
                return;
            }
        }
    }
    PushTaskGlobal(task);
}

TaskBase *TaskScheduler::PopTaskLocalOrPriority()
{
    TaskBase *task = nullptr;
    unsigned int *workerTickPtr = *GetWorkerTick();
    if ((workerTickPtr != nullptr) && (*workerTickPtr % GLOBAL_INTERVAL == 0)) {
        *workerTickPtr = 0;
        task = PopTaskHybridProcess();
        // the worker is not notified when the task attribute is set not to notify worker
        if (NeedNotifyWorker(task)) {
            FFRTFacade::GetEUInstance().NotifyTask<TaskNotifyType::TASK_PICKED>(qos);
        }
        if (task != nullptr) {
            return task;
        }
    }
    // preferentially pick up tasks from the priority unless the priority is empty or occupied
    void **priorityTaskPtr = GetPriorityTask();
    if (*priorityTaskPtr != nullptr) {
        task = reinterpret_cast<TaskBase *>(*priorityTaskPtr);
        *priorityTaskPtr = nullptr;
        if (reinterpret_cast<void *>(task) != &PLACE_HOLDER) {
            return task;
        }
    }
    return reinterpret_cast<TaskBase *>(GetLocalQueue()->PopHead());
}

bool TaskScheduler::PushUVTaskToWaitingQueue(UVTask* task)
{
    std::lock_guard lg(uvMtx);
    if (uvTaskConcurrency_ >= UV_TASK_MAX_CONCURRENCY) {
        uvTaskWaitingQueue_.push_back(task);
        return true;
    }

    return false;
}

bool TaskScheduler::CheckUVTaskConcurrency(UVTask* task)
{
    std::lock_guard lg(uvMtx);
    // the number of workers are executing UV tasks has reached the upper limit.
    // therefore, the current task is placed back to the head of the waiting queue (be preferentially obtained later).
    if (uvTaskConcurrency_ >= UV_TASK_MAX_CONCURRENCY) {
        uvTaskWaitingQueue_.push_front(task);
        return false;
    }

    uvTaskConcurrency_++;
    return true;
}

UVTask* TaskScheduler::PickWaitingUVTask()
{
    std::lock_guard lg(uvMtx);
    if (uvTaskWaitingQueue_.empty()) {
        if (uvTaskConcurrency_ > 0) {
            uvTaskConcurrency_--;
        }
        return nullptr;
    }

    UVTask* task = uvTaskWaitingQueue_.front();
    uvTaskWaitingQueue_.pop_front();
    task->SetDequeued();
    return task;
}

bool TaskScheduler::CancelUVWork(ffrt_executor_task_t* uvWork)
{
    std::lock_guard lg(uvMtx);
    if (!reinterpret_cast<LinkedList*>(uvWork->wq)->InList()) {
        FFRT_SYSEVENT_LOGW("the task has been picked, or has not been inserted");
        return false;
    }

    auto iter = std::remove_if(uvTaskWaitingQueue_.begin(), uvTaskWaitingQueue_.end(), [uvWork](UVTask* task) {
        if (task->uvWork == uvWork) {
            return true;
        }
        return false;
    });
    if (iter != uvTaskWaitingQueue_.end()) {
        uvTaskWaitingQueue_.erase(iter, uvTaskWaitingQueue_.end());
        return true;
    }
    
    auto it = cancelMap_.find(uvWork);
    if (it != cancelMap_.end()) {
        it->second++;
    } else {
        cancelMap_[uvWork] = 1;
    }
    return true;
}

std::mutex* TaskScheduler::GetMutex()
{
    /* We use acquire on load and release on store to enforce the
     * happens-before relationship between the mutex implicit
     * initialization and the publication of its address.
     * i.e. if a thread reads the address of the mutex then
     * it has been already initialized by the thread that published
     * its address.
     */
    auto curMtx = mtx.load(std::memory_order_acquire);
    if (curMtx == nullptr) {
        curMtx = &FFRTFacade::GetEUInstance().GetWorkerGroup(qos).mutex;
        mtx.store(curMtx, std::memory_order_release);
    }
    return curMtx;
}

SchedulerFactory &SchedulerFactory::Instance()
{
    static SchedulerFactory fac;
    return fac;
}

LocalQueue::LocalQueue(int qos, std::unordered_map<pid_t, SpmcQueue *> localQueues)
{
    this->qos = qos;
    localQueue = new SpmcQueue();
    localQueue->Init(LOCAL_QUEUE_SIZE);
    std::lock_guard<std::mutex> lock(*(FFRTFacade::GetSchedInstance()->GetScheduler(qos).GetMutex()));
    localQueues.emplace(syscall(SYS_gettid), localQueue);
}

LocalQueue::~LocalQueue()
{
    std::lock_guard<std::mutex> lock(*(FFRTFacade::GetSchedInstance()->GetScheduler(qos).GetMutex()));
    if (FFRT_LIKELY(localQueue != nullptr)) {
        FFRTFacade::GetSchedInstance()->GetScheduler(qos).RemoveLocalQueue(localQueue);
        delete localQueue;
        localQueue = nullptr;
    }
}
} // namespace ffrt
