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

#ifndef FFRT_TASK_SCHEDULER_HPP
#define FFRT_TASK_SCHEDULER_HPP
#include "sched/task_runqueue.h"
#include "tm/task_base.h"
#include "util/spmc_queue.h"

namespace ffrt {
extern int PLACE_HOLDER;

enum class TaskSchedMode : uint8_t {
    DEFAULT_TASK_SCHED_MODE = 0, // only use global queue
    LOCAL_TASK_SCHED_MODE, // only use local queue and priority slot
};

class TaskScheduler {
public:
    TaskScheduler() = default;
    virtual ~TaskScheduler() {}

    void PushTask(TaskBase *task)
    {
        if (GetTaskSchedMode() == TaskSchedMode::DEFAULT_TASK_SCHED_MODE) {
            PushTaskGlobal(task);
        } else if (GetTaskSchedMode() == TaskSchedMode::LOCAL_TASK_SCHED_MODE) {
            PushTaskLocalOrPriority(task);
        }
    }

    TaskBase* PopTask();

    virtual void SetQos(QoS &q) = 0;

    int qos {0};

    int StealTask();
    void RemoveLocalQueue(SpmcQueue* localQueue);
    SpmcQueue* GetLocalQueue();
    void** GetPriorityTask();
    unsigned int** GetWorkerTick();

    // global_queue.size + totalLocalTaskCnt, not include the PriorityTaskCnt
    uint64_t GetTotalTaskCnt()
    {
        uint64_t totalTaskCnt = GetGlobalTaskCnt();
        for (auto &localQueue : localQueues) {
            totalTaskCnt += localQueue.second->GetLength();
        }
        return totalTaskCnt;
    }
    // global_queue.size
    virtual uint64_t GetGlobalTaskCnt() = 0;

    // thread_local local_queue.size, not totalLocalTaskCnt
    inline uint64_t GetLocalTaskCnt()
    {
        return GetLocalQueue()->GetLength();
    }
    // thread_local priority.size, not totalPriorityTaskCnt
    uint64_t GetPriorityTaskCnt();

    inline void SetTaskSchedMode(const TaskSchedMode& mode)
    {
        taskSchedMode = mode;
    }

    inline const TaskSchedMode& GetTaskSchedMode()
    {
        return taskSchedMode;
    }

    inline SpmcQueue* GetWorkerLocalQueue(pid_t pid)
    {
        std::lock_guard lg(*GetMutex());
        return localQueues[pid];
    }

    virtual bool PushTaskGlobal(TaskBase* task, bool rtb = true) = 0;
    virtual TaskBase* PopTaskGlobal() = 0;

    bool CancelUVWork(ffrt_executor_task_t* uvWork);
    bool PushUVTaskToWaitingQueue(UVTask* task);
    bool CheckUVTaskConcurrency(UVTask* task);
    UVTask* PickWaitingUVTask();

    std::mutex* GetMutex();

    std::atomic_uint64_t stealWorkers { 0 };

protected:
    std::unordered_map<pid_t, SpmcQueue*> localQueues;
    TaskSchedMode taskSchedMode = TaskSchedMode::DEFAULT_TASK_SCHED_MODE;

    void PushTaskLocalOrPriority(TaskBase* task);
    TaskBase* PopTaskLocalOrPriority();

    // gloabl queue -> local queue -> priority slot
    virtual TaskBase* PopTaskHybridProcess() = 0;
    bool PushTaskToPriorityStack(TaskBase *executorTask);

    inline void AddStealingWorker()
    {
        stealWorkers.fetch_add(1);
    }

    void SubStealingWorker()
    {
        while (1) {
            uint64_t stealWorkersNum = stealWorkers.load();
            if (stealWorkersNum == 0) {
                return;
            }
            if (atomic_compare_exchange_weak(&stealWorkers, &stealWorkersNum, stealWorkersNum - 1)) return;
        }
    }

    inline uint64_t GetStealingWorkers()
    {
        return stealWorkers.load(std::memory_order_relaxed);
    }

    TaskBase* GetUVTask(TaskBase* task)
    {
        std::lock_guard<std::mutex> lg(uvMtx);
        UVTask* uvTask = static_cast<UVTask*>(task);
        auto it = cancelMap_.find(uvTask->uvWork);
        if (it != cancelMap_.end()) {
            uvTask->FreeMem();
            // the task has been canceled, remove it
            if (it->second == 1)
                cancelMap_.erase(it);
            else
                it->second--;
            return nullptr;
        }

        uvTask->SetDequeued();
        return task;
    }

private:
    std::atomic<std::mutex*> mtx {nullptr};
    std::mutex uvMtx;
    std::unordered_map<ffrt_executor_task_t*, uint32_t> cancelMap_;
    int uvTaskConcurrency_ = 0;
    std::deque<UVTask*> uvTaskWaitingQueue_;
};

class SchedulerFactory {
public:
    using AllocCB = std::function<TaskScheduler*()>;
    using RecycleCB = std::function<void (TaskScheduler*)>;

    static SchedulerFactory& Instance();

    static TaskScheduler *Alloc()
    {
        return Instance().alloc_();
    }

    static void Recycle(TaskScheduler *schd)
    {
        Instance().recycle_(schd);
    }

    static void RegistCb(const AllocCB &alloc, const RecycleCB &recycle)
    {
        Instance().alloc_ = alloc;
        Instance().recycle_ = recycle;
    }

private:
    AllocCB alloc_;
    RecycleCB recycle_;
};

struct LocalQueue {
    explicit LocalQueue(int qos, std::unordered_map<pid_t, SpmcQueue*> localQueues);
    ~LocalQueue();
    int qos {0};
    SpmcQueue* localQueue;
};
} // namespace ffrt

#endif
