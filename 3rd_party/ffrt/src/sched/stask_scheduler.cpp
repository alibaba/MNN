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
#include "sched/stask_scheduler.h"
#include "util/ffrt_facade.h"

namespace {
constexpr int TASK_OVERRUN_THRESHOLD = 1000;
constexpr int TASK_OVERRUN_ALARM_FREQ = 500;
}

namespace ffrt {
bool STaskScheduler::PushTaskGlobal(TaskBase* task, bool rtb)
{
    int taskCount = 0;
    (void)rtb; // rtb is deprecated here
    FFRT_COND_DO_ERR((task == nullptr), return false, "task is nullptr");

    int level = task->GetQos();
    uint64_t gid = task->gid;
    std::string label = task->GetLabel();

    FFRT_READY_MARKER(gid); // ffrt normal task ready to enqueue
    {
        std::lock_guard lg(*GetMutex());
        // enqueue task and read size under lock-protection
        que->EnQueue(task);
        taskCount = que->Size();
    }
    // The ownership of the task belongs to ReadyTaskQueue, and the task cannot be accessed any more.
    if (taskCount >= TASK_OVERRUN_THRESHOLD && taskCount % TASK_OVERRUN_ALARM_FREQ == 0) {
        FFRT_SYSEVENT_LOGW("qos [%d], task [%s] entered q, task count [%d] exceeds threshold.",
            level, label.c_str(), taskCount);
    }

    return taskCount == 1; // whether it's rising edge
}

TaskBase* STaskScheduler::PopTaskHybridProcess()
{
    TaskBase *task = PopTaskGlobal();
    if (task == nullptr) {
        return nullptr;
    }
    int wakedWorkerNum = FFRTFacade::GetEUInstance().GetWorkerGroup(qos).executingNum;
    // when there is only one worker, the global queue is equivalent to the local queue
    // prevents local queue tasks that cannot be executed due to blocking tasks
    if (wakedWorkerNum <= 1) {
        return task;
    }

    SpmcQueue *queue = GetLocalQueue();
    int expectedTask = GetGlobalTaskCnt() / wakedWorkerNum - 1;
    for (int i = 0; i < expectedTask; i++) {
        if (queue->GetLength() == queue->GetCapacity()) {
            return task;
        }

        TaskBase *task2local = PopTaskGlobal();
        if (task2local == nullptr) {
            return task;
        }
        queue->PushTail(task2local);
    }
    if (NeedNotifyWorker(task)) {
        FFRTFacade::GetEUInstance().NotifyTask<TaskNotifyType::TASK_PICKED>(qos);
    }
    return task;
}
}
