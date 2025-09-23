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

#ifndef FFRT_STASK_SCHEDULER_HPP
#define FFRT_STASK_SCHEDULER_HPP
#include "sched/task_scheduler.h"

namespace ffrt {
class STaskScheduler : public TaskScheduler {
public:
    STaskScheduler()
    {
        que = std::make_unique<FIFOQueue>();
    }

    void SetQos(QoS &q) override
    {
        qos = q;
    }

    uint64_t GetGlobalTaskCnt() override
    {
        return que->Size();
    }

private:
    bool PushTaskGlobal(TaskBase* task, bool rtb) override;
    TaskBase* PopTaskHybridProcess() override;

    TaskBase* PopTaskGlobal() override
    {
        std::unique_lock<std::mutex> lock(*GetMutex());
        TaskBase* task = que->DeQueue();
        lock.unlock();
        if (task && task->type == ffrt_uv_task) {
            return GetUVTask(task);
        }
        return task;
    }
private:
    std::unique_ptr<FIFOQueue> que { nullptr };
};
}
#endif