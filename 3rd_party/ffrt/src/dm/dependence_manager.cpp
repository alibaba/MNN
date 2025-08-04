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

#include "dm/dependence_manager.h"
#include "util/ffrt_facade.h"
#include "util/singleton_register.h"
#include "tm/io_task.h"
#include "tm/uv_task.h"

namespace ffrt {
DependenceManager& DependenceManager::Instance()
{
    return SingletonRegister<DependenceManager>::Instance();
}

void DependenceManager::RegistInsCb(SingleInsCB<DependenceManager>::Instance &&cb)
{
    SingletonRegister<DependenceManager>::RegistInsCb(std::move(cb));
}

void DependenceManager::onSubmitUV(ffrt_executor_task_t *task, const task_attr_private *attr)
{
    FFRT_TRACE_SCOPE(1, onSubmitUV);
    UVTask* uvTask = TaskFactory<UVTask>::Alloc();
    new(uvTask) UVTask(task, attr);
    FFRT_EXECUTOR_TASK_SUBMIT_MARKER(uvTask->gid);
    uvTask->Ready();
}

void DependenceManager::onSubmitIO(const ffrt_io_callable_t& work, const task_attr_private* attr)
{
    FFRT_TRACE_SCOPE(1, onSubmitIO);
    IOTask* ioTask = TaskFactory<IOTask>::Alloc();
    new (ioTask) IOTask(work, attr);
    ioTask->Ready();
}

int DependenceManager::onSkip(ffrt_task_handle_t handle)
{
    ffrt::CPUEUTask *task = static_cast<ffrt::CPUEUTask*>(handle);
    if (task->type == ffrt_queue_task) {
        FFRT_LOGE("use ffrt::queue::cancel instead of ffrt::skip for canceling queue task");
        return ffrt_error;
    }

    auto exp = ffrt::SkipStatus::SUBMITTED;
    if (__atomic_compare_exchange_n(&task->skipped, &exp, ffrt::SkipStatus::SKIPPED, 0, __ATOMIC_ACQUIRE,
        __ATOMIC_RELAXED)) {
        task->Cancel();
        return ffrt_success;
    }
    FFRT_LOGW("skip task [%lu] faild", task->gid);
    return ffrt_error;
}
} // namespace ffrt