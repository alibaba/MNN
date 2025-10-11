/*
* Copyright (c) 2024 Huawei Device Co., Ltd.
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

#include "tm/uv_task.h"
#include "eu/func_manager.h"
#include "dfx/log/ffrt_log_api.h"
#include "dfx/trace_record/ffrt_trace_record.h"
#include "dfx/trace/ffrt_trace.h"
#include "util/ffrt_facade.h"

namespace ffrt {
void UVTask::Ready()
{
    QoS taskQos = qos_;
    FFRTTraceRecord::TaskSubmit<ffrt_uv_task>(taskQos);
    if (FFRTFacade::GetSchedInstance()->GetScheduler(taskQos).PushUVTaskToWaitingQueue(this)) {
        FFRTTraceRecord::TaskEnqueue<ffrt_uv_task>(taskQos);
        return;
    }
    SetStatus(TaskStatus::READY);
    bool isRisingEdge = FFRTFacade::GetSchedInstance()->GetScheduler(taskQos).PushTaskGlobal(this, false);
    FFRTTraceRecord::TaskEnqueue<ffrt_uv_task>(taskQos);
    FFRTFacade::GetEUInstance().NotifyTask<TaskNotifyType::TASK_ADDED_RTQ>(taskQos, false, isRisingEdge);
}

void UVTask::Execute()
{
    if (uvWork == nullptr) {
        FFRT_SYSEVENT_LOGE("task is nullptr");
        DecDeleteRef();
        return;
    }

    // if the concurrency reaches the upper limit, this worker cannot execute UV tasks.
    if (!FFRTFacade::GetSchedInstance()->GetScheduler(qos_).CheckUVTaskConcurrency(this)) {
        return;
    }

    ffrt_executor_task_func func = FuncManager::Instance()->getFunc(ffrt_uv_task);
    if (func == nullptr) {
        FFRT_SYSEVENT_LOGE("Static func is nullptr");
        DecDeleteRef();
        return;
    }

    ExecuteImpl(this, func);
}

void UVTask::ExecuteImpl(UVTask* task, ffrt_executor_task_func func)
{
    ExecuteCtx* ctx = ExecuteCtx::Cur();
    QoS taskQos = task->qos_;
    while (task != nullptr) {
        ctx->task = task;
        ctx->lastGid_ = task->gid;
        task->SetStatus(TaskStatus::EXECUTING);
        FFRTTraceRecord::TaskExecute<ffrt_uv_task>(taskQos);
        FFRT_EXECUTOR_TASK_BEGIN(task->gid);
        func(task->uvWork, taskQos);
        FFRT_TASK_END();
        FFRT_TASKDONE_MARKER(task->gid); // task finish marker for uv task
        FFRTTraceRecord::TaskDone<ffrt_uv_task>(taskQos);
        task->DecDeleteRef();
        task = FFRTFacade::GetSchedInstance()->GetScheduler(taskQos).PickWaitingUVTask();
    }
}
} // namespace ffrt
