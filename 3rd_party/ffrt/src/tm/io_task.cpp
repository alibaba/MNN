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

#include "tm/io_task.h"
#include "dfx/trace_record/ffrt_trace_record.h"
#include "dfx/bbox/bbox.h"

namespace ffrt {
IOTask::IOTask(const ffrt_io_callable_t& work, const task_attr_private* attr)
    : TaskBase(ffrt_io_task, attr), work(work) {}

void IOTask::Execute()
{
    FFRTTraceRecord::TaskExecute<ffrt_io_task>(qos_);
    FFRT_EXECUTOR_TASK_BEGIN(gid);
    SetStatus(TaskStatus::EXECUTING);
    ffrt_coroutine_ptr_t coroutine = work.exec;
    ffrt_coroutine_ret_t ret = coroutine(work.data);
    if (ret == ffrt_coroutine_ready) {
        SetStatus(TaskStatus::FINISH);
        work.destroy(work.data);
        DecDeleteRef();
        FFRT_TASK_END();
        FFRTTraceRecord::TaskDone<ffrt_io_task>(qos_);
        return;
    }
    FFRT_BLOCK_MARKER(gid);
    SetStatus(TaskStatus::PENDING);
#ifdef FFRT_BBOX_ENABLE
    TaskPendingCounterInc();
#endif
    FFRT_TASK_END();
    FFRTTraceRecord::TaskDone<ffrt_io_task>(qos_);
}
} // namespace ffrt
