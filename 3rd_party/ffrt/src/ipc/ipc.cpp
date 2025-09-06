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
#include "c/ffrt_ipc.h"

#include "internal_inc/osal.h"
#include "sched/execute_ctx.h"
#include "tm/cpu_task.h"
#include "dfx/log/ffrt_log_api.h"

#ifdef __cplusplus
extern "C" {
#endif

API_ATTRIBUTE((visibility("default")))
void ffrt_this_task_set_legacy_mode(bool mode)
{
    auto task = ffrt::ExecuteCtx::Cur()->task;
    if (!ffrt::IsCoTask(task)) {
        return;
    }

    ffrt::CoTask* coTask = static_cast<ffrt::CoTask*>(task);

    if (mode) {
        coTask->legacyCountNum++;
    } else {
        coTask->legacyCountNum--;
        if (coTask->legacyCountNum < 0) {
            FFRT_SYSEVENT_LOGE("Legacy count num less than zero");
        }
    }
}

#ifdef __cplusplus
}
#endif