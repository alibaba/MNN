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
#include "cpu_boost_wrapper.h"
#include "cpp/task.h"
#include "dfx/log/ffrt_log_api.h"
#include "internal_inc/osal.h"
#include "sched/execute_ctx.h"
#include "tm/task_base.h"
#include "c/ffrt_cpu_boost.h"

#ifdef __cplusplus
extern "C" {
#endif

API_ATTRIBUTE((visibility("default")))
int ffrt_cpu_boost_start(int ctx_id)
{
    int ret = CpuBoostStart(ctx_id);
    if (ret == 0) {
        if (ffrt::ExecuteCtx::Cur() == nullptr) {
            FFRT_SYSEVENT_LOGW("ExecuteCtx::Cur() is nullptr, save ctxId failed");
            return ret;
        }
        ffrt::CoTask* curTask = ffrt::IsCoTask(ffrt::ExecuteCtx::Cur()->task) ?
            static_cast<ffrt::CoTask*>(ffrt::ExecuteCtx::Cur()->task) : nullptr;
        if (curTask != nullptr && curTask->cpuBoostCtxId < 0) {
            curTask->cpuBoostCtxId = ctx_id;
        }
    }
    return ret;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_cpu_boost_end(int ctx_id)
{
    int ret = CpuBoostEnd(ctx_id);
    if (ret == 0) {
        if (ffrt::ExecuteCtx::Cur() == nullptr) {
            FFRT_SYSEVENT_LOGW("ExecuteCtx::Cur() is nullptr, clear ctxId failed");
            return ret;
        }
        ffrt::CoTask* curTask = ffrt::IsCoTask(ffrt::ExecuteCtx::Cur()->task) ?
            static_cast<ffrt::CoTask*>(ffrt::ExecuteCtx::Cur()->task) : nullptr;
        if (curTask != nullptr && curTask->cpuBoostCtxId == ctx_id) {
            curTask->cpuBoostCtxId = -1;
        }
    }
    return ret;
}
#ifdef __cplusplus
}
#endif