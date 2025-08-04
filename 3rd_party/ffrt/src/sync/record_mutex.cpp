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

#include "record_mutex.h"
#include "sched/execute_ctx.h"
#include "sched/workgroup_internal.h"
#include "tm/cpu_task.h"

namespace {
    constexpr uint64_t MUTEX_TIMEOUT_THRESHOLD_US = 30 * 1000 * 1000;
}

namespace ffrt {
bool RecordMutex::IsTimeout()
{
    return GetDuration() > MUTEX_TIMEOUT_THRESHOLD_US;
}

void RecordMutex::LoadInfo()
{
    if (ExecuteCtx::Cur()->task) {
        owner_.id = ExecuteCtx::Cur()->task->gid;
        owner_.type = MutexOwnerType::MUTEX_OWNER_TYPE_TASK;
    } else {
        owner_.id = static_cast<uint64_t>(gettid());
        owner_.type = MutexOwnerType::MUTEX_OWNER_TYPE_THREAD;
    }

    owner_.timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

uint64_t RecordMutex::GetDuration()
{
    uint64_t now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    uint64_t past = owner_.timestamp;
    return (now > past) ? now - past : 0;
}
}
