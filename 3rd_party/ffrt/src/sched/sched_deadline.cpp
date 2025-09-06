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

#include <cstdio>
#include <cinttypes>
#include <cstdint>
#include <unordered_map>

#include "core/entity.h"
#include "sched/interval.h"
#include "sched/sched_deadline.h"

namespace ffrt {
namespace TaskLoadTracking {
#ifdef PER_TASK_LOAD_TRACKING

static std::unordered_map<const char*, uint64_t> taskLoad;
static __thread uint64_t start;

// Called at eu/co_routine.cpp CoStart(task).
void Begin(CoTask* task)
{
    (void)task;
    start = perf_mmap_read_current();
}

void End(CoTask* task)
{
    uint64_t load = perf_mmap_read_current() - start;

    taskLoad[task->identity] = load;
}

// Get historical load based on its identity. 0 on the first time.
uint64_t GetLoad(CoTask* task)
{
    return taskLoad[task->identity];
}

#else // !PER_TASK_LOAD_TRACKING
void Begin(CoTask* task)
{
    (void)task;
}
void End(CoTask* task)
{
    (void)task;
}
uint64_t GetLoad(CoTask* task)
{
    (void)task;
    return 0;
}
#endif // PER_TASK_LOAD_TRACKING
} // namespace TaskLoadTracking
} // namespace ffrt
