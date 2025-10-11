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

#ifndef FFRT_SCHED_DEADLINE_HPP
#define FFRT_SCHED_DEADLINE_HPP

#include "sched/scheduler.h"
#include "tm/task_base.h"

namespace ffrt {
namespace TaskLoadTracking {
void Begin(CoTask* task);
void End(CoTask* task);
uint64_t GetLoad(CoTask* task);
} // namespace TaskLoadTracking
} // namespace ffrt
#endif
