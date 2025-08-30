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

#include "sched/load_tracking.h"

#include <unordered_set>
#include <unordered_map>

#include <unistd.h>

#include "core/entity.h"
#include "dm/dependence_manager.h"
#include "sched/interval.h"

namespace ffrt {
#define perf_mmap_read_current() (static_cast<uint64_t>(0))

void KernelLoadTracking::BeginImpl()
{
    it.Ctrl().Begin();
}

void KernelLoadTracking::EndImpl()
{
    it.Ctrl().End();
}

uint64_t KernelLoadTracking::GetLoadImpl()
{
    return it.Ctrl().GetLoad();
}
}; // namespace ffrt
