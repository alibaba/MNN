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

#ifndef WATCHDOG_UTIL_H
#define WATCHDOG_UTIL_H
#include "tm/cpu_task.h"

namespace ffrt {
    bool IsValidTimeout(uint64_t gid, uint64_t timeout_us);
    void AddTaskToWatchdog(uint64_t gid);
    void RemoveTaskFromWatchdog(uint64_t gid);
    bool SendTimeoutWatchdog(uint64_t gid, uint64_t timeout, uint64_t delay);
    void RunTimeOutCallback(uint64_t gid, uint64_t timeout);
}
#endif /* WATCHDOG_UTIL_H */