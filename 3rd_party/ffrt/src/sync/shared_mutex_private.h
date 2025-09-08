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

#ifndef _SHARED_MUTEX_PRIVATE_H_
#define _SHARED_MUTEX_PRIVATE_H_

#include "sched/execute_ctx.h"
#include "sync/sync.h"

namespace ffrt {
class SharedMutexPrivate {
public:
    void Lock();
    bool TryLock();
    void LockShared();
    bool TryLockShared();
    void Unlock();

    SharedMutexPrivate() = default;
    ~SharedMutexPrivate() = default;
    SharedMutexPrivate(SharedMutexPrivate const&) = delete;
    void operator = (SharedMutexPrivate const&) = delete;

private:
    fast_mutex mut;

    LinkedList wList1;
    LinkedList wList2;

    unsigned state = 0;
    static constexpr unsigned writeEntered = 1U << (sizeof(unsigned) * __CHAR_BIT__ - 1);
    static constexpr unsigned readersMax = ~writeEntered;

    void Wait(LinkedList& wList, SharedMutexWaitType wtType);
    void NotifyOne(LinkedList& wList);
    void NotifyAll(LinkedList& wList);
};
} // namespace ffrt
#endif
