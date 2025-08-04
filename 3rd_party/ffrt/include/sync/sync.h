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

#ifndef UTIL_SYNC_HPP
#define UTIL_SYNC_HPP
// Provide synchronization primitives

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include "sched/execute_ctx.h"

namespace ffrt {
namespace sync_detail {
const int UNLOCK = 0;
const int LOCK = 1;
const int WAIT = 2;
} // namespace sync_detail

class spin_mutex {
    std::atomic<int> l;
    void lock_contended();

public:
    spin_mutex() : l(sync_detail::UNLOCK)
    {
    }
    spin_mutex(spin_mutex const&) = delete;
    void operator=(spin_mutex const&) = delete;

    void lock()
    {
        if (l.exchange(sync_detail::LOCK, std::memory_order_acquire) == sync_detail::UNLOCK) {
            return;
        }
        lock_contended();
    }

    void unlock()
    {
        l.store(sync_detail::UNLOCK, std::memory_order_release);
    }
};

class fast_mutex {
    int l;
    void lock_contended();

public:
    fast_mutex() : l(sync_detail::UNLOCK)
    {
    }
    fast_mutex(fast_mutex const&) = delete;
    void operator=(fast_mutex const&) = delete;

    void lock()
    {
        int v = sync_detail::UNLOCK;
        if (__atomic_compare_exchange_n(&l, &v, sync_detail::LOCK, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
            return;
        }
        lock_contended();
    }

    bool try_lock()
    {
        int v = sync_detail::UNLOCK;
        return __atomic_compare_exchange_n(&l, &v, sync_detail::LOCK, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED);
    }

    void unlock()
    {
        if (__atomic_exchange_n(&l, sync_detail::UNLOCK, __ATOMIC_RELEASE) == sync_detail::WAIT) {
            syscall(SYS_futex, &l, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
        }
    }
};

bool DelayedWakeup(const TimePoint& to, WaitEntry* we, const std::function<void(WaitEntry*)>& wakeup,
    bool skipTimeCheck = false);
bool DelayedRemove(const TimePoint& to, WaitEntry* we);
} // namespace ffrt
#endif
