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

#include <unistd.h>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/time.h>
#include <sys/syscall.h>

#include <map>
#include <functional>
#include <linux/futex.h>
#include "sync/delayed_worker.h"
#include "util/ffrt_facade.h"
#include "sync/sync.h"

#ifdef NS_PER_SEC
#undef NS_PER_SEC
#endif
namespace ffrt {
bool DelayedWakeup(const TimePoint& to, WaitEntry* we, const std::function<void(WaitEntry*)>& wakeup,
    bool skipTimeCheck)
{
    return FFRTFacade::GetDWInstance().dispatch(to, we, wakeup, skipTimeCheck);
}

bool DelayedRemove(const TimePoint& to, WaitEntry* we)
{
    return FFRTFacade::GetDWInstance().remove(to, we);
}

void spin_mutex::lock_contended()
{
    int v = l.load(std::memory_order_relaxed);
    do {
        while (v != sync_detail::UNLOCK) {
            std::this_thread::yield();
            v = l.load(std::memory_order_relaxed);
        }
    } while (!l.compare_exchange_weak(v, sync_detail::LOCK, std::memory_order_acquire, std::memory_order_relaxed));
}

static void spin()
{
#if defined(__x86_64__)
    asm volatile("pause");
#elif defined(__aarch64__)
    asm volatile("isb sy");
#elif defined(__arm__)
    asm volatile("yield");
#endif
}

void fast_mutex::lock_contended()
{
    int v = 0;
    // lightly contended
    for (uint32_t n = static_cast<uint32_t>(1 + rand() % 4); n <= 64; n <<= 1) {
        for (uint32_t i = 0; i < n; ++i) {
            spin();
        }
        v = __atomic_load_n(&l, __ATOMIC_RELAXED);
        if (v == sync_detail::WAIT) {
            break;
        }
        if (v == sync_detail::UNLOCK) {
            if (__atomic_compare_exchange_n(&l, &v, sync_detail::LOCK, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
                return;
            }
            break;
        }
    }
    // heavily contended
    if (v == sync_detail::WAIT) {
        syscall(SYS_futex, &l, FUTEX_WAIT_PRIVATE, sync_detail::WAIT, nullptr, nullptr, 0);
    }
    while (__atomic_exchange_n(&l, sync_detail::WAIT, __ATOMIC_ACQUIRE) != sync_detail::UNLOCK) {
        syscall(SYS_futex, &l, FUTEX_WAIT_PRIVATE, sync_detail::WAIT, nullptr, nullptr, 0);
    }
}
} // namespace ffrt
