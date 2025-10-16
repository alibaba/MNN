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

#include "shared_mutex_private.h"
#include "dfx/log/ffrt_log_api.h"
#include "dfx/trace/ffrt_trace.h"

#include "internal_inc/osal.h"
#include "internal_inc/types.h"
#include "tm/cpu_task.h"

namespace ffrt {
void SharedMutexPrivate::Lock()
{
    std::lock_guard lg(mut);
    while (state & writeEntered) {
        Wait(wList1, SharedMutexWaitType::WRITE);
    }
    state |= writeEntered;
    while (state & readersMax) {
        Wait(wList2, SharedMutexWaitType::NORMAL);
    }
}

bool SharedMutexPrivate::TryLock()
{
    std::lock_guard lg(mut);
    if (state == 0) {
        state = writeEntered;
        return true;
    }
    return false;
}

void SharedMutexPrivate::LockShared()
{
    std::lock_guard lg(mut);
    while (state >= readersMax) {
        Wait(wList1, SharedMutexWaitType::READ);
    }
    ++state;
}

bool SharedMutexPrivate::TryLockShared()
{
    std::lock_guard lg(mut);
    if (state < readersMax) {
        ++state;
        return true;
    }
    return false;
}

void SharedMutexPrivate::Unlock()
{
    std::lock_guard lg(mut);
    if (state == writeEntered) {
        state = 0;
        NotifyAll(wList1);
        return;
    }

    if (state > 0) {
        --state;
        if (state & writeEntered) {
            if (state == writeEntered) {
                NotifyOne(wList2);
            }
        } else {
            if (state == readersMax - 1) {
                NotifyOne(wList1);
            } else if (!wList1.Empty()) {
                NotifyAll(wList1);
            }
        }
    }
}

void SharedMutexPrivate::Wait(LinkedList& wList, SharedMutexWaitType wtType)
{
    auto ctx = ExecuteCtx::Cur();
    auto task = ctx->task;
    if (task == nullptr || task->Block() == BlockType::BLOCK_THREAD) {
        ctx->wn.task = task;
        ctx->wn.wtType = wtType;
        wList.PushBack(ctx->wn.node);

        std::unique_lock<std::mutex> lk(ctx->wn.wl);
        mut.unlock();
        ctx->wn.cv.wait(lk);
        ctx->wn.task = nullptr;
        if (task) {
            task->Wake();
        }
    } else {
        FFRT_BLOCK_TRACER(task->gid, smx);
        CoWait([&](CoTask* task) -> bool {
            task->we.wtType = wtType;
            wList.PushBack(task->we.node);
            mut.unlock();
            return true;
        });
    }
    mut.lock();
}

void SharedMutexPrivate::NotifyOne(LinkedList& wList)
{
    WaitEntry* we = wList.PopFront(&WaitEntry::node);

    if (we != nullptr) {
        auto task = we->task;
        if (task == nullptr || task->GetBlockType() == BlockType::BLOCK_THREAD) {
            WaitUntilEntry* wue = static_cast<WaitUntilEntry*>(we);
            std::lock_guard<std::mutex> lk(wue->wl);
            wue->cv.notify_one();
        } else {
            CoRoutineFactory::CoWakeFunc(static_cast<CoTask*>(task), CoWakeType::NO_TIMEOUT_WAKE);
        }
    }
}

void SharedMutexPrivate::NotifyAll(LinkedList& wList)
{
    WaitEntry* we = wList.PopFront(&WaitEntry::node);

    while (we != nullptr) {
        auto task = we->task;
        if (task == nullptr || task->GetBlockType() == BlockType::BLOCK_THREAD) {
            WaitUntilEntry* wue = static_cast<WaitUntilEntry*>(we);
            std::lock_guard<std::mutex> lk(wue->wl);
            wue->cv.notify_one();
        } else {
            CoRoutineFactory::CoWakeFunc(static_cast<CoTask*>(task), CoWakeType::NO_TIMEOUT_WAKE);
        }

        if (we->wtType == SharedMutexWaitType::READ) {
            WaitEntry* weNext = wList.Front(&WaitEntry::node);
            if (weNext != nullptr && weNext->wtType == SharedMutexWaitType::WRITE) {
                return;
            }
        } else if (we->wtType == SharedMutexWaitType::WRITE) {
            return;
        }

        we = wList.PopFront(&WaitEntry::node);
    }
}
} // namespace ffrt

#ifdef __cplusplus
extern "C" {
#endif

API_ATTRIBUTE((visibility("default")))
int ffrt_rwlock_init(ffrt_rwlock_t* rwlock, const ffrt_rwlockattr_t* attr)
{
    if (!rwlock) {
        FFRT_LOGE("rwlock should not be empty");
        return ffrt_error_inval;
    }
    if (attr != nullptr) {
        FFRT_LOGE("only support normal rwlock");
        return ffrt_error_inval;
    }
    static_assert(sizeof(ffrt::SharedMutexPrivate) <= ffrt_rwlock_storage_size,
        "size must be less than ffrt_rwlock_storage_size");

    new (rwlock)ffrt::SharedMutexPrivate();
    return ffrt_success;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_rwlock_wrlock(ffrt_rwlock_t* rwlock)
{
    if (!rwlock) {
        FFRT_LOGE("rwlock should not be empty");
        return ffrt_error_inval;
    }
    auto p = reinterpret_cast<ffrt::SharedMutexPrivate*>(rwlock);
    p->Lock();
    return ffrt_success;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_rwlock_trywrlock(ffrt_rwlock_t* rwlock)
{
    if (!rwlock) {
        FFRT_LOGE("rwlock should not be empty");
        return ffrt_error_inval;
    }
    auto p = reinterpret_cast<ffrt::SharedMutexPrivate*>(rwlock);
    return p->TryLock() ? ffrt_success : ffrt_error_busy;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_rwlock_rdlock(ffrt_rwlock_t* rwlock)
{
    if (!rwlock) {
        FFRT_LOGE("rwlock should not be empty");
        return ffrt_error_inval;
    }
    auto p = reinterpret_cast<ffrt::SharedMutexPrivate*>(rwlock);
    p->LockShared();
    return ffrt_success;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_rwlock_tryrdlock(ffrt_rwlock_t* rwlock)
{
    if (!rwlock) {
        FFRT_LOGE("rwlock should not be empty");
        return ffrt_error_inval;
    }
    auto p = reinterpret_cast<ffrt::SharedMutexPrivate*>(rwlock);
    return p->TryLockShared() ? ffrt_success : ffrt_error_busy;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_rwlock_unlock(ffrt_rwlock_t* rwlock)
{
    if (!rwlock) {
        FFRT_LOGE("rwlock should not be empty");
        return ffrt_error_inval;
    }
    auto p = reinterpret_cast<ffrt::SharedMutexPrivate*>(rwlock);
    p->Unlock();
    return ffrt_success;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_rwlock_destroy(ffrt_rwlock_t* rwlock)
{
    if (!rwlock) {
        FFRT_LOGE("rwlock should not be empty");
        return ffrt_error_inval;
    }
    auto p = reinterpret_cast<ffrt::SharedMutexPrivate*>(rwlock);
    p->~SharedMutexPrivate();
    return ffrt_success;
}

#ifdef __cplusplus
}
#endif
