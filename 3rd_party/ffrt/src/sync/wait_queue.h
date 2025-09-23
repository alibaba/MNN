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

#ifndef __WAITQUEUE_H__
#define __WAITQUEUE_H__
#include "sync/sync.h"
#include "cpp/mutex.h"
#include "sched/execute_ctx.h"
#include "util/IntrusiveList.h"
#include "sync/mutex_private.h"
#include "dfx/log/ffrt_log_api.h"

namespace ffrt {
class TaskBase;
struct TaskWithNode;
using TaskListNode = ListNode;
using TaskList = List<TaskWithNode, TaskListNode>;

struct TaskWithNode : public TaskListNode {
    TaskWithNode();
    TaskBase* task = nullptr;
    std::mutex lk;
    std::condition_variable cv;
};

class WaitQueue {
public:
    using TimePoint = std::chrono::steady_clock::time_point;
    void SuspendAndWait(mutexPrivate* lk);
    int SuspendAndWaitUntil(mutexPrivate* lk, const TimePoint& tp) noexcept;
    void NotifyAll() noexcept { Notify(false); }
    void NotifyOne() noexcept { Notify(true); }

    WaitQueue()
    {
        whead = new WaitUntilEntry();
        whead->next = whead;
        whead->prev = whead;
    }
    WaitQueue(WaitQueue const&) = delete;
    void operator=(WaitQueue const&) = delete;

    ~WaitQueue()
    {
        std::lock_guard lg(wqlock);
        ReleaseAll();
        delete whead;
        whead = nullptr;
    }

private:
    fast_mutex wqlock;
    WaitUntilEntry* whead;

private:
    void WeNotifyProc(WaitUntilEntry* we);
    void ThreadWait(WaitUntilEntry* wn, mutexPrivate* lk, TaskBase* task);
    bool ThreadWaitUntil(WaitUntilEntry* wn, mutexPrivate* lk, const TimePoint& tp, TaskBase* task);
    void Notify(bool one) noexcept;

    inline bool empty() const
    {
        if (whead == nullptr) {
            return true;
        }
        return (whead->next == whead);
    }

    void ReleaseAll()
    {
        while (!empty()) {
            FFRT_LOGE("There are still tasks in cv that have not been awakened");
            WaitUntilEntry *wue = pop_front();
            WeNotifyProc(wue);
        }
    }

    inline void push_back(WaitUntilEntry* we)
    {
        if ((we == nullptr) || (whead == nullptr) || (whead->prev == nullptr)) {
            FFRT_LOGE("we or whead or whead->prev is nullptr");
            return;
        }
        we->next = whead;
        we->prev = whead->prev;
        whead->prev->next = we;
        whead->prev = we;
    }

    inline WaitUntilEntry* pop_front()
    {
        if ((whead->next == nullptr) || (whead->next->next == nullptr)) {
            FFRT_LOGE("whead->next or whead->next->next is nullptr");
            return nullptr;
        }
        WaitEntry *we = whead->next;
        whead->next = we->next;
        we->next->prev = whead;
        we->next = nullptr;
        we->prev = nullptr;
        return static_cast<WaitUntilEntry*>(we);
    }

    inline void remove(WaitUntilEntry* we)
    {
        if ((we->next == nullptr) || (we->prev == nullptr)) {
            return;
        }
        we->prev->next = we->next;
        we->next->prev = we->prev;
        we->next = nullptr;
        we->prev = nullptr;
        return;
    }
    friend bool WeTimeoutProc(WaitQueue* wq, WaitUntilEntry* wue);
};
} // namespace ffrt
#endif // _WAITQUEUE_H_
