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

#ifndef TASK_FACTORY_HPP
#define TASK_FACTORY_HPP

#include "tm/task_base.h"
#include "util/cb_func.h"
#include "util/slab.h"

namespace ffrt {
template <typename T>
class TaskFactory {
public:
    static TaskFactory<T>& Instance();

    static T* Alloc()
    {
        return Instance().alloc_();
    }

    static void Free(T* task)
    {
        Instance().free_(task);
    }

    static void Free_(T* task)
    {
        if (Instance().free__ != nullptr) {
            Instance().free__(task);
        }
    }

    static std::vector<void*> GetUnfreedMem()
    {
        if (Instance().getUnfreedMem_ != nullptr) {
            return Instance().getUnfreedMem_();
        }
        return {};
    }

    static std::vector<void*> GetUnfreedTasksFiltered()
    {
        LockMem();
        std::vector<void*> unfreed = GetUnfreedMem();
        // Filter out tasks where the reference count increment failed.
        unfreed.erase(
            std::remove_if(unfreed.begin(), unfreed.end(),
                [](void* task) {
                    return !IncDeleteRefIfPositive(reinterpret_cast<TaskBase*>(task));
                }),
            unfreed.end()
        );
        UnlockMem();
        return unfreed;
    }

    static bool HasBeenFreed(T* task)
    {
        if (Instance().hasBeenFreed_ != nullptr) {
            return Instance().hasBeenFreed_(task);
        }
        return true;
    }

    static void LockMem()
    {
        if (Instance().lockMem_ != nullptr) {
            Instance().lockMem_();
        }
    }

    static void UnlockMem()
    {
        if (Instance().unlockMem_ != nullptr) {
            Instance().unlockMem_();
        }
    }

    static void RegistCb(
        typename TaskAllocCB<T>::Alloc &&alloc,
        typename TaskAllocCB<T>::Free &&free,
        typename TaskAllocCB<T>::Free_ &&free_ = nullptr,
        typename TaskAllocCB<T>::GetUnfreedMem &&getUnfreedMem = nullptr,
        typename TaskAllocCB<T>::HasBeenFreed &&hasBeenFreed = nullptr,
        typename TaskAllocCB<T>::LockMem &&lockMem = nullptr,
        typename TaskAllocCB<T>::UnlockMem &&unlockMem = nullptr)
    {
        Instance().alloc_ = std::move(alloc);
        Instance().free_ = std::move(free);
        Instance().free__ = std::move(free_);
        Instance().getUnfreedMem_ = std::move(getUnfreedMem);
        Instance().hasBeenFreed_ = std::move(hasBeenFreed);
        Instance().lockMem_ = std::move(lockMem);
        Instance().unlockMem_ = std::move(unlockMem);
    }

private:
    typename TaskAllocCB<T>::Alloc alloc_;
    typename TaskAllocCB<T>::Free free_;
    typename TaskAllocCB<T>::Free_ free__;
    typename TaskAllocCB<T>::GetUnfreedMem getUnfreedMem_;
    typename TaskAllocCB<T>::HasBeenFreed hasBeenFreed_;
    typename TaskAllocCB<T>::LockMem lockMem_;
    typename TaskAllocCB<T>::UnlockMem unlockMem_;
};

template <typename T>
class TaskMemScopedLock {
public:
    TaskMemScopedLock()
    {
        TaskFactory<T>::LockMem();
    }

    ~TaskMemScopedLock()
    {
        TaskFactory<T>::UnlockMem();
    }
};
} // namespace ffrt

#endif