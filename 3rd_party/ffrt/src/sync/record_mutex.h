/*
 * Copyright (c) 2024 Huawei Device Co., Ltd.
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

#ifndef RECORD_MUTEX_H
#define RECORD_MUTEX_H

#include <sys/types.h>
#include <chrono>
#include <cstdint>
#include <securec.h>
#include "cpp/mutex.h"

namespace ffrt {
enum class MutexOwnerType {
    MUTEX_OWNER_TYPE_TASK,
    MUTEX_OWNER_TYPE_THREAD,
};

struct MutexOwner {
    uint64_t id; // 持锁task的id或者持锁线程的id
    MutexOwnerType type;
    uint64_t timestamp;
};

class RecordMutex {
public:
    void lock()
    {
        mutex_.lock();
        LoadInfo();
    }

    void unlock()
    {
        ClearInfo();
        mutex_.unlock();
    }

    mutex* GetMutex()
    {
        return &mutex_;
    }

    bool HasLock() const
    {
        return owner_.id != 0;
    }

    bool IsTimeout();

    uint64_t GetOwnerId() const
    {
        return owner_.id;
    }

    MutexOwnerType GetOwnerType() const
    {
        return owner_.type;
    }

    uint64_t GetDuration();

private:
    void LoadInfo();

    void ClearInfo()
    {
        owner_.id = 0;
    }

    ffrt::mutex mutex_;
    MutexOwner owner_ = {};
};
} // namespace ffrt

#endif