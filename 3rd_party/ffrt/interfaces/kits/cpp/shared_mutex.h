/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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

/**
 * @addtogroup FFRT
 * @{
 *
 * @brief Provides FFRT C++ APIs.
 *
 * @since 18
 */

/**
 * @file shared_mutex.h
 *
 * @brief Declares the shared_mutex interfaces in C++.
 *
 * @library libffrt.z.so
 * @kit FunctionFlowRuntimeKit
 * @syscap SystemCapability.Resourceschedule.Ffrt.Core
 * @since 18
 */

#ifndef FFRT_API_CPP_SHARED_MUTEX_H
#define FFRT_API_CPP_SHARED_MUTEX_H

#include "c/shared_mutex.h"

namespace ffrt {
/**
 * @class shared_mutex
 * @brief A class for managing a shared mutex, providing synchronization mechanisms.
 *
 * This class offers methods to support:
 * - Exclusive locking: Ensures only one thread can access critical sections.
 * - Shared locking: Allows multiple threads to read simultaneously.
 *
 * @since 18
 */
class shared_mutex : public ffrt_rwlock_t {
public:
    /**
     * @brief Constructs a shared_mutex object and initializes the underlying lock.
     *
     * @since 18
     */
    shared_mutex()
    {
        ffrt_rwlock_init(this, nullptr);
    }

    /**
     * @brief Destroys the shared_mutex object and releases the underlying lock.
     *
     * @since 18
     */
    ~shared_mutex()
    {
        ffrt_rwlock_destroy(this);
    }

    /**
     * @brief Deleted copy constructor to prevent copying of the shared_mutex object.
     */
    shared_mutex(const shared_mutex&) = delete;

    /**
     * @brief Deleted copy assignment operator to prevent assignment of the shared_mutex object.
     */
    void operator=(const shared_mutex&) = delete;

    /**
     * @brief Acquires an exclusive lock, blocking other threads.
     *
     * @since 18
     */
    inline void lock()
    {
        ffrt_rwlock_wrlock(this);
    }

    /**
     * @brief Attempts to acquire an exclusive lock without blocking.
     *
     * @return true if the lock is successfully acquired; false otherwise.
     * @since 18
     */
    inline bool try_lock()
    {
        return ffrt_rwlock_trywrlock(this) == ffrt_success ? true : false;
    }

    /**
     * @brief Releases the exclusive lock.
     *
     * @since 18
     */
    inline void unlock()
    {
        ffrt_rwlock_unlock(this);
    }

    /**
     * @brief Acquires a shared lock, allowing concurrent access by multiple threads.
     *
     * @since 18
     */
    inline void lock_shared()
    {
        ffrt_rwlock_rdlock(this);
    }

    /**
     * @brief Attempts to acquire a shared lock without blocking.
     *
     * @return true if the lock is successfully acquired; false otherwise.
     * @since 18
     */
    inline bool try_lock_shared()
    {
        return ffrt_rwlock_tryrdlock(this) == ffrt_success ? true : false;
    }

    /**
     * @brief Releases the shared lock.
     *
     * @since 18
     */
    inline void unlock_shared()
    {
        ffrt_rwlock_unlock(this);
    }
};
} // namespace ffrt

#endif // FFRT_API_CPP_SHARED_MUTEX_H
/** @} */