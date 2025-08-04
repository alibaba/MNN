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

/**
 * @addtogroup FFRT
 * @{
 *
 * @brief Provides FFRT C++ APIs.
 *
 * @since 10
 */

/**
 * @file mutex.h
 *
 * @brief Declares the mutex interfaces in C++.
 *
 * @library libffrt.z.so
 * @kit FunctionFlowRuntimeKit
 * @syscap SystemCapability.Resourceschedule.Ffrt.Core
 * @since 10
 */

#ifndef FFRT_API_CPP_MUTEX_H
#define FFRT_API_CPP_MUTEX_H

#include "c/mutex.h"

namespace ffrt {
/**
 * @class mutex
 * @brief Provides a standard mutex for thread synchronization.
 *
 * The `mutex` class offers basic methods for locking, unlocking, and attempting
 * to acquire a lock without blocking. It is designed for safe and efficient
 * synchronization in multithreaded applications.
 *
 * @since 10
 */
class mutex : public ffrt_mutex_t {
public:
    /**
     * @brief Constructs a new mutex object.
     */
    mutex()
    {
        ffrt_mutex_init(this, nullptr);
    }

    /**
     * @brief Destroys the mutex object.
     */
    ~mutex()
    {
        ffrt_mutex_destroy(this);
    }

    /**
     * @brief Deleted copy constructor to prevent copying of the mutex object.
     */
    mutex(const mutex&) = delete;

    /**
     * @brief Deleted copy assignment operator to prevent assignment of the mutex object.
     */
    void operator=(const mutex&) = delete;

    /**
     * @brief Attempts to lock the mutex without blocking.
     *
     * @return true if the lock is successfully acquired, false otherwise.
     * @since 10
     */
    inline bool try_lock()
    {
        return ffrt_mutex_trylock(this) == ffrt_success ? true : false;
    }

    /**
     * @brief Locks the mutex.
     *
     * @since 10
     */
    inline void lock()
    {
        ffrt_mutex_lock(this);
    }

    /**
     * @brief Unlocks the mutex.
     *
     * @since 10
     */
    inline void unlock()
    {
        ffrt_mutex_unlock(this);
    }
};

/**
 * @class recursive_mutex
 * @brief Provides a recursive mutex for thread synchronization.
 *
 * The `recursive_mutex` class allows the same thread to acquire the lock
 * multiple times without causing a deadlock. It is particularly useful in
 * scenarios where a function that acquires a lock may be called recursively.
 *
 * @since 10
 */
class recursive_mutex : public ffrt_mutex_t {
public:
    /**
     * @brief Constructs a new recursive_mutex object.
     *
     * @since 10
     */
    recursive_mutex()
    {
        ffrt_mutexattr_init(&attr);
        ffrt_mutexattr_settype(&attr, ffrt_mutex_recursive);
        ffrt_mutex_init(this, &attr);
    }

    /**
     * @brief Destroys the recursive_mutex object.
     *
     * @since 10
     */
    ~recursive_mutex()
    {
        ffrt_mutexattr_destroy(&attr);
        ffrt_mutex_destroy(this);
    }

    /**
     * @brief Deleted copy constructor to prevent copying of the recursive_mutex object.
     */
    recursive_mutex(const recursive_mutex&) = delete;

    /**
     * @brief Deleted copy assignment operator to prevent assignment of the recursive_mutex object.
     */
    void operator=(const recursive_mutex&) = delete;

    /**
     * @brief Attempts to lock the recursive mutex without blocking.
     *
     * @return true if the lock is successfully acquired, false otherwise.
     * @since 10
     */
    inline bool try_lock()
    {
        return ffrt_mutex_trylock(this) == ffrt_success ? true : false;
    }

    /**
     * @brief Locks the recursive mutex.
     *
     * @since 10
     */
    inline void lock()
    {
        ffrt_mutex_lock(this);
    }

    /**
     * @brief Unlocks the recursive mutex.
     *
     * @since 10
     */
    inline void unlock()
    {
        ffrt_mutex_unlock(this);
    }

private:
    ffrt_mutexattr_t attr; ///< Mutex attribute object used to configure the recursive behavior of the mutex.
};
} // namespace ffrt

#endif // FFRT_API_CPP_MUTEX_H
/** @} */