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
 * @brief Provides FFRT C APIs.
 *
 * @since 10
 */

/**
 * @file type_def.h
 *
 * @brief Declares common types.
 *
 * @library libffrt.z.so
 * @kit FunctionFlowRuntimeKit
 * @syscap SystemCapability.Resourceschedule.Ffrt.Core
 * @since 10
 */

#ifndef FFRT_API_C_TYPE_DEF_H
#define FFRT_API_C_TYPE_DEF_H

#include <stdint.h>
#include <errno.h>

#ifdef __cplusplus
#define FFRT_C_API  extern "C"
#else
#define FFRT_C_API
#endif

/**
 * @brief Enumerates the task priority types.
 *
 * @since 12
 */
typedef enum {
    /** Should be distributed at once if possible, handle time equals to send time, prior to high level. */
    ffrt_queue_priority_immediate = 0,
    /** High priority, sorted by handle time, prior to low level. */
    ffrt_queue_priority_high,
    /** Low priority, sorted by handle time, prior to idle level. */
    ffrt_queue_priority_low,
    /** Lowest priority, sorted by handle time, only distribute when there is no other level inside queue. */
    ffrt_queue_priority_idle,
} ffrt_queue_priority_t;

/**
 * @brief Enumerates the task QoS types.
 *
 * @since 10
 */
typedef enum {
    /** Inheritance. */
    ffrt_qos_inherit = -1,
    /** Background task. */
    ffrt_qos_background,
    /** Real-time tool. */
    ffrt_qos_utility,
    /** Default type. */
    ffrt_qos_default,
    /** User initiated. */
    ffrt_qos_user_initiated,
} ffrt_qos_default_t;

/**
 * @brief Defines the QoS type.
 *
 * @since 10
 */
typedef int ffrt_qos_t;

/**
 * @brief Defines the task function pointer type.
 *
 * @since 10
 */
typedef void(*ffrt_function_t)(void*);

/**
 * @brief Defines a task executor.
 *
 * @since 10
 */
typedef struct {
    /** Function used to execute a task. */
    ffrt_function_t exec;
    /** Function used to destroy a task. */
    ffrt_function_t destroy;
    /** Need to be set to 0. */
    uint64_t reserve[2];
} ffrt_function_header_t;

/**
 * @brief Defines the storage size of multiple types of structs.
 *
 * @since 10
 */
typedef enum {
    /** Task attribute storage size. */
    ffrt_task_attr_storage_size = 128,
    /** Task executor storage size. */
    ffrt_auto_managed_function_storage_size = 64 + sizeof(ffrt_function_header_t),
    /** Mutex storage size. */
    ffrt_mutex_storage_size = 64,
    /** Condition variable storage size. */
    ffrt_cond_storage_size = 64,
    /** Queue storage size. */
    ffrt_queue_attr_storage_size = 128,
    /** Rwlock storage size.
     *
     * @since 18
     */
    ffrt_rwlock_storage_size = 64,
    /** Fiber storage size.
     *
     * This constant defines the fiber storage size.
     * The actual value depends on the target architecture:
     * - __aarch64__: 22
     * - __arm__: 64
     * - __x86_64__: 8
     *
     * @since 20
     */
#if defined(__aarch64__)
    ffrt_fiber_storage_size = 22,
#elif defined(__arm__)
    ffrt_fiber_storage_size = 64,
#elif defined(__x86_64__)
    ffrt_fiber_storage_size = 8,
#else
#error "unsupported architecture"
#endif
} ffrt_storage_size_t;
/**
 * @brief Enumerates the task types.
 *
 * @since 10
 */
typedef enum {
    /** General task. */
    ffrt_function_kind_general,
    /** Queue task. */
    ffrt_function_kind_queue,
} ffrt_function_kind_t;

/**
 * @brief Enumerates the dependency types.
 *
 * @since 10
 */
typedef enum {
    /** Data dependency type. */
    ffrt_dependence_data,
    /** Task dependency type. */
    ffrt_dependence_task,
} ffrt_dependence_type_t;

/**
 * @brief Defines the dependency data structure.
 *
 * @since 10
 */
typedef struct {
    /** Dependency type. */
    ffrt_dependence_type_t type;
    /** Dependency pointer. */
    const void* ptr;
} ffrt_dependence_t;

/**
 * @brief Defines the dependency structure.
 *
 * @since 10
 */
typedef struct {
    /** Number of dependencies. */
    uint32_t len;
    /** Dependency data. */
    const ffrt_dependence_t* items;
} ffrt_deps_t;

/**
 * @brief Defines the task attribute structure.
 *
 * @since 10
 */
typedef struct {
    /** An array of uint32_t used to store the task attribute. */
    uint32_t storage[(ffrt_task_attr_storage_size + sizeof(uint32_t) - 1) / sizeof(uint32_t)];
} ffrt_task_attr_t;

/**
 * @brief Defines the queue attribute structure.
 *
 * @since 10
 */
typedef struct {
    /** An array of uint32_t used to store the queue attribute. */
    uint32_t storage[(ffrt_queue_attr_storage_size + sizeof(uint32_t) - 1) / sizeof(uint32_t)];
} ffrt_queue_attr_t;

/**
 * @brief Defines the task handle, which identifies different tasks.
 *
 * @since 10
 */
typedef void* ffrt_task_handle_t;

/**
 * @brief Enumerates the ffrt error codes.
 *
 * @since 10
 */
typedef enum {
    /** A generic error. */
    ffrt_error = -1,
    /** Success. */
    ffrt_success = 0,
    /** An out of memory error. */
    ffrt_error_nomem = ENOMEM,
    /** A timeout error. */
    ffrt_error_timedout = ETIMEDOUT,
    /** A busy error. */
    ffrt_error_busy = EBUSY,
    /** A invalid value error. */
    ffrt_error_inval = EINVAL
} ffrt_error_t;

/**
 * @brief Defines the condition variable attribute structure.
 *
 * @since 10
 */
typedef struct {
    /** A long integer used to store the condition variable attribute. */
    long storage;
} ffrt_condattr_t;

/**
 * @brief Defines the mutex attribute structure.
 *
 * @since 10
 */
typedef struct {
    /** A long integer used to store the mutex attribute. */
    long storage;
} ffrt_mutexattr_t;

/**
 * @brief Defines the rwlock attribute structure.
 *
 * @since 18
 */
typedef struct {
    /** A long integer used to store the rwlock attribute. */
    long storage;
} ffrt_rwlockattr_t;

/**
 * @brief Enumerates the mutex types.
 *
 * Describes the mutex type, ffrt_mutex_normal is normal mutex;
 * ffrt_mutex_recursive is recursive mutex, ffrt_mutex_default is normal mutex.
 *
 * @since 12
 */
typedef enum {
    /** Normal mutex type. */
    ffrt_mutex_normal = 0,
    /** Recursive mutex type. */
    ffrt_mutex_recursive = 2,
    /** Default mutex type. */
    ffrt_mutex_default = ffrt_mutex_normal
} ffrt_mutex_type;

/**
 * @brief Defines the mutex structure.
 *
 * @since 10
 */
typedef struct {
    /** An array of uint32_t used to store the mutex. */
    uint32_t storage[(ffrt_mutex_storage_size + sizeof(uint32_t) - 1) / sizeof(uint32_t)];
} ffrt_mutex_t;

/**
 * @brief Defines the rwlock structure.
 *
 * @since 18
 */
typedef struct {
    /** An array of uint32_t used to store the rwlock. */
    uint32_t storage[(ffrt_rwlock_storage_size + sizeof(uint32_t) - 1) / sizeof(uint32_t)];
} ffrt_rwlock_t;

/**
 * @brief Defines the condition variable structure.
 *
 * @since 10
 */
typedef struct {
    /** An array of uint32_t used to store the condition variable. */
    uint32_t storage[(ffrt_cond_storage_size + sizeof(uint32_t) - 1) / sizeof(uint32_t)];
} ffrt_cond_t;

/**
 * @brief Defines the fiber structure.
 *
 * @since 20
 */
typedef struct {
    /** An array of uint32_t used to store the fiber. */
    uintptr_t storage[ffrt_fiber_storage_size];
} ffrt_fiber_t;

/**
 * @brief Defines the poller callback function type.
 *
 * @since 12
 */
typedef void (*ffrt_poller_cb)(void* data, uint32_t event);

/**
 * @brief Defines the timer callback function type.
 *
 * @since 12
 */
typedef void (*ffrt_timer_cb)(void* data);

/**
 * @brief Defines the timer handler.
 *
 * @since 12
 */
typedef int ffrt_timer_t;

#ifdef __cplusplus
namespace ffrt {

/**
 * @brief Enumerates the task QoS types.
 *
 * @since 10
 */
enum qos_default {
    /** Inheritance. */
    qos_inherit = ffrt_qos_inherit,
    /** Background task. */
    qos_background = ffrt_qos_background,
    /** Real-time tool. */
    qos_utility = ffrt_qos_utility,
    /** Default type. */
    qos_default = ffrt_qos_default,
    /** User initiated. */
    qos_user_initiated = ffrt_qos_user_initiated,
};

/**
 * @brief Defines the QoS type.
 *
 * @since 10
 */
using qos = int;

}

#endif // __cplusplus
#endif // FFRT_API_C_TYPE_DEF_H
/** @} */