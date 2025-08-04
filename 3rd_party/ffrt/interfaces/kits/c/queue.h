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
 * @file queue.h
 *
 * @brief Declares the queue interfaces in C.
 *
 * @library libffrt.z.so
 * @kit FunctionFlowRuntimeKit
 * @syscap SystemCapability.Resourceschedule.Ffrt.Core
 * @since 10
 */

#ifndef FFRT_API_C_QUEUE_H
#define FFRT_API_C_QUEUE_H

#include <stdbool.h>
#include "type_def.h"

/**
 * @brief Enumerates the queue types.
 *
 * @since 12
 */
typedef enum {
    /** Serial queue. */
    ffrt_queue_serial,
    /** Concurrent queue. */
    ffrt_queue_concurrent,
    /** Invalid queue. */
    ffrt_queue_max
} ffrt_queue_type_t;

/**
 * @brief Defines the queue handle, which identifies different queues.
 *
 * @since 10
 */
typedef void* ffrt_queue_t;

/**
 * @brief Initializes a queue attribute.
 *
 * @param attr Indicates a pointer to the queue attribute.
 * @return Returns <b>0</b> if the queue attribute is initialized;
           returns <b>-1</b> otherwise.
 * @since 10
 */
FFRT_C_API int ffrt_queue_attr_init(ffrt_queue_attr_t* attr);

/**
 * @brief Destroys a queue attribute, the user needs to invoke this interface.
 *
 * @param attr Indicates a pointer to the queue attribute.
 * @since 10
 */
FFRT_C_API void ffrt_queue_attr_destroy(ffrt_queue_attr_t* attr);

/**
 * @brief Sets the QoS for a queue attribute.
 *
 * @param attr Indicates a pointer to the queue attribute.
 * @param qos Indicates the QoS.
 * @since 10
 */
FFRT_C_API void ffrt_queue_attr_set_qos(ffrt_queue_attr_t* attr, ffrt_qos_t qos);

/**
 * @brief Gets the QoS of a queue attribute.
 *
 * @param attr Indicates a pointer to the queue attribute.
 * @return Returns the QoS.
 * @since 10
 */
FFRT_C_API ffrt_qos_t ffrt_queue_attr_get_qos(const ffrt_queue_attr_t* attr);

/**
 * @brief Sets the execution timeout of a serial queue attribute.
 *
 * The lower limit of timeout value is 1 ms, if the value is less than 1 ms, it will be set to 1 ms.
 *
 * @param attr Serial queue attribute pointer.
 * @param timeout_us Serial queue task execution timeout.
 * @since 10
 */
FFRT_C_API void ffrt_queue_attr_set_timeout(ffrt_queue_attr_t* attr, uint64_t timeout_us);

/**
 * @brief Gets the execution timeout of a serial queue attribute.
 *
 * @param attr Serial queue attribute pointer.
 * @return Returns the serial queue task execution timeout.
 * @since 10
 */
FFRT_C_API uint64_t ffrt_queue_attr_get_timeout(const ffrt_queue_attr_t* attr);

/**
 * @brief Sets the timeout callback function of a serial queue attribute.
 *
 * @warning Do not call `exit` in `f` - this my cause unexpected behavior.
 *
 * @param attr Serial queue attribute pointer.
 * @param f Serial queue timeout callback function.
 * @since 10
 */
FFRT_C_API void ffrt_queue_attr_set_callback(ffrt_queue_attr_t* attr, ffrt_function_header_t* f);

/**
 * @brief Gets the timeout callback function of a serial queue attribute.
 *
 * @param attr Serial queue attribute pointer.
 * @return Returns the serial queue task timeout callback function.
 * @since 10
 */
FFRT_C_API ffrt_function_header_t* ffrt_queue_attr_get_callback(const ffrt_queue_attr_t* attr);

/**
 * @brief Sets the queue max concurrency of a queue attribute.
 *
 * @param attr Queue attribute pointer.
 * @param max_concurrency queue max_concurrency.
 * @since 12
 */
FFRT_C_API void ffrt_queue_attr_set_max_concurrency(ffrt_queue_attr_t* attr, const int max_concurrency);

/**
 * @brief Gets the queue max concurrency of a queue attribute.
 *
 * @param attr Queue attribute pointer.
 * @return Returns the queue max concurrency.
 * @since 12
 */
FFRT_C_API int ffrt_queue_attr_get_max_concurrency(const ffrt_queue_attr_t* attr);

/**
 * @brief Sets the execution mode of a queue attribute.
 *
 * This interface specifies whether tasks in the queue are executed in coroutine mode or thread mode.
 * By default, tasks are executed in coroutine mode.
 * Set <b>mode</b> to <b>true</b> to enable thread-based execution.
 *
 * @param attr Queue attribute pointer.
 * @param mode Indicates whether to enable thread-based execution mode.
 *           - <b>true</b>: Tasks are executed as native threads (thread mode).
 *           - <b>false</b>: Tasks are executed as coroutines (default).
 * @since 20
 */
FFRT_C_API void ffrt_queue_attr_set_thread_mode(ffrt_queue_attr_t* attr, bool mode);

/**
 * @brief Gets the execution mode of a queue attribute.
 *
 * This interface returns whether tasks in the queue are configured to run in thread-based execution mode (thread mode).
 *
 * @param attr Queue attribute pointer.
 * @return Returns <b>true</b> if tasks are executed as native threads (thread mode);
 *         returns <b>false</b> if tasks are executed as coroutines (default).
 * @since 20
 */
FFRT_C_API bool ffrt_queue_attr_get_thread_mode(const ffrt_queue_attr_t* attr);

/**
 * @brief Creates a queue.
 *
 * @param type Indicates the queue type.
 * @param name Indicates a pointer to the queue name.
 * @param attr Indicates a pointer to the queue attribute.
 * @return Returns a non-null queue handle if the queue is created;
           returns a null pointer otherwise.
 * @since 10
 */
FFRT_C_API ffrt_queue_t ffrt_queue_create(ffrt_queue_type_t type, const char* name, const ffrt_queue_attr_t* attr);

/**
 * @brief Destroys a queue, the user needs to invoke this interface.
 *
 * @param queue Indicates a queue handle.
 * @since 10
 */
FFRT_C_API void ffrt_queue_destroy(ffrt_queue_t queue);

/**
 * @brief Submits a task to a queue.
 *
 * @param queue Indicates a queue handle.
 * @param f Indicates a pointer to the task executor.
 * @param attr Indicates a pointer to the task attribute.
 * @since 10
 */
FFRT_C_API void ffrt_queue_submit(ffrt_queue_t queue, ffrt_function_header_t* f, const ffrt_task_attr_t* attr);

/**
 * @brief Submits a task to the queue, and obtains a task handle.
 *
 * @param queue Indicates a queue handle.
 * @param f Indicates a pointer to the task executor.
 * @param attr Indicates a pointer to the task attribute.
 * @return Returns a non-null task handle if the task is submitted;
           returns a null pointer otherwise.
 * @since 10
 */
FFRT_C_API ffrt_task_handle_t ffrt_queue_submit_h(
    ffrt_queue_t queue, ffrt_function_header_t* f, const ffrt_task_attr_t* attr);

/**
 * @brief Submits a task to a queue, simplified from of the ffrt_queue_submit interface.
 *
 * This interface wraps the provided task function and its argument into a task wrapper designed
 * for queue submission (ffrt_function_kind_queue). The task destroy callback (after_func), which
 * would normally handle any post-execution cleanup, is automatically set to NULL in this wrapper,
 * thus omitting any additional cleanup actions. The resulting task wrapper is then submitted to
 * the specified queue via the ffrt_queue_submit interface.
 *
 * @param queue Indicates a queue handle.
 * @param func Indicates a task function to be executed.
 * @param arg Indicates a pointer to the argument or closure data that will be passed to the task function.
 * @param attr Indicates a pointer to the task attribute.
 * @see ffrt_queue_submit
 * @since 20
 */
FFRT_C_API void ffrt_queue_submit_f(ffrt_queue_t queue, ffrt_function_t func, void* arg, const ffrt_task_attr_t* attr);

/**
 * @brief Submits a task to a queue, and obtains a task handle, simplified from the ffrt_queue_submit_h interface.
 *
 * This interface wraps the provided task function and its argument into a task wrapper designed
 * for queue submission (ffrt_function_kind_queue). The task destroy callback (after_func), which
 * would normally handle any post-execution cleanup, is automatically set to NULL in this wrapper,
 * thus omitting any additional cleanup actions. The resulting task wrapper is then submitted to
 * the specified queue via the ffrt_queue_submit_h interface.
 *
 * @param queue Indicates a queue handle.
 * @param func Indicates a task function to be executed.
 * @param arg Indicates a pointer to the argument or closure data that will be passed to the task function.
 * @param attr Indicates a pointer to the task attribute.
 * @return Returns a non-null task handle if the task is submitted;
           returns a null pointer otherwise.
 * @see ffrt_queue_submit_h
 * @since 20
 */
FFRT_C_API ffrt_task_handle_t ffrt_queue_submit_h_f(
    ffrt_queue_t queue, ffrt_function_t func, void* arg, const ffrt_task_attr_t* attr);

/**
 * @brief Waits until a task in the queue is complete.
 *
 * @param handle Indicates a task handle.
 * @since 10
 */
FFRT_C_API void ffrt_queue_wait(ffrt_task_handle_t handle);

/**
 * @brief Cancels a task in the queue.
 *
 * @param handle Indicates a task handle.
 * @return Returns <b>0</b> if the task is canceled;
           returns <b>-1</b> otherwise.
 * @since 10
 */
FFRT_C_API int ffrt_queue_cancel(ffrt_task_handle_t handle);

/**
 * @brief Gets the application main thread queue.
 *
 * @return Returns application main thread queue.
 * @since 12
 */
FFRT_C_API ffrt_queue_t ffrt_get_main_queue(void);

/**
 * @brief Gets the application worker(ArkTs) thread queue.
 *
 * @return Returns application worker(ArkTs) thread queue.
 * @deprecated since 18
 * @since 12
 */
FFRT_C_API ffrt_queue_t ffrt_get_current_queue(void);

/**
 * @brief Gets the task count of a queue.
 *
 * @param queue Indicates a queue handle.
 * @return Returns the queue task count.
 * @since 10
 */
FFRT_C_API uint64_t ffrt_queue_get_task_cnt(ffrt_queue_t queue);

/**
 * @brief Submits a task to a queue, for tasks with the same delay, insert the header.
 *
 * @param queue Indicates a queue handle.
 * @param f Indicates a pointer to the task executor.
 * @param attr Indicates a pointer to the task attribute.
 */
FFRT_C_API void ffrt_queue_submit_head(ffrt_queue_t queue, ffrt_function_header_t* f, const ffrt_task_attr_t* attr);

/**
 * @brief Submits a task to the queue, and obtains a task handle, for tasks with the same delay, insert the header.
 *
 * @param queue Indicates a queue handle.
 * @param f Indicates a pointer to the task executor.
 * @param attr Indicates a pointer to the task attribute.
 * @return Returns a non-null task handle if the task is submitted;
           returns a null pointer otherwise.
 */
FFRT_C_API ffrt_task_handle_t ffrt_queue_submit_head_h(
    ffrt_queue_t queue, ffrt_function_header_t* f, const ffrt_task_attr_t* attr);

#endif // FFRT_API_C_QUEUE_H
/** @} */