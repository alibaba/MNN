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
 * @file task.h
 *
 * @brief Declares the task interfaces in C.
 *
 * @library libffrt.z.so
 * @kit FunctionFlowRuntimeKit
 * @syscap SystemCapability.Resourceschedule.Ffrt.Core
 * @since 10
 */

#ifndef FFRT_API_C_TASK_H
#define FFRT_API_C_TASK_H

#include <stdint.h>
#include "type_def.h"

/**
 * @brief Initializes a task attribute.
 *
 * @param attr Indicates a pointer to the task attribute.
 * @return Returns <b>0</b> if the task attribute is initialized;
           returns <b>-1</b> otherwise.
 * @since 10
 */
FFRT_C_API int ffrt_task_attr_init(ffrt_task_attr_t* attr);

/**
 * @brief Sets the name of a task attribute.
 *
 * @param attr Indicates a pointer to the task attribute.
 * @param name Indicates a pointer to the task name.
 * @since 10
 */
FFRT_C_API void ffrt_task_attr_set_name(ffrt_task_attr_t* attr, const char* name);

/**
 * @brief Gets the name of a task attribute.
 *
 * @param attr Indicates a pointer to the task attribute.
 * @return Returns a non-null pointer to the task name if the name is obtained;
           returns a null pointer otherwise.
 * @since 10
 */
FFRT_C_API const char* ffrt_task_attr_get_name(const ffrt_task_attr_t* attr);

/**
 * @brief Destroys a task attribute, the user needs to invoke this interface.
 *
 * @param attr Indicates a pointer to the task attribute.
 * @since 10
 */
FFRT_C_API void ffrt_task_attr_destroy(ffrt_task_attr_t* attr);

/**
 * @brief Sets the QoS of a task attribute.
 *
 * @param attr Indicates a pointer to the task attribute.
 * @param qos Indicates the QoS.
 * @since 10
 */
FFRT_C_API void ffrt_task_attr_set_qos(ffrt_task_attr_t* attr, ffrt_qos_t qos);

/**
 * @brief Gets the QoS of a task attribute.
 *
 * @param attr Indicates a pointer to the task attribute.
 * @return Returns the QoS, which is <b>ffrt_qos_default</b> by default.
 * @since 10
 */
FFRT_C_API ffrt_qos_t ffrt_task_attr_get_qos(const ffrt_task_attr_t* attr);

/**
 * @brief Sets the delay time of a task attribute.
 *
 * @param attr Indicates a pointer to the task attribute.
 * @param delay_us Indicates the delay time, in microseconds.
 * @since 10
 */
FFRT_C_API void ffrt_task_attr_set_delay(ffrt_task_attr_t* attr, uint64_t delay_us);

/**
 * @brief Gets the delay time of a task attribute.
 *
 * @param attr Indicates a pointer to the task attribute.
 * @return Returns the delay time.
 * @since 10
 */
FFRT_C_API uint64_t ffrt_task_attr_get_delay(const ffrt_task_attr_t* attr);

/**
 * @brief Sets the priority of a task attribute.
 *
 * @param attr Indicates a pointer to the task attribute.
 * @param priority Indicates the execute priority of concurrent queue task.
 * @since 12
 */
FFRT_C_API void ffrt_task_attr_set_queue_priority(ffrt_task_attr_t* attr, ffrt_queue_priority_t priority);

/**
 * @brief Gets the priority of a task attribute.
 *
 * @param attr Indicates a pointer to the task attribute.
 * @return Returns the priority of concurrent queue task.
 * @since 12
 */
FFRT_C_API ffrt_queue_priority_t ffrt_task_attr_get_queue_priority(const ffrt_task_attr_t* attr);

/**
 * @brief Sets the stack size of a task attribute.
 *
 * @param attr Indicates a pointer to the task attribute.
 * @param size Indicates the task stack size, unit is byte.
 * @since 12
 */
FFRT_C_API void ffrt_task_attr_set_stack_size(ffrt_task_attr_t* attr, uint64_t size);

/**
 * @brief Gets the stack size of a task attribute.
 *
 * @param attr Indicates a pointer to the task attribute.
 * @return Returns the task stack size, unit is byte.
 * @since 12
 */
FFRT_C_API uint64_t ffrt_task_attr_get_stack_size(const ffrt_task_attr_t* attr);

/**
 * @brief Sets the schedule timeout of a task attribute.
 *
 * The lower limit of timeout value is 1 ms, if the value is less than 1 ms, it will be set to 1 ms.
 *
 * @param attr Indicates a pointer to the task attribute.
 * @param timeout_us task scheduler timeout.
 */
FFRT_C_API void ffrt_task_attr_set_timeout(ffrt_task_attr_t* attr, uint64_t timeout_us);

/**
 * @brief Gets the schedule timeout of a task attribute.
 *
 * @param attr Indicates a pointer to the task attribute.
 * @return Returns the task schedule timeout.
 */
FFRT_C_API uint64_t ffrt_task_attr_get_timeout(const ffrt_task_attr_t* attr);

/**
 * @brief Updates the QoS of this task.
 *
 * @param qos Indicates the new QoS.
 * @return Returns <b>0</b> if the QoS is updated;
           returns <b>-1</b> otherwise.
 * @since 10
 */
FFRT_C_API int ffrt_this_task_update_qos(ffrt_qos_t qos);

/**
 * @brief Gets the QoS of this task.
 *
 * @return Returns the task qos.
 * @since 12
 */
FFRT_C_API ffrt_qos_t ffrt_this_task_get_qos(void);

/**
 * @brief Gets the ID of this task.
 *
 * @return Returns the task ID.
 * @since 10
 */
FFRT_C_API uint64_t ffrt_this_task_get_id(void);

/**
 * @brief Applies memory for the function execution structure.
 *
 * @param kind Indicates the type of the function execution structure, which can be common or queue.
 * @return Returns a non-null pointer if the memory is allocated;
           returns a null pointer otherwise.
 * @since 10
 */
FFRT_C_API void *ffrt_alloc_auto_managed_function_storage_base(ffrt_function_kind_t kind);

/**
 * @brief Submits a task.
 *
 * @param f Indicates a pointer to the task executor.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param out_deps Indicates a pointer to the output dependencies.
 * @param attr Indicates a pointer to the task attribute.
 * @since 10
 */
FFRT_C_API void ffrt_submit_base(ffrt_function_header_t* f, const ffrt_deps_t* in_deps, const ffrt_deps_t* out_deps,
    const ffrt_task_attr_t* attr);

/**
 * @brief Submits a task, and obtains a task handle.
 *
 * @param f Indicates a pointer to the task executor.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param out_deps Indicates a pointer to the output dependencies.
 * @param attr Indicates a pointer to the task attribute.
 * @return Returns a non-null task handle if the task is submitted;
           returns a null pointer otherwise.
 * @since 10
 */
FFRT_C_API ffrt_task_handle_t ffrt_submit_h_base(ffrt_function_header_t* f, const ffrt_deps_t* in_deps,
    const ffrt_deps_t* out_deps, const ffrt_task_attr_t* attr);

/**
 * @brief Submits a task, simplified from the ffrt_submit_base interface.
 *
 * This interface wraps the provided task function and its argument into a task wrapper
 * designated as a general task (ffrt_function_kind_general). During wrapper creation, the
 * task destroy callback (after_func), which is intended to handle any post-execution cleanup,
 * is simplified to NULL. The resulting task wrapper is then submitted using the underlying
 * ffrt_submit_base interface.
 *
 * @param func Indicates a task function to be executed.
 * @param arg Indicates a pointer to the argument or closure data that will be passed to the task function.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param out_deps Indicates a pointer to the output dependencies.
 * @param attr Indicates a pointer to the task attribute.
 * @see ffrt_submit_base
 * @since 20
 */
FFRT_C_API void ffrt_submit_f(ffrt_function_t func, void* arg, const ffrt_deps_t* in_deps, const ffrt_deps_t* out_deps,
    const ffrt_task_attr_t* attr);

/**
 * @brief Submits a task, and obtains a task handle, simplified from the ffrt_submit_h_base interface.
 *
 * This interface wraps the provided task function and its argument into a task wrapper
 * designated as a general task (ffrt_function_kind_general). During wrapper creation, the
 * task destroy callback (after_func), which is intended to handle any post-execution cleanup,
 * is simplified to NULL. The resulting task wrapper is then submitted using the underlying
 * ffrt_submit_h_base interface.
 *
 * @param func Indicates a task function to be executed.
 * @param arg Indicates a pointer to the argument or closure data that will be passed to the task function.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param out_deps Indicates a pointer to the output dependencies.
 * @param attr Indicates a pointer to the task attribute.
 * @return Returns a non-null task handle if the task is submitted;
           returns a null pointer otherwise.
 * @see ffrt_submit_h_base
 * @since 20
 */
FFRT_C_API ffrt_task_handle_t ffrt_submit_h_f(ffrt_function_t func, void* arg, const ffrt_deps_t* in_deps,
    const ffrt_deps_t* out_deps, const ffrt_task_attr_t* attr);

/**
 * @brief Increases reference count of a task.
 *
 * @param handle Indicates a task handle.
 * @return Returns the task handle original reference count.
 * @since 12
 */
FFRT_C_API uint32_t ffrt_task_handle_inc_ref(ffrt_task_handle_t handle);

/**
 * @brief Decreases reference count of a task.
 *
 * @param handle Indicates a task handle.
 * @return Returns the task handle original reference count.
 * @since 12
 */
FFRT_C_API uint32_t ffrt_task_handle_dec_ref(ffrt_task_handle_t handle);

/**
 * @brief Destroys a task handle, the user needs to invoke this interface.
 *
 * @param handle Indicates a task handle.
 * @since 10
 */
FFRT_C_API void ffrt_task_handle_destroy(ffrt_task_handle_t handle);

/**
 * @brief Waits until the dependent tasks are complete.
 *
 * @param deps Indicates a pointer to the dependent tasks.
 * @since 10
 */
FFRT_C_API void ffrt_wait_deps(const ffrt_deps_t* deps);

/**
 * @brief Waits until all submitted tasks are complete.
 *
 * @since 10
 */
FFRT_C_API void ffrt_wait(void);

/**
 * @brief Sets the thread stack size of a specified QoS level.
 *
 * @param qos Indicates the QoS.
 * @param stack_size Indicates worker thread stack size.
 */
FFRT_C_API ffrt_error_t ffrt_set_worker_stack_size(ffrt_qos_t qos, size_t stack_size);

/**
 * @brief Gets gid of a task.
 *
 * @param handle Indicates a task handle.
 * @return Returns gid.
 */
FFRT_C_API uint64_t ffrt_task_handle_get_id(ffrt_task_handle_t handle);

#endif // FFRT_API_C_TASK_H
/** @} */