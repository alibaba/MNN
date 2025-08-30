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
#ifndef FFRT_API_C_EXECUTOR_TASK_H
#define FFRT_API_C_EXECUTOR_TASK_H

#include <stdint.h>
#include <stdbool.h>
#include <sys/epoll.h>
#include "type_def_ext.h"
#include "c/timer.h"

/**
 * @brief Struct of the executor_task, also aligns with the base task class.
 */
typedef struct ffrt_executor_task {
    uintptr_t reserved[2];
    uintptr_t type; // 0: TaskCtx, 1~: Dynamicly Define Task, User Space Address: libuv work
    void* wq[2];
} ffrt_executor_task_t;

/**
 * @brief The executor task types.
 */
typedef enum {
    ffrt_normal_task = 0,
    ffrt_io_task = 1,
    ffrt_uv_task, // only used to register func for libuv
    ffrt_queue_task,
    ffrt_xpu_task,
    ffrt_invalid_task,
} ffrt_executor_task_type_t;

/**
 * @brief Function defined to be executed by the workers.
 *
 * @param data Indicates the args of the function defined by users.
 * @param qos Indicates the qos of the task.
 */
typedef void (*ffrt_executor_task_func)(ffrt_executor_task_t* data, ffrt_qos_t qos);

/**
 * @brief Registers a user-defined function for the workers to execute.
 *
 * @param func Indicates a user-defined function.
 * @param type Indicates which task type the function belongs to.
 */
FFRT_C_API void ffrt_executor_task_register_func(ffrt_executor_task_func func, ffrt_executor_task_type_t type);

/**
 * @brief Submits a UV task or IO task.
 *
 * @param task Indicates a pointer to the task.
 * @param attr Indicates a pointer to the task attribute.
 */
FFRT_C_API void ffrt_executor_task_submit(ffrt_executor_task_t* task, const ffrt_task_attr_t* attr);

/**
 * @brief Cancels a UV task or IO task.
 *
 * @param task Indicates a pointer to the task.
 * @param attr Indicates the qos of the task.
 * @return Returns success or failed.
 *
 */
FFRT_C_API int ffrt_executor_task_cancel(ffrt_executor_task_t* task, const ffrt_qos_t qos);

/**
 * @brief Wakeups the ffrt poller.
 *
 * @param qos Indicates the qos of the poller.
 */
FFRT_C_API void ffrt_poller_wakeup(ffrt_qos_t qos);

/**
 * @brief Gets the number of epoll operations performed.
 *
 * @param qos Indicates the qos of the poller.
 */
FFRT_C_API uint8_t ffrt_epoll_get_count(ffrt_qos_t qos);

/**
 * @brief Querys the ffrt timer.
 *
 * @param handler Indicates the handler of the timer.
 */
FFRT_C_API ffrt_timer_query_t ffrt_timer_query(ffrt_qos_t qos, ffrt_timer_t handle);

/**
 * @brief Submits a fd event to the poller.
 *
 * FFRT provides two ways to deal with fd events:
 *     - Mode 1: execute the cb function
 *     - Mode 2: register fd event into FFRT by ffrt_epoll_ctl, then wait for fd event by ffrt_epoll_wait
 * In Mode 1, ffrt_epoll_ctl can be called in user thread or ffrt task
 *            FFRT will monitor the fd event and then execute the cb function.
 * In Mode 2, both ffrt_epoll_ctl and ffrt_epoll_wait must be called in the same ffrt task.
 *
 * @warning Do not call `exit` in `cb` - this my cause unexpected behavior.
 *
 * @param qos Indicates the qos of the poller.
 * @param op Indicates the option of the event.
 * @param fd Indicates the fd of the event. Only supports eventfd, timerfd, and harware I/O.
 * @param events Indicates the events of the events.
 * @param data Indicates the args of the event callback function.
 * @param cb Indicates the callback function of the event.
 */
FFRT_C_API int ffrt_epoll_ctl(ffrt_qos_t qos, int op, int fd, uint32_t events, void* data, ffrt_poller_cb cb);

/**
 * @brief Waits an epoll event.
 *
 * Calling ffrt_epoll_wait will block the task.
 * Both ffrt_epoll_wait and ffrt_epoll_ctl must be called in the same ffrt task.
 *
 * @param qos Indicates the qos of the poller.
 * @param events Indicates the events of the event.
 * @param max_events Indicates the max event value.
 * @param timeout Indicates the time to timeout.
 */
FFRT_C_API int ffrt_epoll_wait(ffrt_qos_t qos, struct epoll_event* events, int max_events, int timeout);

/**
 * @brief Gets the time a task has waited in the poller.
 *
 * @param taskHandle Indicates the pointer of a task.
 */
FFRT_C_API uint64_t ffrt_epoll_get_wait_time(void* taskHandle);

/**
 * @brief Submit a coroutine IO task.
 *
 * @param co Indicates the args of a task executor function.
 * @param exec Indicates a task executor function.
 * @param destroy Indicates the destroy function of a task.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param out_deps Indicates a pointer to the output dependencies.
 * @param attr Indicates a task attribute.
 */
FFRT_C_API void ffrt_submit_coroutine(void* co, ffrt_coroutine_ptr_t exec, ffrt_function_t destroy,
    const ffrt_deps_t* in_deps, const ffrt_deps_t* out_deps, const ffrt_task_attr_t* attr);

/**
 * @brief Wakeups a coroutine IO task.
 *
 * @param task Indicates a pointer to the task.
 */
FFRT_C_API void ffrt_wake_coroutine(void* task);

/**
 * @brief Get the pointer of the current task.
 *
 * @return Returns a pointer.
 */
FFRT_C_API void* ffrt_get_current_task(void);

/**
 * @brief Obtains current coroutine stack address and size.
 *
 * @param stack_addr Coroutine stack address.
 * @param size Coroutine stack size.
 * @return Returns <b>0</b> if the stack is obtained;
 *         returns <b>-1</b> otherwise.
 * @since 12
 */
FFRT_C_API bool ffrt_get_current_coroutine_stack(void** stack_addr, size_t* size);

/**
 * @brief Obtains current task.
 *
 * @param none.
 * @return Returns current task.
 * @since 12
 */
FFRT_C_API void* ffrt_get_cur_task(void);

/**
 * @brief Set the taskLocal flag in ffrt_task_attr.
 *
 * @param attr The ffrt_task_attr struct.
 * @param task_local The bool value to be set.
 * @return none.
 * @since 12
 */
FFRT_C_API void ffrt_task_attr_set_local(ffrt_task_attr_t* attr, bool task_local);

/**
 * @brief Obtains the taskLocal flag in ffrt_task_attr.
 *
 * @param attr The ffrt_task_attr struct.
 * @return The bool value of task_local.
 * @since 12
 */
FFRT_C_API bool ffrt_task_attr_get_local(ffrt_task_attr_t* attr);

/**
 * @brief Obtains the thread id of the input task handle.
 *
 * @param task_handle The task pointer.
 * @return The thread id of the input task handle.
 */
FFRT_C_API pthread_t ffrt_task_get_tid(void* task_handle);

/**
 * @brief Obtains the task id cached by the current thread.
 *
 * @return Returns the task id.
 */
FFRT_C_API uint64_t ffrt_get_cur_cached_task_id(void);
#endif