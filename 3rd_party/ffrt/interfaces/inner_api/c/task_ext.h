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

#ifndef FFRT_INNER_API_C_TASK_H
#define FFRT_INNER_API_C_TASK_H
#include <stdint.h>
#include <stdbool.h>
#include "type_def_ext.h"

/**
 * @brief Skips a task.
 *
 * @param handle Indicates a task handle.
 * @return Returns <b>0</b> if the task is skipped;
           returns <b>-1</b> otherwise.
 * @since 10
 */
FFRT_C_API int ffrt_skip(ffrt_task_handle_t handle);

/**
 * @brief Sets cgroup attribute.
 *
 * @param qos Indicates the Qos, only support for qos_defined_ive.
 * @param attr Indicates the cgroup attribute.
 * @return Returns <b>0</b> if cgroup attribute set success;
 *         returns <b>-1</b> if cgroup attribute set fail.
 */
FFRT_C_API int ffrt_set_cgroup_attr(ffrt_qos_t qos, ffrt_os_sched_attr* attr);

/**
 * @brief Restore the ffrt threads attribute to the default value for all Qos.
 */
FFRT_C_API void ffrt_restore_qos_config(void);

/**
 * @brief Sets the max num of ffrt threads in a QoS.
 *
 * @param qos Indicates the QoS.
 * @param num Indicates the max num.
 * @return Returns <b>0</b> if max num set success;
 *         return <b>-1</b> if max num set fail;
 */
FFRT_C_API int ffrt_set_cpu_worker_max_num(ffrt_qos_t qos, uint32_t num);

/**
 * @brief Sets whether the task notifies worker, only support for normal task.
 *
 * @param attr Indicates a pointer to the task attribute.
 * @param notify Indicates whether the task notifies worker.
 */
FFRT_C_API void ffrt_task_attr_set_notify_worker(ffrt_task_attr_t* attr, bool notify);

/**
 * @brief Notifies a specified number of workers at a specified QoS level.
 *
 * @param qos Indicates the QoS.
 * @param number Indicates the number of workers to be notified.
 */
FFRT_C_API void ffrt_notify_workers(ffrt_qos_t qos, int number);

/**
 * @brief Obtains the ID of this queue.
 *
 * @return Returns the queue ID.
 */
FFRT_C_API int64_t ffrt_this_queue_get_id(void);

/**
 * @brief Enable the worker escape function (When all the worker threads under a QoS level fully block, the system will
 * temporarily exceed the limit on the number of worker threads and create new worker threads to execute tasks).
 * Delay penalty is added for escape function. As the number of threads increases, the thread creation delay increases.
 * Calling this function does not take effect when the escape function is enabled.
 *
 * @param one_stage_interval_ms Indicates the interval for creating threads in one-stage, default value is 10ms.
 *                              If input parameter value is smaller than the default value, the setting fails.
 * @param two_stage_interval_ms Indicates the interval for creating threads in two-stage, default value is 100ms.
 *                              If input parameter value is smaller than the default value, the setting fails.
 * @param three_stage_interval_ms Indicates the interval for creating threads in three-stage, default value is 1000ms.
 *                              If input parameter value is smaller than the default value, the setting fails.
 * @param one_stage_worker_num Indicates the number of workers in one-stage.
 * @param two_stage_worker_num Indicates the number of workers in two-stage.
 * @return Returns 0 if the parameters are valid and the escape function is enabled successfully;
 *         returns 1 otherwise.
 */
FFRT_C_API int ffrt_enable_worker_escape(uint64_t one_stage_interval_ms, uint64_t two_stage_interval_ms,
    uint64_t three_stage_interval_ms, uint64_t one_stage_worker_num, uint64_t two_stage_worker_num);

/**
 * @brief Disable the worker escape function (When all the worker threads under a QoS level fully block, the system will
 * temporarily exceed the limit on the number of worker threads and create new worker threads to execute tasks).
 */
FFRT_C_API void ffrt_disable_worker_escape(void);

/**
 * @brief Set the sched mode of the QoS.
 */
FFRT_C_API void ffrt_set_sched_mode(ffrt_qos_t qos, ffrt_sched_mode mode);
#endif
