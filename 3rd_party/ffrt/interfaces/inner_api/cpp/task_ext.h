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
 * @file task.h
 *
 * @brief Declares the task inner interfaces in C++.
 *
 * @since 10
 */
#ifndef FFRT_INNER_API_CPP_TASK_H
#define FFRT_INNER_API_CPP_TASK_H
#include <cstdint>
#include "c/task_ext.h"
#include "cpp/task.h"

namespace ffrt {
/**
 * @brief Skips a task.
 *
 * @param handle Indicates a task handle.
 * @return Returns <b>0</b> if the task is skipped;
           returns <b>-1</b> otherwise.
 * @since 10
 */
static inline int skip(task_handle &handle)
{
    return ffrt_skip(handle);
}

void sync_io(int fd);

void set_trace_tag(const char* name);

void clear_trace_tag();

static inline int set_cgroup_attr(qos qos_, ffrt_os_sched_attr *attr)
{
    return ffrt_set_cgroup_attr(qos_, attr);
}

static inline void restore_qos_config()
{
    ffrt_restore_qos_config();
}

static inline int set_cpu_worker_max_num(qos qos_, uint32_t num)
{
    return ffrt_set_cpu_worker_max_num(qos_, num);
}

/**
 * @brief Notifies a specified number of workers at a specified QoS level.
 *
 * @param qos_ Indicates the QoS.
 * @param number Indicates the number of workers to be notified.
 */
static inline void notify_workers(qos qos_, int number)
{
    return ffrt_notify_workers(qos_, number);
}

/**
 * @brief Obtains the ID of this queue.
 *
 * @return Returns the queue ID.
 */
static inline int64_t get_queue_id()
{
    return ffrt_this_queue_get_id();
}

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
static inline int enable_worker_escape(uint64_t one_stage_interval_ms = 10, uint64_t two_stage_interval_ms = 100,
    uint64_t three_stage_interval_ms = 1000, uint64_t one_stage_worker_num = 128, uint64_t two_stage_worker_num = 256)
{
    return ffrt_enable_worker_escape(one_stage_interval_ms, two_stage_interval_ms,
        three_stage_interval_ms, one_stage_worker_num, two_stage_worker_num);
}

/**
 * @brief Disable the worker escape function (When all the worker threads under a QoS level fully block, the system will
 * temporarily exceed the limit on the number of worker threads and create new worker threads to execute tasks).
 *
 */
static inline void disable_worker_escape()
{
    ffrt_disable_worker_escape();
}
} // namespace ffrt
#endif
