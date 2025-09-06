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
#ifndef FFRT_API_CPP_DEADLINE_H
#define FFRT_API_CPP_DEADLINE_H
#include <stdint.h>
#include "c/deadline.h"

namespace ffrt {
using interval = ffrt_interval_t;

/**
 * @brief app create an anonymous interval, the number is limited. should specify the deadline
 */
static inline interval qos_interval_create(uint64_t deadline_us, qos qos_ = static_cast<int>(qos_deadline_request))
{
    return ffrt_interval_create(deadline_us, qos_);
}

/**
 * @brief destroy a interval
 */
static inline void qos_interval_destroy(interval it)
{
    ffrt_interval_destroy(it);
}

/**
 * @brief start the interval
 */
static inline int qos_interval_begin(interval it)
{
    return ffrt_interval_begin(it);
}

/**
 * @brief update interval
 */
static inline int qos_interval_update(interval it, uint64_t new_deadline_us)
{
    return ffrt_interval_update(it, new_deadline_us);
}

/**
 * @brief interval become inactive until next begin
 */
static inline int qos_interval_end(interval it)
{
    return ffrt_interval_end(it);
}

/**
 * @brief current task or thread join an interval, only allow FIXED number of threads to join a interval
 */
static inline int qos_interval_join(interval it)
{
    return ffrt_interval_join(it);
}

/**
 * @brief current task or thread leave an interval
 */
static inline int qos_interval_leave(interval it)
{
    return ffrt_interval_leave(it);
}
}; // namespace ffrt

#endif
