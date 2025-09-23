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
 * @file sleep.h
 *
 * @brief Declares the sleep and yield interfaces in C++.
 *
 * @library libffrt.z.so
 * @kit FunctionFlowRuntimeKit
 * @syscap SystemCapability.Resourceschedule.Ffrt.Core
 * @since 10
 */

#ifndef FFRT_API_CPP_SLEEP_H
#define FFRT_API_CPP_SLEEP_H

#include <chrono>
#include "c/sleep.h"

namespace ffrt {
/**
 * @namespace ffrt::this_task
 * @brief Contains utility functions for the currently executing task.
 */
namespace this_task {
/**
 * @brief Yields the execution of the current task.
 *
 * This function allows other tasks or threads to be scheduled by yielding the current task's execution.
 *
 * @since 10
 */
static inline void yield()
{
    ffrt_yield();
}

/**
 * @brief Suspends the current task for a specified duration.
 *
 * @tparam _Rep The type of the representation of the duration (e.g., int or long).
 * @tparam _Period The period of the duration (e.g., milliseconds, microseconds).
 * @param d The duration for which the task should be suspended.
 * @since 10
 */
template <class _Rep, class _Period>
inline void sleep_for(const std::chrono::duration<_Rep, _Period>& d)
{
    ffrt_usleep(std::chrono::duration_cast<std::chrono::microseconds>(d).count());
}


/**
 * @brief Suspends the current task until a specific time point.
 *
 * @tparam _Clock The clock type that provides the time point (e.g., steady_clock).
 * @tparam _Duration The duration type for the time point.
 * @param abs_time The absolute time point at which the task should resume.
 * @since 10
 */
template<class _Clock, class _Duration>
inline void sleep_until(
    const std::chrono::time_point<_Clock, _Duration>& abs_time)
{
    sleep_for(abs_time.time_since_epoch() - _Clock::now().time_since_epoch());
}
} // namespace this_task
} // namespace ffrt

#endif // FFRT_API_CPP_SLEEP_H
/** @} */