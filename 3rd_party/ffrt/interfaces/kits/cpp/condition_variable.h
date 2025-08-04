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
 * @file condition_variable.h
 *
 * @brief Declares the condition variable interfaces in C++.
 *
 * @library libffrt.z.so
 * @kit FunctionFlowRuntimeKit
 * @syscap SystemCapability.Resourceschedule.Ffrt.Core
 * @since 10
 */

#ifndef FFRT_API_CPP_CONDITION_VARIABLE_H
#define FFRT_API_CPP_CONDITION_VARIABLE_H

#include <chrono>
#include <mutex>
#include "mutex.h"
#include "c/condition_variable.h"

namespace ffrt {
/**
 * @enum cv_status
 * @brief Specifies the result of a condition variable wait operation.
 *
 * @since 10
 */
enum class cv_status {
    no_timeout, /**< Indicates that the wait ended because the condition was met. */
    timeout     /**< Indicates that the wait ended due to a timeout. */
};

/**
 * @class condition_variable
 * @brief A class implementing condition variable synchronization primitives.
 *
 * Provides mechanisms for threads to wait for a notification or a specified condition to be met,
 * supporting both timed and non-timed operations.
 *
 * @since 10
 */
class condition_variable : public ffrt_cond_t {
public:
    /**
     * @brief Constructs a new condition_variable object.
     *
     * @since 10
     */
    condition_variable()
    {
        ffrt_cond_init(this, nullptr);
    }

    /**
     * @brief Destroys the condition_variable object.
     *
     * @since 10
     */
    ~condition_variable() noexcept
    {
        ffrt_cond_destroy(this);
    }

    /**
     * @brief Deleted copy constructor to prevent copying of the condition_variable object.
     */
    condition_variable(const condition_variable&) = delete;

    /**
     * @brief Deleted copy assignment operator to prevent assignment of the condition_variable object.
     */
    condition_variable& operator=(const condition_variable&) = delete;

    /**
     * @brief Waits until a predicate is satisfied or a timeout occurs.
     *
     * @tparam Clock The clock type used for the timeout.
     * @tparam Duration The duration type used for the timeout.
     * @tparam Pred The predicate type.
     * @param lk A unique lock on the associated mutex.
     * @param tp The time point when the wait should time out.
     * @param pred The predicate to be satisfied.
     * @return true if the predicate was satisfied before the timeout, false otherwise.
     * @since 10
     */
    template <typename Clock, typename Duration, typename Pred>
    bool wait_until(
        std::unique_lock<mutex>& lk, const std::chrono::time_point<Clock, Duration>& tp, Pred&& pred) noexcept
    {
        while (!pred()) {
            if (wait_until(lk, tp) == cv_status::timeout) {
                return pred();
            }
        }
        return true;
    }

    /**
     * @brief Waits until a specified time point.
     *
     * @tparam Clock The clock type used for the timeout.
     * @tparam Duration The duration type used for the timeout.
     * @param lk A unique lock on the associated mutex.
     * @param tp The time point when the wait should time out.
     * @return A cv_status value indicating whether the wait ended due to a timeout or a condition.
     * @since 10
     */
    template <typename Clock, typename Duration>
    cv_status wait_until(std::unique_lock<mutex>& lk, const std::chrono::time_point<Clock, Duration>& tp) noexcept
    {
        return _wait_for(lk, tp - Clock::now());
    }

    /**
     * @brief Waits for a specified duration.
     *
     * @tparam Rep The representation type of the duration.
     * @tparam Period The period type of the duration.
     * @param lk A unique lock on the associated mutex.
     * @param sleep_time The duration to wait for.
     * @return A cv_status value indicating whether the wait ended due to a timeout or a condition.
     * @since 10
     */
    template <typename Rep, typename Period>
    cv_status wait_for(std::unique_lock<mutex>& lk, const std::chrono::duration<Rep, Period>& sleep_time) noexcept
    {
        return _wait_for(lk, sleep_time);
    }

    /**
     * @brief Waits for a specified duration or until a predicate is satisfied.
     *
     * @tparam Rep The representation type of the duration.
     * @tparam Period The period type of the duration.
     * @tparam Pred The predicate type.
     * @param lk A unique lock on the associated mutex.
     * @param sleepTime The duration to wait for.
     * @param pred The predicate to be satisfied.
     * @return true if the predicate was satisfied before the timeout, false otherwise.
     * @since 10
     */
    template <typename Rep, typename Period, typename Pred>
    bool wait_for(
        std::unique_lock<mutex>& lk, const std::chrono::duration<Rep, Period>& sleepTime, Pred&& pred) noexcept
    {
        if (sleepTime <= sleepTime.zero()) {
            return pred();
        }

        auto now = std::chrono::steady_clock::now();
        auto nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
        auto sleepNs = std::chrono::duration_cast<std::chrono::nanoseconds>(sleepTime);
        std::chrono::steady_clock::time_point absoluteTime;
        if (sleepNs.count() > ((std::chrono::nanoseconds::max)().count() - nowNs.count())) {
            absoluteTime = (std::chrono::steady_clock::time_point::max)();
        } else {
            absoluteTime = now + sleepNs;
        }

        return wait_until(lk, absoluteTime, std::forward<Pred>(pred));
    }

    /**
     * @brief Waits until a predicate is satisfied.
     *
     * @tparam Pred The predicate type.
     * @param lk A unique lock on the associated mutex.
     * @param pred The predicate to be satisfied.
     * @since 10
     */
    template <typename Pred>
    void wait(std::unique_lock<mutex>& lk, Pred&& pred)
    {
        while (!pred()) {
            wait(lk);
        }
    }

    /**
     * @brief Waits indefinitely until notified.
     *
     * @param lk A unique lock on the associated mutex.
     * @since 10
     */
    void wait(std::unique_lock<mutex>& lk)
    {
        ffrt_cond_wait(this, lk.mutex());
    }

    /**
     * @brief Notifies one thread waiting on the condition variable.
     *
     * @since 10
     */
    void notify_one() noexcept
    {
        ffrt_cond_signal(this);
    }

    /**
     * @brief Notifies all threads waiting on the condition variable.
     *
     * @since 10
     */
    void notify_all() noexcept
    {
        ffrt_cond_broadcast(this);
    }

private:
    /**
     * @brief Internal function to perform a timed wait.
     *
     * @tparam Rep The representation type of the duration.
     * @tparam Period The period type of the duration.
     * @param lk A unique lock on the associated mutex.
     * @param dur The duration to wait for.
     * @return A cv_status value indicating whether the wait ended due to a timeout or a condition.
     * @since 10
     */
    template <typename Rep, typename Period>
    cv_status _wait_for(std::unique_lock<mutex>& lk, const std::chrono::duration<Rep, Period>& dur) noexcept
    {
        if (dur <= dur.zero()) {
            return cv_status::timeout;
        }

        auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch());
        auto dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(dur);

        std::chrono::nanoseconds ns;
        if (now_ns.count() > (std::chrono::nanoseconds::max)().count() - dur_ns.count()) {
            ns = (std::chrono::nanoseconds::max)();
        } else {
            ns = now_ns + dur_ns;
        }

        timespec ts;
        ts.tv_sec = std::chrono::duration_cast<std::chrono::seconds>(ns).count();
        ns -= std::chrono::seconds(ts.tv_sec);
        ts.tv_nsec = static_cast<long>(ns.count());

        auto ret = ffrt_cond_timedwait(this, lk.mutex(), &ts);
        if (ret == ffrt_success) {
            return cv_status::no_timeout;
        }
        return cv_status::timeout;
    }
};
} // namespace ffrt

#endif // FFRT_API_CPP_CONDITION_VARIABLE_H
/** @} */