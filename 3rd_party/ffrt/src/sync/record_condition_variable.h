/*
 * Copyright (c) 2024 Huawei Device Co., Ltd.
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

#ifndef RECORD_CONDITION_VARIABLE_H
#define RECORD_CONDITION_VARIABLE_H

#include "cpp/condition_variable.h"
#include "sync/record_mutex.h"

namespace ffrt {
class RecordConditionVariable {
public:
    template <typename Rep, typename Period>
    cv_status WaitFor(std::unique_lock<RecordMutex>& lk, const std::chrono::duration<Rep, Period>& sleep_time) noexcept
    {
        timespec ts;
        std::chrono::nanoseconds ns = std::chrono::steady_clock::now().time_since_epoch();
        ns += std::chrono::duration_cast<std::chrono::nanoseconds>(sleep_time);
        ts.tv_sec = std::chrono::duration_cast<std::chrono::seconds>(ns).count();
        ns -= std::chrono::seconds(ts.tv_sec);
        ts.tv_nsec = static_cast<long>(ns.count());

        auto ret = ffrt_cond_timedwait(&cv_, lk.mutex()->GetMutex(), &ts);
        if (ret == ffrt_success) {
            return cv_status::no_timeout;
        }
        return cv_status::timeout;
    }

    void NotifyOne() noexcept
    {
        cv_.notify_one();
    }

    void NotifyAll() noexcept
    {
        cv_.notify_all();
    }

private:
    ffrt::condition_variable cv_;
};
} // namespace ffrt

#endif