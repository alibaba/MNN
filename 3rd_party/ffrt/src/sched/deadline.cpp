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

#include "cpp/deadline.h"
#include "c/deadline.h"
#include "internal_inc/osal.h"
#include "sched/interval.h"
#include "dm/dependence_manager.h"
#include "sched/frame_interval.h"
#include "dfx/log/ffrt_log_api.h"

constexpr uint64_t MAX_US_COUNT = static_cast<uint64_t>(std::chrono::microseconds::max().count());

namespace ffrt {
class QosIntervalPrivate {
public:
    template <typename... Args>
    QosIntervalPrivate(uint64_t deadlineUs, const QoS& qos)
    {
        if (qos == qos_user_interactive || qos > qos_max) {
            it = std::make_unique<FrameInterval>(deadlineUs, qos);
        } else {
            it = std::make_unique<DefaultInterval>(deadlineUs, qos);
        }
    }

    Interval* operator->()
    {
        if (it == nullptr) {
            FFRT_SYSEVENT_LOGE("Invalid QoS Interval!");
            return nullptr;
        }
        return it.get();
    }

private:
    std::unique_ptr<Interval> it;
};
}; // namespace ffrt

#ifdef __cplusplus
extern "C" {
#endif
API_ATTRIBUTE((visibility("default")))
ffrt_interval_t ffrt_interval_create(uint64_t deadline_us, ffrt_qos_t qos)
{
    if (qos < static_cast<int>(ffrt_qos_deadline_request)) {
        FFRT_LOGE("Invalid QoS Interval!");
        return nullptr;
    }

    if (deadline_us > MAX_US_COUNT) {
        FFRT_LOGW("Deadline exceeds maximum allowed value %llu us. Clamping to %llu us.", deadline_us, MAX_US_COUNT);
        deadline_us = MAX_US_COUNT;
    }

    return new ffrt::QosIntervalPrivate(deadline_us, qos);
}

API_ATTRIBUTE((visibility("default")))
int ffrt_interval_update(ffrt_interval_t it, uint64_t new_deadline_us)
{
    if (!it) {
        FFRT_LOGE("QoS Interval Not Created Or Has Been Canceled!");
        return ffrt_error;
    }

    if (new_deadline_us > MAX_US_COUNT) {
        FFRT_LOGW("Deadline exceeds maximum allowed value %llu us. Clamping to %llu us.", new_deadline_us,
            MAX_US_COUNT);
        new_deadline_us = MAX_US_COUNT;
    }

    auto _it = static_cast<ffrt::QosIntervalPrivate *>(it);

    (*_it)->Update(new_deadline_us);
    return ffrt_success;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_interval_begin(ffrt_interval_t it)
{
    if (!it) {
        FFRT_LOGE("QoS Interval Not Created Or Has Been Canceled!");
        return ffrt_error;
    }

    auto _it = static_cast<ffrt::QosIntervalPrivate *>(it);

    return (*_it)->Begin();
}

API_ATTRIBUTE((visibility("default")))
int ffrt_interval_end(ffrt_interval_t it)
{
    if (!it) {
        FFRT_LOGE("QoS Interval Not Created Or Has Been Canceled!");
        return ffrt_error;
    }

    auto _it = static_cast<ffrt::QosIntervalPrivate *>(it);

    (*_it)->End();
    return ffrt_success;
}

API_ATTRIBUTE((visibility("default")))
void ffrt_interval_destroy(ffrt_interval_t it)
{
    if (!it) {
        FFRT_LOGE("QoS Interval Not Created Or Has Been Canceled!");
        return;
    }

    delete static_cast<ffrt::QosIntervalPrivate *>(it);
}

API_ATTRIBUTE((visibility("default")))
int ffrt_interval_join(ffrt_interval_t it)
{
    if (!it) {
        FFRT_LOGE("QoS Interval Not Created Or Has Been Canceled!");
        return ffrt_error;
    }

    auto _it = static_cast<ffrt::QosIntervalPrivate *>(it);

    (*_it)->Join();
    return ffrt_success;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_interval_leave(ffrt_interval_t it)
{
    if (!it) {
        FFRT_LOGE("QoS Interval Not Created Or Has Been Canceled!");
        return ffrt_error;
    }

    auto _it = static_cast<ffrt::QosIntervalPrivate *>(it);

    (*_it)->Leave();
    return ffrt_success;
}
#ifdef __cplusplus
}
#endif
