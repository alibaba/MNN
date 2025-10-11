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

#include "cpp/condition_variable.h"
#include "c/condition_variable.h"
#include "sync/wait_queue.h"
#include "sync/mutex_private.h"
#include "internal_inc/osal.h"
#include "dfx/log/ffrt_log_api.h"

namespace ffrt {
using condition_variable_private = WaitQueue;
}

#ifdef __cplusplus
extern "C" {
#endif
API_ATTRIBUTE((visibility("default")))
int ffrt_cond_init(ffrt_cond_t* cond, const ffrt_condattr_t* attr)
{
    if (!cond) {
        FFRT_LOGE("cond should not be empty");
        return ffrt_error_inval;
    }
    static_assert(sizeof(ffrt::condition_variable_private) <= ffrt_cond_storage_size,
        "size must be less than ffrt_cond_storage_size");

    new (cond) ffrt::condition_variable_private();
    return ffrt_success;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_cond_signal(ffrt_cond_t* cond)
{
    if (!cond) {
        FFRT_LOGE("cond should not be empty");
        return ffrt_error_inval;
    }
    auto p = reinterpret_cast<ffrt::condition_variable_private *>(cond);
    p->NotifyOne();
    return ffrt_success;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_cond_broadcast(ffrt_cond_t* cond)
{
    if (!cond) {
        FFRT_LOGE("cond should not be empty");
        return ffrt_error_inval;
    }
    auto p = reinterpret_cast<ffrt::condition_variable_private *>(cond);
    p->NotifyAll();
    return ffrt_success;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_cond_wait(ffrt_cond_t* cond, ffrt_mutex_t* mutex)
{
    if (!cond || !mutex) {
        FFRT_LOGE("cond and mutex should not be empty");
        return ffrt_error_inval;
    }
    auto pc = reinterpret_cast<ffrt::condition_variable_private *>(cond);
    auto pm = reinterpret_cast<ffrt::mutexPrivate *>(mutex);
    pc->SuspendAndWait(pm);
    return ffrt_success;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_cond_timedwait(ffrt_cond_t* cond, ffrt_mutex_t* mutex, const struct timespec* time_point)
{
    if (!cond || !mutex || !time_point) {
        FFRT_LOGE("cond, mutex and time_point should not be empty");
        return ffrt_error_inval;
    }
    auto pc = reinterpret_cast<ffrt::condition_variable_private *>(cond);
    auto pm = reinterpret_cast<ffrt::mutexPrivate *>(mutex);

    using namespace std::chrono;
    auto duration = seconds{ time_point->tv_sec } + nanoseconds{ time_point->tv_nsec };
    auto tp = ffrt::TimePoint {
        duration_cast<steady_clock::duration>(duration_cast<nanoseconds>(duration))
    };

    return pc->SuspendAndWaitUntil(pm, tp);
}

API_ATTRIBUTE((visibility("default")))
int ffrt_cond_destroy(ffrt_cond_t* cond)
{
    if (!cond) {
        FFRT_LOGE("cond should not be empty");
        return ffrt_error_inval;
    }
    auto p = reinterpret_cast<ffrt::condition_variable_private *>(cond);
    p->~WaitQueue();
    return ffrt_success;
}
#ifdef __cplusplus
}
#endif
