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

#include "c/thread.h"
#include "cpp/thread.h"
#include "internal_inc/osal.h"
#include "dfx/log/ffrt_log_api.h"

#ifdef __cplusplus
extern "C" {
#endif
struct ThreadRes {
    bool is_joinable;
    void* result;
};
API_ATTRIBUTE((visibility("default")))
int ffrt_thread_create(ffrt_thread_t* thr, const ffrt_thread_attr_t* attr, void* (*func)(void*), void* arg)
{
    if (!thr || !func) {
        FFRT_LOGE("thr and func should not be empty");
        return ffrt_error_inval;
    }
    if (attr) {
        FFRT_LOGE("attr should be empty");
        return ffrt_error;
    }

    auto p = reinterpret_cast<ThreadRes*>(malloc(sizeof(ThreadRes)));
    if (p == nullptr) {
        FFRT_LOGE("p is empty");
        return ffrt_error_nomem;
    }
    p->is_joinable = true;
    p->result = nullptr;
    ffrt::submit([p, func, arg]() {
        p->result = func(arg);
    }, {}, {p});

    *thr = p;
    return ffrt_success;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_thread_join(ffrt_thread_t thr, void** res)
{
    if (!thr || !res) {
        FFRT_LOGE("thr or res should not be empty");
        return ffrt_error_inval;
    }

    auto p = reinterpret_cast<ThreadRes*>(thr);
    if (p == nullptr || !p->is_joinable) {
        return ffrt_error_inval;
    }
    ffrt::wait({p});
    *res = p->result;
    p->is_joinable = false;
    free(p);
    return ffrt_success;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_thread_detach(ffrt_thread_t thr)
{
    if (!thr) {
        FFRT_LOGE("thr should not be empty");
        return ffrt_error_inval;
    }
    auto p = reinterpret_cast<ThreadRes*>(thr);
    if (p == nullptr || !p->is_joinable) {
        return ffrt_error_inval;
    }
    p->is_joinable = false;
    ffrt::submit([thr]() { free(thr); }, {thr});
    return ffrt_success;
}
#ifdef __cplusplus
}
#endif
