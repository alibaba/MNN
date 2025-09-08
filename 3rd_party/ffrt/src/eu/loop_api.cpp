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

#include "c/loop.h"
#include "loop.h"
#include "queue/queue_handler.h"
#include "internal_inc/osal.h"
#include "dfx/log/ffrt_log_api.h"

using namespace ffrt;

API_ATTRIBUTE((visibility("default")))
ffrt_loop_t ffrt_loop_create(ffrt_queue_t queue)
{
    FFRT_COND_DO_ERR((queue == nullptr), return nullptr, "input invalid, queue is nullptr");
    QueueHandler* handler = static_cast<QueueHandler*>(queue);
    FFRT_COND_DO_ERR((!handler->IsValidForLoop()), return nullptr, "queue invalid for loop");

    Loop* innerLoop = new (std::nothrow) Loop(handler);
    FFRT_COND_DO_ERR((innerLoop == nullptr), return nullptr, "failed to construct loop");

    if (!handler->SetLoop(innerLoop)) {
        FFRT_SYSEVENT_LOGE("failed to set loop for handler");
        delete innerLoop;
        return nullptr;
    }
    return static_cast<ffrt_loop_t>(innerLoop);
}

API_ATTRIBUTE((visibility("default")))
int ffrt_loop_destroy(ffrt_loop_t loop)
{
    FFRT_COND_DO_ERR((loop == nullptr), return -1, "input invalid, loop is nullptr");
    Loop* innerLoop = static_cast<Loop*>(loop);
    delete innerLoop;
    return 0;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_loop_run(ffrt_loop_t loop)
{
    FFRT_COND_DO_ERR((loop == nullptr), return -1, "input invalid, loop is nullptr");
    Loop* innerLoop = static_cast<Loop*>(loop);
    innerLoop->Run();
    return 0;
}


API_ATTRIBUTE((visibility("default")))
void ffrt_loop_stop(ffrt_loop_t loop)
{
    FFRT_COND_DO_ERR((loop == nullptr), return, "input invalid, loop is nullptr");
    Loop* innerLoop = static_cast<Loop*>(loop);
    innerLoop->Stop();
}

API_ATTRIBUTE((visibility("default")))
int ffrt_loop_epoll_ctl(ffrt_loop_t loop, int op, int fd, uint32_t events, void *data, ffrt_poller_cb cb)
{
    FFRT_COND_DO_ERR((loop == nullptr), return -1, "input invalid, loop is nullptr");
    Loop* innerLoop = static_cast<Loop*>(loop);
    return innerLoop->EpollCtl(op, fd, events, data, cb);
}

API_ATTRIBUTE((visibility("default")))
ffrt_timer_t ffrt_loop_timer_start(ffrt_loop_t loop, uint64_t timeout, void* data, ffrt_timer_cb cb, bool repeat)
{
    FFRT_COND_DO_ERR((loop == nullptr), return -1, "input invalid, loop is nullptr");
    FFRT_COND_DO_ERR((cb == nullptr), return -1, "input invalid, cb is nullptr");
    Loop* innerLoop = static_cast<Loop*>(loop);
    return innerLoop->TimerStart(timeout, data, cb, repeat);
}

API_ATTRIBUTE((visibility("default")))
int ffrt_loop_timer_stop(ffrt_loop_t loop, ffrt_timer_t handle)
{
    FFRT_COND_DO_ERR((loop == nullptr), return -1, "input invalid, loop is nullptr");
    Loop* innerLoop = static_cast<Loop*>(loop);
    return innerLoop->TimerStop(handle);
}
