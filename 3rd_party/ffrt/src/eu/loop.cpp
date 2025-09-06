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

#include "loop.h"
#include "tm/queue_task.h"
#include "util/event_handler_adapter.h"

namespace ffrt {
Loop::Loop(QueueHandler* handler) : handler_(handler) {}

Loop::~Loop()
{
    Stop();
    handler_->ClearLoop();
}

void Loop::Run()
{
    if (this->GetQueueType() == ffrt_queue_eventhandler_interactive) {
        FFRT_SYSEVENT_LOGE("main loop no need to run\n");
        return;
    }

    while (!stopFlag_.load()) {
        auto task = handler_->PickUpTask();
        if (task) {
            task->Execute();

            if (!stopFlag_.load() && poller_.DeterminePollerReady()) {
                poller_.PollOnce(0);
            }
            continue;
        }

        poller_.PollOnce(-1);
    }
}

void Loop::Stop()
{
    if (this->GetQueueType() != ffrt_queue_eventhandler_interactive) {
        stopFlag_.store(true);
        WakeUp();
    }
}

void Loop::WakeUp()
{
    poller_.WakeUp();
}

int Loop::GetQueueType()
{
    return handler_->GetQueue()->GetQueueType();
}

int Loop::EpollCtl(int op, int fd, uint32_t events, void *data, ffrt_poller_cb cb)
{
    if (op == EPOLL_CTL_ADD) {
        if (this->GetQueueType() == ffrt_queue_eventhandler_interactive) {
            return EventHandlerAdapter::Instance()->AddFdListener(handler_->GetEventHandler(), fd, events, data, cb);
        } else {
            return poller_.AddFdEvent(op, events, fd, data, cb);
        }
    } else if (op == EPOLL_CTL_DEL) {
        if (this->GetQueueType() == ffrt_queue_eventhandler_interactive) {
            return EventHandlerAdapter::Instance()->RemoveFdListener(handler_->GetEventHandler(), fd);
        } else {
            return poller_.DelFdEvent(fd);
        }
    } else if (op == EPOLL_CTL_MOD) {
        FFRT_SYSEVENT_LOGE("EPOLL_CTL_MOD not supported yet");
        return -1;
    } else {
        FFRT_SYSEVENT_LOGE("EPOLL_CTL op invalid");
        return -1;
    }
}

ffrt_timer_t Loop::TimerStart(uint64_t timeout, void* data, ffrt_timer_cb cb, bool repeat)
{
    return poller_.RegisterTimer(timeout, data, cb, repeat);
}

int Loop::TimerStop(ffrt_timer_t handle)
{
    return poller_.UnregisterTimer(handle);
}
} // ffrt