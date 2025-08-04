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

#ifndef FFRT_LOOP_HPP
#define FFRT_LOOP_HPP
#include "queue/queue_handler.h"
#include "sync/poller.h"

namespace ffrt {
class Loop {
public:
    explicit Loop(QueueHandler* handler);
    ~Loop();

    void Run();
    void Stop();

    int EpollCtl(int op, int fd, uint32_t events, void *data, ffrt_poller_cb cb);
    ffrt_timer_t TimerStart(uint64_t timeout, void* data, ffrt_timer_cb cb, bool repeat);
    int TimerStop(ffrt_timer_t handle);
    void WakeUp();
    int GetQueueType();

private:
    QueueHandler* handler_ = nullptr;
    Poller poller_;
    std::atomic<bool> stopFlag_ { false };
};
}
#endif
