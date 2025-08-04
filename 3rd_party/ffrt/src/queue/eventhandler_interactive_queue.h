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
#ifndef FFRT_EVENTHANDLER_INTERACTIVE_QUEUE_H
#define FFRT_EVENTHANDLER_INTERACTIVE_QUEUE_H

#include "queue/base_queue.h"
#include "tm/queue_task.h"
#include "util/event_handler_adapter.h"

namespace ffrt {
class EventHandlerInteractiveQueue : public BaseQueue {
public:
    explicit EventHandlerInteractiveQueue() {}
    ~EventHandlerInteractiveQueue() override;

    int Push(QueueTask* task) override;

    QueueTask* Pull() override
    {
        return nullptr;
    }

    bool GetActiveStatus() override
    {
        return false;
    }

    int GetQueueType() const override
    {
        return ffrt_queue_eventhandler_interactive;
    }

    int Remove(const QueueTask* task) override
    {
        uintptr_t taskId = static_cast<uintptr_t>(task->gid);
        int ret = EventHandlerAdapter::Instance()->RemoveTask(eventHandler_, taskId);
        return ret;
    }

    void Stop() override
    {
        std::lock_guard lock(mutex_);
        isExit_ = true;
    }

    inline void SetEventHandler(void* eventHandler)
    {
        eventHandler_ = eventHandler;
    }

    inline void* GetEventHandler()
    {
        return eventHandler_;
    }

protected:
    void* eventHandler_ = nullptr;
};

std::unique_ptr<BaseQueue> CreateEventHandlerInteractiveQueue(const ffrt_queue_attr_t* attr);
} // namespace ffrt

#endif // FFRT_EVENTHANDLER_INTERACTIVE_QUEUE_H
