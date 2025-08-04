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

#include "eventhandler_interactive_queue.h"
#include "dfx/log/ffrt_log_api.h"

namespace ffrt {
EventHandlerInteractiveQueue::~EventHandlerInteractiveQueue()
{
    FFRT_LOGI("destruct eventhandler interactive queueId=%u leave", queueId_);
}

int EventHandlerInteractiveQueue::Push(QueueTask* task)
{
    Priority prio = EventHandlerAdapter::Instance()->ConvertPriority(task->GetPriority());
    FFRT_COND_DO_ERR((prio > Priority::IDLE), return FAILED, "Priority invalid.");

    int delayUs = static_cast<int>(task->GetDelay());
    auto f = reinterpret_cast<ffrt_function_header_t*>(task->func_storage);
    std::function<void()> func = [=]() {
        f->exec(f);
        if (f->destroy) {
            f->destroy(f);
        }
        task->DecDeleteRef();
    };

    int msPerSecond = 1000;
    ffrt::TaskOptions taskOptions(
        task->label, delayUs / msPerSecond, static_cast<Priority>(prio), static_cast<uintptr_t>(task->gid));
    bool taskStatus = EventHandlerAdapter::Instance()->PostTask(eventHandler_, func, taskOptions);
    FFRT_COND_DO_ERR((!taskStatus), return FAILED, "post task fail");

    return SUCC;
}

std::unique_ptr<BaseQueue> CreateEventHandlerInteractiveQueue(const ffrt_queue_attr_t* attr)
{
    (void)attr;
    return std::make_unique<EventHandlerInteractiveQueue>();
}
} // namespace ffrt
