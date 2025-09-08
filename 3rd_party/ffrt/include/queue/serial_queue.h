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
#ifndef FFRT_SERIAL_QUEUE_H
#define FFRT_SERIAL_QUEUE_H

#include "base_queue.h"

namespace ffrt {
class SerialQueue : public BaseQueue {
public:
    SerialQueue();
    ~SerialQueue() override;

    int Push(QueueTask* task) override;
    QueueTask* Pull() override;

    bool GetActiveStatus() override
    {
        std::lock_guard lock(mutex_);
        return isActiveState_.load();
    }

    int GetQueueType() const override
    {
        return ffrt_queue_serial;
    }

private:
    uint32_t overloadThreshold_;
};

std::unique_ptr<BaseQueue> CreateSerialQueue(const ffrt_queue_attr_t* attr);
} // namespace ffrt

#endif // FFRT_SERIAL_QUEUE_H
