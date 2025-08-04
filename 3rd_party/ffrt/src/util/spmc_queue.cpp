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

#include "util/spmc_queue.h"
#include <cstdlib>
#include "dfx/log/ffrt_log_api.h"
namespace ffrt {
SpmcQueue::~SpmcQueue()
{
    if (buf_ != nullptr) {
        free(buf_);
        buf_ = nullptr;
    }
}

int SpmcQueue::Init(unsigned int capacity)
{
    if (capacity == 0) {
        return -1;
    }

    buf_ = reinterpret_cast<void**>(malloc(capacity * sizeof(void*)));
    if (buf_ == nullptr) {
        FFRT_LOGE("Queue malloc failed, size: %u", capacity * sizeof(void*));
        return -1;
    }

    capacity_ = capacity;
    return 0;
}

unsigned int SpmcQueue::GetLength() const
{
    return tail_.load() - head_.load();
}

unsigned int SpmcQueue::GetCapacity() const
{
    return capacity_;
}

void* SpmcQueue::PopHead()
{
    if (buf_ == nullptr) {
        return nullptr;
    }

    while (true) {
        unsigned int head = head_.load();
        unsigned int tail = tail_.load();
        if (tail == head) {
            return nullptr;
        }

        void* res = buf_[head % capacity_];
        if (atomic_compare_exchange_weak(&head_, &head, head + 1)) {
            return res;
        }
    }
}

int SpmcQueue::PushTail(void* object)
{
    if (buf_ == nullptr) {
        return -1;
    }

    unsigned int head = head_.load();
    unsigned int tail = tail_.load();
    if ((tail - head) < capacity_) {
        buf_[tail % capacity_] = object;
        tail_.store(tail + 1);
        return 0;
    }

    return -1;
}

unsigned int SpmcQueue::PopHeadToAnotherQueue(SpmcQueue& dstQueue, unsigned int elementNum, PushFunc func)
{
    if (elementNum == 0) {
        return 0;
    }

    unsigned int pushCount = 0;
    while ((dstQueue.GetLength() < dstQueue.GetCapacity()) && (head_.load() != tail_.load())) {
        void* element = PopHead();
        if (element == nullptr) {
            break;
        }

        int ret = dstQueue.PushTail(element);
        if (ret != 0) {
            func(element);
            return pushCount;
        }

        if (++pushCount == elementNum) {
            break;
        }
    }

    return pushCount;
}
}