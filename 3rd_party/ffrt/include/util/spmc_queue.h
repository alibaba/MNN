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

#ifndef FFRT_SPMC_QUEUE_H
#define FFRT_SPMC_QUEUE_H

#include <atomic>

namespace ffrt {
class SpmcQueue {
public:
    ~SpmcQueue();

    unsigned int GetLength() const;
    unsigned int GetCapacity() const;

    /**
    * @brief 初始化队列。
    * @param capacity 队列容量。
    * @retval 成功返回0，失败返回-1。
    */
    int Init(unsigned int capacity);

    /**
    * @brief 取出队列首部元素。
    * @retval 指向首部元素的指针，若队列为空则返回nullptr。
    */
    void* PopHead();

    /**
    * @brief 将元素推入队列尾部。
    * @param object 要推入队列的元素。
    * @retval 成功返回0，失败返回-1。
    */
    int PushTail(void* object);

    /**
    * @brief 从队列首部批量取出元素后将元素批量推入目标队列尾部。
    * @param dstQueue 目标队列。
    * @param elementNum 取出元素数量。
    * @param qos        全局队列qos等级。
    * @param func       元素入队操作。
    * @retval 返回被推入队列尾部的元素数量。
    */
    using PushFunc = void(*)(void*);
    unsigned int PopHeadToAnotherQueue(SpmcQueue& dstQueue, unsigned int elementNum, PushFunc func);

private:
    void** buf_ = nullptr;
    unsigned int capacity_ = 0;
    std::atomic<unsigned int> head_ {0};
    std::atomic<unsigned int> tail_ {0};
};
}
#endif