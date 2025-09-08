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

#ifndef FFRT_TS_QUEUE_HPP
#define FFRT_TS_QUEUE_HPP
#include <queue>
#include <mutex>
#include <condition_variable>

namespace ffrt {
// 线程安全队列，支持线程阻塞
template <typename T>
class TSQueue {
public:
    TSQueue() = default;
    ~TSQueue() = default;

    void Push(const T& data)
    {
        {
            std::lock_guard<decltype(mutex_)> lg(mutex_);
            queue_.push(data);
        }
        cond_.notify_one();
    }

    T Pop()
    {
        std::unique_lock<decltype(mutex_)> lg(mutex_);
        cond_.wait(lg, [this] { return !queue_.empty(); });
        auto& res = queue_.front();
        queue_.pop();
        return res;
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
};
} // namespace ffrt
#endif
