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

#ifndef _TS_LIST_
#define _TS_LIST_
#include <list>
#include <shared_mutex>

namespace ffrt {
template <typename T>
class TSList {
public:
    TSList() = default;
    ~TSList() = default;

    T* emplace_back(T&& val)
    {
        std::lock_guard<std::shared_mutex> lck(mtx_);
        list_.emplace_back(std::move(val));
        return &list_.back();
    }

    void push_back(T val)
    {
        std::lock_guard<std::shared_mutex> lck(mtx_);
        list_.push_back(val);
    }

    std::list<T>& get_all()
    {
        std::shared_lock<std::shared_mutex> lck(mtx_);
        return list_;
    }

    std::list<T> claim()
    {
        std::lock_guard<std::shared_mutex> lck(mtx_);
        std::list<T> copy = list_;
        list_.clear();
        return copy;
    }

private:
    std::list<T> list_;
    std::shared_mutex mtx_;
};
} // namespace ffrt

#endif
