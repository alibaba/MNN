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
#ifndef FFRT_API_CPP_THREAD_H
#define FFRT_API_CPP_THREAD_H
#include <memory>
#include "cpp/task.h"

namespace ffrt {
class thread {
public:
    thread() noexcept
    {
    }

    template <typename Fn, typename... Args,
        class = std::enable_if_t<!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Fn>>, thread>>>
    explicit thread(const char* name, qos qos_, Fn&& fn, Args&&... args)
    {
        is_joinable = std::make_unique<task_handle>();
        using Target = std::tuple<std::decay_t<Fn>, std::decay_t<Args>...>;
        auto tup = new Target(std::forward<Fn>(fn), std::forward<Args>(args)...);
        *is_joinable = ffrt::submit_h([tup]() {
            execute(*tup, std::make_index_sequence<std::tuple_size_v<Target>>());
            delete tup;
            }, {}, {}, ffrt::task_attr().name(name).qos(qos_));
    }

    template <typename Fn, typename... Args,
        class = std::enable_if_t<!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Fn>>, thread>>>
    explicit thread(qos qos_, Fn&& fn, Args&&... args)
    {
        is_joinable = std::make_unique<task_handle>();
        using Target = std::tuple<std::decay_t<Fn>, std::decay_t<Args>...>;
        auto tup = new Target(std::forward<Fn>(fn), std::forward<Args>(args)...);
        *is_joinable = ffrt::submit_h([tup]() {
            execute(*tup, std::make_index_sequence<std::tuple_size_v<Target>>());
            delete tup;
            }, {}, {}, ffrt::task_attr().qos(qos_));
    }

    template <class Fn, class... Args,
        class = std::enable_if_t<!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Fn>>, thread>>,
        class = std::enable_if_t<!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Fn>>, char*>>,
        class = std::enable_if_t<!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Fn>>, qos>>>
    explicit thread(Fn&& fn, Args&& ... args)
    {
        is_joinable = std::make_unique<task_handle>();
        using Target = std::tuple<std::decay_t<Fn>, std::decay_t<Args>...>;
        auto tup = new Target (std::forward<Fn>(fn), std::forward<Args>(args)...);
        *is_joinable = ffrt::submit_h([tup]() {
            execute(*tup, std::make_index_sequence<std::tuple_size_v<Target>>());
            delete tup;
            });
    }

    thread(const thread&) = delete;
    thread& operator=(const thread&) = delete;

    thread(thread&& th) noexcept
    {
        swap(th);
    }

    thread& operator=(thread&& th) noexcept
    {
        if (this != &th) {
            thread tmp(std::move(th));
            swap(tmp);
        }
        return *this;
    }

    bool joinable() const noexcept
    {
        return is_joinable.get() && *is_joinable;
    }

    void detach() noexcept
    {
        is_joinable = nullptr;
    }

    void join() noexcept
    {
        if (joinable()) {
            ffrt::wait({*is_joinable});
            is_joinable = nullptr;
        }
    }

    ~thread()
    {
        if (joinable()) {
            std::terminate();
        }
    }

private:
    template<class Target, size_t... Idxs>
    static inline void execute(Target& tup,
        std::index_sequence<Idxs...>)
    {
        std::invoke(std::move(std::get<Idxs>(tup))...);
    }

    void swap(thread& other) noexcept
    {
        is_joinable.swap(other.is_joinable);
    };
    std::unique_ptr<task_handle> is_joinable;
};
} // namespace ffrt
#endif
