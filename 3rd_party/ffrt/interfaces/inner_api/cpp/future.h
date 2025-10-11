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
#ifndef FFRT_API_CPP_FUTURE_H
#define FFRT_API_CPP_FUTURE_H
#include <memory>
#include <optional>
#include <chrono>
#include "cpp/condition_variable.h"
#include "thread.h"

namespace ffrt {
struct non_copyable {
protected:
    non_copyable() = default;
    ~non_copyable() = default;
    non_copyable(const non_copyable&) = delete;
    non_copyable& operator=(const non_copyable&) = delete;
};
enum class future_status { ready, timeout, deferred };

namespace detail {
template <typename Derived>
struct shared_state_base : private non_copyable {
    void wait() const noexcept
    {
        std::unique_lock lk(this->m_mtx);
        wait_(lk);
    }

    template <typename Rep, typename Period>
    future_status wait_for(const std::chrono::duration<Rep, Period>& waitTime) const noexcept
    {
        std::unique_lock<mutex> lk(m_mtx);
        return m_cv.wait_for(lk, waitTime, [this] { return get_derived().has_value(); }) ? future_status::ready :
            future_status::timeout;
    }

    template <typename Clock, typename Duration>
    future_status wait_until(const std::chrono::time_point<Clock, Duration>& tp) const noexcept
    {
        std::unique_lock<mutex> lk(m_mtx);
        return m_cv.wait_until(lk, tp, [this] { return get_derived().has_value(); }) ? future_status::ready :
            future_status::timeout;
    }

protected:
    void wait_(std::unique_lock<mutex>& lk) const noexcept
    {
        m_cv.wait(lk, [this] { return get_derived().has_value(); });
    }

    mutable mutex m_mtx;
    mutable condition_variable m_cv;

private:
    const Derived& get_derived() const
    {
        return *static_cast<const Derived*>(this);
    }
};

template <typename R>
struct shared_state : public shared_state_base<shared_state<R>> {
    void set_value(const R& value) noexcept
    {
        std::unique_lock<mutex> lk(this->m_mtx);
        m_res.emplace(value);
        this->m_cv.notify_all();
    }

    void set_value(R&& value) noexcept
    {
        std::unique_lock<mutex> lk(this->m_mtx);
        m_res.emplace(std::move(value));
        this->m_cv.notify_all();
    }

    R& get() noexcept
    {
        std::unique_lock lk(this->m_mtx);
        this->wait_(lk);
        return m_res.value();
    }

    bool has_value() const noexcept
    {
        return m_res.has_value();
    }

private:
    std::optional<R> m_res;
};

template <>
struct shared_state<void> : public shared_state_base<shared_state<void>> {
    void set_value() noexcept
    {
        std::unique_lock<mutex> lk(this->m_mtx);
        m_hasValue = true;
        this->m_cv.notify_all();
    }

    void get() noexcept
    {
        std::unique_lock lk(this->m_mtx);
        this->wait_(lk);
    }

    bool has_value() const noexcept
    {
        return m_hasValue;
    }

private:
    bool m_hasValue {false};
};
}; // namespace detail

template <typename R>
class future : private non_copyable {
    template <typename>
    friend struct promise;

    template <typename>
    friend struct packaged_task;

public:
    explicit future(const std::shared_ptr<detail::shared_state<R>>& state) noexcept : m_state(state)
    {
    }

    future() noexcept = default;

    future(future&& fut) noexcept
    {
        swap(fut);
    }
    future& operator=(future&& fut) noexcept
    {
        if (this != &fut) {
            future tmp(std::move(fut));
            swap(tmp);
        }
        return *this;
    }

    bool valid() const noexcept
    {
        return m_state != nullptr;
    }

    R get() noexcept
    {
        auto tmp = std::move(m_state);
        if constexpr(!std::is_void_v<R>) {
            return std::move(tmp->get());
        } else {
            return tmp->get();
        }
    }

    template <typename Rep, typename Period>
    future_status wait_for(const std::chrono::duration<Rep, Period>& waitTime) const noexcept
    {
        return m_state->wait_for(waitTime);
    }

    template <typename Clock, typename Duration>
    future_status wait_until(const std::chrono::time_point<Clock, Duration>& tp) const noexcept
    {
        return m_state->wait_until(tp);
    }

    void wait() const noexcept
    {
        m_state->wait();
    }

    void swap(future<R>& rhs) noexcept
    {
        std::swap(m_state, rhs.m_state);
    }

private:
    std::shared_ptr<detail::shared_state<R>> m_state;
};

template <typename R>
struct promise : private non_copyable {
    promise() noexcept : m_state {std::make_shared<detail::shared_state<R>>()}
    {
    }
    promise(promise&& p) noexcept
    {
        swap(p);
    }
    promise& operator=(promise&& p) noexcept
    {
        if (this != &p) {
            promise tmp(std::move(p));
            swap(tmp);
        }
        return *this;
    }

    void set_value(const R& value) noexcept
    {
        m_state->set_value(value);
    }

    void set_value(R&& value) noexcept
    {
        m_state->set_value(std::move(value));
    }

    future<R> get_future() noexcept
    {
        return future<R> {m_state};
    }

    void swap(promise<R>& rhs) noexcept
    {
        std::swap(m_state, rhs.m_state);
    }

private:
    std::shared_ptr<detail::shared_state<R>> m_state;
};

template <>
struct promise<void> : private non_copyable {
    promise() noexcept : m_state {std::make_shared<detail::shared_state<void>>()}
    {
    }
    promise(promise&& p) noexcept
    {
        swap(p);
    }
    promise& operator=(promise&& p) noexcept
    {
        if (this != &p) {
            promise tmp(std::move(p));
            swap(tmp);
        }
        return *this;
    }

    void set_value() noexcept
    {
        m_state->set_value();
    }

    future<void> get_future() noexcept
    {
        return future<void> {m_state};
    }

    void swap(promise<void>& rhs) noexcept
    {
        std::swap(m_state, rhs.m_state);
    }

private:
    std::shared_ptr<detail::shared_state<void>> m_state;
};

template <typename F>
struct packaged_task;

template <typename R, typename... Args>
struct packaged_task<R(Args...)> {
    packaged_task() noexcept = default;

    packaged_task(const packaged_task& pt) noexcept
    {
        m_fn = pt.m_fn;
        m_state = pt.m_state;
    }

    packaged_task(packaged_task&& pt) noexcept
    {
        swap(pt);
    }

    packaged_task& operator=(packaged_task&& pt) noexcept
    {
        if (this != &pt) {
            packaged_task tmp(std::move(pt));
            swap(tmp);
        }
        return *this;
    }

    template <typename F>
    explicit packaged_task(F&& f) noexcept
        : m_fn {std::forward<F>(f)}, m_state {std::make_shared<detail::shared_state<R>>()}
    {
    }

    bool valid() const noexcept
    {
        return bool(m_fn) && m_state != nullptr;
    }

    future<R> get_future() noexcept
    {
        return future<R> {m_state};
    }

    void operator()(Args... args)
    {
        if constexpr(!std::is_void_v<R>) {
            m_state->set_value(m_fn(std::forward<Args>(args)...));
        } else {
            m_fn(std::forward<Args>(args)...);
            m_state->set_value();
        }
    }

    void swap(packaged_task& pt) noexcept
    {
        std::swap(m_fn, pt.m_fn);
        std::swap(m_state, pt.m_state);
    }

private:
    std::function<R(Args...)> m_fn;
    std::shared_ptr<detail::shared_state<R>> m_state;
};

template <typename F, typename... Args>
future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>> async(F&& f, Args&& ... args)
{
    using R = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
    packaged_task<R(std::decay_t<Args>...)> pt {std::forward<F>(f)};
    auto fut {pt.get_future()};
    auto th = ffrt::thread(std::move(pt), std::forward<Args>(args)...);
    th.detach();
    return fut;
}
} // namespace ffrt
#endif
