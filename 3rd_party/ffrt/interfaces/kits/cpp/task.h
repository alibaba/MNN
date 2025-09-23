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

/**
 * @addtogroup FFRT
 * @{
 *
 * @brief Provides FFRT C++ APIs.
 *
 * @since 10
 */

/**
 * @file task.h
 *
 * @brief Declares the task interfaces in C++.
 *
 * @library libffrt.z.so
 * @kit FunctionFlowRuntimeKit
 * @syscap SystemCapability.Resourceschedule.Ffrt.Core
 * @since 10
 */

#ifndef FFRT_API_CPP_TASK_H
#define FFRT_API_CPP_TASK_H

#include <string>
#include <vector>
#include <functional>
#include "c/task.h"

namespace ffrt {
/**
 * @class task_attr
 * @brief Represents the attributes of a task, such as its name, QoS level, delay, priority, and timeout.
 *
 * @since 10
 */
class task_attr : public ffrt_task_attr_t {
public:
#if __has_builtin(__builtin_FUNCTION)
    /**
     * @brief Constructs a task_attr object with an optional function name as its name.
     *
     * If supported, the function name is automatically set as the task name.
     *
     * @since 10
     */
    task_attr(const char* func = __builtin_FUNCTION())
    {
        ffrt_task_attr_init(this);
        ffrt_task_attr_set_name(this, func);
    }
#else
    /**
     * @brief Constructs a task_attr object.
     *
     * @since 10
     */
    task_attr()
    {
        ffrt_task_attr_init(this);
    }
#endif

    /**
     * @brief Destroys the task_attr object, releasing its resources.
     *
     * @since 10
     */
    ~task_attr()
    {
        ffrt_task_attr_destroy(this);
    }

    /**
     * @brief Deleted copy constructor to prevent copying of the task_attr object.
     */
    task_attr(const task_attr&) = delete;

    /**
     * @brief Deleted copy assignment operator to prevent assignment of the task_attr object.
     */
    task_attr& operator=(const task_attr&) = delete;

    /**
     * @brief Sets a task name.
     *
     * @param name Indicates a pointer to the task name.
     * @since 10
     */
    inline task_attr& name(const char* name)
    {
        ffrt_task_attr_set_name(this, name);
        return *this;
    }

    /**
     * @brief Obtains the task name.
     *
     * @return Returns a pointer to the task name.
     * @since 10
     */
    inline const char* name() const
    {
        return ffrt_task_attr_get_name(this);
    }

    /**
     * @brief Sets the QoS for this task.
     *
     * @param qos Indicates the QoS.
     * @since 10
     */
    inline task_attr& qos(qos qos_)
    {
        ffrt_task_attr_set_qos(this, qos_);
        return *this;
    }

    /**
     * @brief Obtains the QoS of this task.
     *
     * @return Returns the QoS.
     * @since 10
     */
    inline int qos() const
    {
        return ffrt_task_attr_get_qos(this);
    }

    /**
     * @brief Sets the delay time for this task.
     *
     * @param delay_us Indicates the delay time, in microseconds.
     * @since 10
     */
    inline task_attr& delay(uint64_t delay_us)
    {
        ffrt_task_attr_set_delay(this, delay_us);
        return *this;
    }

    /**
     * @brief Obtains the delay time of this task.
     *
     * @return Returns the delay time.
     * @since 10
     */
    inline uint64_t delay() const
    {
        return ffrt_task_attr_get_delay(this);
    }

    /**
     * @brief Sets the priority for this task.
     *
     * @param priority Indicates the execute priority of concurrent queue task.
     * @since 12
     */
    inline task_attr& priority(ffrt_queue_priority_t prio)
    {
        ffrt_task_attr_set_queue_priority(this, prio);
        return *this;
    }

    /**
     * @brief Obtains the priority of this task.
     *
     * @return Returns the priority of concurrent queue task.
     * @since 12
     */
    inline ffrt_queue_priority_t priority() const
    {
        return ffrt_task_attr_get_queue_priority(this);
    }

    /**
     * @brief Sets the stack size for this task.
     *
     * @param size Indicates the task stack size, unit is byte.
     * @since 12
     */
    inline task_attr& stack_size(uint64_t size)
    {
        ffrt_task_attr_set_stack_size(this, size);
        return *this;
    }

    /**
     * @brief Obtains the stack size of this task.
     *
     * @return Returns task stack size, unit is byte.
     * @since 12
     */
    inline uint64_t stack_size() const
    {
        return ffrt_task_attr_get_stack_size(this);
    }

    /**
     * @brief Sets the task schedule timeout.
     *
     * The lower limit of timeout value is 1 ms, if the value is less than 1 ms, it will be set to 1 ms.
     *
     * @param timeout_us task scheduler timeout.
     */
    inline task_attr& timeout(uint64_t timeout_us)
    {
        ffrt_task_attr_set_timeout(this, timeout_us);
        return *this;
    }

    /**
     * @brief Obtains the task schedule timeout.
     *
     * @return Returns task scheduler timeout.
     */
    inline uint64_t timeout() const
    {
        return ffrt_task_attr_get_timeout(this);
    }
};

/**
 * @class task_handle
 * @brief Represents a handle for a submitted task, allowing operations such as
 *        querying task IDs and managing task resources.
 *
 * @since 10
 */
class task_handle {
public:
    /**
     * @brief Default constructor for task_handle.
     *
     * @since 10
     */
    task_handle() : p(nullptr)
    {
    }

    /**
     * @brief Constructs a task_handle object from a raw task handle pointer.
     *
     * @param p The raw task handle pointer.
     * @since 10
     */
    task_handle(ffrt_task_handle_t p) : p(p)
    {
    }

    /**
     * @brief Destroys the task_handle object, releasing any associated resources.
     *
     * @since 10
     */
    ~task_handle()
    {
        if (p) {
            ffrt_task_handle_destroy(p);
        }
    }

    /**
     * @brief Deleted copy constructor to prevent copying of the task_handle object.
     */
    task_handle(task_handle const&) = delete;

    /**
     * @brief Deleted copy assignment operator to prevent assignment of the task_handle object.
     */
    task_handle& operator=(task_handle const&) = delete;

    /**
     * @brief Move constructor for task_handle.
     *
     * @param h The task_handle object to move from.
     * @since 10
     */
    inline task_handle(task_handle&& h)
    {
        *this = std::move(h);
    }

    /**
     * @brief get gid from task handle.
     *
     * @return Return gid.
     */
    inline uint64_t get_id() const
    {
        return ffrt_task_handle_get_id(p);
    }

    /**
     * @brief Move assignment operator for task_handle.
     *
     * @param h The task_handle object to move from.
     * @return Returns the current task_handle object.
     * @since 10
     */
    inline task_handle& operator=(task_handle&& h)
    {
        if (this != &h) {
            if (p) {
                ffrt_task_handle_destroy(p);
            }
            p = h.p;
            h.p = nullptr;
        }
        return *this;
    }

    /**
     * @brief Implicit conversion to a void pointer.
     *
     * @return Returns the raw task handle pointer.
     * @since 10
     */
    inline operator void* () const
    {
        return p;
    }

private:
    ffrt_task_handle_t p = nullptr; ///< Handle to the underlying task.
};

/**
 * @struct dependence
 * @brief Represents a dependency for a task, which can either be a data dependency or a task dependency.
 */
struct dependence : ffrt_dependence_t {
    /**
     * @brief Constructs a data dependency.
     *
     * @param d A pointer to the data dependency.
     * @since 10
     */
    dependence(const void* d)
    {
        type = ffrt_dependence_data;
        ptr = d;
    }

    /**
     * @brief Constructs a task dependency.
     *
     * @param h A reference to a task_handle representing the dependency.
     * @since 10
     */
    dependence(const task_handle& h)
    {
        type = ffrt_dependence_task;
        ptr = h;
        ffrt_task_handle_inc_ref(const_cast<ffrt_task_handle_t>(ptr));
    }

    /**
     * @brief Copy constructor for dependence.
     *
     * @param other The dependence object to copy from.
     * @since 10
     */
    dependence(const dependence& other)
    {
        (*this) = other;
    }

    /**
     * @brief Move constructor for dependence.
     *
     * @param other The dependence object to move from.
     * @since 10
     */
    dependence(dependence&& other)
    {
        (*this) = std::move(other);
    }

    /**
     * @brief Copy assignment operator for dependence.
     *
     * @param other The dependence object to copy from.
     * @return Returns the current dependence object.
     * @since 10
     */
    dependence& operator=(const dependence& other)
    {
        if (this != &other) {
            type = other.type;
            ptr = other.ptr;
            if (type == ffrt_dependence_task) {
                ffrt_task_handle_inc_ref(const_cast<ffrt_task_handle_t>(ptr));
            }
        }
        return *this;
    }

    /**
     * @brief Move assignment operator for dependence.
     *
     * @param other The dependence object to move from.
     * @return Returns the current dependence object.
     * @since 10
     */
    dependence& operator=(dependence&& other)
    {
        if (this != &other) {
            type = other.type;
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    /**
     * @brief Destructor for dependence.
     *
     * @since 10
     */
    ~dependence()
    {
        if (type == ffrt_dependence_task && ptr) {
            ffrt_task_handle_dec_ref(const_cast<ffrt_task_handle_t>(ptr));
        }
    }
};

/**
 * @struct function
 * @brief Represents a function wrapper for task execution.
 *
 * This template struct is used to wrap a function closure for task execution.
 *
 * @tparam T The type of the function closure.
 * @since 10
 */
template<class T>
struct function {
    ffrt_function_header_t header;
    T closure;
};

/**
 * @brief Executes a function wrapper.
 *
 * This function is used to execute a function wrapper.
 *
 * @tparam T The type of the function closure.
 * @param t A pointer to the function wrapper.
 * @since 10
 */
template<class T>
void exec_function_wrapper(void* t)
{
    auto f = reinterpret_cast<function<std::decay_t<T>>*>(t);
    f->closure();
}

/**
 * @brief Destroys a function wrapper.
 *
 * This function is used to destroy a function wrapper.
 *
 * @tparam T The type of the function closure.
 * @param t A pointer to the function wrapper.
 * @since 10
 */
template<class T>
void destroy_function_wrapper(void* t)
{
    auto f = reinterpret_cast<function<std::decay_t<T>>*>(t);
    f->closure = nullptr;
}

/**
 * @brief Creates a function wrapper.
 *
 * This function is used to create a function wrapper for task submission.
 *
 * @tparam T The type of the function closure.
 * @param func The function closure.
 * @param kind The function kind (optional).
 * @return Returns a pointer to the function wrapper header.
 * @since 10
 */
template<class T>
inline ffrt_function_header_t* create_function_wrapper(T&& func,
    ffrt_function_kind_t kind = ffrt_function_kind_general)
{
    using function_type = function<std::decay_t<T>>;
    static_assert(sizeof(function_type) <= ffrt_auto_managed_function_storage_size,
        "size of function must be less than ffrt_auto_managed_function_storage_size");

    auto p = ffrt_alloc_auto_managed_function_storage_base(kind);
    auto f = new (p)function_type;
    f->header.exec = exec_function_wrapper<T>;
    f->header.destroy = destroy_function_wrapper<T>;
    f->header.reserve[0] = 0;
    f->closure = std::forward<T>(func);
    return reinterpret_cast<ffrt_function_header_t*>(f);
}

/**
 * @brief Submits a task without input and output dependencies.
 *
 * @param func Indicates a task executor function closure.
 * @param attr Indicates a task attribute.
 * @since 10
 */
static inline void submit(std::function<void()>&& func, const task_attr& attr = {})
{
    return ffrt_submit_base(create_function_wrapper(std::move(func)), nullptr, nullptr, &attr);
}

/**
 * @brief Submits a task with input dependencies only.
 *
 * @param func Indicates a task executor function closure.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param attr Indicates a task attribute.
 * @since 10
 */
static inline void submit(std::function<void()>&& func, std::initializer_list<dependence> in_deps,
    const task_attr& attr = {})
{
    ffrt_deps_t in{static_cast<uint32_t>(in_deps.size()), in_deps.begin()};
    return ffrt_submit_base(create_function_wrapper(std::move(func)), &in, nullptr, &attr);
}

/**
 * @brief Submits a task with input and output dependencies.
 *
 * @param func Indicates a task executor function closure.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param out_deps Indicates a pointer to the output dependencies.
 * @param attr Indicates a task attribute.
 * @since 10
 */
static inline void submit(std::function<void()>&& func, std::initializer_list<dependence> in_deps,
    std::initializer_list<dependence> out_deps, const task_attr& attr = {})
{
    ffrt_deps_t in{static_cast<uint32_t>(in_deps.size()), in_deps.begin()};
    ffrt_deps_t out{static_cast<uint32_t>(out_deps.size()), out_deps.begin()};
    return ffrt_submit_base(create_function_wrapper(std::move(func)), &in, &out, &attr);
}

/**
 * @brief Submits a task with input dependencies only.
 *
 * @param func Indicates a task executor function closure.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param attr Indicates a task attribute.
 * @since 10
 */
static inline void submit(std::function<void()>&& func, const std::vector<dependence>& in_deps,
    const task_attr& attr = {})
{
    ffrt_deps_t in{static_cast<uint32_t>(in_deps.size()), in_deps.data()};
    return ffrt_submit_base(create_function_wrapper(std::move(func)), &in, nullptr, &attr);
}

/**
 * @brief Submits a task with input and output dependencies.
 *
 * @param func Indicates a task executor function closure.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param out_deps Indicates a pointer to the output dependencies.
 * @param attr Indicates a task attribute.
 * @since 10
 */
static inline void submit(std::function<void()>&& func, const std::vector<dependence>& in_deps,
    const std::vector<dependence>& out_deps, const task_attr& attr = {})
{
    ffrt_deps_t in{static_cast<uint32_t>(in_deps.size()), in_deps.data()};
    ffrt_deps_t out{static_cast<uint32_t>(out_deps.size()), out_deps.data()};
    return ffrt_submit_base(create_function_wrapper(std::move(func)), &in, &out, &attr);
}

/**
 * @brief Submits a task without input and output dependencies.
 *
 * @param func Indicates a task executor function closure.
 * @param attr Indicates a task attribute.
 * @since 10
 */
static inline void submit(const std::function<void()>& func, const task_attr& attr = {})
{
    return ffrt_submit_base(create_function_wrapper(func), nullptr, nullptr, &attr);
}

/**
 * @brief Submits a task with input dependencies only.
 *
 * @param func Indicates a task executor function closure.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param attr Indicates a task attribute.
 * @since 10
 */
static inline void submit(const std::function<void()>& func, std::initializer_list<dependence> in_deps,
    const task_attr& attr = {})
{
    ffrt_deps_t in{static_cast<uint32_t>(in_deps.size()), in_deps.begin()};
    return ffrt_submit_base(create_function_wrapper(func), &in, nullptr, &attr);
}

/**
 * @brief Submits a task with input and output dependencies.
 *
 * @param func Indicates a task executor function closure.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param out_deps Indicates a pointer to the output dependencies.
 * @param attr Indicates a task attribute.
 * @since 10
 */
static inline void submit(const std::function<void()>& func, std::initializer_list<dependence> in_deps,
    std::initializer_list<dependence> out_deps, const task_attr& attr = {})
{
    ffrt_deps_t in{static_cast<uint32_t>(in_deps.size()), in_deps.begin()};
    ffrt_deps_t out{static_cast<uint32_t>(out_deps.size()), out_deps.begin()};
    return ffrt_submit_base(create_function_wrapper(func), &in, &out, &attr);
}

/**
 * @brief Submits a task with input dependencies only.
 *
 * @param func Indicates a task executor function closure.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param attr Indicates a task attribute.
 * @since 10
 */
static inline void submit(const std::function<void()>& func, const std::vector<dependence>& in_deps,
    const task_attr& attr = {})
{
    ffrt_deps_t in{static_cast<uint32_t>(in_deps.size()), in_deps.data()};
    return ffrt_submit_base(create_function_wrapper(func), &in, nullptr, &attr);
}

/**
 * @brief Submits a task with input and output dependencies.
 *
 * @param func Indicates a task executor function closure.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param out_deps Indicates a pointer to the output dependencies.
 * @param attr Indicates a task attribute.
 * @since 10
 */
static inline void submit(const std::function<void()>& func, const std::vector<dependence>& in_deps,
    const std::vector<dependence>& out_deps, const task_attr& attr = {})
{
    ffrt_deps_t in{static_cast<uint32_t>(in_deps.size()), in_deps.data()};
    ffrt_deps_t out{static_cast<uint32_t>(out_deps.size()), out_deps.data()};
    return ffrt_submit_base(create_function_wrapper(func), &in, &out, &attr);
}

/**
 * @brief Submits a task without input and output dependencies, and obtains a task handle.
 *
 * @param func Indicates a task executor function closure.
 * @param attr Indicates a task attribute.
 * @return Returns a non-null task handle if the task is submitted;
           returns a null pointer otherwise.
 * @since 10
 */
static inline task_handle submit_h(std::function<void()>&& func, const task_attr& attr = {})
{
    return ffrt_submit_h_base(create_function_wrapper(std::move(func)), nullptr, nullptr, &attr);
}

/**
 * @brief Submits a task with input dependencies only, and obtains a task handle.
 *
 * @param func Indicates a task executor function closure.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param attr Indicates a task attribute.
 * @return Returns a non-null task handle if the task is submitted;
           returns a null pointer otherwise.
 * @since 10
 */
static inline task_handle submit_h(std::function<void()>&& func, std::initializer_list<dependence> in_deps,
    const task_attr& attr = {})
{
    ffrt_deps_t in{static_cast<uint32_t>(in_deps.size()), in_deps.begin()};
    return ffrt_submit_h_base(create_function_wrapper(std::move(func)), &in, nullptr, &attr);
}

/**
 * @brief Submits a task with input and output dependencies, and obtains a task handle.
 *
 * @param func Indicates a task executor function closure.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param out_deps Indicates a pointer to the output dependencies.
 * @param attr Indicates a task attribute.
 * @return Returns a non-null task handle if the task is submitted;
           returns a null pointer otherwise.
 * @since 10
 */
static inline task_handle submit_h(std::function<void()>&& func, std::initializer_list<dependence> in_deps,
    std::initializer_list<dependence> out_deps, const task_attr& attr = {})
{
    ffrt_deps_t in{static_cast<uint32_t>(in_deps.size()), in_deps.begin()};
    ffrt_deps_t out{static_cast<uint32_t>(out_deps.size()), out_deps.begin()};
    return ffrt_submit_h_base(create_function_wrapper(std::move(func)), &in, &out, &attr);
}

/**
 * @brief Submits a task with input dependencies only, and obtains a task handle.
 *
 * @param func Indicates a task executor function closure.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param attr Indicates a task attribute.
 * @return Returns a non-null task handle if the task is submitted;
           returns a null pointer otherwise.
 * @since 10
 */
static inline task_handle submit_h(std::function<void()>&& func, const std::vector<dependence>& in_deps,
    const task_attr& attr = {})
{
    ffrt_deps_t in{static_cast<uint32_t>(in_deps.size()), in_deps.data()};
    return ffrt_submit_h_base(create_function_wrapper(std::move(func)), &in, nullptr, &attr);
}

/**
 * @brief Submits a task with input and output dependencies, and obtains a task handle.
 *
 * @param func Indicates a task executor function closure.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param out_deps Indicates a pointer to the output dependencies.
 * @param attr Indicates a task attribute.
 * @return Returns a non-null task handle if the task is submitted;
           returns a null pointer otherwise.
 * @since 10
 */
static inline task_handle submit_h(std::function<void()>&& func, const std::vector<dependence>& in_deps,
    const std::vector<dependence>& out_deps, const task_attr& attr = {})
{
    ffrt_deps_t in{static_cast<uint32_t>(in_deps.size()), in_deps.data()};
    ffrt_deps_t out{static_cast<uint32_t>(out_deps.size()), out_deps.data()};
    return ffrt_submit_h_base(create_function_wrapper(std::move(func)), &in, &out, &attr);
}

/**
 * @brief Submits a task without input and output dependencies, and obtains a task handle.
 *
 * @param func Indicates a task executor function closure.
 * @param attr Indicates a task attribute.
 * @return Returns a non-null task handle if the task is submitted;
           returns a null pointer otherwise.
 * @since 10
 */
static inline task_handle submit_h(const std::function<void()>& func, const task_attr& attr = {})
{
    return ffrt_submit_h_base(create_function_wrapper(func), nullptr, nullptr, &attr);
}

/**
 * @brief Submits a task with input dependencies only, and obtains a task handle.
 *
 * @param func Indicates a task executor function closure.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param attr Indicates a task attribute.
 * @return Returns a non-null task handle if the task is submitted;
           returns a null pointer otherwise.
 * @since 10
 */
static inline task_handle submit_h(const std::function<void()>& func, std::initializer_list<dependence> in_deps,
    const task_attr& attr = {})
{
    ffrt_deps_t in{static_cast<uint32_t>(in_deps.size()), in_deps.begin()};
    return ffrt_submit_h_base(create_function_wrapper(func), &in, nullptr, &attr);
}

/**
 * @brief Submits a task with input and output dependencies, and obtains a task handle.
 *
 * @param func Indicates a task executor function closure.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param out_deps Indicates a pointer to the output dependencies.
 * @param attr Indicates a task attribute.
 * @return Returns a non-null task handle if the task is submitted;
           returns a null pointer otherwise.
 * @since 10
 */
static inline task_handle submit_h(const std::function<void()>& func, std::initializer_list<dependence> in_deps,
    std::initializer_list<dependence> out_deps, const task_attr& attr = {})
{
    ffrt_deps_t in{static_cast<uint32_t>(in_deps.size()), in_deps.begin()};
    ffrt_deps_t out{static_cast<uint32_t>(out_deps.size()), out_deps.begin()};
    return ffrt_submit_h_base(create_function_wrapper(func), &in, &out, &attr);
}

/**
 * @brief Submits a task with input dependencies only, and obtains a task handle.
 *
 * @param func Indicates a task executor function closure.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param attr Indicates a task attribute.
 * @return Returns a non-null task handle if the task is submitted;
           returns a null pointer otherwise.
 * @since 10
 */
static inline task_handle submit_h(const std::function<void()>& func, const std::vector<dependence>& in_deps,
    const task_attr& attr = {})
{
    ffrt_deps_t in{static_cast<uint32_t>(in_deps.size()), in_deps.data()};
    return ffrt_submit_h_base(create_function_wrapper(func), &in, nullptr, &attr);
}

/**
 * @brief Submits a task with input and output dependencies, and obtains a task handle.
 *
 * @param func Indicates a task executor function closure.
 * @param in_deps Indicates a pointer to the input dependencies.
 * @param out_deps Indicates a pointer to the output dependencies.
 * @param attr Indicates a task attribute.
 * @return Returns a non-null task handle if the task is submitted;
           returns a null pointer otherwise.
 * @since 10
 */
static inline task_handle submit_h(const std::function<void()>& func, const std::vector<dependence>& in_deps,
    const std::vector<dependence>& out_deps, const task_attr& attr = {})
{
    ffrt_deps_t in{static_cast<uint32_t>(in_deps.size()), in_deps.data()};
    ffrt_deps_t out{static_cast<uint32_t>(out_deps.size()), out_deps.data()};
    return ffrt_submit_h_base(create_function_wrapper(func), &in, &out, &attr);
}

/**
 * @brief Waits until all submitted tasks are complete.
 *
 * @since 10
 */
static inline void wait()
{
    ffrt_wait();
}

/**
 * @brief Waits until dependent tasks are complete.
 *
 * @param deps Indicates a pointer to the dependent tasks.
 * @since 10
 */
static inline void wait(std::initializer_list<dependence> deps)
{
    ffrt_deps_t d{static_cast<uint32_t>(deps.size()), deps.begin()};
    ffrt_wait_deps(&d);
}

/**
 * @brief Waits until dependent tasks are complete.
 *
 * @param deps Indicates a pointer to the dependent tasks.
 * @since 10
 */
static inline void wait(const std::vector<dependence>& deps)
{
    ffrt_deps_t d{static_cast<uint32_t>(deps.size()), deps.data()};
    ffrt_wait_deps(&d);
}

/**
 * @brief Sets the thread stack size of a specified QoS level.
 *
 * @param qos_ Indicates the QoS.
 * @param stack_size Indicates the thread stack size.
 * @return Returns ffrt_success if the stack size set success;
           returns ffrt_error_inval if qos_ or stack_size invalid;
           returns ffrt_error otherwise.
 */
static inline ffrt_error_t set_worker_stack_size(qos qos_, size_t stack_size)
{
    return ffrt_set_worker_stack_size(qos_, stack_size);
}

/**
 * @namespace ffrt::this_task
 * @brief Contains utility functions for the currently executing task.
 */
namespace this_task {
/**
 * @brief Updates the QoS level of the currently executing task.
 *
 * @param qos_ The new QoS level.
 * @return Returns the updated QoS level.
 * @since 10
 */
static inline int update_qos(qos qos_)
{
    return ffrt_this_task_update_qos(qos_);
}

/**
 * @brief Obtains the ID of this task.
 *
 * @return Returns the task ID.
 * @since 10
 */
static inline uint64_t get_id()
{
    return ffrt_this_task_get_id();
}
} // namespace this_task
} // namespace ffrt

#endif // FFRT_API_CPP_TASK_H
/** @} */