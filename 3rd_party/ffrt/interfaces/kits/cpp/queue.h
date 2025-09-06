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
 * @file queue.h
 *
 * @brief Declares the queue interfaces in C++.
 *
 * @library libffrt.z.so
 * @kit FunctionFlowRuntimeKit
 * @syscap SystemCapability.Resourceschedule.Ffrt.Core
 * @since 10
 */

#ifndef FFRT_API_CPP_QUEUE_H
#define FFRT_API_CPP_QUEUE_H

#include "c/queue.h"
#include "task.h"

namespace ffrt {
/**
 * @enum queue_type
 * @brief Defines the types of queues supported.
 *
 * @since 12
 */
enum queue_type {
    queue_serial = ffrt_queue_serial,         ///< A serial queue that processes tasks sequentially.
    queue_concurrent = ffrt_queue_concurrent, ///< A concurrent queue that processes tasks in parallel.
    queue_max = ffrt_queue_max,               ///< Defines the maximum type for validation.
};

/**
 * @class queue_attr
 * @brief Represents attributes for configuring a queue.
 *
 * This class provides methods to set and retrieve queue attributes such as QoS,
 * timeout values, callback functions, and maximum concurrency.
 *
 * @since 10
 */
class queue_attr : public ffrt_queue_attr_t {
public:
    /**
     * @brief Constructs a queue_attr object with default values.
     *
     * @since 10
     */
    queue_attr()
    {
        ffrt_queue_attr_init(this);
    }

    /**
     * @brief Destroys the queue_attr object and releases its resources.
     *
     * @since 10
     */
    ~queue_attr()
    {
        ffrt_queue_attr_destroy(this);
    }

    /**
     * @brief Deleted copy constructor to prevent copying of queue_attr object.
     */
    queue_attr(const queue_attr&) = delete;

    /**
     * @brief Deleted copy assignment operator to prevent assignment of queue_attr object.
     */
    queue_attr& operator=(const queue_attr&) = delete;

    /**
     * @brief Sets the QoS for this queue attribute.
     *
     * @param attr Indicates the QoS.
     * @since 10
     */
    inline queue_attr& qos(qos qos_)
    {
        ffrt_queue_attr_set_qos(this, qos_);
        return *this;
    }

    /**
     * @brief Gets the QoS level of this queue attribute.
     *
     * @return Returns the QoS level.
     * @since 10
     */
    inline int qos() const
    {
        return ffrt_queue_attr_get_qos(this);
    }

    /**
     * @brief Sets the timeout value for this queue attribute.
     *
     * The lower limit of timeout value is 1 ms, if the value is less than 1 ms, it will be set to 1 ms.
     *
     * @param timeout_us Indicates the timeout value in microseconds.
     * @return Returns the current queue_attr object for chaining.
     * @since 10
     */
    inline queue_attr& timeout(uint64_t timeout_us)
    {
        ffrt_queue_attr_set_timeout(this, timeout_us);
        return *this;
    }

    /**
     * @brief Gets the timeout value of this queue attribute.
     *
     * @return Returns the timeout value in microseconds.
     * @since 10
     */
    inline uint64_t timeout() const
    {
        return ffrt_queue_attr_get_timeout(this);
    }

    /**
     * @brief Sets the timeout callback function for this queue attribute.
     *
     * @warning Do not call `exit` in `func` - this my cause unexpected behavior.
     *
     * @param func Indicates the callback function.
     * @return Returns the current queue_attr object for chaining.
     * @since 10
     */
    inline queue_attr& callback(const std::function<void()>& func)
    {
        ffrt_queue_attr_set_callback(this, create_function_wrapper(func, ffrt_function_kind_queue));
        return *this;
    }

    /**
     * @brief Gets the timeout callback function of this queue attribute.
     *
     * @return Returns a pointer to the callback function header.
     * @since 10
     */
    inline ffrt_function_header_t* callback() const
    {
        return ffrt_queue_attr_get_callback(this);
    }

    /**
     * @brief Sets the maximum concurrency level for this queue attribute.
     *
     * @param max_concurrency Indicates the maximum concurrency level.
     * @return Returns the current queue_attr object for chaining.
     * @since 12
     */
    inline queue_attr& max_concurrency(const int max_concurrency)
    {
        ffrt_queue_attr_set_max_concurrency(this, max_concurrency);
        return *this;
    }

    /**
     * @brief Gets the maximum concurrency level of this queue attribute.
     *
     * @return Returns the maximum concurrency level.
     * @since 12
     */
    inline int max_concurrency() const
    {
        return ffrt_queue_attr_get_max_concurrency(this);
    }

    /**
     * @brief Sets the mode for this queue attribute.
     *
     * @param legacy_mode Indicates the queue mode.
     * @return Returns the current queue_attr object for chaining.
     * @since 20
     */
    inline queue_attr& thread_mode(bool mode)
    {
        ffrt_queue_attr_set_thread_mode(this, mode);
        return *this;
    }

    /**
     * @brief Gets the mode of this queue attribute.
     *
     * @return Returns the queue mode.
     * @since 20
     */
    inline bool thread_mode() const
    {
        return ffrt_queue_attr_get_thread_mode(this);
    }
};

/**
 * @class queue
 * @brief Represents a task queue for managing and submitting tasks.
 *
 * This class provides methods to submit tasks, cancel tasks, wait for completion,
 * and retrieve the number of pending tasks in the queue.
 *
 * @since 10
 */
class queue {
public:
    /**
     * @brief Constructs a queue object with the specified type, name, and attributes.
     *
     * @param type Indicates the type of queue.
     * @param name Indicates the name of the queue.
     * @param attr Specifies the attributes for the queue.
     * @since 10
     */
    queue(const queue_type type, const char* name, const queue_attr& attr = {})
    {
        queue_handle = ffrt_queue_create(ffrt_queue_type_t(type), name, &attr);
        deleter = ffrt_queue_destroy;
    }

    /**
     * @brief Constructs a serial queue object with the specified name and attributes.
     *
     * @param name Indicates the name of the queue.
     * @param attr Specifies the attributes for the queue.
     * @since 10
     */
    queue(const char* name, const queue_attr& attr = {})
    {
        queue_handle = ffrt_queue_create(ffrt_queue_serial, name, &attr);
        deleter = ffrt_queue_destroy;
    }

    /**
     * @brief Destroys the queue object and releases its resources.
     * @since 10
     */
    ~queue()
    {
        if (deleter) {
            deleter(queue_handle);
        }
    }

    /**
     * @brief Deleted copy constructor to prevent copying of the queue object.
     */
    queue(const queue&) = delete;

    /**
     * @brief Deleted copy assignment operator to prevent assignment of the queue object.
     */
    void operator=(const queue&) = delete;

    /**
     * @brief Submits a task with a specified attribute to this queue.
     *
     * @param func Indicates a task executor function closure.
     * @param attr Indicates a task attribute.
     * @since 10
     */
    inline void submit(const std::function<void()>& func, const task_attr& attr = {})
    {
        ffrt_queue_submit(queue_handle, create_function_wrapper(func, ffrt_function_kind_queue), &attr);
    }

    /**
     * @brief Submits a task with a specified attribute to this queue.
     *
     * @param func Indicates a task executor function closure.
     * @param attr Indicates a task attribute.
     * @since 10
     */
    inline void submit(std::function<void()>&& func, const task_attr& attr = {})
    {
        ffrt_queue_submit(queue_handle, create_function_wrapper(std::move(func), ffrt_function_kind_queue), &attr);
    }

    /**
     * @brief Submits a task with a specified attribute to this queue, and obtains a task handle.
     *
     * @param func Indicates a task executor function closure.
     * @param attr Indicates a task attribute.
     * @return Returns a non-null task handle if the task is submitted;
               returns a null pointer otherwise.
     * @since 10
     */
    inline task_handle submit_h(const std::function<void()>& func, const task_attr& attr = {})
    {
        return ffrt_queue_submit_h(queue_handle, create_function_wrapper(func, ffrt_function_kind_queue), &attr);
    }

    /**
     * @brief Submits a task with a specified attribute to this queue, and obtains a task handle.
     *
     * @param func Indicates a task executor function closure.
     * @param attr Indicates a task attribute.
     * @return Returns a non-null task handle if the task is submitted;
               returns a null pointer otherwise.
     * @since 10
     */
    inline task_handle submit_h(std::function<void()>&& func, const task_attr& attr = {})
    {
        return ffrt_queue_submit_h(
            queue_handle, create_function_wrapper(std::move(func), ffrt_function_kind_queue), &attr);
    }

    /**
     * @brief Submits a task with a specified attribute to this queue.
     *
     * @param func Indicates a task executor function closure.
     * @param attr Indicates a task attribute.
     */
    inline void submit_head(const std::function<void()>& func, const task_attr& attr = {})
    {
        ffrt_queue_submit_head(queue_handle, create_function_wrapper(func, ffrt_function_kind_queue), &attr);
    }

    /**
     * @brief Submits a task with a specified attribute to this queue.
     *
     * @param func Indicates a task executor function closure.
     * @param attr Indicates a task attribute.
     */
    inline void submit_head(std::function<void()>&& func, const task_attr& attr = {})
    {
        ffrt_queue_submit_head(queue_handle, create_function_wrapper(std::move(func), ffrt_function_kind_queue), &attr);
    }

    /**
     * @brief Submits a task with a specified attribute to this queue, and obtains a task handle.
     *
     * @param func Indicates a task executor function closure.
     * @param attr Indicates a task attribute.
     * @return Returns a non-null task handle if the task is submitted;
               returns a null pointer otherwise.
     */
    inline task_handle submit_head_h(const std::function<void()>& func, const task_attr& attr = {})
    {
        return ffrt_queue_submit_head_h(queue_handle, create_function_wrapper(func, ffrt_function_kind_queue), &attr);
    }

    /**
     * @brief Submits a task with a specified attribute to this queue, and obtains a task handle.
     *
     * @param func Indicates a task executor function closure.
     * @param attr Indicates a task attribute.
     * @return Returns a non-null task handle if the task is submitted;
               returns a null pointer otherwise.
     */
    inline task_handle submit_head_h(std::function<void()>&& func, const task_attr& attr = {})
    {
        return ffrt_queue_submit_head_h(
            queue_handle, create_function_wrapper(std::move(func), ffrt_function_kind_queue), &attr);
    }

    /**
     * @brief Cancels a task.
     *
     * @param handle Indicates a task handle.
     * @return Returns 0 if the task is canceled; -1 otherwise.
     * @since 10
     */
    inline int cancel(const task_handle& handle)
    {
        return ffrt_queue_cancel(handle);
    }

    /**
     * @brief Waits until a task is complete.
     *
     * @param handle Indicates a task handle.
     * @since 10
     */
    inline void wait(const task_handle& handle)
    {
        return ffrt_queue_wait(handle);
    }

    /**
     * @brief Get queue task count.
     *
     * @param queue Indicates a queue handle.
     * @return Returns the queue task count.
     */
    inline uint64_t get_task_cnt()
    {
        return ffrt_queue_get_task_cnt(queue_handle);
    }

    /**
     * @brief Get application main thread queue.
     *
     * @return Returns application main thread queue.
     * @since 12
     */
    static inline queue* get_main_queue()
    {
        ffrt_queue_t q = ffrt_get_main_queue();
        // corner case: main queue is not ready.
        if (q == nullptr) {
            return nullptr;
        }
        static queue main_queue(q);
        return &main_queue;
    }

private:
    /**
     * @brief Type alias for the function pointer used to delete or destroy a queue.
     */
    using QueueDeleter = void (*)(ffrt_queue_t);

    /**
     * @brief Constructs a queue object from an existing queue handle and an optional deleter.
     *
     * @param queue_handle The handle to the existing queue.
     * @param deleter The function pointer used to destroy the queue.
     */
    queue(ffrt_queue_t queue_handle, QueueDeleter deleter = nullptr) : queue_handle(queue_handle), deleter(deleter) {}

    ffrt_queue_t queue_handle = nullptr; ///< Handle to the underlying queue.
    QueueDeleter deleter = nullptr;      ///< Function pointer used to delete or destroy the queue.
};
} // namespace ffrt

#endif // FFRT_API_CPP_QUEUE_H
/** @} */