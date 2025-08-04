/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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
 * @since 20
 */

/**
 * @file job_partner.h
 *
 * @brief Declares the job partner interfaces in C++.
 *
 * @library libffrt.z.so
 * @kit FunctionFlowRuntimeKit
 * @syscap SystemCapability.Resourceschedule.Ffrt.Core
 * @since 20
 */

#ifndef FFRT_JOB_PARTNER_H
#define FFRT_JOB_PARTNER_H

#include <functional>
#include <string>
#include <atomic>
#include "job_utils.h"
#include "cpp/task.h"

namespace ffrt {

/**
 * @struct job_partner_attr
 * @brief Defines the job partner attribute structure for controlling worker concurrency.
 *
 * This structure provides initialization and configuration for job partner attributes,
 * including QoS, maximum worker number, ratio, threshold, busy wait time, and queue depth.
 *
 * The relationship between job number and partner number is illustrated as follows:
 * @verbatim
 * partner_num
 *     ^
 *     |
 * max |            ------------------
 *     |           /
 *     |    ratio /
 *     |         /
 *     |        /
 *     |       /
 *     |      /
 *     +------------------------------> job_num
 *         threshold
 * @endverbatim
 *
 * - The vertical axis is partner_num, and the horizontal axis is job_num.
 * - Threshold: When job_num is less than threshold, partner_num is 0.
 * - Ratio control: When job_num is between threshold and "max * ratio + threshold",
 *   partner_num is calculated as "round((job_num - threshold) / ratio)".
 * - Maximum value: When job_num is greater than "max * ratio + threshold", partner_num is the maximum value.
 *
 * @since 20
 */
struct job_partner_attr {
    /**
     * @brief Set QoS level.
     *
     * @param q QoS value.
     * @return Reference to this attribute object.
     */
    inline job_partner_attr& qos(qos q)
    {
        this->qos_ = q;
        return *this;
    }

    /**
     * @brief Set max number of partner workers.
     *
     * @param v Maximum number of workers.
     * @return Reference to this attribute object.
     */
    inline job_partner_attr& max_num(uint64_t v)
    {
        this->max_num_ = v;
        return *this;
    }

    /**
     * @brief Set the ratio parameter for controlling the number of workers.
     *
     * @param v Ratio value.
     * @return Reference to this attribute object.
     */
    inline job_partner_attr& ratio(uint64_t v)
    {
        this->ratio_ = v;
        return *this;
    }

    /**
     * @brief Set the threshold parameter for controlling the number of workers.
     *
     * @param v Threshold value.
     * @return Reference to this attribute object.
     */
    inline job_partner_attr& threshold(uint64_t v)
    {
        this->threshold_ = v;
        return *this;
    }

    /**
     * @brief Set last worker's retry busy time (in microseconds).
     *
     * @param us Busy wait time in microseconds.
     * @return Reference to this attribute object.
     */
    inline job_partner_attr& busy(uint64_t us)
    {
        this->busy_us_ = us;
        return *this;
    }

    /**
     * @brief Set the depth of job queue.
     *
     * @param depth Queue depth.
     * @return Reference to this attribute object.
     */
    inline job_partner_attr& queue_depth(uint64_t depth)
    {
        this->queue_depth_ = depth;
        return *this;
    }

    /**
     * @brief Get QoS level.
     *
     * @return QoS value.
     */
    inline int qos() const
    {
        return this->qos_;
    }

    /**
     * @brief Get max number of partner workers.
     *
     * @return Maximum number of workers.
     */
    inline uint64_t max_num() const
    {
        return this->max_num_;
    }

    /**
     * @brief Get the ratio parameter for controlling the number of workers.
     *
     * @return Ratio value.
     */
    inline uint64_t ratio() const
    {
        return this->ratio_;
    }

    /**
     * @brief Get the threshold parameter for controlling the number of workers.
     *
     * @return Threshold value.
     */
    inline uint64_t threshold() const
    {
        return this->threshold_;
    }

    /**
     * @brief Get last worker's retry busy time (in microseconds).
     *
     * @return Busy wait time in microseconds.
     */
    inline uint64_t busy() const
    {
        return this->busy_us_;
    }

    /**
     * @brief Get the depth of job queue.
     *
     * @return Queue depth.
     */
    inline uint64_t queue_depth() const
    {
        return this->queue_depth_;
    }

private:
    int qos_ = ffrt::qos_user_initiated;             ///< QoS level for the job partner.
    uint64_t max_num_ = default_partner_max;         ///< Maximum number of partner workers.
    uint64_t ratio_ = default_partner_ratio;         ///< Ratio for scaling the number of workers.
    uint64_t threshold_ = default_partner_threshold; ///< Threshold for scaling the number of workers.
    uint64_t busy_us_ = default_partner_delay_us;    ///< Busy wait time (us) for the last worker before exit.
    uint64_t queue_depth_ = default_q_depth;         ///< Depth of the job queue.

    static constexpr uint64_t default_partner_max = 2;        ///< Default max number of partner workers.
    static constexpr uint64_t default_partner_ratio = 20;     ///< Default ratio for worker scaling.
    static constexpr uint64_t default_partner_threshold = 0;  ///< Default threshold for worker scaling.
    static constexpr uint64_t default_partner_delay_us = 100; ///< Default busy wait time (us) for last worker.
    static constexpr uint64_t default_q_depth = 1024;         ///< Default depth of the job queue.
};

/**
 * @struct job_partner
 * @brief Provide the function of submitting tasks and waiting for task completion.
 *
 * @tparam UsageId The user-defined job type.
 * @since 20
 */
template <int UsageId = 0>
struct job_partner : ref_obj<job_partner<UsageId>>, detail::non_copyable {
    /**
    * @brief Retrieves the job_partner instance in the current thread.
    *
    * @param attr Job partner attributes.
    * @return Reference to the job_partner instance.
    * @since 20
    */
    static __attribute__((noinline)) auto& get_partner_of_this_thread(const job_partner_attr& attr = {})
    {
        static thread_local auto s = ref_obj<job_partner<UsageId>>::make(attr);
        return s;
    }

    /**
     * @brief Submits a suspendable job to the partner thread (blocking).
     *
     * This function submits a job that can be suspended and resumed, using the specified stack and stack size.
     * The function is blocking: it will block the current thread until the job is successfully executed.
     * It can be called from both master and non-master threads. If the queue is full, it will retry until successful.
     *
     * @tparam boost Indicates whether to dynamically add workers.
     * @param suspendable_job The job executor function closure.
     * @param stack Pointer to the stack memory for the job.
     * @param stack_size Size of the stack memory.
     * @return Returns 1 if job initialization failed (e.g., invalid stack_size); 0 if submission succeeded.
     * @since 20
     */
    template <bool boost = true>
    inline int submit(std::function<void()>&& suspendable_job, void* stack, size_t stack_size)
    {
        auto p = job_t::init(std::forward<std::function<void()>>(suspendable_job), stack, stack_size);
        if (p == nullptr) {
            FFRT_API_LOGE("job initialize failed, maybe invalid stack_size");
            return 1;
        }
        FFRT_API_LOGD("submit %lu", p->id);
        p->local().partner = this;
        submit<boost>(suspendable_job_func, p);
        return 0;
    }

    /**
     * @brief Submits a non-suspendable job to the partner thread (non-blocking).
     *
     * This function submits a job that cannot be suspended. The function is non-blocking:
     * it will not block the current thread, and the job will be asynchronously executed by a partner worker thread.
     * If the queue is full, it will retry until successful.
     *
     * @tparam boost Indicates whether to dynamically add workers.
     * @param non_suspendable_job The job executor function closure.
     * @since 20
     */
    template <bool boost = true>
    inline void submit(std::function<void()>&& non_suspendable_job)
    {
        auto p = new non_suspendable_job_t(std::forward<std::function<void()>>(non_suspendable_job), this);
        FFRT_API_LOGD("non-suspendable job submit: %p", p);
        submit<boost>(non_suspendable_job_func, p);
    }

    /**
     * @brief Submits a job to the master thread and suspends the current task until completion.
     *
     * This function submits a job to the master thread. The current task will be paused after submitting the closure,
     * and will resume only after the master thread finishes executing the closure. If called outside a job context,
     * the closure will be executed directly. If the queue is full, it will retry until successful.
     *
     * @param job The job executor function closure.
     * @since 20
     */
    static inline void submit_to_master(std::function<void()>&& job)
    {
        auto& e = job_t::env();
        auto j = e.cur;
        if (j == nullptr || j->local().partner == e.tl.token) {
            return job();
        }
        j->local().partner->submit_to_master(e, j, std::forward<std::function<void()>>(job));
    }

    /**
     * @brief Waits until all submitted tasks are complete.
     *
     * This function blocks the calling thread until all submitted jobs have finished execution.
     * It can only be called from the master thread; calling from a non-master thread will fail.
     *
     * @tparam help_partner If true, the current thread will also consume jobs from the worker queue.
     * @tparam busy_wait_us If the worker queue is empty, the current thread will busy-wait for
     *                      this duration (in microseconds) before sleeping.
     *                      If a job is submitted during this time, the thread will consume it.
     * @return Returns 1 if called from a non-master thread (wait fails); 0 if wait succeeds.
     * @since 20
     */
    template<bool help_partner = true, uint64_t busy_wait_us = 100>
    int wait()
    {
        if (!this_thread_is_master()) {
            FFRT_API_LOGE("wait only can be called on master thread");
            return 1;
        }
        FFRT_API_TRACE_SCOPE("%s wait on master", name.c_str());
        FFRT_API_LOGD("wait on master");

        for (;;) {
_begin_consume_master_job:
            int idx = 0;
            while (master_q.try_run()) {
                if (((++idx & 0xF) == 0) && partner_num.load(std::memory_order_relaxed) == 0) {
                    job_partner_task();
                }
            }

            auto concurrency = job_num.load();
            auto wn = partner_num.load(std::memory_order_relaxed);
            if (wn < attr.max_num() && partner_q.size() > wn * attr.ratio() + attr.threshold()) {
                job_partner_task();
            }
            if (help_partner && partner_q.try_run()) {
                goto _begin_consume_master_job;
            }
            if (concurrency == 0) {
                break;
            } else {
                auto s = clock::now();
                while (!help_partner && busy_wait_us > 0 && clock::ns(s) < busy_wait_us * 1000) {
                    if (master_q.try_run()) {
                        goto _begin_consume_master_job;
                    }
                    wfe();
                }
                master_wait.wait(0);
                master_wait = 0;
            }
        }

        FFRT_API_LOGD("wait success");
        return 0;
    }

    /**
     * @brief Judge whether the current thread is the job_partner master.
     *
     * @return true if the current thread is the job_partner master; false otherwise.
     * @since 20
     */
    inline bool this_thread_is_master()
    {
        return job_t::env().tl.token == this;
    }

private:
    friend ref_obj<job_partner>; ///< Allows ref_obj to access private members for reference counting.

    /**
     * @brief Fiber-local storage structure for master function and partner pointer.
     */
    struct fls {
        std::function<void()> master_f; ///< Function to be executed by the master.
        job_partner* partner;           ///< Pointer to the associated job_partner instance.
    };

    /**
     * @brief Thread-local storage structure for token identification.
     */
    struct tls {
        void* token = nullptr; ///< Token used to identify the current job_partner instance.
    };

    /**
     * @brief Alias for the fiber type used by this job_partner.
     */
    using job_t = fiber<UsageId, fls, tls>;

    /**
     * @brief Structure representing a non-suspendable job.
     */
    struct non_suspendable_job_t {
        /**
         * @brief Constructs a non_suspendable_job_t object.
         *
         * @param fn Function to execute.
         * @param p Pointer to the associated job_partner.
         */
        non_suspendable_job_t(std::function<void()>&& fn, job_partner* p)
            : fn(std::forward<std::function<void()>>(fn)), partner(p) {}

        std::function<void()> fn; ///< Function to execute.
        job_partner* partner;     ///< Pointer to the associated job_partner.
    };

    /**
     * @brief Constructs a job_partner object with the given attributes.
     *
     * @param attr Job partner attributes.
     */
    job_partner(const job_partner_attr& attr = {})
        : name("partner<" + std::to_string(UsageId) + ">" + std::to_string(syscall(SYS_gettid))),
        attr(attr), partner_q(attr.queue_depth(), name + "_pq"), master_q(attr.queue_depth(), name + "_mq")
    {
        concurrency_name = name + "_cc#";
        partner_num_name = name + "_p#";
        job_t::env().tl.token = this;
    }

    /**
     * @brief Submits a job to the partner queue.
     *
     * @tparam boost Indicates whether to dynamically add workers.
     * @param f Function pointer for the job.
     * @param p Pointer to the job data.
     */
    template <bool boost>
    void submit(func_ptr f, void* p)
    {
        auto concurrency = job_num.fetch_add(1, std::memory_order_relaxed) + 1;
        (void)(concurrency);
        FFRT_API_TRACE_INT64(concurrency_name.c_str(), concurrency);
        partner_q.template push<1>(f, p);

        auto wn = partner_num.load(std::memory_order_relaxed);
        if (boost || attr.threshold()) {
            if (wn < attr.max_num() && partner_q.size() > wn * attr.ratio() + attr.threshold()) {
                job_partner_task();
            }
        } else {
            if (wn == 0) {
                job_partner_task();
            }
        }
    }

    /**
     * @brief Submits a job to the master queue and suspends the current fiber.
     *
     * @tparam Env Environment type.
     * @param e Reference to the environment.
     * @param p Pointer to the job fiber.
     * @param job Function to execute.
     */
    template<class Env>
    void submit_to_master(Env& e, job_t* p, std::function<void()>&& job)
    {
        FFRT_API_LOGD("job %lu submit to master", (p ? p->id : -1UL));
        p->local().master_f = std::forward<std::function<void()>>(job);
        p->suspend(e, submit_to_master_suspend_func);
    }

    /**
     * @brief Suspend function used when submitting to master.
     *
     * @param p Pointer to the job fiber.
     * @return True if suspension is successful.
     */
    static bool submit_to_master_suspend_func(void* p)
    {
        auto partner = (static_cast<job_t*>(p))->local().partner;
        partner->master_q.template push<0>(master_run_func, p);
        partner->notify_master();
        return true;
    }

    /**
     * @brief Notifies the master thread to wake up if waiting.
     */
    inline void notify_master()
    {
        if (master_wait.exchange(1) == 0) {
            master_wait.notify_one();
        }
    }

    /**
     * @brief Launches a new partner worker task.
     */
    void job_partner_task()
    {
        auto partner_n = partner_num.fetch_add(1) + 1;
        (void)(partner_n);
        FFRT_API_TRACE_INT64(partner_num_name.c_str(), partner_n);
        FFRT_API_TRACE_SCOPE("%s add task", name.c_str());

        ref_obj<job_partner<UsageId>>::inc_ref();
        ffrt::submit([this] {
            auto wn = partner_num.load(std::memory_order_relaxed);
            if (wn < attr.max_num() && partner_q.size() > wn * attr.ratio() + attr.threshold()) {
                job_partner_task();
            }
_re_run_partner:
            while (partner_q.try_run());
            if (partner_num.load() == 1 && attr.busy() > 0) { // last partner delay
                FFRT_API_TRACE_SCOPE("stall");
                auto s = clock::now();
                while (clock::ns(s) < attr.busy() * 1000) {
                    if (partner_q.try_run()) {
                        goto _re_run_partner;
                    }
                    wfe();
                }
            }
            auto partner_n = partner_num.fetch_sub(1) - 1;
            (void)(partner_n);
            FFRT_API_TRACE_INT64(partner_num_name.c_str(), partner_n);
            if (partner_q.try_run()) {
                auto partner_n = partner_num.fetch_add(1) + 1;
                (void)(partner_n);
                FFRT_API_TRACE_INT64(partner_num_name.c_str(), partner_n);
                goto _re_run_partner;
            }
            ref_obj<job_partner<UsageId>>::dec_ref();
            }, {}, {}, task_attr().qos(attr.qos()).name(name.c_str()));
    }

    /**
     * @brief Function executed by a suspendable job.
     *
     * @param p_ Pointer to the job fiber.
     */
    static void suspendable_job_func(void* p_)
    {
        auto p = (static_cast<job_t*>(p_));
        FFRT_API_LOGD("run partner job %lu", p->id);
        FFRT_API_TRACE_SCOPE("pjob%lu", p->id);
        if (p->start()) { // job done
            auto partner = p->local().partner;
            auto concurrency = partner->job_num.fetch_sub(1, std::memory_order_acquire) - 1;
            FFRT_API_TRACE_INT64(partner->concurrency_name.c_str(), concurrency);
            if (concurrency == 0) {
                partner->notify_master();
            }
            p->destroy();
        }
    }

    /**
     * @brief Function executed by a non-suspendable job.
     *
     * @param p_ Pointer to the non_suspendable_job_t object.
     */
    static void non_suspendable_job_func(void* p_)
    {
        auto p = static_cast<non_suspendable_job_t*>(p_);
        FFRT_API_LOGD("run non-suspendable job %p", p);
        FFRT_API_TRACE_SCOPE("nsjob");
        (p->fn)();
        auto partner = p->partner;
        auto concurrency = partner->job_num.fetch_sub(1, std::memory_order_acquire) - 1;
        FFRT_API_TRACE_INT64(partner->concurrency_name.c_str(), concurrency);
        if (concurrency == 0) {
            partner->notify_master();
        }
        delete p;
    }

    /**
     * @brief Function executed by the master to run a job.
     *
     * @param p_ Pointer to the job fiber.
     */
    static void master_run_func(void* p_)
    {
        auto p = static_cast<job_t*>(p_);
        {
            FFRT_API_LOGD("run master job %lu", p->id);
            FFRT_API_TRACE_SCOPE("mjob%lu", p->id);
            p->local().master_f();
            p->local().master_f = nullptr;
        }
        p->local().partner->partner_q.template push<1>(suspendable_job_func, p);
    }

    std::string name;             ///< Name of the job_partner instance.
    std::string concurrency_name; ///< Name used for concurrency tracing.
    std::string partner_num_name; ///< Name used for partner number tracing.
    job_partner_attr attr;        ///< Attributes for configuring the job_partner.

    alignas(detail::cacheline_size) std::atomic_uint64_t partner_num{0}; ///< Number of active partner workers.
    alignas(detail::cacheline_size) std::atomic_uint64_t job_num{0};     ///< Number of active jobs.

    runnable_queue<mpmc_queue> partner_q; ///< Runnable queue for partner jobs.
    runnable_queue<mpmc_queue> master_q;  ///< Runnable queue for master jobs.
    atomic_wait master_wait = 0;          ///< Synchronization primitive for master waiting.
};

} // namespace ffrt

#endif // FFRT_JOB_PARTNER_H
/** @} */