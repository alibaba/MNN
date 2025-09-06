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

#ifndef FFRT_CPU_WORKER_HPP
#define FFRT_CPU_WORKER_HPP

#include <atomic>
#include <unistd.h>
#ifdef FFRT_PTHREAD_ENABLE
#include <pthread.h>
#endif
#include <thread>
#ifdef OHOS_THREAD_STACK_DUMP
#include <sstream>
#endif
#ifdef USE_OHOS_QOS
#include "qos.h"
#else
#include "staging_qos/sched/qos.h"
#endif
#include "tm/task_base.h"
#include "dfx/log/ffrt_log_api.h"
#include "c/executor_task.h"
#include "util/spmc_queue.h"

namespace ffrt {
constexpr int PTHREAD_CREATE_NO_MEM_CODE = 11;
constexpr int FFRT_RETRY_MAX_COUNT = 12;
const std::vector<uint64_t> FFRT_RETRY_CYCLE_LIST = {
    10 * 1000, 50 * 1000, 100 * 1000, 200 * 1000, 500 * 1000, 1000 * 1000, 2 * 1000 * 1000,
    5 * 1000 * 1000, 10 * 1000 * 1000, 50 * 1000 * 1000, 100 * 1000 * 1000, 500 * 1000 * 1000
};

enum class WorkerAction {
    RETRY = 0,
    RETIRE,
    MAX,
};

enum class WorkerStatus {
    EXECUTING = 0,
    SLEEPING,
    DESTROYED,
};

class CPUWorker;
struct CpuWorkerOps {
    std::function<WorkerAction (CPUWorker*)> WorkerIdleAction;
    std::function<void (CPUWorker*)> WorkerRetired;
    std::function<void (CPUWorker*)> WorkerPrepare;
#ifdef FFRT_WORKERS_DYNAMIC_SCALING
    std::function<bool (void)> IsBlockAwareInit;
#endif
};

class CPUWorker {
public:
    explicit CPUWorker(const QoS& qos, CpuWorkerOps&& ops, size_t stackSize);
    ~CPUWorker();

    bool Exited() const
    {
        return exited.load(std::memory_order_relaxed);
    }

    void SetExited()
    {
        exited.store(true, std::memory_order_relaxed);
    }

    pid_t Id() const
    {
        while (!exited && tid < 0) {
        }
        return tid;
    }

    const QoS& GetQos() const
    {
        return qos;
    }

    const WorkerStatus& GetWorkerState() const
    {
        return state;
    }

    void SetWorkerState(const WorkerStatus& newState)
    {
        this->state = newState;
    }

#ifdef FFRT_WORKERS_DYNAMIC_SCALING
    unsigned int GetDomainId() const
    {
        return domain_id;
    }
#endif
#ifdef FFRT_PTHREAD_ENABLE
    void Start(void*(*ThreadFunc)(void*), void* args)
    {
        int ret = pthread_create(&thread_, &attr_, ThreadFunc, args);
        if (ret == PTHREAD_CREATE_NO_MEM_CODE) {
            int count = 0;
            while (ret == PTHREAD_CREATE_NO_MEM_CODE && count < FFRT_RETRY_MAX_COUNT) {
                usleep(FFRT_RETRY_CYCLE_LIST[count]);
                count++;
                FFRT_LOGW("pthread_create failed due to shortage of system memory, FFRT retry %d times...", count);
                ret = pthread_create(&thread_, &attr_, ThreadFunc, args);
            }
        }
        if (ret != 0) {
            FFRT_LOGE("pthread_create failed, ret = %d", ret);
            exited = true;
        }
        pthread_attr_destroy(&attr_);
    }

    void Join()
    {
        if (tid > 0 && thread_ != 0) {
            pthread_join(thread_, nullptr);
        }
        tid = -1;
    }

    void Detach()
    {
        if (tid > 0 && thread_ != 0) {
            pthread_detach(thread_);
        } else {
            FFRT_LOGD("qos %d thread not joinable.", qos());
        }
        tid = -1;
    }

    pthread_t& GetThread()
    {
        return this->thread_;
    }
#else
    template <typename F, typename... Args>
    void Start(F&& f, Args&&... args)
    {
        auto wrap = [&](Args&&... args) {
            NativeConfig();
            return f(args...);
        };
        thread = std::thread(wrap, args...);
    }

    void Join()
    {
        if (thread.joinable()) {
            thread.join();
        }
        tid = -1;
    }

    void Detach()
    {
        if (thread.joinable()) {
            thread.detach();
        } else {
            FFRT_LOGD("qos %d thread not joinable\n", qos());
        }
        tid = -1;
    }

    pthread_t GetThread()
    {
        return this->thread.native_handle();
    }
#endif

    void SetThreadAttr(const QoS& newQos);
    TaskBase* curTask = nullptr;
    std::atomic<uintptr_t> curTaskType_ {ffrt_invalid_task};
    std::string curTaskLabel_ = ""; // 需要打开宏WORKER_CAHCE_NAMEID才会赋值
    uint64_t curTaskGid_ = UINT64_MAX;
    unsigned int tick = 0;

private:
    void NativeConfig();
    static void WorkerLooper(CPUWorker* worker);
    static void* WrapDispatch(void* worker);
    void WorkerSetup();
    static void Dispatch(CPUWorker* worker);
    static void RunTask(TaskBase* task, CPUWorker* worker);
    static bool RunSingleTask(int qos, CPUWorker *worker);
#ifdef FFRT_SEND_EVENT
    int cacheQos; // cache int qos
    std::string cacheLabel; // cache string label
    uint64_t cacheFreq = 1000000; // cache cpu freq
#endif
    std::atomic_bool exited {false};
    std::atomic<pid_t> tid {-1};
    QoS qos;
    CpuWorkerOps ops;
    WorkerStatus state {WorkerStatus::EXECUTING};
#ifdef FFRT_PTHREAD_ENABLE
    pthread_t thread_{0};
    pthread_attr_t attr_;
#else
    std::thread thread;
#endif
#ifdef FFRT_WORKERS_DYNAMIC_SCALING
    unsigned int domain_id;
#endif
};
} // namespace ffrt
#endif
