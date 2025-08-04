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

#ifndef _TASK_BASE_H_
#define _TASK_BASE_H_
#include <atomic>
#include <vector>
#include "eu/co_routine.h"
#include "internal_inc/types.h"
#ifdef USE_OHOS_QOS
#include "qos.h"
#else
#include "staging_qos/sched/qos.h"
#endif
#include "sched/execute_ctx.h"
#include "internal_inc/non_copyable.h"
#include "util/time_format.h"
#include "internal_inc/types.h"
#include "core/task_attr_private.h"

namespace ffrt {
static constexpr uint64_t cacheline_size = 64;

typedef struct HiTraceIdStruct {
#if __BYTE_ORDER == __BIG_ENDIAN
    uint64_t chainId : 60;
    uint64_t ver : 3;
    uint64_t valid : 1;

    uint64_t parentSpanId : 26;
    uint64_t spanId : 26;
    uint64_t flags : 12;
#elif __BYTE_ORDER == __LITTLE_ENDIAN
    uint64_t valid : 1;
    uint64_t ver : 3;
    uint64_t chainId : 60;

    uint64_t flags : 12;
    uint64_t spanId : 26;
    uint64_t parentSpanId : 26;
#else
#error "ERROR: No BIG_LITTLE_ENDIAN defines."
#endif
} HiTraceIdStruct;
constexpr uint64_t HITRACE_ID_VALID = 1;

struct TimeoutTask {
    uint64_t taskGid{0};
    uint64_t timeoutCnt{0};
    TaskStatus taskStatus{TaskStatus::PENDING};
    TimeoutTask() = default;
    TimeoutTask(uint64_t gid, uint64_t timeoutcnt, TaskStatus status)
        : taskGid(gid), timeoutCnt(timeoutcnt), taskStatus(status) {}
};

class TaskBase : private NonCopyable {
public:
    TaskBase(ffrt_executor_task_type_t type, const task_attr_private *attr);
    virtual ~TaskBase() = default;

    // lifecycle actions
    virtual void Prepare() = 0;
    virtual void Ready() = 0;
    virtual void Pop() = 0;
    // must be called by a sync primitive when blocking this task.
    // return value indicates this task need to be blocked on thread or yield from it's coroutine.
    virtual BlockType Block() = 0;
    virtual void Wake() = 0;
    virtual void Execute() = 0;
    virtual void Finish() = 0;
    virtual void Cancel() = 0;
    virtual void FreeMem() = 0;

    // must be called by a sync primitive when it wakes a blocking task.
    // return value indicates this task has been blocked whether on thread or yield from it's coroutine.
    virtual BlockType GetBlockType() const = 0;

    // getters and setters
    virtual std::string GetLabel() const = 0;
    virtual void SetQos(const QoS& newQos) = 0;

    inline int GetQos() const
    {
        return qos_();
    }

    inline void SetStatus(TaskStatus statusIn)
    {
        /* Note this function can be called concurrently.
         * The following accesses can be interleaved.
         * We use atomic relaxed accesses in order to
         * combat data-races without incurring performance
         * overhead. Currently statusTime & preStatus
         * are only used in printing debug information
         * and don't play a role in the logic.
         */
        statusTime.store(TimeStampCntvct(), std::memory_order_relaxed);
        preStatus.store(curStatus, std::memory_order_relaxed);
        curStatus.store(statusIn, std::memory_order_relaxed);
    }

    // delete ref setter functions, for memory management
    inline uint32_t IncDeleteRef()
    {
        auto v = rc.fetch_add(1);
        return v;
    }

    inline uint32_t DecDeleteRef()
    {
        auto v = rc.fetch_sub(1);
        if (v == 1) {
            FreeMem();
        }
        return v;
    }

    // returns the current g_taskId value
    static uint32_t GetLastGid();

    // properties
    LinkedList node; // used on fifo fast que
    ffrt_executor_task_type_t type;
    const uint64_t gid; // global unique id in this process
    QoS qos_ = qos_default;
    std::atomic_uint32_t rc = 1; // reference count for delete
    std::atomic<TaskStatus> preStatus = TaskStatus::PENDING;
    std::atomic<TaskStatus> curStatus = TaskStatus::PENDING;
    std::atomic<uint64_t> statusTime = TimeStampCntvct();
    std::atomic<AliveStatus> aliveStatus {AliveStatus::UNITINITED};

#ifdef FFRT_ASYNC_STACKTRACE
    uint64_t stackId = 0;
#endif

    struct HiTraceIdStruct traceId_ = {};
    uint64_t createTime {0};
    uint64_t executeTime {0};
    int32_t fromTid {0};
};

class CoTask : public TaskBase {
public:
    CoTask(ffrt_executor_task_type_t type, const task_attr_private *attr)
        : TaskBase(type, attr)
    {
        we.task = this;
        if (attr) {
            stack_size = std::max(attr->stackSize_, MIN_STACK_SIZE);
        }
    }
    ~CoTask() override = default;

    uint8_t func_storage[ffrt_auto_managed_function_storage_size]; // 函数闭包、指针或函数对象

    std::string label;
    CoWakeType coWakeType { CoWakeType::NO_TIMEOUT_WAKE };
    int cpuBoostCtxId = -1;
    WaitEntry we; // Used on syncprimitive wait que
    WaitUntilEntry* wue = nullptr; // used on syncprimitive wait que and delayed wait que
    // lifecycle connection between task and coroutine is shown as below:
    // |*task pending*|*task ready*|*task executing*|*task done*|*task release*|
    //                             |**********coroutine*********|
    CoRoutine* coRoutine = nullptr;
    uint64_t stack_size = STACK_SIZE;
    std::atomic<pthread_t> runningTid = 0;
    int legacyCountNum = 0; // dynamic switch controlled by set_legacy_mode api
    std::mutex mutex_; // used in coroute
    std::condition_variable waitCond_; // cv for thread wait

    bool pollerEnable = false; // set true if task call ffrt_epoll_ctl
    bool threadMode_ = false;
    std::string GetLabel() const override
    {
        return label;
    }

    inline void UnbindCoRoutione()
    {
        std::lock_guard lck(mutex_);
        coRoutine = nullptr;
    }

protected:
    BlockType blockType { BlockType::BLOCK_COROUTINE }; // block type for lagacy mode changing
};

std::string StatusToString(TaskStatus status);

void ExecuteTask(TaskBase* task);

inline bool IsCoTask(TaskBase* task)
{
    return task && (task->type == ffrt_normal_task ||
        (task->type == ffrt_queue_task && (!reinterpret_cast<ffrt::CoTask*>(task)->threadMode_)));
}

inline bool IncDeleteRefIfPositive(TaskBase* task)
{
    uint32_t expected = task->rc.load();
    while (expected > 0) {
        if (task->rc.compare_exchange_weak(expected, expected + 1)) {
            return true;
        }
    }
    return false;
}
} // namespace ffrt
#endif
