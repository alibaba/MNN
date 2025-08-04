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
#ifndef _CPU_TASK_H_
#define _CPU_TASK_H_

#include <string>
#include <functional>
#include <unordered_set>
#include <vector>
#include <mutex>
#include <atomic>
#include <string>
#include <set>
#include <list>
#include <memory>
#include <unistd.h>
#include "task_base.h"
#include "eu/co_routine.h"
#include "core/task_attr_private.h"
#include "dfx/log/ffrt_log_api.h"
#include "eu/func_manager.h"
#ifdef FFRT_ASYNC_STACKTRACE
#include "dfx/async_stack/ffrt_async_stack.h"
#endif

namespace ffrt {
constexpr int CO_CREATE_RETRY_INTERVAL = 500 * 1000;
constexpr uint64_t MASK_FOR_HCS_TASK = 0xFF000000000000;
struct VersionCtx;
class SCPUEUTask;

#ifdef FFRT_TASK_LOCAL_ENABLE
struct TaskLocalAttr {
    bool taskLocal = false; // 是否开启taskLocal特性
    void** threadTsd = nullptr; // 指向保存线程tsd数据内存空间的指针
    void** tsd = nullptr; // 指向task自身tsd数据内存空间的指针
};
#endif

class CPUEUTask : public CoTask {
public:
    CPUEUTask(const task_attr_private *attr, CPUEUTask *parent, const uint64_t &id);
    ~CPUEUTask() override;
    SkipStatus skipped = SkipStatus::SUBMITTED;

    CPUEUTask* parent = nullptr;
    uint64_t delayTime = 0;
    TimeoutTask timeoutTask;

    std::vector<CPUEUTask*>* in_handles_ = nullptr;
    /* The current number of child nodes does not represent the real number of child nodes,
     * because the dynamic graph child nodes will grow to assist in the generation of id
     */
    std::atomic<uint64_t> childNum {0};
    bool isWatchdogEnable = false;
    bool notifyWorker_ = true;
    bool isDelaying = false;

#ifdef FFRT_TASK_LOCAL_ENABLE
    TaskLocalAttr* tlsAttr = nullptr;
#endif

    inline bool IsRoot() const
    {
        return parent == nullptr;
    }

    inline void SetInHandles(std::vector<CPUEUTask*>& in_handles)
    {
        if (in_handles.empty()) {
            return;
        }
        in_handles_ = new std::vector<CPUEUTask*>(in_handles);
    }

    inline const std::vector<CPUEUTask*>& GetInHandles()
    {
        static const std::vector<CPUEUTask*> empty;
        if (!in_handles_) {
            return empty;
        }
        return *in_handles_;
    }

    void Prepare() override;
    void Ready() override;

    void Pop() override
    {
        SetStatus(TaskStatus::POPPED);
    }

    void Execute() override;

    BlockType Block() override
    {
        if (USE_COROUTINE && !IsRoot() && legacyCountNum <= 0) {
            blockType = BlockType::BLOCK_COROUTINE;
            SetStatus(TaskStatus::COROUTINE_BLOCK);
        } else {
            blockType = BlockType::BLOCK_THREAD;
            SetStatus(TaskStatus::THREAD_BLOCK);
        }
        return blockType;
    }

    void Wake() override
    {
        SetStatus(TaskStatus::EXECUTING);
        blockType = BlockType::BLOCK_COROUTINE;
    }

    void Cancel() override
    {
        SetStatus(TaskStatus::CANCELED);
    }

    void FreeMem() override;
    void SetQos(const QoS& newQos) override;

    BlockType GetBlockType() const override
    {
        if (IsRoot()) {
            return BlockType::BLOCK_THREAD;
        }
        return blockType;
    }
};

inline bool NeedNotifyWorker(TaskBase* task)
{
    if (task == nullptr) {
        return false;
    }
    bool needNotify = true;
    if (task->type == ffrt_normal_task) {
        CPUEUTask* cpuTask = static_cast<CPUEUTask*>(task);
        needNotify = cpuTask->notifyWorker_;
        cpuTask->notifyWorker_ = true;
    }
    return needNotify;
}

inline bool isDelayingTask(CPUEUTask* task)
{
    return task->isDelaying;
}
} /* namespace ffrt */
#endif
