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

#ifndef UV_TASK_H_
#define UV_TASK_H_

#include "task_base.h"
#include "c/executor_task.h"
#include "core/task_io.h"
#include "core/task_attr_private.h"
#include "tm/task_factory.h"
#ifdef USE_OHOS_QOS
#include "qos.h"
#else
#include "staging_qos/sched/qos.h"
#endif

namespace ffrt {
class UVTask : public TaskBase {
public:
    UVTask(ffrt_executor_task* uvWork, const task_attr_private *attr)
        : TaskBase(ffrt_uv_task, attr), uvWork(uvWork)
    {
        /*
            uvWork是libuv传进来的ffrt_executor_task指针，其中的wq成员为双向链表节点，和LinkedList的内存布局一致，
            曾经用于软化队列的节点和TaskBase类一样插入ReadyQueue。
            目前由于已组合进入UVTask类型，不再作为链表节点，此处就可以使用uvWorker的wq来标记任务是否已经出队的状态。
            libuv里面，在cancel时也会判断任务是否已经出队（uv__queue_empty）:
            q== q->next || q != q->next->prev;其中next对应wq[0]，prev对应wq[1]
            将wq入队，即可满足v__queue_empty返回false的需要
        */
        if (uvWork != nullptr) {
            uvWorkList.PushBack(reinterpret_cast<LinkedList*>(&uvWork->wq));
        } else {
            FFRT_LOGE("executor_task is nullptr");
        }
    }
    ffrt_executor_task* uvWork;

    void Prepare() override {}

    void Ready() override;

    void Pop() override
    {
        SetStatus(TaskStatus::POPPED);
    }

    void Execute() override;

    BlockType Block() override
    {
        SetStatus(TaskStatus::THREAD_BLOCK);
        return BlockType::BLOCK_THREAD;
    }

    void Wake() override
    {
        SetStatus(TaskStatus::EXECUTING);
    }

    void Finish() override {}
    void Cancel() override {}

    void FreeMem() override
    {
        TaskFactory<UVTask>::Free(this);
    }

    void SetQos(const QoS& newQos) override
    {
        qos_ = newQos;
    }

    std::string GetLabel() const override
    {
        return "uv-task";
    }

    BlockType GetBlockType() const override
    {
        return BlockType::BLOCK_THREAD;
    }

    inline void SetDequeued()
    {
        if (uvWork == nullptr) {
            return;
        }
        /*
            uvWork是libuv传进来的ffrt_executor_task指针，其中的wq成员为双向链表节点，和LinkedList的内存布局一致，
            曾经用于软化队列的节点和TaskBase类一样插入ReadyQueue。
            目前由于已组合进入UVTask类型，不再作为链表节点，此处就可以使用uvWorker的wq来标记任务是否已经出队的状态。
            libuv里面，在cancel时也会判断任务是否已经出队（uv__queue_empty）:
            q== q->next || q != q->next->prev;其中next对应wq[0]，prev对应wq[1]
            将wq出队，即可满足v__queue_empty返回false的需要
        */
        LinkedList::RemoveCur(reinterpret_cast<LinkedList*>(&uvWork->wq));
    }

    static void ExecuteImpl(UVTask* task, ffrt_executor_task_func func);
private:
    LinkedList uvWorkList;
};
} /* namespace ffrt */
#endif
