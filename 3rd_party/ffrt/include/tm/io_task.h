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
#ifndef _IO_TASK_H_
#define _IO_TASK_H_

#include "task_base.h"
#include "core/task_io.h"
#include "core/task_attr_private.h"
#include "tm/task_factory.h"
#include "util/ffrt_facade.h"

namespace ffrt {
class IOTask : public TaskBase {
public:
    IOTask(const ffrt_io_callable_t& work, const task_attr_private* attr);

    void Prepare() override {}

    void Ready() override
    {
        QoS taskQos = qos_;
        FFRTTraceRecord::TaskSubmit<ffrt_io_task>(taskQos);
        SetStatus(TaskStatus::READY);
        FFRTFacade::GetSchedInstance()->GetScheduler(taskQos).PushTaskGlobal(this);
        FFRTTraceRecord::TaskEnqueue<ffrt_io_task>(taskQos);
        FFRTFacade::GetEUInstance().NotifyTask<TaskNotifyType::TASK_LOCAL>(taskQos);
    }

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
        TaskFactory<IOTask>::Free(this);
    }

    void SetQos(const QoS& newQos) override
    {
        qos_ = newQos;
    }

    std::string GetLabel() const override
    {
        return "io-task";
    }

    BlockType GetBlockType() const override
    {
        return BlockType::BLOCK_THREAD;
    }

private:
    ffrt_io_callable_t work;
};
} /* namespace ffrt */
#endif
