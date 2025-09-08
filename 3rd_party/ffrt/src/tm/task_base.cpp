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

#include "tm/task_base.h"
#include "util/ffrt_facade.h"

namespace {
    std::atomic_uint64_t g_taskId(0);
} // namespace

namespace ffrt {
TaskBase::TaskBase(ffrt_executor_task_type_t type, const task_attr_private *attr) : type(type), gid(++g_taskId)
{
    // qos_inherit is an abnormal enum value, which must to be replaced by another.
    qos_ = (attr && attr->qos_ != qos_inherit) ? QoS(attr->qos_) : QoS();
}

uint32_t TaskBase::GetLastGid()
{
    return g_taskId.load(std::memory_order_acquire);
}

void ExecuteTask(TaskBase* task)
{
    bool isCoTask = IsCoTask(task);

    // set current task info to context
    ExecuteCtx* ctx = ExecuteCtx::Cur();
    ctx->task = task;
    ctx->lastGid_ = task->gid;

    // run task with coroutine
    if (USE_COROUTINE && isCoTask) {
        while (CoStart(static_cast<CoTask*>(task), GetCoEnv()) != 0) {
            usleep(CO_CREATE_RETRY_INTERVAL);
        }
    } else {
    // run task on thread
#ifdef FFRT_ASYNC_STACKTRACE
        if (isCoTask) {
            FFRTSetStackId(task->stackId);
        }
#endif
        task->Execute();
    }

    // reset task info in context
    ctx->task = nullptr;
}

std::string StatusToString(TaskStatus status)
{
    static const std::unordered_map<TaskStatus, std::string> statusMap = {
        {TaskStatus::PENDING,         "PENDING"},
        {TaskStatus::SUBMITTED,       "SUBMITTED"},
        {TaskStatus::ENQUEUED,        "ENQUEUED"},
        {TaskStatus::DEQUEUED,        "DEQUEUED"},
        {TaskStatus::READY,           "READY"},
        {TaskStatus::POPPED,          "POPPED"},
        {TaskStatus::EXECUTING,       "EXECUTING"},
        {TaskStatus::THREAD_BLOCK,    "THREAD_BLOCK"},
        {TaskStatus::COROUTINE_BLOCK, "COROUTINE_BLOCK"},
        {TaskStatus::FINISH,          "FINISH"},
        {TaskStatus::WAIT_RELEASING,  "WAIT_RELEASING"},
        {TaskStatus::CANCELED,        "CANCELED"}
    };

    auto it = statusMap.find(status);
    return (it != statusMap.end()) ? it->second : "Unknown";
}
} // namespace ffrt
