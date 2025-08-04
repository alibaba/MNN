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
#include "tm/queue_task.h"
#include "dfx/trace/ffrt_trace.h"
#include "dfx/log/ffrt_log_api.h"
#include "dfx/trace_record/ffrt_trace_record.h"
#include "c/task.h"
#include "util/ffrt_facade.h"
#include "tm/task_factory.h"

namespace {
constexpr uint64_t MIN_SCHED_TIMEOUT = 100000; // 0.1s
}
namespace ffrt {
QueueTask::QueueTask(QueueHandler* handler, const task_attr_private* attr, bool insertHead)
    : CoTask(ffrt_queue_task, attr), handler_(handler), insertHead_(insertHead)
{
    if (handler) {
        if (attr) {
            label = handler->GetName() + "_" + attr->name_ + "_" + std::to_string(gid);
        } else {
            label = handler->GetName() + "_" + std::to_string(gid);
        }
        threadMode_ = handler->GetMode();
    }

    uptime_ = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());

    if (attr) {
        delay_ = attr->delay_;
        qos_ = attr->qos_;
        uptime_ += delay_;
        prio_ = attr->prio_;
        if (delay_ && attr->timeout_) {
            FFRT_SYSEVENT_LOGW("task [gid=%llu] not support delay and timeout at the same time, timeout ignored", gid);
        } else if (attr->timeout_) {
            schedTimeout_ = std::max(attr->timeout_, MIN_SCHED_TIMEOUT); // min 0.1s
        }
    }

    FFRT_LOGD("ctor task [gid=%llu], delay=%lluus, type=%lu, prio=%d, timeout=%luus", gid, delay_, type, prio_,
        schedTimeout_);
}

QueueTask::~QueueTask()
{
    FFRT_LOGD("dtor task [gid=%llu]", gid);
}

void QueueTask::Prepare()
{
    SetStatus(TaskStatus::ENQUEUED);
    FFRTTraceRecord::TaskSubmit<ffrt_queue_task>(qos_, &createTime, &fromTid);
#ifdef FFRT_ENABLE_HITRACE_CHAIN
    if (TraceChainAdapter::Instance().HiTraceChainGetId().valid == HITRACE_ID_VALID) {
        traceId_ = TraceChainAdapter::Instance().HiTraceChainCreateSpan();
    }
#endif
}

void QueueTask::Ready()
{
    QoS taskQos = qos_;
    FFRTTraceRecord::TaskSubmit<ffrt_queue_task>(taskQos);
    this->SetStatus(TaskStatus::READY);
    bool isRisingEdge = FFRTFacade::GetSchedInstance()->GetScheduler(taskQos).PushTaskGlobal(this, false);
    FFRTTraceRecord::TaskEnqueue<ffrt_queue_task>(taskQos);
    FFRTFacade::GetEUInstance().NotifyTask<TaskNotifyType::TASK_ADDED_RTQ>(taskQos, false, isRisingEdge);
}

void QueueTask::Finish()
{
    if (createTime != 0) {
        FFRTTraceRecord::TaskDone<ffrt_queue_task>(qos_(), this);
    }
    auto f = reinterpret_cast<ffrt_function_header_t *>(func_storage);
    if (f->destroy) {
        f->destroy(f);
    }
    Notify();
    FFRT_TASKDONE_MARKER(gid);
}

void QueueTask::FreeMem()
{
    // only tasks which called ffrt_poll_ctl may have cached events
    if (pollerEnable) {
        FFRTFacade::GetPPInstance().ClearCachedEvents(this);
    }
    TaskFactory<QueueTask>::Free(this);
}

void QueueTask::Destroy()
{
    // release user func
    auto f = reinterpret_cast<ffrt_function_header_t*>(func_storage);
    if (f->destroy) {
        f->destroy(f);
    }
    // free serial task object
    DecDeleteRef();
}

void QueueTask::Notify()
{
    std::lock_guard lock(mutex_);
    isFinished_.store(true);
    if (onWait_) {
        waitCond_.notify_all();
    }
}

void QueueTask::Execute()
{
    IncDeleteRef();
    FFRT_LOGD("Execute stask[%lu], name[%s]", gid, label.c_str());
    if (isFinished_.load()) {
        FFRT_SYSEVENT_LOGE("task [gid=%llu] is complete, no need to execute again", gid);
        DecDeleteRef();
        return;
    }
    handler_->Dispatch(this);
    UnbindCoRoutione();
    DecDeleteRef();
}

void QueueTask::Wait()
{
    std::unique_lock lock(mutex_);
    onWait_ = true;
    while (!isFinished_.load()) {
        waitCond_.wait(lock);
    }
}

uint32_t QueueTask::GetQueueId() const
{
    return handler_->GetQueueId();
}
} // namespace ffrt
