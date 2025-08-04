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

#include "queue/queue_handler.h"
#include <sstream>
#include "concurrent_queue.h"
#include "eventhandler_adapter_queue.h"
#include "dfx/log/ffrt_log_api.h"
#include "dfx/trace_record/ffrt_trace_record.h"
#include "dfx/sysevent/sysevent.h"
#include "util/event_handler_adapter.h"
#include "util/ffrt_facade.h"
#include "util/slab.h"
#include "util/time_format.h"
#include "tm/queue_task.h"
#include "sched/scheduler.h"

namespace {
constexpr uint32_t STRING_SIZE_MAX = 128;
constexpr uint32_t TASK_DONE_WAIT_UNIT = 10;
constexpr uint64_t SCHED_TIME_ACC_ERROR_US = 5000; // 5ms
constexpr uint64_t MIN_TIMEOUT_THRESHOLD_US = 1000000; // 1s
constexpr uint64_t TASK_WAIT_COUNT = 50000; // 5s
constexpr uint64_t INVALID_GID = 0;
}

namespace ffrt {
QueueHandler::QueueHandler(const char* name, const ffrt_queue_attr_t* attr, const int type)
{
    // parse queue attribute
    if (attr) {
        qos_ = (ffrt_queue_attr_get_qos(attr) >= ffrt_qos_background) ? ffrt_queue_attr_get_qos(attr) : qos_;
        timeout_ = ffrt_queue_attr_get_timeout(attr);
        timeoutCb_ = ffrt_queue_attr_get_callback(attr);
        maxConcurrency_ = ffrt_queue_attr_get_max_concurrency(attr);
        threadMode_ = ffrt_queue_attr_get_thread_mode(attr);
    }

    // callback reference counting is to ensure life cycle
    if (timeout_ > 0 && timeoutCb_ != nullptr) {
        QueueTask* cbTask = GetQueueTaskByFuncStorageOffset(timeoutCb_);
        cbTask->IncDeleteRef();
    }
    trafficRecord_.SetTimeInterval(trafficRecordInterval_);
    curTaskVec_.resize(maxConcurrency_);
    timeoutTaskVec_.resize(maxConcurrency_);

    queue_ = CreateQueue(type, attr);
    FFRT_COND_DO_ERR((queue_ == nullptr), return, "[queueId=%u] constructed failed", GetQueueId());

    if (name != nullptr && std::string(name).size() <= STRING_SIZE_MAX) {
        name_ = "sq_" + std::string(name) + "_" + std::to_string(GetQueueId());
    } else {
        name_ += "sq_unnamed_" + std::to_string(GetQueueId());
        FFRT_LOGW("failed to set [queueId=%u] name due to invalid name or length.", GetQueueId());
    }

    FFRTFacade::GetQMInstance().RegisterQueue(this);
    FFRT_LOGI("construct %s succ, qos[%d]", name_.c_str(), qos_);
}

QueueHandler::~QueueHandler()
{
    FFRT_LOGI("destruct %s enter", name_.c_str());
    // clear tasks in queue
    CancelAndWait();
    FFRTFacade::GetQMInstance().DeregisterQueue(this);

    // release callback resource
    if (timeout_ > 0) {
        // wait for all delayedWorker to complete.
        while (delayedCbCnt_.load() > 0) {
            this_task::sleep_for(std::chrono::microseconds(timeout_));
        }

        if (timeoutCb_ != nullptr) {
            QueueTask* cbTask = GetQueueTaskByFuncStorageOffset(timeoutCb_);
            cbTask->DecDeleteRef();
        }
    }

    if (we_ != nullptr) {
        DelayedRemove(we_->tp, we_);
        SimpleAllocator<WaitUntilEntry>::FreeMem(we_);
    }
    FFRT_LOGI("destruct %s leave", name_.c_str());
}

bool QueueHandler::SetLoop(Loop* loop)
{
    FFRT_COND_DO_ERR((queue_ == nullptr), return false, "[queueId=%u] constructed failed", GetQueueId());
    if (queue_->GetQueueType() == ffrt_queue_eventhandler_interactive) {
        return true;
    }
    FFRT_COND_DO_ERR((queue_->GetQueueType() != ffrt_queue_concurrent),
        return false, "[queueId=%u] type invalid", GetQueueId());
    return reinterpret_cast<ConcurrentQueue*>(queue_.get())->SetLoop(loop);
}

bool QueueHandler::ClearLoop()
{
    FFRT_COND_DO_ERR((queue_ == nullptr), return false, "[queueId=%u] constructed failed", GetQueueId());
    FFRT_COND_DO_ERR((queue_->GetQueueType() != ffrt_queue_concurrent),
        return false, "[queueId=%u] type invalid", GetQueueId());
    return reinterpret_cast<ConcurrentQueue*>(queue_.get())->ClearLoop();
}

QueueTask* QueueHandler::PickUpTask()
{
    FFRT_COND_DO_ERR((queue_ == nullptr), return nullptr, "[queueId=%u] constructed failed", GetQueueId());
    return queue_->Pull();
}

void QueueHandler::Submit(QueueTask* task)
{
    FFRT_COND_DO_ERR((queue_ == nullptr), return, "cannot submit, [queueId=%u] constructed failed", GetQueueId());
    FFRT_COND_DO_ERR((task == nullptr), return, "input invalid, serial task is nullptr");

    // if qos not specified, qos of the queue is inherited by task
    if (task->GetQos() == qos_inherit || task->GetQos() == qos_default) {
        task->SetQos(qos_);
    }

    uint64_t gid = task->gid;
    task->Prepare();

    trafficRecord_.SubmitTraffic(this);

#if (FFRT_TRACE_RECORD_LEVEL < FFRT_TRACE_RECORD_LEVEL_1)
    if (queue_->GetQueueType() == ffrt_queue_eventhandler_adapter) {
        task->fromTid = ExecuteCtx::Cur()->tid;
    }
#endif

    // work after that schedule timeout is set for queue
    if (task->GetSchedTimeout() > 0) {
        AddSchedDeadline(task);
    }

    int ret = queue_->Push(task);
    if (ret == SUCC) {
        FFRT_LOGD("submit task[%lu] into %s", gid, name_.c_str());
        return;
    }
    if (ret == FAILED) {
        FFRT_SYSEVENT_LOGE("push task failed");
        return;
    }

    if (!isUsed_.load()) {
        isUsed_.store(true);
    }

    // activate queue
    if (task->GetDelay() == 0) {
        FFRT_LOGD("task [%llu] activate %s", gid, name_.c_str());
        {
            std::lock_guard lock(mutex_);
            UpdateCurTask(task);
        }
        TransferTask(task);
    } else {
        FFRT_LOGD("task [%llu] with delay [%llu] activate %s", gid, task->GetDelay(), name_.c_str());
        if (ret == INACTIVE) {
            queue_->Push(task);
        }
        TransferInitTask();
    }
    FFRTFacade::GetQMInstance().UpdateQueueInfo();
}

void QueueHandler::Cancel()
{
    FFRT_COND_DO_ERR((queue_ == nullptr), return, "cannot cancel, [queueId=%u] constructed failed", GetQueueId());
    std::lock_guard lock(mutex_);
    std::vector<QueueTask*> taskVec = queue_->GetHeadTask();
    for (auto& task : taskVec) {
        for (auto& curtask : curTaskVec_) {
            if (task == curtask) {
                curtask = nullptr;
            }
        }
    }
    int count = queue_->Remove();
    trafficRecord_.DoneTraffic(count);
}

void QueueHandler::CancelAndWait()
{
    FFRT_COND_DO_ERR((queue_ == nullptr), return, "cannot cancelAndWait, [queueId=%u] constructed failed",
        GetQueueId());

    {
        std::unique_lock lock(mutex_);
        for (auto& curTask : curTaskVec_) {
            if (curTask != nullptr && curTask->curStatus != TaskStatus::EXECUTING) {
                curTask = nullptr;
            }
        }
    }
    queue_->Stop();
    while (CheckExecutingTask() || queue_->GetActiveStatus() || deliverCnt_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(TASK_DONE_WAIT_UNIT));
        desWaitCnt_++;
        if (desWaitCnt_ == TASK_WAIT_COUNT) {
            std::lock_guard lock(mutex_);
            for (int i = 0; i < static_cast<int>(curTaskVec_.size()); i++) {
                FFRT_LOGI("Queue Destruct blocked for 5s, %s", GetDfxInfo(i).c_str());
            }
            desWaitCnt_ = 0;
        }
    }
}

bool QueueHandler::CheckExecutingTask()
{
    std::lock_guard lock(mutex_);
    for (const auto& curTask : curTaskVec_) {
        if (curTask != nullptr && curTask->curStatus == TaskStatus::EXECUTING) {
            return true;
        }
    }
    return false;
}

int QueueHandler::Cancel(const char* name)
{
    FFRT_COND_DO_ERR((queue_ == nullptr), return INACTIVE,
        "cannot cancel, [queueId=%u] constructed failed", GetQueueId());
    std::lock_guard lock(mutex_);
    std::vector<QueueTask*> taskVec = queue_->GetHeadTask();
    for (auto& task : taskVec) {
        for (auto& curtask : curTaskVec_) {
            if (task == curtask && task->IsMatch(name)) {
                curtask = nullptr;
            }
        }
    }
    int ret = queue_->Remove(name);
    if (ret <= 0) {
        FFRT_LOGD("cancel task %s failed, task may have been executed", name);
    } else {
        trafficRecord_.DoneTraffic(ret);
    }
    return ret > 0 ? SUCC : FAILED;
}

int QueueHandler::Cancel(QueueTask* task)
{
    FFRT_COND_DO_ERR((queue_ == nullptr), return INACTIVE,
         "cannot cancel, [queueId=%u] constructed failed", GetQueueId());
    FFRT_COND_DO_ERR((task == nullptr), return INACTIVE, "input invalid, serial task is nullptr");

    if (task->GetSchedTimeout() > 0) {
        RemoveSchedDeadline(task);
    }

    int ret = queue_->Remove(task);
    if (ret == SUCC) {
        FFRT_LOGD("cancel task[%llu] %s succ", task->gid, task->label.c_str());
        for (int i = 0; i < static_cast<int>(curTaskVec_.size()); i++) {
            std::lock_guard lock(mutex_);
            if (curTaskVec_[i] == task) {
                curTaskVec_[i] = nullptr;
                break;
            }
        }
        trafficRecord_.DoneTraffic();
        task->Cancel();
    } else {
        FFRT_LOGD("cancel task[%llu] %s failed, task may have been executed", task->gid, task->GetLabel().c_str());
    }
    return ret;
}

void QueueHandler::Dispatch(QueueTask* inTask)
{
    QueueTask* nextTask = nullptr;
    for (QueueTask* task = inTask; task != nullptr; task = nextTask) {
        // dfx watchdog
        SetTimeoutMonitor(task);
        SetCurTask(task);
        execTaskId_.store(task->gid);

        // run user task
        task->SetStatus(TaskStatus::EXECUTING);
        FFRT_LOGD("run task [gid=%llu], queueId=%u", task->gid, GetQueueId());
        auto f = reinterpret_cast<ffrt_function_header_t*>(task->func_storage);
        FFRTTraceRecord::TaskExecute(&(task->executeTime));
        if (task->GetSchedTimeout() > 0) {
            RemoveSchedDeadline(task);
        }

        uint64_t triggerTime{0};
        if (queue_->GetQueueType() == ffrt_queue_eventhandler_adapter) {
            triggerTime = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        }

        f->exec(f);

        if (queue_->GetQueueType() == ffrt_queue_eventhandler_adapter) {
            uint64_t completeTime = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
            reinterpret_cast<EventHandlerAdapterQueue*>(queue_.get())->PushHistoryTask(task, triggerTime, completeTime);
        }

        if (task != inTask) {
            task->SetStatus(TaskStatus::FINISH);
        }

        task->Finish();
        RemoveTimeoutMonitor(task);

        trafficRecord_.DoneTraffic();
        // run task batch
        nextTask = task->GetNextTask();
        {
            std::lock_guard lock(mutex_);
            curTaskVec_[task->curTaskIdx] = nextTask;
        }
        task->DecDeleteRef();
        if (nextTask == nullptr) {
            if (!queue_->IsOnLoop()) {
            execTaskId_.store(0);
                Deliver();
            }
        }
    }
    inTask->SetStatus(TaskStatus::FINISH);
}

void QueueHandler::Deliver()
{
    deliverCnt_.fetch_add(1);
    {
        // curtask has to be updated to headtask of whenmap before pull
        std::lock_guard lock(mutex_);
        std::vector<QueueTask*> taskMap = queue_->GetHeadTask();
        if (!taskMap.empty()) {
            std::unordered_set<QueueTask*> curTaskSet(curTaskVec_.begin(), curTaskVec_.end());
            for (auto& task : taskMap) {
                if (curTaskSet.find(task) == curTaskSet.end()) {
                    UpdateCurTask(task);
                    break;
                }
            }
        }
    }
    QueueTask* task = queue_->Pull();
    deliverCnt_.fetch_sub(1);
    if (task != nullptr) {
        SetCurTask(task);
        TransferTask(task);
    }
}

void QueueHandler::TransferTask(QueueTask* task)
{
    if (queue_->GetQueueType() == ffrt_queue_eventhandler_adapter) {
        reinterpret_cast<EventHandlerAdapterQueue*>(queue_.get())->SetCurrentRunningTask(task);
    }
    task->Ready();
}

void QueueHandler::TransferInitTask()
{
    std::function<void()> initFunc = [] {};
    auto f = create_function_wrapper(initFunc, ffrt_function_kind_queue);
    QueueTask* initTask = GetQueueTaskByFuncStorageOffset(f);
    new (initTask)ffrt::QueueTask(this);
    initTask->SetQos(qos_);
    trafficRecord_.SubmitTraffic(this);
    TransferTask(initTask);
}

void QueueHandler::SetTimeoutMonitor(QueueTask* task)
{
    if (timeout_ <= 0) {
        return;
    }

    task->IncDeleteRef();

    // set delayed worker callback
    auto timeoutWe = new (SimpleAllocator<WaitUntilEntry>::AllocMem()) WaitUntilEntry();
    timeoutWe->cb = ([this, task](WaitEntry* timeoutWe) {
        task->MonitorTaskStart();
        if (!task->GetFinishStatus()) {
            RunTimeOutCallback(task);
        }
        delayedCbCnt_.fetch_sub(1);
        task->DecDeleteRef();
        SimpleAllocator<WaitUntilEntry>::FreeMem(static_cast<WaitUntilEntry*>(timeoutWe));
    });

    // set delayed worker wakeup time
    std::chrono::microseconds timeout(timeout_);
    auto now = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::steady_clock::now());
    timeoutWe->tp = std::chrono::time_point_cast<std::chrono::steady_clock::duration>(now + timeout);
    task->SetMonitorTask(timeoutWe);

    if (!DelayedWakeup(timeoutWe->tp, timeoutWe, timeoutWe->cb, true)) {
        task->DecDeleteRef();
        SimpleAllocator<WaitUntilEntry>::FreeMem(timeoutWe);
        FFRT_LOGW("failed to set watchdog for task gid=%llu in %s with timeout [%llu us] ", task->gid,
            name_.c_str(), timeout_);
        return;
    }

    FFRT_LOGD("set watchdog of task gid=%llu of %s succ", task->gid, name_.c_str());
}

void QueueHandler::RemoveTimeoutMonitor(QueueTask* task)
{
    if (timeout_ <= 0 || task->IsMonitorTaskStart()) {
        return;
    }

    if (DelayedRemove(task->GetMonitorTask()->tp, task->GetMonitorTask())) {
        delayedCbCnt_.fetch_sub(1);
        task->DecDeleteRef();
        SimpleAllocator<WaitUntilEntry>::FreeMem(static_cast<WaitUntilEntry*>(task->GetMonitorTask()));
    }
}

void QueueHandler::RunTimeOutCallback(QueueTask* task)
{
    std::stringstream ss;
    std::string processNameStr = std::string(GetCurrentProcessName());
    ss << "[Serial_Queue_Timeout_Callback] process name:[" << processNameStr << "], serial queue:[" <<
        name_ << "], queueId:[" << GetQueueId() << "], serial task gid:[" << task->gid << "], task name:["
        << task->label << "], execution time exceeds[" << timeout_ << "] us";
    FFRT_LOGE("%s", ss.str().c_str());
    if (timeoutCb_ != nullptr) {
        delayedCbCnt_.fetch_add(1);
        FFRTFacade::GetDWInstance().SubmitAsyncTask([this] {
            timeoutCb_->exec(timeoutCb_);
            delayedCbCnt_.fetch_sub(1);
        });
    }
}

std::string QueueHandler::GetDfxInfo(int index) const
{
    std::stringstream ss;
    if (queue_ != nullptr && curTaskVec_[index] != nullptr) {
        TaskStatus curTaskStatus = curTaskVec_[index]->curStatus;
        uint64_t curTaskTime = curTaskVec_[index]->statusTime.load(std::memory_order_relaxed);
        TaskStatus preTaskStatus = curTaskVec_[index]->preStatus.load(std::memory_order_relaxed);
        ss << "Queue task: tskname[" << curTaskVec_[index]->label.c_str() << "], qname=[" << name_ <<
                "], with delay of[" <<  curTaskVec_[index]->GetDelay() << "]us, qos[" << curTaskVec_[index]->GetQos() <<
                "], current status[" << StatusToString(curTaskStatus) << "], start at[" <<
                FormatDateString4SteadyClock(curTaskTime) << "], last status[" << StatusToString(preTaskStatus)
                << "], type=[" << queue_->GetQueueType() << "]";
    } else {
        ss << "Current queue or task nullptr";
    }
    return ss.str();
}

std::pair<std::vector<uint64_t>, uint64_t> QueueHandler::EvaluateTaskTimeout(uint64_t timeoutThreshold,
    uint64_t timeoutUs, std::stringstream& ss)
{
    uint64_t whenmapTskCount = GetTaskCnt();
    std::lock_guard lock(mutex_);
    std::pair<std::vector<uint64_t>, uint64_t> curTaskInfo;
    uint64_t minTime = UINT64_MAX;
    for (int i = 0; i < static_cast<int>(curTaskVec_.size()); i++) {
        QueueTask* curTask = curTaskVec_[i];
        if (curTask == nullptr) {
            curTaskInfo.first.emplace_back(INVALID_GID);
            continue;
        }

        uint64_t curTaskTime = curTask->statusTime.load(std::memory_order_relaxed);
        if (curTaskTime == 0 || CheckDelayStatus()) {
            curTaskInfo.first.emplace_back(INVALID_GID);
            // Update the next inspection time if current task is delaying and there are still tasks in whenMap.
            // Otherwise, pause the monitor timer.
            if (whenmapTskCount > 0) {
                minTime = std::min(minTime, TimeStampCntvct());
            }
            continue;
        }

        if (curTaskTime > timeoutThreshold) {
            curTaskInfo.first.emplace_back(INVALID_GID);
            minTime = std::min(minTime, curTaskTime);
            continue;
        }

        TimeoutTask& timeoutTaskInfo = timeoutTaskVec_[i];
        if (curTask->gid == timeoutTaskInfo.taskGid &&
            curTask->curStatus == timeoutTaskInfo.taskStatus) {
                // Check if current timeout task needs to update timeout count.
                uint64_t nextSchedule = curTaskTime + timeoutUs * timeoutTaskInfo.timeoutCnt;
                if (nextSchedule < timeoutThreshold) {
                    timeoutTaskInfo.timeoutCnt += 1;
                } else {
                    curTaskInfo.first.emplace_back(INVALID_GID);
                    minTime = std::min(minTime, nextSchedule);
                    continue;
                }
        } else {
            timeoutTaskInfo.timeoutCnt = 1;
            timeoutTaskInfo.taskGid = curTask->gid;
            timeoutTaskInfo.taskStatus = curTask->curStatus;
        }

        // When the same task is reported multiple times, the next inspection time is updated by adding the
        // accumulated time based on the current status time.
        curTaskInfo.first.emplace_back(timeoutTaskInfo.taskGid);
        uint64_t nextTimeSchedule = curTaskTime + (timeoutUs * timeoutTaskInfo.timeoutCnt);
        minTime = std::min(minTime, nextTimeSchedule);

        if (ControlTimeoutFreq(timeoutTaskInfo.timeoutCnt)) {
            ReportTaskTimeout(timeoutUs, ss, i);
        }
    }

    curTaskInfo.second = minTime;
    return curTaskInfo;
}

bool QueueHandler::ControlTimeoutFreq(uint64_t timeoutCnt)
{
    return (timeoutCnt < 10) || (timeoutCnt < 100 && timeoutCnt % 10 == 0) || (timeoutCnt % 100 == 0);
}

void QueueHandler::ReportTaskTimeout(uint64_t timeoutUs, std::stringstream& ss, int index)
{
    ss.str("");
    ss << GetDfxInfo(index) << ", timeout for[" << timeoutUs / MIN_TIMEOUT_THRESHOLD_US <<
        "]s, reported count: " << timeoutTaskVec_[index].timeoutCnt;
    FFRT_LOGW("%s", ss.str().c_str());
#ifdef FFRT_SEND_EVENT
    if (timeoutTaskVec_[index].timeoutCnt == 1) {
        std::string senarioName = "Serial_Queue_Timeout";
        TaskTimeoutReport(ss, GetCurrentProcessName(), senarioName);
    }
#endif
}

bool QueueHandler::IsIdle()
{
    FFRT_COND_DO_ERR((queue_ == nullptr), return false, "[queueId=%u] constructed failed", GetQueueId());
    FFRT_COND_DO_ERR((queue_->GetQueueType() != ffrt_queue_eventhandler_adapter),
        return false, "[queueId=%u] type invalid", GetQueueId());

    return reinterpret_cast<EventHandlerAdapterQueue*>(queue_.get())->IsIdle();
}

void QueueHandler::SetEventHandler(void* eventHandler)
{
    FFRT_COND_DO_ERR((queue_ == nullptr), return, "[queueId=%u] constructed failed", GetQueueId());

    bool typeInvalid = (queue_->GetQueueType() != ffrt_queue_eventhandler_interactive) &&
        (queue_->GetQueueType() != ffrt_queue_eventhandler_adapter);
    FFRT_COND_DO_ERR(typeInvalid, return, "[queueId=%u] type invalid", GetQueueId());

    reinterpret_cast<EventHandlerInteractiveQueue*>(queue_.get())->SetEventHandler(eventHandler);
}

void* QueueHandler::GetEventHandler()
{
    FFRT_COND_DO_ERR((queue_ == nullptr), return nullptr, "[queueId=%u] constructed failed", GetQueueId());

    bool typeInvalid = (queue_->GetQueueType() != ffrt_queue_eventhandler_interactive) &&
        (queue_->GetQueueType() != ffrt_queue_eventhandler_adapter);
    FFRT_COND_DO_ERR(typeInvalid, return nullptr, "[queueId=%u] type invalid", GetQueueId());

    return reinterpret_cast<EventHandlerInteractiveQueue*>(queue_.get())->GetEventHandler();
}

int QueueHandler::Dump(const char* tag, char* buf, uint32_t len, bool historyInfo)
{
    FFRT_COND_DO_ERR((queue_ == nullptr), return -1, "[queueId=%u] constructed failed", GetQueueId());
    FFRT_COND_DO_ERR((queue_->GetQueueType() != ffrt_queue_eventhandler_adapter),
        return -1, "[queueId=%u] type invalid", GetQueueId());
    return reinterpret_cast<EventHandlerAdapterQueue*>(queue_.get())->Dump(tag, buf, len, historyInfo);
}

int QueueHandler::DumpSize(ffrt_inner_queue_priority_t priority)
{
    FFRT_COND_DO_ERR((queue_ == nullptr), return -1, "[queueId=%u] constructed failed", GetQueueId());
    FFRT_COND_DO_ERR((queue_->GetQueueType() != ffrt_queue_eventhandler_adapter),
        return -1, "[queueId=%u] type invalid", GetQueueId());
    return reinterpret_cast<EventHandlerAdapterQueue*>(queue_.get())->DumpSize(priority);
}

void QueueHandler::SendSchedTimer(TimePoint delay)
{
    we_->tp = delay;
    if (!DelayedWakeup(we_->tp, we_, we_->cb, true)) {
        FFRT_LOGW("failed to set delayedworker");
    }
}

void QueueHandler::CheckSchedDeadline()
{
    std::vector<std::pair<uint64_t, std::string>> timeoutTaskInfo;
    // Collecting Timeout Tasks
    {
        std::lock_guard lock(mutex_);
        uint64_t threshold = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count() + SCHED_TIME_ACC_ERROR_US;

        auto it = schedDeadline_.begin();
        uint64_t nextDeadline = UINT64_MAX;
        while (it != schedDeadline_.end()) {
            if (it->second < threshold) {
                timeoutTaskInfo.push_back(std::make_pair(it->first->gid, it->first->label));
                it = schedDeadline_.erase(it);
            } else {
                nextDeadline = std::min(nextDeadline, it->second);
                ++it;
            }
        }

        if (schedDeadline_.empty()) {
            initSchedTimer_ = false;
        } else {
            std::chrono::microseconds timeout(nextDeadline);
            TimePoint tp = std::chrono::time_point_cast<std::chrono::steady_clock::duration>(
                std::chrono::steady_clock::time_point() + timeout);
            FFRT_LOGI("queueId=%u set sched timer", GetQueueId());
            SendSchedTimer(tp);
        }
    }

    // Reporting Timeout Information
    if (!timeoutTaskInfo.empty()) {
        ReportTimeout(timeoutTaskInfo);
    }
}

void QueueHandler::AddSchedDeadline(QueueTask* task)
{
    // sched timeout only support serial queues, other queue types will be supported based on service requirements.
    if (queue_->GetQueueType() != ffrt_queue_serial) {
        return;
    }

    std::lock_guard lock(mutex_);
    schedDeadline_.insert({task, task->GetSchedTimeout() + task->GetUptime()});

    if (!initSchedTimer_) {
        if (we_ == nullptr) {
            we_ = new (SimpleAllocator<WaitUntilEntry>::AllocMem()) WaitUntilEntry();
            we_->cb = ([this](WaitEntry* we) { CheckSchedDeadline(); });
        }
        std::chrono::microseconds timeout(schedDeadline_[task]);
        TimePoint tp = std::chrono::time_point_cast<std::chrono::steady_clock::duration>(
            std::chrono::steady_clock::time_point() + timeout);
        SendSchedTimer(tp);
        initSchedTimer_ = true;
    }
}

void QueueHandler::RemoveSchedDeadline(QueueTask* task)
{
    std::lock_guard lock(mutex_);
    schedDeadline_.erase(task);
}

void QueueHandler::ReportTimeout(const std::vector<std::pair<uint64_t, std::string>>& timeoutTaskInfo)
{
    std::stringstream ss;
    ss << "Queue_Schedule_Timeout, queueId=" << GetQueueId() << ", timeout task gid: ";
    for (auto& info : timeoutTaskInfo) {
        ss << info.first << ", name " << info.second.c_str() << " ";
    }

    FFRT_LOGE("%s", ss.str().c_str());

    uint32_t queueId = GetQueueId();
    std::string ssStr = ss.str();
    if (ffrt_task_timeout_get_cb()) {
        FFRTFacade::GetDWInstance().SubmitAsyncTask([queueId, ssStr] {
            ffrt_task_timeout_cb func = ffrt_task_timeout_get_cb();
            if (func) {
                func(queueId, ssStr.c_str(), ssStr.size());
            }
        });
    }
}

void QueueHandler::SetCurTask(QueueTask* task)
{
    std::lock_guard lock(mutex_);
    if (task != nullptr) {
        curTaskVec_[task->curTaskIdx] = task;
    }
}

void QueueHandler::UpdateCurTask(QueueTask* task)
{
    for (int i = 0; i < static_cast<int>(curTaskVec_.size()); i++) {
        if (curTaskVec_[i] == nullptr) {
            curTaskVec_[i] = task;
            task->curTaskIdx = i;
            return;
        }
    }
}
} // namespace ffrt
