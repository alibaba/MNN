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
#include "c/queue_ext.h"
#include "cpp/queue.h"
#include "core/task_wrapper.h"
#include "util/event_handler_adapter.h"
#include "dm/dependence_manager.h"
#include "tm/queue_task.h"
#include "queue/queue_handler.h"
#include "util/common_const.h"

constexpr uint64_t MAX_TIMEOUT_US_COUNT = 1000000ULL * 100 * 60 * 60 * 24 * 365; // 100 year

using namespace std;
using namespace ffrt;

namespace {
inline void ResetTimeoutCb(ffrt::queue_attr_private* p)
{
    if (p->timeoutCb_ == nullptr) {
        return;
    }
    QueueTask* cbTask = GetQueueTaskByFuncStorageOffset(p->timeoutCb_);
    cbTask->DecDeleteRef();
    p->timeoutCb_ = nullptr;
}

inline QueueTask* ffrt_queue_submit_base(ffrt_queue_t queue, ffrt_function_header_t* f, bool withHandle,
    bool insertHead, const ffrt_task_attr_t* attr)
{
    FFRT_COND_DO_ERR(unlikely(queue == nullptr), return nullptr, "input invalid, queue == nullptr");
    FFRT_COND_DO_ERR(unlikely(f == nullptr), return nullptr, "input invalid, function header == nullptr");
    QueueHandler* handler = static_cast<QueueHandler*>(queue);
    ffrt::task_attr_private *p = reinterpret_cast<ffrt::task_attr_private *>(const_cast<ffrt_task_attr_t *>(attr));
    QueueTask* task = GetQueueTaskByFuncStorageOffset(f);
    new (task)ffrt::QueueTask(handler, p, insertHead);
    if (withHandle) {
        task->IncDeleteRef();
    }

    handler->Submit(task);
    return task;
}

constexpr uint64_t MIN_TRAFFIC_INTERVAL_US = 1000000;
constexpr uint64_t MAX_TRAFFIC_INTERVAL_US = 600000000;
constexpr uint64_t DEFAULT_TRAFFIC_INTERVAL_US = 6000000;
} // namespace

API_ATTRIBUTE((visibility("default")))
int ffrt_queue_attr_init(ffrt_queue_attr_t* attr)
{
    FFRT_COND_DO_ERR((attr == nullptr), return -1, "input invalid, attr == nullptr");
    static_assert(sizeof(ffrt::queue_attr_private) <= ffrt_queue_attr_storage_size,
        "size must be less than ffrt_queue_attr_storage_size");

    new (attr) ffrt::queue_attr_private();
    return 0;
}

API_ATTRIBUTE((visibility("default")))
void ffrt_queue_attr_destroy(ffrt_queue_attr_t* attr)
{
    FFRT_COND_DO_ERR((attr == nullptr), return, "input invalid, attr == nullptr");
    auto p = reinterpret_cast<ffrt::queue_attr_private*>(attr);
    ResetTimeoutCb(p);
    p->~queue_attr_private();
}

API_ATTRIBUTE((visibility("default")))
void ffrt_queue_attr_set_qos(ffrt_queue_attr_t* attr, ffrt_qos_t qos)
{
    FFRT_COND_DO_ERR((attr == nullptr), return, "input invalid, attr == nullptr");
    FFRT_COND_DO_ERR((ffrt::GetFuncQosMap() == nullptr), return, "input invalid, FuncQosMap has not regist");

    (reinterpret_cast<ffrt::queue_attr_private*>(attr))->qos_ = ffrt::GetFuncQosMap()(qos);
}

API_ATTRIBUTE((visibility("default")))
ffrt_qos_t ffrt_queue_attr_get_qos(const ffrt_queue_attr_t* attr)
{
    FFRT_COND_DO_ERR((attr == nullptr), return ffrt_qos_default, "input invalid, attr == nullptr");
    ffrt_queue_attr_t* p = const_cast<ffrt_queue_attr_t*>(attr);
    return (reinterpret_cast<ffrt::queue_attr_private*>(p))->qos_;
}

API_ATTRIBUTE((visibility("default")))
void ffrt_queue_attr_set_timeout(ffrt_queue_attr_t* attr, uint64_t timeout_us)
{
    FFRT_COND_DO_ERR((attr == nullptr), return, "input invalid, attr == nullptr");
    if (timeout_us < ONE_THOUSAND) {
        (reinterpret_cast<ffrt::queue_attr_private*>(attr))->timeout_ = ONE_THOUSAND;
        return;
    }

    if (timeout_us > MAX_TIMEOUT_US_COUNT) {
        FFRT_LOGW("timeout_us exceeds maximum allowed value %llu us. Clamping to %llu us.", timeout_us,
            MAX_TIMEOUT_US_COUNT);
        timeout_us = MAX_TIMEOUT_US_COUNT;
    }

    (reinterpret_cast<ffrt::queue_attr_private*>(attr))->timeout_ = timeout_us;
}

API_ATTRIBUTE((visibility("default")))
uint64_t ffrt_queue_attr_get_timeout(const ffrt_queue_attr_t* attr)
{
    FFRT_COND_DO_ERR((attr == nullptr), return 0, "input invalid, attr == nullptr");
    ffrt_queue_attr_t* p = const_cast<ffrt_queue_attr_t*>(attr);
    return (reinterpret_cast<ffrt::queue_attr_private*>(p))->timeout_;
}

API_ATTRIBUTE((visibility("default")))
void ffrt_queue_attr_set_callback(ffrt_queue_attr_t* attr, ffrt_function_header_t* f)
{
    FFRT_COND_DO_ERR((attr == nullptr), return, "input invalid, attr == nullptr");
    FFRT_COND_DO_ERR((f == nullptr), return, "input invalid, f == nullptr");
    ffrt::queue_attr_private* p = reinterpret_cast<ffrt::queue_attr_private*>(attr);
    ResetTimeoutCb(p);
    p->timeoutCb_ = f;
    // the memory of timeoutCb are managed in the form of QueueTask
    QueueTask* task = GetQueueTaskByFuncStorageOffset(f);
    new (task)ffrt::QueueTask(nullptr);
}

API_ATTRIBUTE((visibility("default")))
ffrt_function_header_t* ffrt_queue_attr_get_callback(const ffrt_queue_attr_t* attr)
{
    FFRT_COND_DO_ERR((attr == nullptr), return nullptr, "input invalid, attr == nullptr");
    ffrt_queue_attr_t* p = const_cast<ffrt_queue_attr_t*>(attr);
    return (reinterpret_cast<ffrt::queue_attr_private*>(p))->timeoutCb_;
}

API_ATTRIBUTE((visibility("default")))
void ffrt_queue_attr_set_max_concurrency(ffrt_queue_attr_t* attr, const int max_concurrency)
{
    FFRT_COND_DO_ERR((attr == nullptr), return, "input invalid, attr == nullptr");

    FFRT_COND_DO_ERR((max_concurrency <= 0), return,
        "max concurrency should be a valid value");

    (reinterpret_cast<ffrt::queue_attr_private*>(attr))->maxConcurrency_ = max_concurrency;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_queue_attr_get_max_concurrency(const ffrt_queue_attr_t* attr)
{
    FFRT_COND_DO_ERR((attr == nullptr), return 0, "input invalid, attr == nullptr");
    ffrt_queue_attr_t* p = const_cast<ffrt_queue_attr_t*>(attr);
    return (reinterpret_cast<ffrt::queue_attr_private*>(p))->maxConcurrency_;
}

API_ATTRIBUTE((visibility("default")))
void ffrt_queue_attr_set_thread_mode(ffrt_queue_attr_t* attr, bool mode)
{
    FFRT_COND_DO_ERR((attr == nullptr), return, "input invalid, attr == nullptr");

    (reinterpret_cast<ffrt::queue_attr_private*>(attr))->threadMode_ = mode;
}

API_ATTRIBUTE((visibility("default")))
bool ffrt_queue_attr_get_thread_mode(const ffrt_queue_attr_t* attr)
{
    FFRT_COND_DO_ERR((attr == nullptr), return 0, "input invalid, attr == nullptr");
    ffrt_queue_attr_t* p = const_cast<ffrt_queue_attr_t*>(attr);
    return (reinterpret_cast<ffrt::queue_attr_private*>(p))->threadMode_;
}

API_ATTRIBUTE((visibility("default")))
ffrt_queue_t ffrt_queue_create(ffrt_queue_type_t type, const char* name, const ffrt_queue_attr_t* attr)
{
    bool invalidType = (type == ffrt_queue_max) || (type < ffrt_queue_serial) ||
        (type >= static_cast<ffrt_queue_type_t>(ffrt_queue_inner_max));
    FFRT_COND_DO_ERR(invalidType, return nullptr, "input invalid, type unsupport");
    QueueHandler* handler = new (std::nothrow) QueueHandler(name, attr, type);
    FFRT_COND_DO_ERR((handler == nullptr), return nullptr, "failed to construct QueueHandler");
    return static_cast<ffrt_queue_t>(handler);
}

API_ATTRIBUTE((visibility("default")))
void ffrt_queue_destroy(ffrt_queue_t queue)
{
    FFRT_COND_DO_ERR((queue == nullptr), return, "input invalid, queue is nullptr");
    QueueHandler* handler = static_cast<QueueHandler*>(queue);
    delete handler;
}

API_ATTRIBUTE((visibility("default")))
void ffrt_queue_submit(ffrt_queue_t queue, ffrt_function_header_t* f, const ffrt_task_attr_t* attr)
{
    FFRT_COND_DO_ERR((f == nullptr), return, "input invalid, function is nullptr");
    QueueTask* task = ffrt_queue_submit_base(queue, f, false, false, attr);
    FFRT_COND_DO_ERR((task == nullptr), return, "failed to submit serial task");
}

API_ATTRIBUTE((visibility("default")))
void ffrt_queue_submit_f(ffrt_queue_t queue, ffrt_function_t func, void* arg, const ffrt_task_attr_t* attr)
{
    ffrt_function_header_t* f = ffrt_create_function_wrapper(func, nullptr, arg, ffrt_function_kind_queue);
    ffrt_queue_submit(queue, f, attr);
}

API_ATTRIBUTE((visibility("default")))
void ffrt_queue_submit_head(ffrt_queue_t queue, ffrt_function_header_t* f, const ffrt_task_attr_t* attr)
{
    FFRT_COND_DO_ERR((f == nullptr), return, "input invalid, function is nullptr");
    QueueTask* task = ffrt_queue_submit_base(queue, f, false, true, attr);
    FFRT_COND_DO_ERR((task == nullptr), return, "failed to submit serial task");
}

API_ATTRIBUTE((visibility("default")))
ffrt_task_handle_t ffrt_queue_submit_h(ffrt_queue_t queue, ffrt_function_header_t* f, const ffrt_task_attr_t* attr)
{
    FFRT_COND_DO_ERR((f == nullptr), return nullptr, "input invalid, function is nullptr");
    QueueTask* task = ffrt_queue_submit_base(queue, f, true, false, attr);
    FFRT_COND_DO_ERR((task == nullptr), return nullptr, "failed to submit serial task");
    return static_cast<ffrt_task_handle_t>(task);
}

API_ATTRIBUTE((visibility("default")))
ffrt_task_handle_t ffrt_queue_submit_h_f(ffrt_queue_t queue, ffrt_function_t func, void* arg,
    const ffrt_task_attr_t* attr)
{
    ffrt_function_header_t* f = ffrt_create_function_wrapper(func, nullptr, arg, ffrt_function_kind_queue);
    return ffrt_queue_submit_h(queue, f, attr);
}

API_ATTRIBUTE((visibility("default")))
ffrt_task_handle_t ffrt_queue_submit_head_h(ffrt_queue_t queue, ffrt_function_header_t* f, const ffrt_task_attr_t* attr)
{
    FFRT_COND_DO_ERR((f == nullptr), return nullptr, "input invalid, function is nullptr");
    QueueTask* task = ffrt_queue_submit_base(queue, f, true, true, attr);
    FFRT_COND_DO_ERR((task == nullptr), return nullptr, "failed to submit serial task");
    return static_cast<ffrt_task_handle_t>(task);
}

API_ATTRIBUTE((visibility("default")))
void ffrt_queue_wait(ffrt_task_handle_t handle)
{
    FFRT_COND_DO_ERR((handle == nullptr), return, "input invalid, task_handle is nullptr");
    QueueTask* task = static_cast<QueueTask*>(handle);
    task->Wait();
}

API_ATTRIBUTE((visibility("default")))
uint64_t ffrt_queue_get_task_cnt(ffrt_queue_t queue)
{
    FFRT_COND_DO_ERR(unlikely(queue == nullptr), return 0, "input invalid, queue == nullptr");
    QueueHandler* handler = static_cast<QueueHandler*>(queue);
    return handler->GetTaskCnt();
}

API_ATTRIBUTE((visibility("default")))
int ffrt_queue_cancel(ffrt_task_handle_t handle)
{
    FFRT_COND_DO_ERR((handle == nullptr), return -1, "input invalid, handle is nullptr");
    QueueTask* task = reinterpret_cast<QueueTask*>(static_cast<CPUEUTask*>(handle));
    QueueHandler* handler = task->GetHandler();
    FFRT_COND_DO_ERR((handler == nullptr), return -1, "task handler is nullptr");
    int ret = handler->Cancel(task);
    return ret;
}

API_ATTRIBUTE((visibility("default")))
void ffrt_queue_cancel_all(ffrt_queue_t queue)
{
    FFRT_COND_DO_ERR(unlikely(queue == nullptr), return, "input invalid, queue is nullptr");
    QueueHandler* handler = static_cast<QueueHandler*>(queue);
    handler->Cancel();
}

API_ATTRIBUTE((visibility("default")))
void ffrt_queue_cancel_and_wait(ffrt_queue_t queue)
{
    FFRT_COND_DO_ERR(unlikely(queue == nullptr), return, "input invalid, queue is nullptr");
    QueueHandler* handler = static_cast<QueueHandler*>(queue);
    handler->CancelAndWait();
}

API_ATTRIBUTE((visibility("default")))
int ffrt_queue_cancel_by_name(ffrt_queue_t queue, const char* name)
{
    FFRT_COND_DO_ERR(unlikely(queue == nullptr), return -1, "input invalid, queue is nullptr");
    FFRT_COND_DO_ERR(unlikely(name == nullptr), return -1, "input invalid, name is nullptr");
    QueueHandler* handler = static_cast<QueueHandler*>(queue);
    return handler->Cancel(name);
}

API_ATTRIBUTE((visibility("default")))
bool ffrt_queue_has_task(ffrt_queue_t queue, const char* name)
{
    FFRT_COND_DO_ERR(unlikely(queue == nullptr), return false, "input invalid, queue is nullptr");
    FFRT_COND_DO_ERR(unlikely(name == nullptr), return false, "input invalid, name is nullptr");
    QueueHandler* handler = static_cast<QueueHandler*>(queue);
    return handler->HasTask(name);
}

API_ATTRIBUTE((visibility("default")))
bool ffrt_queue_is_idle(ffrt_queue_t queue)
{
    FFRT_COND_DO_ERR(unlikely(queue == nullptr), return false, "input invalid, queue is nullptr");
    QueueHandler* handler = static_cast<QueueHandler*>(queue);
    return handler->IsIdle();
}

API_ATTRIBUTE((visibility("default")))
void ffrt_queue_set_eventhandler(ffrt_queue_t queue, void* eventhandler)
{
    FFRT_COND_DO_ERR(unlikely(queue == nullptr), return, "input invalid, queue is nullptr");
    QueueHandler* handler = static_cast<QueueHandler*>(queue);
    handler->SetEventHandler(eventhandler);
}

API_ATTRIBUTE((visibility("default")))
void* ffrt_get_current_queue_eventhandler(void)
{
    TaskBase* curTask = ffrt::ExecuteCtx::Cur()->task;
    if (curTask == nullptr || curTask->type != ffrt_queue_task) {
        FFRT_LOGW("Current task is nullptr or is not a serial task.");
        return nullptr;
    }

    QueueHandler* handler = static_cast<QueueTask*>(curTask)->GetHandler();
    FFRT_COND_DO_ERR((handler == nullptr), return nullptr, "task handler is nullptr");
    return handler->GetEventHandler();
}

API_ATTRIBUTE((visibility("default")))
int ffrt_concurrent_queue_wait_all(ffrt_queue_t queue)
{
    FFRT_COND_DO_ERR(unlikely(queue == nullptr), return -1, "input invalid, queue is nullptr");
    return static_cast<QueueHandler*>(queue)->WaitAll();
}

API_ATTRIBUTE((visibility("default")))
ffrt_queue_t ffrt_get_main_queue(void)
{
    FFRT_COND_DO_ERR((EventHandlerAdapter::Instance()->GetMainEventHandler == nullptr),
        return nullptr, "failed to load GetMainEventHandler Func.");
    static QueueHandler handler = QueueHandler("main_queue", nullptr, ffrt_queue_eventhandler_interactive);
    if (!handler.GetEventHandler()) {
        void* mainHandler = EventHandlerAdapter::Instance()->GetMainEventHandler();
        FFRT_COND_DO_ERR((mainHandler == nullptr), return nullptr, "failed to get main queue.");
        handler.SetEventHandler(mainHandler);
    }
    return static_cast<ffrt_queue_t>(&handler);
}

API_ATTRIBUTE((visibility("default")))
ffrt_queue_t ffrt_get_current_queue(void)
{
    FFRT_COND_DO_ERR((EventHandlerAdapter::Instance()->GetCurrentEventHandler == nullptr),
        return nullptr, "failed to load GetCurrentEventHandler Func.");
    void* workerHandler = EventHandlerAdapter::Instance()->GetCurrentEventHandler();
    FFRT_COND_DO_ERR((workerHandler == nullptr), return nullptr, "failed to get ArkTs worker queue.");
    QueueHandler *handler = new (std::nothrow) QueueHandler(
        "current_queue", nullptr, ffrt_queue_eventhandler_interactive);
    FFRT_COND_DO_ERR((handler == nullptr), return nullptr, "failed to construct WorkerThreadQueueHandler");
    handler->SetEventHandler(workerHandler);
    return static_cast<ffrt_queue_t>(handler);
}

API_ATTRIBUTE((visibility("default")))
int ffrt_queue_dump(ffrt_queue_t queue, const char* tag, char* buf, uint32_t len, bool history_info)
{
    FFRT_COND_DO_ERR((queue == nullptr), return -1, "input invalid, queue is nullptr");
    FFRT_COND_DO_ERR((tag == nullptr || buf == nullptr), return -1, "input invalid, tag or buf is nullptr");
    QueueHandler* handler = static_cast<QueueHandler*>(queue);
    return handler->Dump(tag, buf, len, history_info);
}

API_ATTRIBUTE((visibility("default")))
int ffrt_queue_size_dump(ffrt_queue_t queue, ffrt_inner_queue_priority_t priority)
{
    if (priority > ffrt_inner_queue_priority_idle || priority < ffrt_inner_queue_priority_vip) {
        FFRT_LOGE("priority:%d is not valid", priority);
        return -1;
    }
    FFRT_COND_DO_ERR((queue == nullptr), return -1, "input invalid, queue is nullptr");
    QueueHandler* handler = static_cast<QueueHandler*>(queue);
    return handler->DumpSize(priority);
}