/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <memory>
#include <vector>

#include "ffrt_inner.h"
#include "internal_inc/osal.h"
#ifdef USE_OHOS_QOS
#include "qos.h"
#else
#include "staging_qos/sched/qos.h"
#endif
#include "sched/task_scheduler.h"
#include "core/task_attr_private.h"
#include "internal_inc/config.h"
#include "eu/osattr_manager.h"
#include "eu/cpu_worker.h"
#include "dfx/log/ffrt_log_api.h"
#include "dfx/trace_record/ffrt_trace_record.h"
#include "dfx/watchdog/watchdog_util.h"
#include "eu/func_manager.h"
#include "util/ffrt_facade.h"
#include "util/slab.h"
#include "eu/sexecute_unit.h"
#include "core/task_io.h"
#include "core/task_wrapper.h"
#include "sync/poller.h"
#include "util/spmc_queue.h"
#include "tm/task_factory.h"
#include "tm/queue_task.h"
#include "util/common_const.h"
#include "util/ref_function_header.h"

constexpr uint64_t MAX_DELAY_US_COUNT = 1000000ULL * 100 * 60 * 60 * 24 * 365; // 100 year
constexpr uint64_t MAX_TIMEOUT_US_COUNT = 1000000ULL * 100 * 60 * 60 * 24 * 365; // 100 year

namespace ffrt {
inline void submit_impl(bool has_handle, ffrt_task_handle_t &handle, ffrt_function_header_t *f,
    const ffrt_deps_t *ins, const ffrt_deps_t *outs, const task_attr_private *attr)
{
    FFRTFacade::GetDMInstance().onSubmit(has_handle, handle, f, ins, outs, attr);
}

void DestroyFunctionWrapper(ffrt_function_header_t* f,
    ffrt_function_kind_t kind = ffrt_function_kind_general)
{
    if (f == nullptr || f->destroy == nullptr) {
        return;
    }
    f->destroy(f);
    // 按照kind转化为对应类型，释放内存
    if (kind == ffrt_function_kind_general) {
        CPUEUTask *t = reinterpret_cast<CPUEUTask *>(static_cast<uintptr_t>(
            static_cast<size_t>(reinterpret_cast<uintptr_t>(f)) - OFFSETOF(CPUEUTask, func_storage)));
        TaskFactory<CPUEUTask>::Free_(t);
        return;
    }
    QueueTask *t = reinterpret_cast<QueueTask *>(static_cast<uintptr_t>(
        static_cast<size_t>(reinterpret_cast<uintptr_t>(f)) - OFFSETOF(QueueTask, func_storage)));
    TaskFactory<QueueTask>::Free_(t);
}

API_ATTRIBUTE((visibility("default")))
void sync_io(int fd)
{
    FFRTFacade::GetPPInstance().WaitFdEvent(fd);
}

API_ATTRIBUTE((visibility("default")))
void set_trace_tag(const char* name)
{
    // !deprecated
    (void)name;
}

API_ATTRIBUTE((visibility("default")))
void clear_trace_tag()
{
    // !deprecated
}

void CreateDelayDeps(
    ffrt_task_handle_t &handle, const ffrt_deps_t *in_deps, const ffrt_deps_t *out_deps, task_attr_private *p)
{
    // setting dependences is not supportted for delayed task
    if (unlikely(((in_deps != nullptr) && (in_deps->len != 0)) || ((out_deps != nullptr) && (out_deps->len != 0)))) {
        FFRT_LOGE("delayed task do not support dependence, in_deps/out_deps ignored.");
    }

    // delay task
    uint64_t delayUs = p->delay_;
    std::function<void()> &&func = [delayUs]() {
        this_task::sleep_for(std::chrono::microseconds(delayUs));
        FFRT_LOGD("submit task delay time [%d us] has ended.", delayUs);
    };
    ffrt_function_header_t *delay_func = create_function_wrapper(std::move(func));
    submit_impl(true, handle, delay_func, nullptr, nullptr, reinterpret_cast<task_attr_private *>(p));
}
} // namespace ffrt

#ifdef __cplusplus
extern "C" {
#endif
API_ATTRIBUTE((visibility("default")))
int ffrt_task_attr_init(ffrt_task_attr_t *attr)
{
    if (unlikely(!attr)) {
        FFRT_LOGE("attr should be a valid address");
        return -1;
    }
    static_assert(sizeof(ffrt::task_attr_private) <= ffrt_task_attr_storage_size,
        "size must be less than ffrt_task_attr_storage_size");

    new (attr)ffrt::task_attr_private();
    return 0;
}

API_ATTRIBUTE((visibility("default")))
void ffrt_task_attr_destroy(ffrt_task_attr_t *attr)
{
    if (unlikely(!attr)) {
        FFRT_LOGE("attr should be a valid address");
        return;
    }
    auto p = reinterpret_cast<ffrt::task_attr_private *>(attr);
    p->~task_attr_private();
}

API_ATTRIBUTE((visibility("default")))
void ffrt_task_attr_set_name(ffrt_task_attr_t *attr, const char *name)
{
    if (unlikely(!attr || !name)) {
        FFRT_LOGE("invalid attr or name");
        return;
    }
    (reinterpret_cast<ffrt::task_attr_private *>(attr))->name_ = name;
}

API_ATTRIBUTE((visibility("default")))
const char *ffrt_task_attr_get_name(const ffrt_task_attr_t *attr)
{
    if (unlikely(!attr)) {
        FFRT_LOGE("attr should be a valid address");
        return nullptr;
    }
    ffrt_task_attr_t *p = const_cast<ffrt_task_attr_t *>(attr);
    return (reinterpret_cast<ffrt::task_attr_private *>(p))->name_.c_str();
}

API_ATTRIBUTE((visibility("default")))
void ffrt_task_attr_set_qos(ffrt_task_attr_t *attr, ffrt_qos_t qos)
{
    if (unlikely(!attr)) {
        FFRT_LOGE("attr should be a valid address");
        return;
    }
    if (ffrt::GetFuncQosMap() == nullptr) {
        FFRT_LOGE("FuncQosMap has not regist");
        return;
    }
    (reinterpret_cast<ffrt::task_attr_private *>(attr))->qos_ = ffrt::GetFuncQosMap()(qos);
}

API_ATTRIBUTE((visibility("default")))
ffrt_qos_t ffrt_task_attr_get_qos(const ffrt_task_attr_t *attr)
{
    if (unlikely(!attr)) {
        FFRT_LOGE("attr should be a valid address");
        return static_cast<int>(ffrt_qos_default);
    }
    ffrt_task_attr_t *p = const_cast<ffrt_task_attr_t *>(attr);
    return (reinterpret_cast<ffrt::task_attr_private *>(p))->qos_;
}

API_ATTRIBUTE((visibility("default")))
void ffrt_task_attr_set_delay(ffrt_task_attr_t *attr, uint64_t delay_us)
{
    if (unlikely(!attr)) {
        FFRT_LOGE("attr should be a valid address");
        return;
    }

    if (delay_us > MAX_DELAY_US_COUNT) {
        FFRT_LOGW("delay_us exceeds maximum allowed value %llu us. Clamping to %llu us.", delay_us, MAX_DELAY_US_COUNT);
        delay_us = MAX_DELAY_US_COUNT;
    }

    (reinterpret_cast<ffrt::task_attr_private *>(attr))->delay_ = delay_us;
}

API_ATTRIBUTE((visibility("default")))
uint64_t ffrt_task_attr_get_delay(const ffrt_task_attr_t *attr)
{
    if (unlikely(!attr)) {
        FFRT_LOGE("attr should be a valid address");
        return 0;
    }
    ffrt_task_attr_t *p = const_cast<ffrt_task_attr_t *>(attr);
    return (reinterpret_cast<ffrt::task_attr_private *>(p))->delay_;
}

API_ATTRIBUTE((visibility("default")))
void ffrt_task_attr_set_timeout(ffrt_task_attr_t *attr, uint64_t timeout_us)
{
    if (unlikely(!attr)) {
        FFRT_LOGE("attr should be a valid address");
        return;
    }
    if (timeout_us < ONE_THOUSAND) {
        (reinterpret_cast<ffrt::task_attr_private *>(attr))->timeout_ = ONE_THOUSAND;
        return;
    }

    if (timeout_us > MAX_TIMEOUT_US_COUNT) {
        FFRT_LOGW("timeout_us exceeds maximum allowed value %llu us. Clamping to %llu us.", timeout_us,
            MAX_TIMEOUT_US_COUNT);
        timeout_us = MAX_TIMEOUT_US_COUNT;
    }

    (reinterpret_cast<ffrt::task_attr_private *>(attr))->timeout_ = timeout_us;
}

API_ATTRIBUTE((visibility("default")))
uint64_t ffrt_task_attr_get_timeout(const ffrt_task_attr_t *attr)
{
    if (unlikely(!attr)) {
        FFRT_LOGE("attr should be a valid address");
        return 0;
    }
    ffrt_task_attr_t *p = const_cast<ffrt_task_attr_t *>(attr);
    return (reinterpret_cast<ffrt::task_attr_private *>(p))->timeout_;
}


API_ATTRIBUTE((visibility("default")))
void ffrt_task_attr_set_notify_worker(ffrt_task_attr_t* attr, bool notify)
{
    if (unlikely(!attr)) {
        FFRT_LOGE("attr should be a valid address");
        return;
    }
    (reinterpret_cast<ffrt::task_attr_private *>(attr))->notifyWorker_ = notify;
}

API_ATTRIBUTE((visibility("default")))
void ffrt_task_attr_set_queue_priority(ffrt_task_attr_t* attr, ffrt_queue_priority_t priority)
{
    if (unlikely(!attr)) {
        FFRT_LOGE("attr should be a valid address");
        return;
    }

    // eventhandler inner priority is one more than the kits priority
    int prio = static_cast<int>(priority);
    if (prio < static_cast<int>(ffrt_queue_priority_immediate) ||
        prio > static_cast<int>(ffrt_queue_priority_idle) + 1) {
        FFRT_LOGE("priority should be a valid priority");
        return;
    }

    (reinterpret_cast<ffrt::task_attr_private *>(attr))->prio_ = priority;
}

API_ATTRIBUTE((visibility("default")))
ffrt_queue_priority_t ffrt_task_attr_get_queue_priority(const ffrt_task_attr_t* attr)
{
    if (unlikely(!attr)) {
        FFRT_LOGE("attr should be a valid address");
        return ffrt_queue_priority_immediate;
    }
    ffrt_task_attr_t *p = const_cast<ffrt_task_attr_t *>(attr);
    return static_cast<ffrt_queue_priority_t>((reinterpret_cast<ffrt::task_attr_private *>(p))->prio_);
}

API_ATTRIBUTE((visibility("default")))
void ffrt_task_attr_set_stack_size(ffrt_task_attr_t* attr, uint64_t size)
{
    if (unlikely(!attr)) {
        FFRT_LOGE("attr should be a valid address");
        return;
    }
    (reinterpret_cast<ffrt::task_attr_private *>(attr))->stackSize_ = size;
}

API_ATTRIBUTE((visibility("default")))
uint64_t ffrt_task_attr_get_stack_size(const ffrt_task_attr_t* attr)
{
    if (unlikely(!attr)) {
        FFRT_LOGE("attr should be a valid address");
        return 0;
    }
    return (reinterpret_cast<const ffrt::task_attr_private *>(attr))->stackSize_;
}

// submit
API_ATTRIBUTE((visibility("default")))
void *ffrt_alloc_auto_managed_function_storage_base(ffrt_function_kind_t kind)
{
    if (kind == ffrt_function_kind_general) {
        return ffrt::TaskFactory<ffrt::CPUEUTask>::Alloc()->func_storage;
    }
    return ffrt::TaskFactory<ffrt::QueueTask>::Alloc()->func_storage;
}

API_ATTRIBUTE((visibility("default")))
void ffrt_submit_base(ffrt_function_header_t *f, const ffrt_deps_t *in_deps, const ffrt_deps_t *out_deps,
    const ffrt_task_attr_t *attr)
{
    if (unlikely(!f)) {
        FFRT_LOGE("function handler should not be empty");
        return;
    }
    ffrt_task_handle_t handle;
    ffrt::task_attr_private *p = reinterpret_cast<ffrt::task_attr_private *>(const_cast<ffrt_task_attr_t *>(attr));
    if (likely(attr == nullptr || ffrt_task_attr_get_delay(attr) == 0)) {
        ffrt::submit_impl(false, handle, f, in_deps, out_deps, p);
        return;
    }

    // task after delay
    ffrt_task_handle_t delay_handle;
    uint64_t timeout = p->timeout_;
    p->timeout_ = 0;
    p->isDelaying_ = true;
    ffrt::CreateDelayDeps(delay_handle, in_deps, out_deps, p);
    p->isDelaying_ = false;
    p->timeout_ = timeout;
    std::vector<ffrt_dependence_t> deps = {{ffrt_dependence_task, delay_handle}};
    ffrt_deps_t delay_deps {static_cast<uint32_t>(deps.size()), deps.data()};
    ffrt::submit_impl(false, handle, f, &delay_deps, nullptr, p);
    ffrt_task_handle_destroy(delay_handle);
}

API_ATTRIBUTE((visibility("default")))
void ffrt_submit_f(ffrt_function_t func, void* arg, const ffrt_deps_t* in_deps, const ffrt_deps_t* out_deps,
    const ffrt_task_attr_t* attr)
{
    ffrt_function_header_t* f = ffrt_create_function_wrapper(func, nullptr, arg, ffrt_function_kind_general);
    ffrt_submit_base(f, in_deps, out_deps, attr);
}

API_ATTRIBUTE((visibility("default")))
ffrt_task_handle_t ffrt_submit_h_base(ffrt_function_header_t *f, const ffrt_deps_t *in_deps,
    const ffrt_deps_t *out_deps, const ffrt_task_attr_t *attr)
{
    if (unlikely(!f)) {
        FFRT_LOGE("function handler should not be empty");
        return nullptr;
    }
    ffrt_task_handle_t handle = nullptr;
    ffrt::task_attr_private *p = reinterpret_cast<ffrt::task_attr_private *>(const_cast<ffrt_task_attr_t *>(attr));
    if (likely(attr == nullptr || ffrt_task_attr_get_delay(attr) == 0)) {
        ffrt::submit_impl(true, handle, f, in_deps, out_deps, p);
        return handle;
    }

    // task after delay
    ffrt_task_handle_t delay_handle = nullptr;
    uint64_t timeout = p->timeout_;
    p->timeout_ = 0;
    ffrt::CreateDelayDeps(delay_handle, in_deps, out_deps, p);
    p->timeout_ = timeout;
    std::vector<ffrt_dependence_t> deps = {{ffrt_dependence_task, delay_handle}};
    ffrt_deps_t delay_deps {static_cast<uint32_t>(deps.size()), deps.data()};
    ffrt::submit_impl(true, handle, f, &delay_deps, nullptr, p);
    ffrt_task_handle_destroy(delay_handle);
    return handle;
}

API_ATTRIBUTE((visibility("default")))
ffrt_task_handle_t ffrt_submit_h_f(ffrt_function_t func, void* arg, const ffrt_deps_t* in_deps,
    const ffrt_deps_t* out_deps, const ffrt_task_attr_t* attr)
{
    ffrt_function_header_t* f = ffrt_create_function_wrapper(func, nullptr, arg, ffrt_function_kind_general);
    return ffrt_submit_h_base(f, in_deps, out_deps, attr);
}

API_ATTRIBUTE((visibility("default")))
uint32_t ffrt_task_handle_inc_ref(ffrt_task_handle_t handle)
{
    if (handle == nullptr) {
        FFRT_LOGE("input task handle is invalid");
        return -1;
    }
    return static_cast<ffrt::CPUEUTask*>(handle)->IncDeleteRef();
}

API_ATTRIBUTE((visibility("default")))
uint32_t ffrt_task_handle_dec_ref(ffrt_task_handle_t handle)
{
    if (handle == nullptr) {
        FFRT_LOGE("input task handle is invalid");
        return -1;
    }
    return static_cast<ffrt::CPUEUTask*>(handle)->DecDeleteRef();
}

API_ATTRIBUTE((visibility("default")))
void ffrt_task_handle_destroy(ffrt_task_handle_t handle)
{
    ffrt_task_handle_dec_ref(handle);
}

API_ATTRIBUTE((visibility("default")))
uint64_t ffrt_task_handle_get_id(ffrt_task_handle_t handle)
{
    FFRT_COND_DO_ERR((handle == nullptr), return 0, "input task handle is invalid");
    return static_cast<ffrt::TaskBase*>(handle)->gid;
}

// wait
API_ATTRIBUTE((visibility("default")))
void ffrt_wait_deps(const ffrt_deps_t *deps)
{
    if (unlikely(!deps)) {
        FFRT_LOGE("deps should not be empty");
        return;
    }
    std::vector<ffrt_dependence_t> v(deps->len);
    for (uint64_t i = 0; i < deps->len; ++i) {
        v[i] = deps->items[i];
    }
    ffrt_deps_t d = { deps->len, v.data() };
    ffrt::FFRTFacade::GetDMInstance().onWait(&d);
}

API_ATTRIBUTE((visibility("default")))
void ffrt_wait()
{
    ffrt::FFRTFacade::GetDMInstance().onWait();
}

API_ATTRIBUTE((visibility("default")))
int ffrt_set_cgroup_attr(ffrt_qos_t qos, ffrt_os_sched_attr *attr)
{
    if (unlikely(!attr)) {
        FFRT_LOGE("attr should not be empty");
        return -1;
    }
    if (ffrt::GetFuncQosMap() == nullptr) {
        FFRT_LOGE("FuncQosMap has not regist");
        return -1;
    }
    ffrt::QoS _qos = ffrt::GetFuncQosMap()(qos);
    return ffrt::OSAttrManager::Instance()->UpdateSchedAttr(_qos, attr);
}

API_ATTRIBUTE((visibility("default")))
void ffrt_restore_qos_config()
{
    ffrt::FFRTFacade::GetEUInstance().RestoreThreadConfig();
}

API_ATTRIBUTE((visibility("default")))
int ffrt_set_cpu_worker_max_num(ffrt_qos_t qos, uint32_t num)
{
    if (num == 0 || num > ffrt::QOS_WORKER_MAXNUM) {
        FFRT_LOGE("qos[%d] worker num[%u] is invalid.", qos, num);
        return -1;
    }
    if (ffrt::GetFuncQosMap() == nullptr) {
        FFRT_LOGE("FuncQosMap has not regist");
        return -1;
    }
    ffrt::QoS _qos = ffrt::GetFuncQosMap()(qos);
    if (((qos != ffrt::qos_default) && (_qos() == ffrt::qos_default)) || (qos <= ffrt::qos_inherit)) {
        FFRT_LOGE("qos[%d] is invalid.", qos);
        return -1;
    }
    return ffrt::FFRTFacade::GetEUInstance().SetWorkerMaxNum(_qos, num);
}

API_ATTRIBUTE((visibility("default")))
void ffrt_notify_workers(ffrt_qos_t qos, int number)
{
    if (qos < ffrt::QoS::Min() || qos >= ffrt::QoS::Max() || number <= 0) {
        FFRT_LOGE("qos [%d] or number [%d] or is invalid.", qos, number);
        return;
    }

    ffrt::FFRTFacade::GetEUInstance().NotifyWorkers(qos, number);
}

API_ATTRIBUTE((visibility("default")))
ffrt_error_t ffrt_set_worker_stack_size(ffrt_qos_t qos, size_t stack_size)
{
    if (qos < ffrt::QoS::Min() || qos >= ffrt::QoS::Max() || stack_size < PTHREAD_STACK_MIN) {
        FFRT_LOGE("qos [%d] or stack size [%d] is invalid.", qos, stack_size);
        return ffrt_error_inval;
    }

    if (ffrt::FFRTFacade::GetEUInstance().SetWorkerStackSize(ffrt::QoS(qos), stack_size) != 0) {
        return ffrt_error;
    }

    return ffrt_success;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_this_task_update_qos(ffrt_qos_t qos)
{
    if (ffrt::GetFuncQosMap() == nullptr) {
        FFRT_LOGE("FuncQosMap has not regist");
        return 1;
    }
    ffrt::QoS _qos = ffrt::GetFuncQosMap()(qos);
    auto curTask = ffrt::ExecuteCtx::Cur()->task;
    if (curTask == nullptr) {
        FFRT_SYSEVENT_LOGW("task is nullptr");
        return 1;
    }

    FFRT_COND_DO_ERR((curTask->type != ffrt_normal_task), return 1, "update qos task type invalid");
    if (_qos() == curTask->qos_) {
        FFRT_LOGW("the target qos is equal to current qos, no need update");
        return 0;
    }

    curTask->SetQos(_qos);
    ffrt_yield();

    return 0;
}

API_ATTRIBUTE((visibility("default")))
ffrt_qos_t ffrt_this_task_get_qos(void)
{
    if (ffrt::ExecuteCtx::Cur()->task == nullptr) {
        FFRT_LOGW("task is nullptr");
        return static_cast<int>(ffrt_qos_default);
    }
    return ffrt::ExecuteCtx::Cur()->qos();
}

API_ATTRIBUTE((visibility("default")))
uint64_t ffrt_this_task_get_id()
{
    auto curTask = ffrt::ExecuteCtx::Cur()->task;
    if (curTask == nullptr) {
        return 0;
    }

    if (curTask->type == ffrt_normal_task) {
        return curTask->gid;
    } else if (curTask->type == ffrt_queue_task) {
        return reinterpret_cast<ffrt::QueueTask*>(curTask)->GetHandler()->GetExecTaskId();
    }

    return 0;
}

API_ATTRIBUTE((visibility("default")))
int64_t ffrt_this_queue_get_id()
{
    auto curTask = ffrt::ExecuteCtx::Cur()->task;
    if (curTask == nullptr || curTask->type != ffrt_queue_task) {
        // not serial queue task
        return -1;
    }

    ffrt::QueueTask* task = static_cast<ffrt::QueueTask*>(curTask);
    return task->GetQueueId();
}

API_ATTRIBUTE((visibility("default")))
int ffrt_skip(ffrt_task_handle_t handle)
{
    FFRT_COND_DO_ERR((handle == nullptr), return ffrt_error_inval, "input ffrt task handle is invalid.");
    return ffrt::FFRTFacade::GetDMInstance().onSkip(handle);
}

API_ATTRIBUTE((visibility("default")))
void ffrt_executor_task_submit(ffrt_executor_task_t* task, const ffrt_task_attr_t* attr)
{
    if (task == nullptr) {
        FFRT_LOGE("function handler should not be empty");
        return;
    }
    ffrt::task_attr_private* p = reinterpret_cast<ffrt::task_attr_private *>(const_cast<ffrt_task_attr_t *>(attr));
    if (likely(attr == nullptr || ffrt_task_attr_get_delay(attr) == 0)) {
        ffrt::FFRTFacade::GetDMInstance().onSubmitUV(task, p);
        return;
    }
    FFRT_LOGE("uv function does not support delay");
}

API_ATTRIBUTE((visibility("default")))
void ffrt_executor_task_register_func(ffrt_executor_task_func func, ffrt_executor_task_type_t type)
{
    FFRT_COND_DO_ERR((func == nullptr), return, "function handler should not be empty.");
    ffrt::FuncManager* func_mg = ffrt::FuncManager::Instance();
    func_mg->insert(type, func);
}

API_ATTRIBUTE((visibility("default")))
int ffrt_executor_task_cancel(ffrt_executor_task_t* task, const ffrt_qos_t qos)
{
    FFRT_COND_DO_ERR((qos == ffrt::qos_inherit), return 0, "Level incorrect");
    FFRT_COND_DO_ERR((task == nullptr), return 0, "libuv task is NULL");

    ffrt::Scheduler* sch = ffrt::FFRTFacade::GetSchedInstance();
    bool ret = sch->CancelUVWork(task, qos);
    if (ret) {
        ffrt::FFRTTraceRecord::TaskCancel<ffrt_uv_task>(qos);
    }
    return static_cast<int>(ret);
}

API_ATTRIBUTE((visibility("default")))
void* ffrt_get_cur_task(void)
{
    if (ffrt::IsCoTask(ffrt::ExecuteCtx::Cur()->task)) {
        return ffrt::ExecuteCtx::Cur()->task;
    }
    return nullptr;
}

API_ATTRIBUTE((visibility("default")))
bool ffrt_get_current_coroutine_stack(void** stack_addr, size_t* size)
{
    if (stack_addr == nullptr || size == nullptr) {
        return false;
    }

    if (!ffrt::USE_COROUTINE) {
        return false;
    }

    // init is false to avoid the crash issue caused by nested calls to malloc during initialization.
    auto ctx = ffrt::ExecuteCtx::Cur(false);
    if (ctx == nullptr) {
        return false;
    }

    auto curTask = ffrt::ExecuteCtx::Cur()->task;
    if (IsCoTask(curTask)) {
        auto co = static_cast<ffrt::CoTask*>(curTask)->coRoutine;
        if (co) {
            *size = co->stkMem.size;
            *stack_addr = GetCoStackAddr(co);
            return true;
        }
    }
    return false;
}

API_ATTRIBUTE((visibility("default")))
void ffrt_task_attr_set_local(ffrt_task_attr_t* attr, bool task_local)
{
    if (unlikely(!attr)) {
        FFRT_LOGE("attr should be a valid address");
        return;
    }
    (reinterpret_cast<ffrt::task_attr_private *>(attr))->taskLocal_ = task_local;
}

API_ATTRIBUTE((visibility("default")))
bool ffrt_task_attr_get_local(ffrt_task_attr_t* attr)
{
    if (unlikely(!attr)) {
        FFRT_LOGE("attr should be a valid address");
        return false;
    }
    return (reinterpret_cast<ffrt::task_attr_private *>(attr))->taskLocal_;
}

API_ATTRIBUTE((visibility("default")))
pthread_t ffrt_task_get_tid(void* task_handle)
{
    if (task_handle == nullptr) {
        FFRT_LOGE("invalid task handle");
        return 0;
    }

    auto task = reinterpret_cast<ffrt::CPUEUTask*>(task_handle);
    return task->runningTid.load();
}

API_ATTRIBUTE((visibility("default")))
uint64_t ffrt_get_cur_cached_task_id(void)
{
    uint64_t gid = ffrt_this_task_get_id();
    if (gid == 0) {
        return ffrt::ExecuteCtx::Cur()->lastGid_;
    }

    return gid;
}

API_ATTRIBUTE((visibility("default")))
int ffrt_enable_worker_escape(uint64_t one_stage_interval_ms, uint64_t two_stage_interval_ms,
    uint64_t three_stage_interval_ms, uint64_t one_stage_worker_num, uint64_t two_stage_worker_num)
{
    return ffrt::FFRTFacade::GetEUInstance().SetEscapeEnable(one_stage_interval_ms, two_stage_interval_ms,
        three_stage_interval_ms, one_stage_worker_num, two_stage_worker_num);
}

API_ATTRIBUTE((visibility("default")))
void ffrt_disable_worker_escape(void)
{
    ffrt::FFRTFacade::GetEUInstance().SetEscapeDisable();
}

API_ATTRIBUTE((visibility("default")))
void ffrt_set_sched_mode(ffrt_qos_t qos, ffrt_sched_mode mode)
{
    if (qos < ffrt::QoS::Min() || qos >= ffrt::QoS::Max()) {
        FFRT_LOGE("Currently, the energy saving mode is unavailable or qos [%d] is invalid..", qos);
        return;
    }
    ffrt::FFRTFacade::GetEUInstance().SetSchedMode(ffrt::QoS(qos), static_cast<ffrt::sched_mode_type>(mode));
}
#ifdef __cplusplus
}
#endif
