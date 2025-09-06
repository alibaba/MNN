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

#include "eu/co_routine.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <securec.h>
#include <string>
#include <sys/mman.h>
#include "util/cpu_boost_wrapper.h"
#include "dfx/trace/ffrt_trace.h"
#include "dm/dependence_manager.h"
#include "core/entity.h"
#include "tm/queue_task.h"
#include "sched/scheduler.h"
#include "sync/sync.h"
#include "util/slab.h"
#include "sched/sched_deadline.h"
#include "sync/perf_counter.h"
#include "dfx/bbox/bbox.h"
#include "dfx/trace_record/ffrt_trace_record.h"
#include "eu/co_routine_factory.h"
#include "util/ffrt_facade.h"
#ifdef FFRT_TASK_LOCAL_ENABLE
#include "pthread_ffrt.h"
#endif
#ifdef FFRT_ASYNC_STACKTRACE
#include "dfx/async_stack/ffrt_async_stack.h"
#ifdef FFRT_TASK_LOCAL_ENABLE
#include "pthread_ffrt.h"
#endif
#endif

#ifdef FFRT_ENABLE_HITRACE_CHAIN
#include "dfx/trace/ffrt_trace_chain.h"
#endif

using namespace ffrt;

static inline void CoStackCheck(CoRoutine* co)
{
    if (unlikely(co->stkMem.magic != STACK_MAGIC)) {
        FFRT_SYSEVENT_LOGE("sp offset:%llx.\n", co->stkMem.stk +
            co->stkMem.size - co->ctx.storage[FFRT_REG_SP]);
        FFRT_SYSEVENT_LOGE("stack over flow, check local variable in you tasks"
            " or use api 'ffrt_task_attr_set_stack_size'.\n");
        if (ExecuteCtx::Cur()->task != nullptr) {
            auto curTask = ExecuteCtx::Cur()->task;
            FFRT_SYSEVENT_LOGE("task name[%s], gid[%lu], submit_tid[%d]",
                curTask->GetLabel().c_str(), curTask->gid, curTask->fromTid);
        }
        abort();
    }
}

extern pthread_key_t g_executeCtxTlsKey;
pthread_key_t g_coThreadTlsKey = 0;
pthread_once_t g_coThreadTlsKeyOnce = PTHREAD_ONCE_INIT;
void CoEnvDestructor(void* args)
{
    auto coEnv = static_cast<CoRoutineEnv*>(args);
    if (coEnv) {
        delete coEnv;
    }
}

void MakeCoEnvTlsKey()
{
    pthread_key_create(&g_coThreadTlsKey, CoEnvDestructor);
}

CoRoutineEnv* GetCoEnv()
{
    CoRoutineEnv* coEnv = nullptr;
    pthread_once(&g_coThreadTlsKeyOnce, MakeCoEnvTlsKey);

    void *curTls = pthread_getspecific(g_coThreadTlsKey);
    if (curTls != nullptr) {
        coEnv = reinterpret_cast<CoRoutineEnv *>(curTls);
    } else {
        coEnv = new CoRoutineEnv();
        pthread_setspecific(g_coThreadTlsKey, coEnv);
    }
    return coEnv;
}

#ifdef FFRT_TASK_LOCAL_ENABLE
namespace {
bool IsTaskLocalEnable(ffrt::CoTask* task)
{
    if ((task->type != ffrt_normal_task)) {
        return false;
    }

    TaskLocalAttr* attr = static_cast<CPUEUTask*>(task)->tlsAttr;

    if (attr == nullptr || !attr->taskLocal) {
        return false;
    }

    if (attr->tsd == nullptr) {
        FFRT_SYSEVENT_LOGE("taskLocal enabled but task tsd invalid");
        return false;
    }

    return true;
}

void InitWorkerTsdValueToTask(void** taskTsd)
{
    const pthread_key_t updKeyMap[] = {g_executeCtxTlsKey, g_coThreadTlsKey};
    auto threadTsd = pthread_gettsd();
    for (const auto& key : updKeyMap) {
        FFRT_UNLIKELY_COND_DO_ABORT(key <= 0, "FFRT abort: key[%u] invalid", key);
        auto addr = threadTsd[key];
        if (addr) {
            taskTsd[key] = addr;
        }
    }
}

void SwitchTsdAddrToTask(ffrt::CPUEUTask* task)
{
    auto threadTsd = pthread_gettsd();
    task->tlsAttr->threadTsd = threadTsd;
    pthread_settsd(task->tlsAttr->tsd);
}

void SwitchTsdToTask(ffrt::CoTask* task)
{
    if (!IsTaskLocalEnable(task)) {
        return;
    }

    CPUEUTask* cpuTask = static_cast<CPUEUTask*>(task);

    InitWorkerTsdValueToTask(cpuTask->tlsAttr->tsd);

    SwitchTsdAddrToTask(cpuTask);

    cpuTask->runningTid.store(pthread_self());
    FFRT_LOGD("switch tsd to task Success");
}

bool SwitchTsdAddrToThread(ffrt::CPUEUTask* task)
{
    if (!task->tlsAttr->threadTsd) {
        FFRT_SYSEVENT_LOGE("threadTsd is null");
        return false;
    }
    pthread_settsd(task->tlsAttr->threadTsd);
    task->tlsAttr->threadTsd = nullptr;
    return true;
}

void UpdateWorkerTsdValueToThread(void** taskTsd)
{
    const pthread_key_t updKeyMap[] = {g_executeCtxTlsKey, g_coThreadTlsKey};
    auto threadTsd = pthread_gettsd();
    for (const auto& key : updKeyMap) {
        FFRT_UNLIKELY_COND_DO_ABORT(key <= 0, "FFRT abort: key[%u] invalid", key);
        auto threadVal = threadTsd[key];
        auto taskVal = taskTsd[key];
        if (!threadVal && taskVal) {
            threadTsd[key] = taskVal;
        } else {
            FFRT_UNLIKELY_COND_DO_ABORT((threadVal && taskVal && (threadVal != taskVal)),
                "FFRT abort: mismatch key=[%u]", key);
            FFRT_UNLIKELY_COND_DO_ABORT((threadVal && !taskVal),
                "FFRT abort: unexpected: thread exists but task not exists, key=[%u]", key);
        }
        taskTsd[key] = nullptr;
    }
}

void SwitchTsdToThread(ffrt::CoTask* task)
{
    if (!IsTaskLocalEnable(task)) {
        return;
    }

    CPUEUTask* cpuTask = static_cast<CPUEUTask*>(task);

    if (!SwitchTsdAddrToThread(cpuTask)) {
        return;
    }

    UpdateWorkerTsdValueToThread(cpuTask->tlsAttr->tsd);

    cpuTask->runningTid.store(0);
    FFRT_LOGD("switch tsd to thread Success");
}

void TaskTsdRunDtors(ffrt::CPUEUTask* task)
{
    SwitchTsdAddrToTask(task);
    pthread_tsd_run_dtors();
    SwitchTsdAddrToThread(task);
}
} // namespace

void TaskTsdDeconstruct(ffrt::CPUEUTask* task)
{
    if (!IsTaskLocalEnable(task)) {
        return;
    }

    TaskTsdRunDtors(task);
    if (task->tlsAttr->tsd != nullptr) {
        free(task->tlsAttr->tsd);
        task->tlsAttr->tsd = nullptr;
        task->tlsAttr->taskLocal = false;
    }
    FFRT_LOGD("tsd deconstruct done, task[%lu], name[%s]", task->gid, task->GetLabel().c_str());
}
#endif

static inline void CoSwitch(CoCtx* from, CoCtx* to)
{
    co2_switch_context(from, to);
}

static inline void CoExit(CoRoutine* co, bool isNormalTask)
{
#ifdef FFRT_ENABLE_HITRACE_CHAIN
    TraceChainAdapter::Instance().HiTraceChainClearId();
#endif
#ifdef FFRT_TASK_LOCAL_ENABLE
    if (isNormalTask) {
        SwitchTsdToThread(co->task);
    }
#endif
    CoStackCheck(co);
#ifdef ASAN_MODE
    /* co to thread start */
    __sanitizer_start_switch_fiber((void **)&co->asanFakeStack, co->asanFiberAddr, co->asanFiberSize);
    /* clear remaining shadow stack */
    __asan_handle_no_return();
#endif
    /* co switch to thread, and do not switch back again */
    CoSwitch(&co->ctx, &co->thEnv->schCtx);
}

static inline void CoStartEntry(void* arg)
{
    CoRoutine* co = reinterpret_cast<CoRoutine*>(arg);
#ifdef ASAN_MODE
    /* thread to co finish first */
    __sanitizer_finish_switch_fiber(co->asanFakeStack, (const void**)&co->asanFiberAddr, &co->asanFiberSize);
#endif
    ffrt::CoTask* task = co->task;
    bool isNormalTask = task->type == ffrt_normal_task;
    task->Execute();
    co->status.store(static_cast<int>(CoStatus::CO_UNINITIALIZED));
    CoExit(co, isNormalTask);
}

static void CoSetStackProt(CoRoutine* co, int prot)
{
    /* set the attribute of the page table closest to the stack top in the user stack to read-only,
     * and 1~2 page table space will be wasted
     */
    size_t p_size = getpagesize();
    uint64_t mp = reinterpret_cast<uint64_t>(co->stkMem.stk);
    mp = (mp + p_size - 1) / p_size * p_size;
    int ret = mprotect(reinterpret_cast<void *>(static_cast<uintptr_t>(mp)), p_size, prot);
    FFRT_UNLIKELY_COND_DO_ABORT(ret < 0, "coroutine size:%lu, mp:0x%lx, page_size:%zu,result:%d,prot:%d, err:%d,%s",
                                static_cast<unsigned long>(sizeof(struct CoRoutine)), static_cast<unsigned long>(mp),
                                p_size, ret, prot, errno, strerror(errno));
}

static inline CoRoutine* AllocNewCoRoutine(size_t stackSize)
{
    std::size_t defaultStackSize = FFRTFacade::GetCSAInstance()->size;
    CoRoutine* co = nullptr;
    if (likely(stackSize == defaultStackSize)) {
        co = ffrt::CoRoutineAllocMem(stackSize);
    } else {
        co = static_cast<CoRoutine*>(mmap(nullptr, stackSize,
            PROT_READ | PROT_WRITE,  MAP_ANONYMOUS | MAP_PRIVATE, -1, 0));
        if (co == reinterpret_cast<CoRoutine*>(MAP_FAILED)) {
            FFRT_SYSEVENT_LOGE("memory mmap failed.");
            return nullptr;
        }
    }
    if (!co) {
        FFRT_SYSEVENT_LOGE("memory not enough");
        return nullptr;
    }
    new (co)CoRoutine();
    co->allocatedSize = stackSize;
    co->stkMem.size = static_cast<uint64_t>(stackSize - sizeof(CoRoutine) + 8);
    co->stkMem.magic = STACK_MAGIC;
    if (FFRTFacade::GetCSAInstance()->type == CoStackProtectType::CO_STACK_STRONG_PROTECT) {
        CoSetStackProt(co, PROT_READ);
    }
    return co;
}

static inline void CoMemFree(CoRoutine* co)
{
    if (FFRTFacade::GetCSAInstance()->type == CoStackProtectType::CO_STACK_STRONG_PROTECT) {
        CoSetStackProt(co, PROT_WRITE | PROT_READ);
    }
    std::size_t defaultStackSize = FFRTFacade::GetCSAInstance()->size;
    if (likely(co->allocatedSize == defaultStackSize)) {
        ffrt::CoRoutineFreeMem(co);
    } else {
        int ret = munmap(co, co->allocatedSize);
        if (ret != 0) {
            FFRT_SYSEVENT_LOGE("munmap failed with errno: %d", errno);
        }
    }
}

void CoStackFree(void)
{
    if (GetCoEnv()) {
        if (GetCoEnv()->runningCo) {
            CoMemFree(GetCoEnv()->runningCo);
            GetCoEnv()->runningCo = nullptr;
        }
    }
}

void CoWorkerExit(void)
{
    CoStackFree();
}

static inline void BindNewCoRoutione(ffrt::CoTask* task)
{
    task->coRoutine = GetCoEnv()->runningCo;
    task->coRoutine->task = task;
    task->coRoutine->thEnv = GetCoEnv();
}

static inline int CoAlloc(ffrt::CoTask* task)
{
    if (task->coRoutine) { // use allocated coroutine stack
        if (GetCoEnv()->runningCo) { // free cached stack if it exist
            CoMemFree(GetCoEnv()->runningCo);
        }
        GetCoEnv()->runningCo = task->coRoutine;
    } else {
        if (!GetCoEnv()->runningCo) { // if no cached stack, alloc one
            GetCoEnv()->runningCo = AllocNewCoRoutine(task->stack_size);
        } else { // exist cached stack
            if (GetCoEnv()->runningCo->allocatedSize != task->stack_size) { // stack size not match, alloc one
                CoMemFree(GetCoEnv()->runningCo); // free cached stack
                GetCoEnv()->runningCo = AllocNewCoRoutine(task->stack_size);
            }
        }
    }
    return 0;
}

// call CoCreat when task creat
static inline int CoCreat(ffrt::CoTask* task)
{
    CoAlloc(task);
    if (GetCoEnv()->runningCo == nullptr) { // retry once if alloc failed
        CoAlloc(task);
        if (GetCoEnv()->runningCo == nullptr) { // retry still failed
            FFRT_LOGE("alloc co routine failed");
            return -1;
        }
    }
    BindNewCoRoutione(task);
    auto co = task->coRoutine;
    if (co->status.load() == static_cast<int>(CoStatus::CO_UNINITIALIZED)) {
        ffrt_fiber_init(&co->ctx, CoStartEntry, static_cast<void*>(co), co->stkMem.stk, co->stkMem.size);
    }
    return 0;
}

static inline void CoSwitchInTransaction(ffrt::CoTask* task)
{
    if (task->cpuBoostCtxId >= 0) {
        CpuBoostRestore(task->cpuBoostCtxId);
    }
}

static inline void CoSwitchOutTransaction(ffrt::CoTask* task)
{
    if (task->cpuBoostCtxId >= 0) {
        CpuBoostSave(task->cpuBoostCtxId);
    }
}

static inline bool CoBboxPreCheck(ffrt::CoTask* task)
{
    if (task->coRoutine) {
        int ret = task->coRoutine->status.exchange(static_cast<int>(CoStatus::CO_RUNNING));
        if (ret == static_cast<int>(CoStatus::CO_RUNNING) && GetBboxEnableState() != 0) {
            FFRT_SYSEVENT_LOGE("executed by worker suddenly, ignore backtrace");
            return false;
        }
    }

    return true;
}

// called by thread work
int CoStart(ffrt::CoTask* task, CoRoutineEnv* coRoutineEnv)
{
    if (!CoBboxPreCheck(task)) {
        return 0;
    }

    if (CoCreat(task) != 0) {
        return -1;
    }
    auto co = task->coRoutine;

    FFRTTraceRecord::TaskRun(task->GetQos(), task);

    for (;;) {
        ffrt::TaskLoadTracking::Begin(task);
#ifdef FFRT_ASYNC_STACKTRACE
        FFRTSetStackId(task->stackId);
#endif
        FFRT_TASK_BEGIN(task->label, task->gid);
        CoSwitchInTransaction(task);
#ifdef FFRT_TASK_LOCAL_ENABLE
        SwitchTsdToTask(co->task);
#endif
#ifdef FFRT_ENABLE_HITRACE_CHAIN
        if (task->traceId_.valid == HITRACE_ID_VALID) {
            TraceChainAdapter::Instance().HiTraceChainRestoreId(&task->traceId_);
        }
#endif
#ifdef ASAN_MODE
        /* thread to co start */
        __sanitizer_start_switch_fiber((void **)&co->asanFakeStack, GetCoStackAddr(co), co->stkMem.size);
#endif
        // mark tasks as EXECUTING which waked by CoWakeFunc that blocked by CoWait before
        task->SetStatus(TaskStatus::EXECUTING);
        /* thread switch to co */
        CoSwitch(&co->thEnv->schCtx, &co->ctx);
#ifdef ASAN_MODE
        /* co to thread finish */
        __sanitizer_finish_switch_fiber(co->asanFakeStack, (const void**)&co->asanFiberAddr, &co->asanFiberSize);
#endif
        FFRT_TASK_END();
#ifdef FFRT_ENABLE_HITRACE_CHAIN
        if (co->status.load() != static_cast<int>(CoStatus::CO_UNINITIALIZED)) {
            task->traceId_ = TraceChainAdapter::Instance().HiTraceChainGetId();
            if (task->traceId_.valid == HITRACE_ID_VALID) {
                TraceChainAdapter::Instance().HiTraceChainClearId();
            }
        }
#endif
        ffrt::TaskLoadTracking::End(task); // Todo: deal with CoWait()
        CoStackCheck(co);

        // 1. coroutine task done, exit normally, need to exec next coroutine task
        if (co->isTaskDone) {
            ffrt::FFRTFacade::GetDMInstance().onTaskDone(static_cast<CPUEUTask*>(task));
            co->isTaskDone = false;
            return 0;
        }

        // 2. couroutine task block, switch to thread
        // need suspend the coroutine task or continue to execute the coroutine task.
        auto pending = coRoutineEnv->pending;
        if (pending == nullptr) {
            return 0;
        }
        coRoutineEnv->pending = nullptr;
        FFRTTraceRecord::TaskCoSwitchOut(task);
        // Fast path: skip state transition
        if ((*pending)(task)) {
            // The ownership of the task belongs to other host(cv/mutex/epoll etc)
            // And the task cannot be accessed any more.
            return 0;
        }
        FFRT_WAKE_TRACER(task->gid); // fast path wk
        coRoutineEnv->runningCo = co;
    }
}

// called by thread work
void CoYield(void)
{
    CoRoutine* co = static_cast<CoRoutine*>(GetCoEnv()->runningCo);
    co->status.store(static_cast<int>(CoStatus::CO_NOT_FINISH));
    GetCoEnv()->runningCo = nullptr;
    CoSwitchOutTransaction(co->task);
    FFRT_BLOCK_MARKER(co->task->gid);
#ifdef FFRT_TASK_LOCAL_ENABLE
    SwitchTsdToThread(co->task);
#endif
    CoStackCheck(co);
#ifdef ASAN_MODE
    /* co to thread start */
    __sanitizer_start_switch_fiber((void **)&co->asanFakeStack, co->asanFiberAddr, co->asanFiberSize);
#endif
    /* co switch to thread */
    CoSwitch(&co->ctx, &GetCoEnv()->schCtx);
#ifdef ASAN_MODE
    /* thread to co finish */
    __sanitizer_finish_switch_fiber(co->asanFakeStack, (const void**)&co->asanFiberAddr, &co->asanFiberSize);
#else
    while (GetBboxEnableState() != 0) {
        if (GetBboxEnableState() != gettid()) {
            BboxFreeze(); // freeze non-crash thread
            return;
        }
        const int IGNORE_DEPTH = 3;
        backtrace(IGNORE_DEPTH);
        co->status.store(static_cast<int>(CoStatus::CO_NOT_FINISH)); // recovery to old state
        CoExit(co, co->task->type == ffrt_normal_task);
    }
#endif
}

void CoWait(const std::function<bool(ffrt::CoTask*)>& pred)
{
    GetCoEnv()->pending = &pred;
    CoYield();
}

void CoWake(ffrt::CoTask* task, CoWakeType type)
{
    if (task == nullptr) {
        FFRT_SYSEVENT_LOGE("task is nullptr");
        return;
    }
    // Fast path: state transition without lock
    task->coWakeType = type;
    FFRT_WAKE_TRACER(task->gid);
    switch (task->type) {
        case ffrt_normal_task: {
            task->Ready();
            break;
        }
        case ffrt_queue_task: {
            QueueTask* sTask = reinterpret_cast<QueueTask*>(task);
            auto handle = sTask->GetHandler();
            handle->TransferTask(sTask);
            break;
        }
        default: {
            FFRT_LOGE("CoWake unsupport task[%lu], type=%d, name[%s]", task->gid, task->type, task->GetLabel().c_str());
            break;
        }
    }
}

CoRoutineFactory &CoRoutineFactory::Instance()
{
    static CoRoutineFactory fac;
    return fac;
}
