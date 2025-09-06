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
#ifdef FFRT_BBOX_ENABLE

#include "dfx/bbox/bbox.h"
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>
#include <cstdlib>
#include <string>
#include <sstream>
#include <vector>
#include "dfx/log/ffrt_log_api.h"
#include "dfx/trace_record/ffrt_trace_record.h"
#include "sched/scheduler.h"
#include "tm/queue_task.h"
#include "queue/queue_monitor.h"
#include "queue/traffic_record.h"
#include "tm/task_factory.h"
#include "eu/execute_unit.h"
#include "util/time_format.h"
#include "tm/uv_task.h"
#include "tm/io_task.h"
#ifdef OHOS_STANDARD_SYSTEM
#include "dfx/bbox/fault_logger_fd_manager.h"
#endif
#include "dfx/dump/dump.h"
#include "util/ffrt_facade.h"
#include "util/slab.h"

using namespace ffrt;

constexpr static size_t EACH_QUEUE_TASK_DUMP_SIZE = 64;
constexpr static unsigned int WAIT_PID_SLEEP_MS = 2;
constexpr static unsigned int WAIT_PID_MAX_RETRIES = 1000;
static std::atomic<unsigned int> g_taskPendingCounter(0);
static std::atomic<unsigned int> g_taskWakeCounter(0);
static TaskBase* g_cur_task;
static unsigned int g_cur_pid;
static unsigned int g_cur_tid;
static const char* g_cur_signame;

static struct sigaction s_oldSa[SIGSYS + 1]; // SIGSYS = 31

static FuncSaveKeyStatusInfo saveKeyStatusInfo = nullptr;
static FuncSaveKeyStatus saveKeyStatus = nullptr;
static FuncGetKeyStatus getKeyStatus = nullptr;
void SetFuncSaveKeyStatus(FuncGetKeyStatus getFunc, FuncSaveKeyStatus saveFunc, FuncSaveKeyStatusInfo infoFunc)
{
    getKeyStatus = getFunc;
    saveKeyStatus = saveFunc;
    saveKeyStatusInfo = infoFunc;
}

void TaskWakeCounterInc(void)
{
    ++g_taskWakeCounter;
}

void TaskPendingCounterInc(void)
{
    ++g_taskPendingCounter;
}

static void SignalUnReg(int signo)
{
    sigaction(signo, &s_oldSa[signo], nullptr);
}

__attribute__((destructor)) static void BBoxDeInit()
{
    SignalUnReg(SIGABRT);
    SignalUnReg(SIGBUS);
    SignalUnReg(SIGFPE);
    SignalUnReg(SIGILL);
    SignalUnReg(SIGSTKFLT);
    SignalUnReg(SIGSYS);
    SignalUnReg(SIGTRAP);
    SignalUnReg(SIGINT);
    SignalUnReg(SIGKILL);
}

static inline void SaveCurrent()
{
    FFRT_BBOX_LOG("<<<=== current status ===>>>");
    FFRT_BBOX_LOG("signal %s triggered: source pid %d, tid %d", g_cur_signame, g_cur_pid, g_cur_tid);
    auto t = g_cur_task;
    if (t) {
        FFRT_BBOX_LOG("task id %lu, qos %d, name %s, status %s",
            t->gid, t->qos_(), t->GetLabel().c_str(), StatusToString(t->curStatus).c_str());
    }
}

#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_2)
static inline void SaveTaskCounter()
{
    FFRT_BBOX_LOG("<<<=== task counter ===>>>");
    FFRT_BBOX_LOG("FFRT BBOX TaskSubmitCounter:%u TaskEnQueueCounter:%u TaskDoneCounter:%u",
        FFRTTraceRecord::GetSubmitCount(), FFRTTraceRecord::GetEnqueueCount(), FFRTTraceRecord::GetDoneCount());
    FFRT_BBOX_LOG("FFRT BBOX TaskRunCounter:%u TaskSwitchCounter:%u TaskFinishCounter:%u",
        FFRTTraceRecord::GetRunCount(), FFRTTraceRecord::GetCoSwitchCount(), FFRTTraceRecord::GetFinishCount());
    FFRT_BBOX_LOG("FFRT BBOX TaskWakeCounterInc:%u, TaskPendingCounter:%u",
        g_taskWakeCounter.load(), g_taskPendingCounter.load());
    if (FFRTTraceRecord::GetCoSwitchCount() + FFRTTraceRecord::GetFinishCount() == FFRTTraceRecord::GetRunCount()) {
        FFRT_BBOX_LOG("TaskRunCounter equals TaskSwitchCounter + TaskFinishCounter");
    } else {
        FFRT_BBOX_LOG("TaskRunCounter is not equal to TaskSwitchCounter + TaskFinishCounter");
    }
}
#endif

static inline void SaveLocalFifoStatus(int qos, CPUWorker* worker)
{
    auto sched = FFRTFacade::GetSchedInstance();
    if (sched->GetTaskSchedMode(qos) == TaskSchedMode::DEFAULT_TASK_SCHED_MODE) { return; }
    TaskBase* t = reinterpret_cast<TaskBase*>(sched->GetWorkerLocalQueue(qos, worker->Id())->PopHead());
    while (t != nullptr) {
        FFRT_BBOX_LOG("qos %d: worker tid %d is localFifo task id %lu name %s",
            qos, worker->Id(), t->gid, t->GetLabel().c_str());
        t = reinterpret_cast<TaskBase*>(sched->GetWorkerLocalQueue(qos, worker->Id())->PopHead());
    }
}

static inline void SaveWorkerStatus()
{
    FFRT_BBOX_LOG("<<<=== worker status ===>>>");
    for (int i = 0; i < QoS::MaxNum(); i++) {
        CPUWorkerGroup& workerGroup = FFRTFacade::GetEUInstance().GetWorkerGroup(i);
        for (auto& thread : workerGroup.threads) {
            SaveLocalFifoStatus(i, thread.first);
            TaskBase* t = thread.first->curTask;
            if (t == nullptr) {
                FFRT_BBOX_LOG("qos %d: worker tid %d is running nothing", i, thread.first->Id());
                continue;
            }
            FFRT_BBOX_LOG("qos %d: worker tid %d is running task", i, thread.first->Id());
        }
    }
}

static inline void SaveKeyStatus()
{
    FFRT_BBOX_LOG("<<<=== key status ===>>>");
    if (saveKeyStatus == nullptr) {
        FFRT_BBOX_LOG("no key status");
        return;
    }
    saveKeyStatus();
}

static inline void SaveNormalTaskStatus()
{
    auto unfree = TaskFactory<CPUEUTask>::GetUnfreedMem();
    auto apply = [&](const char* tag, const std::function<bool(CPUEUTask*)>& filter) {
        std::vector<CPUEUTask*> tmp;
        for (auto task : unfree) {
            auto t = reinterpret_cast<CPUEUTask*>(task);
            auto f = reinterpret_cast<ffrt_function_header_t*>(t->func_storage);
            if (((f->reserve[0] & MASK_FOR_HCS_TASK) != MASK_FOR_HCS_TASK) && filter(t)) {
                tmp.emplace_back(t);
            }
        }

        if (tmp.size() > 0) {
            FFRT_BBOX_LOG("<<<=== %s ===>>>", tag);
        }
        size_t idx = 1;
        for (auto t : tmp) {
            if (t->type == ffrt_normal_task) {
                FFRT_BBOX_LOG("<%zu/%lu> id %lu qos %d name %s", idx,
                    tmp.size(), t->gid, t->qos_(), t->GetLabel().c_str());
                idx++;
            }
            if (t->coRoutine && (t->coRoutine->status.load() == static_cast<int>(CoStatus::CO_NOT_FINISH)) &&
                t != g_cur_task) {
#ifdef FFRT_CO_BACKTRACE_OH_ENABLE
                std::string dumpInfo;
                DumpTask(t, dumpInfo, 1);
                if (!dumpInfo.empty()) {
                    FFRT_BBOX_LOG("%s", dumpInfo.c_str());
                }
#else
                CoStart(t, GetCoEnv());
#endif // FFRT_CO_BACKTRACE_OH_ENABLE
            }
        }
    };

    // Do not dump tasks marked with a final status (e.g., FINISH or CANCELED),
    // as they may be allocated by another submit and not initialized yet.
    apply("pending task", [](CPUEUTask* t) {
        return t->curStatus == TaskStatus::SUBMITTED;
    });
    apply("ready task", [](CPUEUTask* t) {
        return t->curStatus == TaskStatus::READY;
    });
    apply("POPPED task", [](CPUEUTask* t) {
        return t->curStatus == TaskStatus::POPPED;
    });
    apply("executing task", [](CPUEUTask* t) {
        return t->curStatus == TaskStatus::EXECUTING;
    });
    apply("blocked by synchronization primitive(mutex etc) or wait dependence", [](CPUEUTask* t) {
        return (t->curStatus == TaskStatus::THREAD_BLOCK) || (t->curStatus == TaskStatus::COROUTINE_BLOCK);
    });
}

static void DumpQueueTask(const char* tag, const std::vector<QueueTask*>& tasks,
    const std::function<bool(QueueTask*)>& filter, size_t limit = EACH_QUEUE_TASK_DUMP_SIZE)
{
    std::vector<QueueTask*> tmp;
    for (auto t : tasks) {
        if (tmp.size() < limit && filter(t)) {
            tmp.emplace_back(t);
        }
    }
    if (tmp.size() == 0) {
        return;
    }

    FFRT_BBOX_LOG("<<<=== %s ===>>>", tag);
    size_t idx = 1;
    for (auto t : tmp) {
        if (t->type == ffrt_queue_task) {
            FFRT_BBOX_LOG("<%zu/%lu> id %lu qos %d name %s", idx, tmp.size(), t->gid, t->GetQos(), t->label.c_str());
            idx++;
        }
        if (t->coRoutine && (t->coRoutine->status.load() == static_cast<int>(CoStatus::CO_NOT_FINISH))) {
#ifdef FFRT_CO_BACKTRACE_OH_ENABLE
            std::string dumpInfo;
            DumpTask(t, dumpInfo, 1);
            if (!dumpInfo.empty()) {
                FFRT_BBOX_LOG("%s", dumpInfo.c_str());
            }
#else
            CoStart(reinterpret_cast<CPUEUTask*>(t), GetCoEnv());
#endif // FFRT_CO_BACKTRACE_OH_ENABLE
        }
    }
}

static inline void SaveQueueTaskStatus()
{
    auto unfreeQueueTask = SimpleAllocator<QueueTask>::getUnfreedMem();
    if (unfreeQueueTask.size() == 0) {
        return;
    }

    std::map<QueueHandler*, std::vector<QueueTask*>> taskMap;
    for (auto t : unfreeQueueTask) {
        auto task = reinterpret_cast<QueueTask*>(t);
        if (task->type == ffrt_queue_task && task->curStatus != TaskStatus::FINISH && task->GetHandler() != nullptr) {
            taskMap[task->GetHandler()].push_back(task);
        }
    }
    if (taskMap.empty()) {
        return;
    }

    for (auto entry : taskMap) {
        std::sort(entry.second.begin(), entry.second.end(), [](QueueTask* first, QueueTask* second) {
            return first->GetUptime() < second->GetUptime();
        });
    }

    // Do not dump tasks marked with a final status (e.g., FINISH or CANCELLED),
    // as they may be allocated by another submit and not initialized yet.
    for (auto entry : taskMap) {
        DumpQueueTask("queue task enqueued", entry.second, [](QueueTask* t) {
            return t->curStatus == TaskStatus::ENQUEUED;
        });
        DumpQueueTask("queue task dequeued", entry.second, [](QueueTask* t) {
            return t->curStatus == TaskStatus::DEQUEUED;
        });
        DumpQueueTask("queue task ready", entry.second, [](QueueTask* t) {
            return t->curStatus == TaskStatus::READY;
        });
        DumpQueueTask("queue task POPPED", entry.second, [](QueueTask* t) {
            return t->curStatus == TaskStatus::POPPED;
        });
        DumpQueueTask("queue task executing", entry.second, [](QueueTask* t) {
            return t->curStatus == TaskStatus::EXECUTING;
        });
        DumpQueueTask("queue task blocked by synchronization primitive(mutex etc)", entry.second,
            [](QueueTask* t) {
                return (t->curStatus == TaskStatus::THREAD_BLOCK) || (t->curStatus == TaskStatus::COROUTINE_BLOCK);
        });
    }
}

static inline void SaveTimeoutTask()
{
    FFRT_BBOX_LOG("<<<=== Timeout Task Info ===>>>");

    std::string normaltskTimeoutInfo = FFRTFacade::GetWMInstance().DumpTimeoutInfo();
    std::string queueTimeoutInfo = FFRTFacade::GetQMInstance().DumpQueueTimeoutInfo();
    std::stringstream ss;
    ss << normaltskTimeoutInfo << queueTimeoutInfo;
    FFRT_BBOX_LOG("%s", ss.str().c_str());
}

static inline void SaveQueueTrafficRecord()
{
    FFRT_BBOX_LOG("<<<=== Queue Traffic Record ===>>>");

    std::string trafficInfo = TrafficRecord::DumpTrafficInfo(false);
    std::stringstream ss;
    ss << trafficInfo;
    FFRT_BBOX_LOG("%s", ss.str().c_str());
}

static std::atomic_uint g_bbox_tid_is_dealing {0};
static std::atomic_uint g_bbox_called_times {0};

void BboxFreeze()
{
    while (g_bbox_tid_is_dealing.load() != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_PID_SLEEP_MS));
    }
}

void backtrace(int ignoreDepth)
{
#ifdef FFRT_CO_BACKTRACE_OH_ENABLE
    std::string dumpInfo;
    DumpTask(nullptr, dumpInfo, 1);
    if (!dumpInfo.empty()) {
        FFRT_BBOX_LOG("%s", dumpInfo.c_str());
    }
#endif // FFRT_CO_BACKTRACE_OH_ENABLE
}

unsigned int GetBboxEnableState(void)
{
    return g_bbox_tid_is_dealing.load();
}

unsigned int GetBboxCalledTimes(void)
{
    return g_bbox_called_times.load();
}

bool FFRTIsWork()
{
    return FFRTTraceRecord::FfrtBeUsed();
}

void RecordDebugInfo(void)
{
    auto t = ExecuteCtx::Cur()->task;
    FFRT_BBOX_LOG("<<<=== ffrt debug log start ===>>>");

    if (t != nullptr) {
        FFRT_BBOX_LOG("debug log: tid %d, task id %lu, qos %d, name %s, status %s", gettid(), t->gid, t->qos_(),
            t->GetLabel().c_str(), StatusToString(t->curStatus).c_str());
    }
    FFRT_BBOX_LOG("<<<=== key status ===>>>");
    if (saveKeyStatusInfo == nullptr) {
        FFRT_BBOX_LOG("no key status");
    } else {
        FFRT_BBOX_LOG("%s", saveKeyStatusInfo().c_str());
    }
    FFRT_BBOX_LOG("<<<=== ffrt debug log finish ===>>>");
}

/**
 * @brief BBOX信息记录，包括task、queue、worker相关信息
 *
 * @param void
 * @return void
 * @约束：
 *  1、FFRT模块收到信号，记录BBOX信息，支持信号如下：
 *     SIGABRT、SIGBUS、SIGFPE、SIGILL、SIGSTKFLT、SIGSTOP、SIGSYS、SIGTRAP
 * @规格：
 *  1.调用时机：FFRT模块收到信号时
 *  2.影响：1）FFRT功能不可用，FFRT任务不再执行
 *          2）影响范围仅影响FFRT任务运行，不能造成处理过程中的空指针等异常，如ffrt处理过程造成进行Crash
 */
void SaveTheBbox()
{
    FFRT_BBOX_LOG("<<<=== ffrt black box(BBOX) start ===>>>");
    SaveCurrent();
#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_2)
    SaveTaskCounter();
#endif
    SaveWorkerStatus();
    SaveKeyStatus();
    SaveNormalTaskStatus();
    SaveQueueTaskStatus();
    SaveTimeoutTask();
    SaveQueueTrafficRecord();
    FFRT_BBOX_LOG("<<<=== ffrt black box(BBOX) finish ===>>>");
}

static void ResendSignal(siginfo_t* info)
{
    int rc = syscall(SYS_rt_tgsigqueueinfo, getpid(), syscall(SYS_gettid), info->si_signo, info);
    if (rc != 0) {
        FFRT_LOGE("ffrt failed to resend signal during crash");
    }
}

static const char* GetSigName(const siginfo_t* info)
{
    switch (info->si_signo) {
        case SIGABRT: return "SIGABRT";
        case SIGBUS: return "SIGBUS";
        case SIGFPE: return "SIGFPE";
        case SIGILL: return "SIGILL";
        case SIGSTKFLT: return "SIGSTKFLT";
        case SIGSTOP: return "SIGSTOP";
        case SIGSYS: return "SIGSYS";
        case SIGTRAP: return "SIGTRAP";
        default: return "?";
    }
}

static void HandleChildProcess()
{
    BBoxDeInit();
    pid_t childPid = (pid_t)syscall(SYS_clone, SIGCHLD, 0);
    if (childPid == 0) {
        // init is false to avoid deadlock occurs in the signal handling function due to memory allocation calls.
        auto ctx = ExecuteCtx::Cur(false);
        g_cur_task = ctx != nullptr ? ctx->task : nullptr;
        g_bbox_tid_is_dealing.store(gettid());
        SaveTheBbox();
        g_bbox_tid_is_dealing.store(0);
#ifdef OHOS_STANDARD_SYSTEM
        FaultLoggerFdManager::CloseFd();
#endif
        _exit(0);
    } else if (childPid > 0) {
        pid_t wpid;
        unsigned int remainingRetries = WAIT_PID_MAX_RETRIES;
        while ((wpid = waitpid(childPid, nullptr, WNOHANG)) == 0 && remainingRetries-- > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_PID_SLEEP_MS));
        }
        if (wpid == 0) {
            (void)kill(childPid, SIGKILL);
        }
    }
}

static void SignalHandler(int signo, siginfo_t* info, void* context __attribute__((unused)))
{
    unsigned int pid = static_cast<unsigned int>(getpid());
    unsigned int tid = static_cast<unsigned int>(gettid());
    unsigned int defaultTid = 0;
    if (g_bbox_tid_is_dealing.compare_exchange_strong(defaultTid, tid)&&
        FFRTIsWork() && g_bbox_called_times.fetch_add(1) == 0) { // only save once
        g_cur_pid = pid;
        g_cur_tid = tid;
        g_cur_signame = GetSigName(info);
        if (getKeyStatus != nullptr) {
            getKeyStatus();
        }
#ifdef OHOS_STANDARD_SYSTEM
        FaultLoggerFdManager::InitFaultLoggerFd();
#endif
        pid_t childPid = static_cast<pid_t>(syscall(SYS_clone, SIGCHLD, 0));
        if (childPid == 0) {
            HandleChildProcess();
            _exit(0);
        } else if (childPid > 0) {
            waitpid(childPid, nullptr, 0);
            g_bbox_tid_is_dealing.store(0);
        }
    } else {
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = WAIT_PID_SLEEP_MS * 1000000;
        if (tid == g_bbox_tid_is_dealing.load()) {
            g_bbox_tid_is_dealing.store(0);
        } else {
            while (g_bbox_tid_is_dealing.load() != 0) {
                nanosleep(&ts, nullptr);
            }
        }
    }
    // we need to deregister our signal handler for that signal before continuing.
    sigaction(signo, &s_oldSa[signo], nullptr);
    ResendSignal(info);
}

static void SignalReg(int signo)
{
    sigaction(signo, nullptr, &s_oldSa[signo]);
    struct sigaction newAction = {};
    newAction.sa_flags = SA_RESTART | SA_SIGINFO;
    newAction.sa_sigaction = SignalHandler;
    sigaction(signo, &newAction, nullptr);
}

__attribute__((constructor)) static void BBoxInit()
{
    SignalReg(SIGABRT);
    SignalReg(SIGBUS);
    SignalReg(SIGFPE);
    SignalReg(SIGILL);
    SignalReg(SIGSTKFLT);
    SignalReg(SIGSYS);
    SignalReg(SIGTRAP);
    SignalReg(SIGINT);
    SignalReg(SIGKILL);
}

std::string GetDumpPreface(void)
{
    std::ostringstream ss;
    ss << "|-> Launcher proc ffrt, now:" << FormatDateToString(TimeStamp()) << " pid:" << GetPid()
        << std::endl;
    return ss.str();
}

#ifdef FFRT_CO_BACKTRACE_OH_ENABLE
#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_2)
std::string SaveTaskCounterInfo(void)
{
    std::ostringstream ss;
    ss << "    |-> task counter" << std::endl;
    ss << "        TaskSubmitCounter:" << FFRTTraceRecord::GetSubmitCount() << " TaskEnQueueCounter:"
       << FFRTTraceRecord::GetEnqueueCount() << " TaskDoneCounter:" << FFRTTraceRecord::GetDoneCount() << std::endl;

    ss << "        TaskRunCounter:" << FFRTTraceRecord::GetRunCount() << " TaskSwitchCounter:"
       << FFRTTraceRecord::GetCoSwitchCount() << " TaskFinishCounter:" << FFRTTraceRecord::GetFinishCount()
       << std::endl;

    if (FFRTTraceRecord::GetCoSwitchCount() + FFRTTraceRecord::GetFinishCount() == FFRTTraceRecord::GetRunCount()) {
        ss << "        TaskRunCounter equals TaskSwitchCounter + TaskFinishCounter" << std::endl;
    } else {
        ss << "        TaskRunCounter is not equal to TaskSwitchCounter + TaskFinishCounter" << std::endl;
    }
    return ss.str();
}
#endif // FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_2

void AppendTaskInfo(std::ostringstream& oss, TaskBase* task)
{
    if (task->fromTid) {
        oss << " fromTid " << task->fromTid;
    }
    if (task->createTime) {
        oss << " createTime " << FormatDateToString(task->createTime);
    }
    if (task->executeTime) {
        oss << " executeTime " << FormatDateToString(task->executeTime);
    }
}

std::string SaveKeyInfo(void)
{
    std::ostringstream oss;

    ffrt::FFRTFacade::GetEUInstance().WorkerInit();
    oss << "    |-> key status" << std::endl;
    if (saveKeyStatusInfo == nullptr) {
        oss << "no key status info" << std::endl;
        return oss.str();
    }
    oss << saveKeyStatusInfo();
    return oss.str();
}

void DumpNormalTaskInfo(std::ostringstream& ss, int qos, pid_t tid, TaskBase* t)
{
    {
        TaskMemScopedLock<CPUEUTask> lock;
        if (TaskFactory<CPUEUTask>::HasBeenFreed(static_cast<CPUEUTask*>(t))) {
            return;
        }
        if (t->curStatus == TaskStatus::FINISH) {
            return;
        }
        if (!IncDeleteRefIfPositive(t)) {
            return;
        }
    }
    ss << "        qos " << qos
        << ": worker tid " << tid
        << " normal task is running, task id " << t->gid
        << " name " << t->GetLabel().c_str()
        << " status " << StatusToString(t->curStatus).c_str();
    AppendTaskInfo(ss, t);
    t->DecDeleteRef();
    ss << std::endl;
}

void DumpQueueTaskInfo(std::ostringstream& ss, int qos, pid_t tid, TaskBase* t)
{
    {
        TaskMemScopedLock<QueueTask> lock;
        auto queueTask = reinterpret_cast<QueueTask*>(t);
        if (TaskFactory<QueueTask>::HasBeenFreed(queueTask)) {
            return;
        }
        if (queueTask->GetFinishStatus()) {
            return;
        }
        if (!IncDeleteRefIfPositive(queueTask)) {
            return;
        }
    }
    ss << "        qos " << qos
        << ": worker tid " << tid
        << " queue task is running, task id " << t->gid
        << " name " << t->GetLabel().c_str()
        << " status " << StatusToString(t->curStatus).c_str();
    AppendTaskInfo(ss, t);
    t->DecDeleteRef();
    ss << std::endl;
}

void DumpThreadTaskInfo(CPUWorker* thread, int qos, std::ostringstream& ss)
{
    TaskBase* t = thread->curTask;
    pid_t tid = thread->Id();
    if (t == nullptr) {
        ss << "        qos " << qos << ": worker tid " << tid << " is running nothing" << std::endl;
        return;
    }

    switch (thread->curTaskType_.load(std::memory_order_relaxed)) {
        case ffrt_normal_task: {
            DumpNormalTaskInfo(ss, qos, tid, t);
            return;
        }
        case ffrt_queue_task: {
            DumpQueueTaskInfo(ss, qos, tid, t);
            return;
        }
        case ffrt_io_task: {
            ss << "        qos "
                << qos << ": worker tid "
                << tid << " io task is running"
                << std::endl;
            return;
        }
        case ffrt_uv_task: {
            ss << "        qos " << qos
                << ": worker tid " << tid
                << " uv task is running"
                << std::endl;
            return;
        }
        default: {
            return;
        }
    }
}

std::string SaveWorkerStatusInfo(void)
{
    std::ostringstream ss;
    std::ostringstream oss;
    oss << "    |-> worker count" << std::endl;
    ss << "    |-> worker status" << std::endl;
    for (int i = 0; i < QoS::MaxNum(); i++) {
        std::vector<int> tidArr;
        CPUWorkerGroup& workerGroup = FFRTFacade::GetEUInstance().GetWorkerGroup(i);
        std::shared_lock<std::shared_mutex> lck(workerGroup.tgMutex);
        for (auto& thread : workerGroup.threads) {
            tidArr.push_back(thread.first->Id());
            DumpThreadTaskInfo(thread.first, i, ss);
        }
        if (tidArr.size() == 0) {
            continue;
        }
        oss << "        qos " << i << ": worker num:" << tidArr.size() << " tid:";
        std::for_each(tidArr.begin(), tidArr.end(), [&](const int &t) {
            if (&t == &tidArr.back()) {
                oss << t;
            } else {
                oss << t << ", ";
            }
        });
        oss << std::endl;
    }
    oss << ss.str();
    return oss.str();
}

void DumpCoYieldTaskBacktrace(CoTask* coTask, std::ostringstream& oss)
{
    std::string dumpInfo;
    std::unique_lock<std::mutex> lck(coTask->mutex_);
    if (coTask->coRoutine && (coTask->coRoutine->status.load() == static_cast<int>(CoStatus::CO_NOT_FINISH))) {
        DumpTask(coTask, dumpInfo, 1);
        lck.unlock();
        oss << dumpInfo.c_str();
    }
}

std::string SaveNormalTaskStatusInfo(void)
{
    std::string ffrtStackInfo;
    std::ostringstream ss;
    std::vector<void*> unfree = TaskFactory<CPUEUTask>::GetUnfreedTasksFiltered();
    if (unfree.size() == 0) {
        return ffrtStackInfo;
    }

    auto apply = [&](const char* tag, const std::function<bool(CPUEUTask*)>& filter) {
        std::vector<CPUEUTask*> tmp;
        for (auto task : unfree) {
            auto t = reinterpret_cast<CPUEUTask*>(task);
            auto f = reinterpret_cast<ffrt_function_header_t*>(t->func_storage);
            if (((f->reserve[0] & MASK_FOR_HCS_TASK) != MASK_FOR_HCS_TASK) && filter(t)) {
                tmp.emplace_back(reinterpret_cast<CPUEUTask*>(t));
            }
        }

        if (tmp.size() > 0) {
            ss << "    |-> " << tag << std::endl;
            ffrtStackInfo += ss.str();
        }
        size_t idx = 1;
        for (auto t : tmp) {
            ss.str("");
            if (t->type == ffrt_normal_task) {
                ss << "        <" << idx++ << "/" << tmp.size() << ">" << "stack: task id " << t->gid << ",qos "
                    << t->qos_() << ",name " << t->GetLabel().c_str();
                AppendTaskInfo(ss, t);
                ss << std::endl;
            }
            DumpCoYieldTaskBacktrace(t, ss);
            ffrtStackInfo += ss.str();
        }
    };

    // Do not dump tasks marked with a final status (e.g., FINISH or CANCELLED),
    // as they may be allocated by another submit and not initialized yet.
    apply("pending task", [](CPUEUTask* t) {
        return t->curStatus == TaskStatus::SUBMITTED;
    });
    apply("ready task", [](CPUEUTask* t) {
        return t->curStatus == TaskStatus::READY;
    });
    apply("POPPED task", [](CPUEUTask* t) {
        return t->curStatus == TaskStatus::POPPED;
    });
    apply("executing task", [](CPUEUTask* t) {
        return t->curStatus == TaskStatus::EXECUTING;
    });
    apply("blocked by synchronization primitive(mutex etc) or wait dependence", [](CPUEUTask* t) {
        return (t->curStatus == TaskStatus::THREAD_BLOCK) || (t->curStatus == TaskStatus::COROUTINE_BLOCK);
    });
    for (auto& task : unfree) {
        reinterpret_cast<CPUEUTask*>(task)->DecDeleteRef();
    }
    return ffrtStackInfo;
}

void DumpQueueTaskInfo(std::string& ffrtStackInfo, const char* tag, const std::vector<QueueTask*>& tasks,
    const std::function<bool(QueueTask*)>& filter, size_t limit = EACH_QUEUE_TASK_DUMP_SIZE)
{
    std::vector<QueueTask*> tmp;
    for (auto t : tasks) {
        if (tmp.size() < limit && filter(t)) {
            tmp.emplace_back(t);
        }
    }
    if (tmp.size() == 0) {
        return;
    }
    std::ostringstream ss;
    ss << "<<<=== " << tag << "===>>>" << std::endl;
    ffrtStackInfo += ss.str();

    size_t idx = 1;
    for (auto t : tmp) {
        ss.str("");
        if (t->type == ffrt_queue_task) {
            ss << "<" << idx++ << "/" << tmp.size() << ">" << "id " << t->gid << " qos "
                << t->GetQos() << " name " << t->GetLabel().c_str();
            AppendTaskInfo(ss, t);
            ss << std::endl;
        }
        DumpCoYieldTaskBacktrace(t, ss);
        ffrtStackInfo += ss.str();
    }
}

std::string SaveQueueTaskStatusInfo()
{
    std::string ffrtStackInfo;
    std::vector<void*> unfree = TaskFactory<QueueTask>::GetUnfreedTasksFiltered();
    if (unfree.size() == 0) {
        return ffrtStackInfo;
    }

    std::map<QueueHandler*, std::vector<QueueTask*>> taskMap;
    for (auto t : unfree) {
        auto task = reinterpret_cast<QueueTask*>(t);
        if (task->type == ffrt_queue_task && task->GetFinishStatus() == false && task->GetHandler() != nullptr) {
            taskMap[task->GetHandler()].push_back(task);
        }
    }

    for (auto entry : taskMap) {
        std::sort(entry.second.begin(), entry.second.end(), [](QueueTask* first, QueueTask* second) {
            return first->GetUptime() < second->GetUptime();
        });
    }

    // Do not dump tasks marked with a final status (e.g., FINISH or CANCELLED),
    // as they may be allocated by another submit and not initialized yet.
    for (auto entry : taskMap) {
        ffrtStackInfo += "\n";
        DumpQueueTaskInfo(ffrtStackInfo, "queue task enqueued", entry.second, [](QueueTask* t) {
            return t->curStatus == TaskStatus::ENQUEUED;
        });
        DumpQueueTaskInfo(ffrtStackInfo, "queue task dequeued", entry.second, [](QueueTask* t) {
            return t->curStatus == TaskStatus::DEQUEUED;
        });
        DumpQueueTaskInfo(ffrtStackInfo, "queue task ready", entry.second, [](QueueTask* t) {
            return t->curStatus == TaskStatus::READY;
        });
        DumpQueueTaskInfo(ffrtStackInfo, "queue task POPPED", entry.second, [](QueueTask* t) {
            return t->curStatus == TaskStatus::POPPED;
        });
        DumpQueueTaskInfo(ffrtStackInfo, "queue task executing", entry.second, [](QueueTask* t) {
            return t->curStatus == TaskStatus::EXECUTING;
        });
        DumpQueueTaskInfo(ffrtStackInfo, "queue task blocked by synchronization primitive(mutex etc)", entry.second,
            [](QueueTask* t) {
                return (t->curStatus == TaskStatus::THREAD_BLOCK) || (t->curStatus == TaskStatus::COROUTINE_BLOCK);
        });
    }

    for (auto& task : unfree) {
        reinterpret_cast<QueueTask*>(task)->DecDeleteRef();
    }
    return ffrtStackInfo;
}

std::string SaveTimeoutTaskInfo()
{
    std::string ffrtStackInfo;
    std::ostringstream ss;
    ss << "<<<=== Timeout Task Info ===>>>" << std::endl;
    ffrtStackInfo += ss.str();
    std::string timeoutInfo = FFRTFacade::GetWMInstance().DumpTimeoutInfo();
    std::string queueTimeoutInfo = FFRTFacade::GetQMInstance().DumpQueueTimeoutInfo();
    ffrtStackInfo += timeoutInfo;
    ffrtStackInfo += queueTimeoutInfo;
    return ffrtStackInfo;
}

std::string SaveQueueTrafficRecordInfo()
{
    std::string ffrtStackInfo;
    std::ostringstream ss;
    ss << "<<<=== Queue Traffic Record ===>>>" << std::endl;
    ffrtStackInfo += ss.str();
    std::string trafficInfo = TrafficRecord::DumpTrafficInfo();
    ffrtStackInfo += trafficInfo;
    return ffrtStackInfo;
}
#endif
#endif /* FFRT_BBOX_ENABLE */
