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

#include "eu/execute_unit.h"
#include "eu/sexecute_unit.h"

#include <climits>
#include <cstring>
#include <sys/stat.h>
#include "dfx/trace_record/ffrt_trace_record.h"
#include "eu/co_routine_factory.h"
#include "eu/qos_interface.h"
#include "sched/scheduler.h"
#include "sched/workgroup_internal.h"
#include "util/ffrt_facade.h"
#include "util/slab.h"
#include "dfx/perf/ffrt_perf.h"
#include "dfx/sysevent/sysevent.h"
#include "internal_inc/config.h"

namespace {
/* SUPPORT_WORKER_DESTRUCT indicates that the idle thread destruction function is supported.
 * The stack canary is saved or restored during coroutine switch-out and switch-in,
 * currently, only the stack canary used by the ohos compiler stack protection is global
 * and is not affected by worker destruction.
 */
#if !defined(SUPPORT_WORKER_DESTRUCT)
constexpr int waiting_seconds = 10;
#else
constexpr int waiting_seconds = 5;
#endif
const size_t TIGGER_SUPPRESS_WORKER_COUNT = 4;
const size_t TIGGER_SUPPRESS_EXECUTION_NUM = 2;
const size_t MAX_ESCAPE_WORKER_NUM = 1024;
const int SEXECUTE_DESTRY_SLEEP_TIME = 1000;

const std::map<std::string,
    void(*)(ffrt::SExecuteUnit*, const ffrt::QoS&, ffrt::TaskNotifyType)> NOTIFY_FUNCTION_FACTORY = {
    { "CameraDaemon", ffrt::SExecuteUnit::HandleTaskNotifyConservative },
    { "bluetooth", ffrt::SExecuteUnit::HandleTaskNotifyUltraConservative },
};
}

namespace ffrt {
constexpr int MANAGER_DESTRUCT_TIMESOUT = 1000;
constexpr uint64_t DELAYED_WAKED_UP_TASK_TIME_INTERVAL = 5 * 1000 * 1000;

SExecuteUnit::SExecuteUnit() : ExecuteUnit(), handleTaskNotify(SExecuteUnit::HandleTaskNotifyDefault)
{
#ifdef OHOS_STANDARD_SYSTEM
    for (const auto& notifyFunc : NOTIFY_FUNCTION_FACTORY) {
        if (strstr(GetCurrentProcessName(), notifyFunc.first.c_str())) {
            handleTaskNotify = notifyFunc.second;
            break;
        }
    }
#endif

#ifdef FFRT_WORKERS_DYNAMIC_SCALING
    int ret = BlockawareInit(&keyPtr);
    if (ret != 0) {
        FFRT_SYSEVENT_LOGE("blockaware init fail, ret[%d], key[0x%lx]", ret, keyPtr);
    } else {
        blockAwareInit = true;
    }
#endif
    FFRT_LOGD("Construction completed.");
}

SExecuteUnit::~SExecuteUnit()
{
    tearDown = true;
    for (auto qos = QoS::Min(); qos < QoS::Max(); ++qos) {
        workerGroup[qos].SetTearDown();
    }
    // Before destroying this object, we need to make sure that all threads that
    // might access this object or its members have exited.
    // If the destruction of this object happens before or in parallel of
    // these threads access to freed memory can occur.
    for (auto qos = QoS::Min(); qos < QoS::Max(); ++qos) {
        int try_cnt = MANAGER_DESTRUCT_TIMESOUT;
        while (try_cnt-- > 0) {
            {
                std::lock_guard lk(workerGroup[qos].mutex);
                workerGroup[qos].cv.notify_all();
            }
            {
                usleep(SEXECUTE_DESTRY_SLEEP_TIME);
                std::shared_lock<std::shared_mutex> lck(workerGroup[qos].tgMutex);
                if (workerGroup[qos].threads.empty()) {
                    break;
                }
            }
        }

        if (try_cnt <= 0) {
            FFRT_SYSEVENT_LOGE("erase qos[%d] threads failed", qos);
        }
    }
    // Note that delayedWorker might
    // call ffrt::SExecuteUnit::WakeupWorkers
    // We need to ensure the object is still
    // alive when that happens. Hence, we
    // delay the destruction till we ensure
    // this access cannot happen.
    FFRTFacade::GetDWInstance().Terminate();
    FFRT_LOGD("Destruction completed.");
}

WorkerAction SExecuteUnit::WorkerIdleAction(CPUWorker* thread)
{
    if (tearDown) {
        return WorkerAction::RETIRE;
    }
    auto& group = workerGroup[thread->GetQos()];
    std::unique_lock lk(group.mutex);
    IntoSleep(thread->GetQos());
    thread->SetWorkerState(WorkerStatus::SLEEPING);
#ifdef FFRT_WORKERS_DYNAMIC_SCALING
    BlockawareEnterSleeping();
#endif
    if (group.cv.wait_for(lk, std::chrono::seconds(waiting_seconds), [this, thread] {
        bool taskExistence = FFRTFacade::GetSchedInstance()->GetGlobalTaskCnt(thread->GetQos());
        return tearDown || taskExistence;
        })) {
        workerGroup[thread->GetQos()].OutOfSleep();
        thread->SetWorkerState(WorkerStatus::EXECUTING);
#ifdef FFRT_WORKERS_DYNAMIC_SCALING
        BlockawareLeaveSleeping();
#endif
        return WorkerAction::RETRY;
    } else {
#if !defined(SUPPORT_WORKER_DESTRUCT)
        workerGroup[thread->GetQos()].IntoDeepSleep();
        CoStackFree();
        if (IsExceedDeepSleepThreshold()) {
            ffrt::CoRoutineReleaseMem();
        }
        group.cv.wait(lk, [this, thread] {
            return tearDown ||
                FFRTFacade::GetSchedInstance()->GetTotalTaskCnt(thread->GetQos()) > 0;
        });
        workerGroup[thread->GetQos()].OutOfDeepSleep();
        thread->SetWorkerState(WorkerStatus::EXECUTING);
        return WorkerAction::RETRY;
#else
        workerGroup[thread->GetQos()].WorkerDestroy();
        return WorkerAction::RETIRE;
#endif
    }
}

void SExecuteUnit::WakeupWorkers(const QoS& qos)
{
    if (tearDown) {
        FFRT_SYSEVENT_LOGE("CPU Worker Manager exit");
        return;
    }
    workerGroup[qos].cv.notify_one();
}

// default strategy which is kind of radical for poking workers
void SExecuteUnit::HandleTaskNotifyDefault(SExecuteUnit* manager, const QoS& qos, TaskNotifyType notifyType)
{
    size_t taskCount = FFRTFacade::GetSchedInstance()->GetGlobalTaskCnt(qos);
    switch (notifyType) {
        case TaskNotifyType::TASK_ADDED:
        case TaskNotifyType::TASK_PICKED:
        case TaskNotifyType::TASK_ESCAPED:
            if (taskCount > 0) {
                manager->PokeImpl(qos, taskCount, notifyType);
            }
            break;
        case TaskNotifyType::TASK_LOCAL:
                manager->PokeImpl(qos, taskCount, notifyType);
            break;
        default:
            break;
    }
}

// conservative strategy for poking workers
void SExecuteUnit::HandleTaskNotifyConservative(SExecuteUnit* manager, const QoS& qos, TaskNotifyType notifyType)
{
    int taskCount = FFRTFacade::GetSchedInstance()->GetGlobalTaskCnt(qos);
    if (taskCount == 0) {
        // no available task in global queue, skip
        return;
    }
    constexpr double thresholdTaskPick = 1.0;
    CPUWorkerGroup& workerCtrl = manager->workerGroup[qos];
    workerCtrl.lock.lock();

    if (notifyType == TaskNotifyType::TASK_PICKED) {
        int wakedWorkerCount = workerCtrl.executingNum;
        double remainingLoadRatio = (wakedWorkerCount == 0) ? static_cast<double>(workerCtrl.maxConcurrency) :
            static_cast<double>(taskCount) / static_cast<double>(wakedWorkerCount);
        if (remainingLoadRatio <= thresholdTaskPick) {
            // for task pick, wake worker when load ratio > 1
            workerCtrl.lock.unlock();
            return;
        }
    }

    if (static_cast<uint32_t>(workerCtrl.executingNum) < workerCtrl.maxConcurrency) {
        if (workerCtrl.sleepingNum == 0) {
            FFRT_LOGI("begin to create worker, notifyType[%d]"
                "execnum[%d], maxconcur[%d], slpnum[%d], dslpnum[%d]",
                notifyType, workerCtrl.executingNum, workerCtrl.maxConcurrency,
                workerCtrl.sleepingNum, workerCtrl.deepSleepingWorkerNum);
            workerCtrl.WorkerCreate();
            workerCtrl.lock.unlock();
            if (!manager->IncWorker(qos)) {
                workerCtrl.RollBackCreate();
            }
        } else {
            workerCtrl.lock.unlock();
            manager->WakeupWorkers(qos);
        }
    } else {
        workerCtrl.lock.unlock();
    }
}

void SExecuteUnit::HandleTaskNotifyUltraConservative(SExecuteUnit* manager, const QoS& qos, TaskNotifyType notifyType)
{
    (void)notifyType;
    int taskCount = FFRTFacade::GetSchedInstance()->GetGlobalTaskCnt(qos);
    if (taskCount == 0) {
        // no available task in global queue, skip
        return;
    }

    CPUWorkerGroup& workerCtrl = manager->workerGroup[qos];
    std::lock_guard lock(workerCtrl.lock);

    int runningNum = static_cast<int>(manager->GetRunningNum(qos));
#ifdef FFRT_WORKERS_DYNAMIC_SCALING
    if (manager->blockAwareInit && !manager->stopMonitor && taskCount == runningNum) {
        return;
    }
#endif

    if (taskCount < runningNum) {
        return;
    }

    if (runningNum < static_cast<int>(workerCtrl.maxConcurrency)) {
        if (workerCtrl.sleepingNum == 0) {
            workerCtrl.WorkerCreate();
            if (!manager->IncWorker(qos)) {
                workerCtrl.RollBackCreate();
            }
        } else {
            manager->WakeupWorkers(qos);
        }
    }
}

void SExecuteUnit::PokeImpl(const QoS& qos, uint32_t taskCount, TaskNotifyType notifyType)
{
    CPUWorkerGroup& workerCtrl = workerGroup[qos];
    workerCtrl.lock.lock();
    size_t runningNum = GetRunningNum(qos);
    size_t totalNum = static_cast<size_t>(workerCtrl.sleepingNum + workerCtrl.executingNum);

    bool tiggerSuppression = (totalNum > TIGGER_SUPPRESS_WORKER_COUNT) &&
        (runningNum > TIGGER_SUPPRESS_EXECUTION_NUM) && (taskCount < runningNum);
    if (notifyType != TaskNotifyType::TASK_ADDED && notifyType != TaskNotifyType::TASK_ESCAPED && tiggerSuppression) {
        workerCtrl.lock.unlock();
        return;
    }

    if ((static_cast<uint32_t>(workerCtrl.sleepingNum) > 0) && (runningNum < workerCtrl.maxConcurrency)) {
        workerCtrl.lock.unlock();
        WakeupWorkers(qos);
    } else if ((runningNum < workerCtrl.maxConcurrency) && (totalNum < workerCtrl.hardLimit)) {
        workerCtrl.WorkerCreate();
        FFRTTraceRecord::WorkRecord(qos(), workerCtrl.executingNum);
        workerCtrl.lock.unlock();
        if (!IncWorker(qos)) {
            workerCtrl.RollBackCreate();
        }
    } else if ((runningNum == 0) && (totalNum < MAX_ESCAPE_WORKER_NUM)) {
        SubmitEscape(qos, totalNum);
        workerCtrl.lock.unlock();
    } else {
        workerCtrl.lock.unlock();
    }
}

void SExecuteUnit::ExecuteEscape(int qos)
{
    if (FFRTFacade::GetSchedInstance()->GetGlobalTaskCnt(qos) > 0) {
        CPUWorkerGroup& workerCtrl = workerGroup[qos];
        workerCtrl.lock.lock();

        size_t runningNum = GetRunningNum(qos);
        size_t totalNum = static_cast<size_t>(workerCtrl.sleepingNum + workerCtrl.executingNum);
        if ((workerCtrl.sleepingNum > 0) && (runningNum < workerCtrl.maxConcurrency)) {
            workerCtrl.lock.unlock();
            WakeupWorkers(qos);
        } else if ((runningNum == 0) && (totalNum < MAX_ESCAPE_WORKER_NUM)) {
            if (IsEscapeEnable()) {
                workerCtrl.WorkerCreate();
                FFRTTraceRecord::WorkRecord(qos, workerCtrl.executingNum);
                workerCtrl.lock.unlock();
                if (!IncWorker(qos)) {
                    workerCtrl.RollBackCreate();
                }
            } else {
                workerCtrl.lock.unlock();
            }
            ReportEscapeEvent(qos, totalNum);
        } else {
            workerCtrl.lock.unlock();
        }
    }
}
} // namespace ffrt
