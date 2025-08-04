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

#include <sys/resource.h>
#include "internal_inc/config.h"
#include "util/singleton_register.h"
#include "eu/co_routine_factory.h"
#include "util/ffrt_facade.h"
#include "dfx/sysevent/sysevent.h"

namespace {
const size_t MAX_ESCAPE_WORKER_NUM = 1024;
constexpr uint64_t MAX_ESCAPE_INTERVAL_MS_COUNT = 1000ULL * 100 * 60 * 60 * 24 * 365; // 100 year
constexpr size_t MAX_TID_SIZE = 100;
ffrt::WorkerStatusInfo g_workerStatusInfo[ffrt::QoS::MaxNum()];
ffrt::fast_mutex g_workerStatusMutex[ffrt::QoS::MaxNum()];
}

namespace ffrt {
const std::map<QoS, std::vector<std::pair<QoS, bool>>> DEFAULT_WORKER_SHARE_CONFIG = {
    {0, {std::make_pair(1, false), std::make_pair(2, false), std::make_pair(3, false), std::make_pair(4, false)}},
    {1, {std::make_pair(0, false), std::make_pair(2, false), std::make_pair(3, false), std::make_pair(4, false)}},
    {2, {std::make_pair(0, false), std::make_pair(1, false), std::make_pair(3, false), std::make_pair(4, false)}},
    {3, {std::make_pair(0, false), std::make_pair(1, false), std::make_pair(2, false), std::make_pair(4, false)}},
    {4, {std::make_pair(0, false), std::make_pair(1, false), std::make_pair(2, false), std::make_pair(3, false)}},
};
const std::map<QoS, std::vector<std::pair<QoS, bool>>> WORKER_SHARE_CONFIG = {
    {1, {std::make_pair(0, false), std::make_pair(2, false), std::make_pair(3, false), std::make_pair(4, false)}},
    {2, {std::make_pair(0, false), std::make_pair(1, false), std::make_pair(3, false), std::make_pair(4, false)}},
    {3, {std::make_pair(0, false), std::make_pair(1, false), std::make_pair(2, false), std::make_pair(4, false)}},
    {4, {std::make_pair(0, false), std::make_pair(1, false), std::make_pair(2, false), std::make_pair(3, false)}},
    {6, {std::make_pair(5, false)}},
};
const std::set<QoS> DEFAULT_TASK_BACKLOG_CONFIG = {0, 1, 2, 3, 4, 5};
const std::set<QoS> TASK_BACKLOG_CONFIG = {0, 2, 4, 5};

ExecuteUnit::ExecuteUnit()
{
    ffrt::CoRoutineInstance(CoStackAttr::Instance()->size);

    workerGroup[qos_deadline_request].tg = std::make_unique<ThreadGroup>();

    for (auto qos = QoS::Min(); qos < QoS::Max(); ++qos) {
        workerGroup[qos].hardLimit = DEFAULT_HARDLIMIT;
        workerGroup[qos].maxConcurrency = GlobalConfig::Instance().getCpuWorkerNum(qos);
    }
#ifdef FFRT_WORKERS_DYNAMIC_SCALING
    memset_s(&domainInfoMonitor, sizeof(domainInfoMonitor), 0, sizeof(domainInfoMonitor));
    wakeupCond.check_ahead = false;
    wakeupCond.global.low = 0;
    wakeupCond.global.high = 0;
    for (int i = 0; i < BLOCKAWARE_DOMAIN_ID_MAX + 1; i++) {
        wakeupCond.local[i].low = 0;
        if (i < qosMonitorMaxNum) {
            wakeupCond.local[i].high = UINT_MAX;
            wakeupCond.global.low += wakeupCond.local[i].low;
            wakeupCond.global.high = UINT_MAX;
        } else {
            wakeupCond.local[i].high = 0;
        }
    }
#endif
    for (int idx = 0; idx < QoS::MaxNum(); idx++) {
        we_[idx] = new WaitUntilEntry;
        we_[idx]->cb = nullptr;
    }

    if (strstr(GetCurrentProcessName(), "CameraDaemon")) {
        SetWorkerShare(WORKER_SHARE_CONFIG);
        SetTaskBacklog(TASK_BACKLOG_CONFIG);
    } else {
        SetWorkerShare(DEFAULT_WORKER_SHARE_CONFIG);
        SetTaskBacklog(DEFAULT_TASK_BACKLOG_CONFIG);
    }
}

ExecuteUnit::~ExecuteUnit()
{
    // worker escape event
    FFRT_LOGI("Destructor.");
    for (int idx = 0; idx < QoS::MaxNum(); idx++) {
        if (we_[idx] != nullptr) {
            delete we_[idx];
            we_[idx] = nullptr;
        }
    }
}

ExecuteUnit &ExecuteUnit::Instance()
{
    return SingletonRegister<ExecuteUnit>::Instance();
}

void ExecuteUnit::RegistInsCb(SingleInsCB<ExecuteUnit>::Instance &&cb)
{
    SingletonRegister<ExecuteUnit>::RegistInsCb(std::move(cb));
}

ThreadGroup *ExecuteUnit::BindTG(QoS& qos)
{
    auto &tgwrap = workerGroup[qos];
    if (!tgwrap.tg) {
        return nullptr;
    }

    std::lock_guard<std::shared_mutex> lck(tgwrap.tgMutex);

    if (tgwrap.tgRefCount++ > 0) {
        return tgwrap.tg.get();
    }

    if (!(tgwrap.tg->Init())) {
        FFRT_SYSEVENT_LOGE("Init Thread Group Failed");
        return tgwrap.tg.get();
    }

    for (auto &thread : tgwrap.threads) {
        pid_t tid = thread.first->Id();
        if (!(tgwrap.tg->Join(tid))) {
            FFRT_SYSEVENT_LOGE("Failed to Join Thread %d", tid);
        }
    }
    return tgwrap.tg.get();
}

void ExecuteUnit::BindWG(QoS& qos)
{
    auto &tgwrap = workerGroup[qos];
    std::shared_lock<std::shared_mutex> lck(tgwrap.tgMutex);
    for (auto &thread : tgwrap.threads) {
        pid_t tid = thread.first->Id();
        if (!JoinWG(tid, qos)) {
            FFRT_SYSEVENT_LOGE("Failed to Join Thread %d", tid);
        }
    }
}

void ExecuteUnit::UnbindTG(QoS& qos)
{
    auto &tgwrap = workerGroup[qos];
    if (!tgwrap.tg) {
        return;
    }

    std::lock_guard<std::shared_mutex> lck(tgwrap.tgMutex);

    if (tgwrap.tgRefCount == 0) {
        return;
    }

    if (--tgwrap.tgRefCount == 0) {
        if (qos != qos_user_interactive) {
            for (auto &thread : tgwrap.threads) {
                pid_t tid = thread.first->Id();
                if (!(tgwrap.tg->Leave(tid))) {
                    FFRT_SYSEVENT_LOGE("Failed to Leave Thread %d", tid);
                }
            }
        }

        if (!(tgwrap.tg->Release())) {
            FFRT_SYSEVENT_LOGE("Release Thread Group Failed");
        }
    }
}

int ExecuteUnit::SetWorkerStackSize(const QoS &qos, size_t stack_size)
{
    CPUWorkerGroup &group = workerGroup[qos];
    std::lock_guard<std::shared_mutex> lck(group.tgMutex);
    if (!group.threads.empty()) {
        FFRT_SYSEVENT_LOGE("stack size can be set only when there is no worker.");
        return -1;
    }
    int pageSize = getpagesize();
    if (pageSize < 0) {
        FFRT_SYSEVENT_LOGE("Invalid pagesize : %d", pageSize);
        return -1;
    }
    group.workerStackSize = (stack_size - 1 + static_cast<size_t>(pageSize)) & -(static_cast<size_t>(pageSize));
    return 0;
}

void ClampValue(uint64_t& value, uint64_t maxValue)
{
    if (value > maxValue) {
        FFRT_LOGW("exceeds maximum allowed value %llu ms. Clamping to %llu ms.", value, maxValue);
        value = maxValue;
    }
}

int ExecuteUnit::SetEscapeEnable(uint64_t oneStageIntervalMs, uint64_t twoStageIntervalMs,
    uint64_t threeStageIntervalMs, uint64_t oneStageWorkerNum, uint64_t twoStageWorkerNum)
{
    if (escapeConfig.enableEscape_) {
        FFRT_LOGW("Worker escape is enabled, the interface cannot be invoked repeatedly.");
        return 1;
    }

    if (oneStageIntervalMs < escapeConfig.oneStageIntervalMs_ ||
        twoStageIntervalMs < escapeConfig.twoStageIntervalMs_ ||
        threeStageIntervalMs < escapeConfig.threeStageIntervalMs_ || oneStageWorkerNum > twoStageWorkerNum) {
        FFRT_LOGE("Setting failed, each stage interval value [%lu, %lu, %lu] "
                  "cannot be smaller than default value [%lu, %lu, %lu], "
                  "and one-stage worker number [%lu] cannot be larger than two-stage worker number [%lu].",
            oneStageIntervalMs,
            twoStageIntervalMs,
            threeStageIntervalMs,
            escapeConfig.oneStageIntervalMs_,
            escapeConfig.twoStageIntervalMs_,
            escapeConfig.threeStageIntervalMs_,
            oneStageWorkerNum,
            twoStageWorkerNum);
        return 1;
    }

    ClampValue(oneStageIntervalMs, MAX_ESCAPE_INTERVAL_MS_COUNT);
    ClampValue(twoStageIntervalMs, MAX_ESCAPE_INTERVAL_MS_COUNT);
    ClampValue(threeStageIntervalMs, MAX_ESCAPE_INTERVAL_MS_COUNT);

    escapeConfig.enableEscape_ = true;
    escapeConfig.oneStageIntervalMs_ = oneStageIntervalMs;
    escapeConfig.twoStageIntervalMs_ = twoStageIntervalMs;
    escapeConfig.threeStageIntervalMs_ = threeStageIntervalMs;
    escapeConfig.oneStageWorkerNum_ = oneStageWorkerNum;
    escapeConfig.twoStageWorkerNum_ = twoStageWorkerNum;
    FFRT_LOGI("Enable worker escape success, one-stage interval ms %lu, two-stage interval ms %lu, "
              "three-stage interval ms %lu, one-stage worker number %lu, two-stage worker number %lu.",
        escapeConfig.oneStageIntervalMs_,
        escapeConfig.twoStageIntervalMs_,
        escapeConfig.threeStageIntervalMs_,
        escapeConfig.oneStageWorkerNum_,
        escapeConfig.twoStageWorkerNum_);
    return 0;
}

void ExecuteUnit::SubmitEscape(int qos, uint64_t totalWorkerNum)
{
    // escape event has been triggered and will not be submitted repeatedly
    if (submittedDelayedTask_[qos]) {
        return;
    }

    we_[qos]->tp = std::chrono::steady_clock::now() + std::chrono::milliseconds(CalEscapeInterval(totalWorkerNum));
    if (we_[qos]->cb == nullptr) {
        we_[qos]->cb = [this, qos](WaitEntry *we) {
            (void)we;
            ExecuteEscape(qos);
            submittedDelayedTask_[qos].store(false, std::memory_order_relaxed);
        };
    }

    if (!DelayedWakeup(we_[qos]->tp, we_[qos], we_[qos]->cb, true)) {
        FFRT_LOGW("Failed to set qos %d escape task.", qos);
        return;
    }

    submittedDelayedTask_[qos].store(true, std::memory_order_relaxed);
}

std::array<std::atomic<sched_mode_type>, QoS::MaxNum()> ExecuteUnit::schedMode{};

bool ExecuteUnit::IncWorker(const QoS &qos)
{
    int workerQos = qos();
    if (workerQos < 0 || workerQos >= QoS::MaxNum()) {
        FFRT_SYSEVENT_LOGE("IncWorker qos:%d is invaild", workerQos);
        return false;
    }
    if (tearDown) {
        FFRT_SYSEVENT_LOGE("CPU Worker Manager exit");
        return false;
    }

    workerNum.fetch_add(1);
    auto worker = CreateCPUWorker(qos);
    auto uniqueWorker = std::unique_ptr<CPUWorker>(worker);
    if (uniqueWorker == nullptr) {
        workerNum.fetch_sub(1);
        FFRT_SYSEVENT_LOGE("IncWorker failed: worker is nullptr\n");
        return false;
    }
    {
        std::lock_guard<std::shared_mutex> lock(workerGroup[workerQos].tgMutex);
        if (uniqueWorker->Exited()) {
            FFRT_SYSEVENT_LOGW("IncWorker failed: worker has exited\n");
            goto create_success;
        }

        auto result = workerGroup[workerQos].threads.emplace(worker, std::move(uniqueWorker));
        if (!result.second) {
            FFRT_SYSEVENT_LOGW("qos:%d worker insert fail:%d", workerQos, result.second);
        }
    }
create_success:
#ifdef FFRT_WORKER_MONITOR
    FFRTFacade::GetWMInstance().SubmitTask();
#endif
    FFRTTraceRecord::UseFfrt();
    return true;
}

void ExecuteUnit::RestoreThreadConfig()
{
    for (auto qos = ffrt::QoS::Min(); qos < ffrt::QoS::Max(); ++qos) {
        ffrt::CPUWorkerGroup &group = workerGroup[qos];
        std::lock_guard<std::shared_mutex> lck(group.tgMutex);
        for (auto &thread : group.threads) {
            thread.first->SetThreadAttr(qos);
        }
    }
}

void ExecuteUnit::NotifyWorkers(const QoS &qos, int number)
{
    CPUWorkerGroup &group = workerGroup[qos];
    std::lock_guard lg(group.lock);
    int increasableNumber = static_cast<int>(group.maxConcurrency) - (group.executingNum + group.sleepingNum);
    int wakeupNumber = std::min(number, group.sleepingNum);
    for (int idx = 0; idx < wakeupNumber; idx++) {
        WakeupWorkers(qos);
    }

    int incNumber = std::min(number - wakeupNumber, increasableNumber);
    for (int idx = 0; idx < incNumber; idx++) {
        group.executingNum++;
        IncWorker(qos);
    }
    FFRT_LOGD("qos[%d] inc [%d] workers, wakeup [%d] workers", static_cast<int>(qos), incNumber, wakeupNumber);
}

bool ExecuteUnit::WorkerShare(CPUWorker* worker, std::function<bool(int, CPUWorker*)> taskFunction)
{
    for (const auto& pair : workerGroup[worker->GetQos()].workerShareConfig) {
        int shareQos = pair.first;
        bool isChangePriority = pair.second;
        if (!isChangePriority) {
            if (taskFunction(shareQos, worker)) {
                return true;
            }
        } else {}
    }
    return false;
}

void ExecuteUnit::WorkerRetired(CPUWorker *thread)
{
    thread->SetWorkerState(WorkerStatus::DESTROYED);
    pid_t tid = thread->Id();
    int qos = static_cast<int>(thread->GetQos());

    {
        std::lock_guard<std::shared_mutex> lck(workerGroup[qos].tgMutex);
        thread->SetExited();
        thread->Detach();
        auto worker = std::move(workerGroup[qos].threads[thread]);
        int ret = workerGroup[qos].threads.erase(thread);
        if (ret != 1) {
            FFRT_SYSEVENT_LOGE("erase qos[%d] thread failed, %d elements removed", qos, ret);
        }
        WorkerLeaveTg(qos, tid);
#ifdef FFRT_WORKERS_DYNAMIC_SCALING
        if (IsBlockAwareInit()) {
            ret = BlockawareUnregister();
            if (ret != 0) {
                FFRT_SYSEVENT_LOGE("blockaware unregister fail, ret[%d]", ret);
            }
        }
#endif
        worker = nullptr;
        workerNum.fetch_sub(1);
    }
    FFRT_LOGD("to exit, qos[%d], tid[%d]", qos, tid);
}

void ExecuteUnit::WorkerJoinTg(const QoS &qos, pid_t pid)
{
    std::shared_lock<std::shared_mutex> lock(workerGroup[qos()].tgMutex);
    if (qos == qos_user_interactive || qos > qos_max) {
        (void)JoinWG(pid, qos);
        return;
    }
    auto &tgwrap = workerGroup[qos()];
    if (!tgwrap.tg) {
        return;
    }

    if ((tgwrap.tgRefCount) == 0) {
        return;
    }

    tgwrap.tg->Join(pid);
}

void ExecuteUnit::WorkerLeaveTg(const QoS &qos, pid_t pid)
{
    if (qos == qos_user_interactive || qos > qos_max) {
        (void)LeaveWG(pid, qos);
        return;
    }
    auto &tgwrap = workerGroup[qos()];
    if (!tgwrap.tg) {
        return;
    }

    if ((tgwrap.tgRefCount) == 0) {
        return;
    }

    tgwrap.tg->Leave(pid);
}

CPUWorker *ExecuteUnit::CreateCPUWorker(const QoS &qos)
{
    // default strategy of worker ops
    CpuWorkerOps ops{
        [this](CPUWorker *thread) { return this->WorkerIdleAction(thread); },
        [this](CPUWorker *thread) { this->WorkerRetired(thread); },
        [this](CPUWorker *thread) { this->WorkerPrepare(thread); },
#ifdef FFRT_WORKERS_DYNAMIC_SCALING
        [this]() { return this->IsBlockAwareInit(); },
#endif
    };

    return new (std::nothrow) CPUWorker(qos, std::move(ops), workerGroup[qos].workerStackSize);
}

#ifdef FFRT_WORKERS_DYNAMIC_SCALING
bool ExecuteUnit::IsBlockAwareInit()
{
    return blockAwareInit;
}

BlockawareWakeupCond *ExecuteUnit::WakeupCond(void)
{
    return &wakeupCond;
}

void ExecuteUnit::MonitorMain()
{
    (void)WorkerInit();
    int ret = BlockawareLoadSnapshot(keyPtr, &domainInfoMonitor);
    if (ret != 0) {
        FFRT_SYSEVENT_LOGE("blockaware load snapshot fail, ret[%d]", ret);
        return;
    }
    for (int i = 0; i < qosMonitorMaxNum; i++) {
        auto &info = domainInfoMonitor.localinfo[i];
        if (info.nrRunning <= wakeupCond.local[i].low &&
            (info.nrRunning + info.nrBlocked + info.nrSleeping) < MAX_ESCAPE_WORKER_NUM) {
            NotifyTask<TaskNotifyType::TASK_ESCAPED>(i);
        }
    }
    stopMonitor = true;
}
#endif

size_t ExecuteUnit::GetRunningNum(const QoS &qos)
{
    CPUWorkerGroup &group = workerGroup[qos()];
    size_t runningNum = group.executingNum;

#ifdef FFRT_WORKERS_DYNAMIC_SCALING
    /* There is no need to update running num when executingNum < maxConcurrency */
    if (static_cast<size_t>(group.executingNum) >= group.maxConcurrency && blockAwareInit) {
        auto nrBlocked = BlockawareLoadSnapshotNrBlockedFast(keyPtr, qos());
        if (static_cast<unsigned int>(group.executingNum) >= nrBlocked) {
            /* nrRunning may not be updated in a timely manner */
            runningNum = group.executingNum - nrBlocked;
        } else {
            FFRT_SYSEVENT_LOGE(
                "qos [%d] nrBlocked [%u] is larger than executingNum [%d].", qos(), nrBlocked, group.executingNum);
        }
    }
#endif

    return runningNum;
}

void ExecuteUnit::ReportEscapeEvent(int qos, size_t totalNum)
{
#ifdef FFRT_SEND_EVENT
    WorkerEscapeReport(GetCurrentProcessName(), qos, totalNum);
#endif
}

void ExecuteUnit::WorkerStart(int qos)
{
    auto& workerStatusInfo = g_workerStatusInfo[qos];
    std::lock_guard lk(g_workerStatusMutex[qos]);
    workerStatusInfo.startedCnt++;
    auto& tids = workerStatusInfo.startedTids;
    tids.push_back(ExecuteCtx::Cur()->tid);
    if (tids.size() > MAX_TID_SIZE) {
        tids.pop_front();
    }
}

void ExecuteUnit::WorkerExit(int qos)
{
    auto& workerStatusInfo = g_workerStatusInfo[qos];
    std::lock_guard lk(g_workerStatusMutex[qos]);
    workerStatusInfo.exitedCnt++;
    auto& tids = workerStatusInfo.exitedTids;
    tids.push_back(ExecuteCtx::Cur()->tid);
    if (tids.size() > MAX_TID_SIZE) {
        tids.pop_front();
    }
}

WorkerStatusInfo ExecuteUnit::GetWorkerStatusInfoAndReset(int qos)
{
    auto& workerStatusInfo = g_workerStatusInfo[qos];
    WorkerStatusInfo result;
    std::lock_guard<fast_mutex> lock(g_workerStatusMutex[qos]);
    result = workerStatusInfo;
    workerStatusInfo.startedCnt = 0;
    workerStatusInfo.exitedCnt = 0;
    std::deque<pid_t> startedEmptyDeque;
    workerStatusInfo.startedTids.swap(startedEmptyDeque);
    std::deque<pid_t> exitedEmptyDeque;
    workerStatusInfo.exitedTids.swap(exitedEmptyDeque);
    return result;
}
} // namespace ffrt
