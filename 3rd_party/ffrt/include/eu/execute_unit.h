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

#ifndef FFRT_EXECUTE_UNIT_HPP
#define FFRT_EXECUTE_UNIT_HPP

#include <memory>
#include <atomic>
#include <deque>
#include <vector>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <unordered_map>
#include <set>
#include <map>
#include <array>
#include "cpp/mutex.h"
#include "sched/workgroup_internal.h"
#include "eu/thread_group.h"
#include "eu/cpu_worker.h"
#include "sync/sync.h"
#include "internal_inc/osal.h"
#include "util/cb_func.h"
#ifdef FFRT_WORKERS_DYNAMIC_SCALING
#include "eu/blockaware.h"
#endif

namespace {
constexpr uint64_t ONE_STAGE_INTERVAL = 10;
constexpr uint64_t TWO_STAGE_INTERVAL = 100;
constexpr uint64_t THREE_STAGE_INTERVAL = 1000;
constexpr uint64_t ONE_STAGE_WORKER_NUM = 128;
constexpr uint64_t TWO_STAGE_WORKER_NUM = 256;
constexpr int DEEP_SLEEP_NUM_DOUBLE = 2;
}

namespace ffrt {
enum class TaskNotifyType {
    TASK_PICKED = 0,
    TASK_ADDED,
    TASK_LOCAL,
    TASK_ESCAPED,
    TASK_ADDED_RTQ,
};

struct WorkerStatusInfo {
    unsigned int startedCnt = 0;
    unsigned int exitedCnt = 0;
    std::deque<pid_t> startedTids;
    std::deque<pid_t> exitedTids;
};

struct CPUWorkerGroup {
    // rtg parameters
    std::unique_ptr<ThreadGroup> tg;
    uint64_t tgRefCount = 0;
    mutable std::shared_mutex tgMutex;

    // worker manage parameters
    size_t hardLimit{0};
    size_t maxConcurrency{0};
    size_t workerStackSize{0};
    bool setWorkerMaxNum{false};
    std::unordered_map<CPUWorker *, std::unique_ptr<CPUWorker>> threads;
    std::mutex mutex;
    std::condition_variable cv;

    // group status parameters
    alignas(cacheline_size) fast_mutex lock;
    alignas(cacheline_size) int executingNum{0};
    alignas(cacheline_size) int sleepingNum{0};
    alignas(cacheline_size) bool irqEnable{false};
    /* used for performance mode */
    alignas(cacheline_size) bool fastWakeEnable = false; // directly wakeup first worker by futex
    alignas(cacheline_size) int pendingWakeCnt = 0;      // number of workers waking but not waked-up yet
    alignas(cacheline_size) int pendingTaskCnt = 0;      // number of tasks submitted to RTB but not picked-up yet

    // used for worker share
    std::vector<std::pair<QoS, bool>> workerShareConfig;
    int deepSleepingWorkerNum{0};
    bool retryBeforeDeepSleep{true};

    inline void WorkerCreate()
    {
        executingNum++;
    }

    inline void RollBackCreate()
    {
        std::lock_guard lk(lock);
        executingNum--;
    }

    inline void IntoDeepSleep()
    {
        std::lock_guard lk(lock);
        deepSleepingWorkerNum++;
    }

    inline void OutOfDeepSleep(bool irqWake = false)
    {
        std::lock_guard lk(lock);
        if (irqWake) {
            irqEnable = false;
        }
        sleepingNum--;
        deepSleepingWorkerNum--;
        executingNum++;
    }

    inline void OutOfSleep(bool irqWake = false)
    {
        std::lock_guard lk(lock);
        if (irqWake) {
            irqEnable = false;
        }
        if (pendingWakeCnt > 0) {
            pendingWakeCnt--;
        }
        sleepingNum--;
        executingNum++;
    }

    inline void WorkerDestroy()
    {
        std::lock_guard lk(lock);
        sleepingNum--;
    }

    inline bool TryDestroy()
    {
        std::lock_guard lk(lock);
        sleepingNum--;
        return sleepingNum > 0;
    }

    inline void RollbackDestroy(bool irqWake = false)
    {
        std::lock_guard lk(lock);
        if (irqWake) {
            irqEnable = false;
        }
        executingNum++;
    }

    inline void SetTearDown()
    {
        std::shared_lock<std::shared_mutex> lck(tgMutex);
        for (const auto& pair : threads) {
            pair.second->SetExited();
        }
    }
};

struct EscapeConfig {
    bool enableEscape_ = false;
    uint64_t oneStageIntervalMs_ = ONE_STAGE_INTERVAL;
    uint64_t twoStageIntervalMs_ = TWO_STAGE_INTERVAL;
    uint64_t threeStageIntervalMs_ = THREE_STAGE_INTERVAL;
    uint64_t oneStageWorkerNum_ = ONE_STAGE_WORKER_NUM;
    uint64_t twoStageWorkerNum_ = TWO_STAGE_WORKER_NUM;
};

class ExecuteUnit {
public:
    static ExecuteUnit &Instance();

    static void RegistInsCb(SingleInsCB<ExecuteUnit>::Instance &&cb);

    ThreadGroup *BindTG(QoS& qos);
    void UnbindTG(QoS& qos);
    void BindWG(QoS& qos);

    // event notify
    template <TaskNotifyType TYPE>
    void NotifyTask(const QoS &qos, bool isPollWait = false, bool isRisingEdge = false)
    {
        if constexpr (TYPE == TaskNotifyType::TASK_ADDED) {
            PokeAdd(qos);
        } else if constexpr (TYPE == TaskNotifyType::TASK_PICKED) {
            PokePick(qos);
        } else if constexpr (TYPE == TaskNotifyType::TASK_ESCAPED) {
            PokeEscape(qos, isPollWait);
        } else if constexpr (TYPE == TaskNotifyType::TASK_LOCAL) {
            PokeLocal(qos);
        } else if constexpr (TYPE == TaskNotifyType::TASK_ADDED_RTQ) {
            PokeAddRtq(qos, isRisingEdge);
        }
    }

    // dfx op
    virtual void WorkerInit() = 0;
    CPUWorkerGroup &GetWorkerGroup(int qos)
    {
        return workerGroup[qos];
    }

    inline int SetWorkerMaxNum(const QoS &qos, uint32_t num)
    {
        CPUWorkerGroup &group = workerGroup[qos];
        std::lock_guard lk(group.lock);
        if (group.setWorkerMaxNum) {
            FFRT_SYSEVENT_LOGE("qos[%d] worker num can only been setup once", qos());
            return -1;
        }
        group.hardLimit = static_cast<size_t>(num);
        group.setWorkerMaxNum = true;
        return 0;
    }

    int SetWorkerStackSize(const QoS &qos, size_t stack_size);

    // worker escape
    int SetEscapeEnable(uint64_t oneStageIntervalMs, uint64_t twoStageIntervalMs, uint64_t threeStageIntervalMs,
        uint64_t oneStageWorkerNum, uint64_t twoStageWorkerNum);

    inline void SetEscapeDisable()
    {
        escapeConfig.enableEscape_ = false;
        // after the escape function is disabled, parameters are restored to default values
        escapeConfig.oneStageIntervalMs_ = ONE_STAGE_INTERVAL;
        escapeConfig.twoStageIntervalMs_ = TWO_STAGE_INTERVAL;
        escapeConfig.threeStageIntervalMs_ = THREE_STAGE_INTERVAL;
        escapeConfig.oneStageWorkerNum_ = ONE_STAGE_WORKER_NUM;
        escapeConfig.twoStageWorkerNum_ = TWO_STAGE_WORKER_NUM;
    }

    inline bool IsEscapeEnable()
    {
        return escapeConfig.enableEscape_;
    }

    void SubmitEscape(int qos, uint64_t totalWorkerNum);

    inline uint64_t GetWorkerNum()
    {
        return workerNum.load();
    }

    inline void SetSchedMode(const QoS qos, const sched_mode_type mode)
    {
        schedMode[qos].store(mode);
    }

    inline sched_mode_type GetSchedMode(const QoS qos)
    {
        return schedMode[qos].load();
    }

    inline void SetWorkerShare(const std::map<QoS, std::vector<std::pair<QoS, bool>>> workerShareConfig)
    {
        for (const auto& item : workerShareConfig) {
            workerGroup[item.first].workerShareConfig = item.second;
        }
    }

    inline void SetTaskBacklog(const std::set<QoS> userTaskBacklogConfig)
    {
        for (const QoS& qos : userTaskBacklogConfig) {
            taskBacklogConfig[qos] = true;
        }
    }

    void RestoreThreadConfig();

    void NotifyWorkers(const QoS &qos, int number);
    // used for worker sharing
    bool WorkerShare(CPUWorker* worker, std::function<bool(int, CPUWorker*)> taskFunction);
// worker dynamic scaling
#ifdef FFRT_WORKERS_DYNAMIC_SCALING
    void MonitorMain();
    BlockawareWakeupCond *WakeupCond(void);
#endif
    void WorkerStart(int qos);
    void WorkerExit(int qos);
    WorkerStatusInfo GetWorkerStatusInfoAndReset(int qos);

protected:
    virtual void WakeupWorkers(const QoS &qos) = 0;

    // worker manipulate op
    bool IncWorker(const QoS &qos);
    virtual void WorkerPrepare(CPUWorker *thread) = 0;
    virtual WorkerAction WorkerIdleAction(CPUWorker *thread) = 0;
    void WorkerRetired(CPUWorker *thread);

    // worker rtg config
    void WorkerJoinTg(const QoS &qos, pid_t pid);
    void WorkerLeaveTg(const QoS &qos, pid_t pid);

    // worker group info
    inline bool IsExceedDeepSleepThreshold()
    {
        int totalWorker = 0;
        int deepSleepingWorkerNum = 0;
        for (unsigned int i = 0; i < static_cast<unsigned int>(QoS::Max()); i++) {
            CPUWorkerGroup &group = workerGroup[i];
            std::lock_guard lk(group.lock);
            deepSleepingWorkerNum += group.deepSleepingWorkerNum;
            totalWorker += group.executingNum + group.sleepingNum;
        }
        return deepSleepingWorkerNum * DEEP_SLEEP_NUM_DOUBLE > totalWorker;
    }

    // worker group state
    virtual void IntoSleep(const QoS &qos) = 0;

    ExecuteUnit();
    virtual ~ExecuteUnit();

    size_t GetRunningNum(const QoS &qos);
    void ReportEscapeEvent(int qos, size_t totalNum);

    CPUWorkerGroup workerGroup[QoS::MaxNum()];
    std::atomic_uint64_t workerNum = 0;
    std::atomic_bool tearDown{false};
#ifdef FFRT_WORKERS_DYNAMIC_SCALING
    bool blockAwareInit{false};
    bool stopMonitor{false};
    unsigned long keyPtr{0};
    int qosMonitorMaxNum{std::min(QoS::Max(), BLOCKAWARE_DOMAIN_ID_MAX + 1)};
    BlockawareWakeupCond wakeupCond;
    BlockawareDomainInfoArea domainInfoMonitor;
#endif

    // eu sched task bacllog array
    bool taskBacklogConfig[QoS::MaxNum()] = {};
private:
    CPUWorker *CreateCPUWorker(const QoS &qos);

    virtual void PokeAdd(const QoS &qos) = 0;
    virtual void PokePick(const QoS &qos) = 0;
    virtual void PokeLocal(const QoS &qos) = 0;
    virtual void PokeEscape(const QoS &qos, bool isPollWait) = 0;
    virtual void PokeAddRtq(const QoS &qos, bool isRisingEdge) = 0;

    // eu sched mode array
    static std::array<std::atomic<sched_mode_type>, QoS::MaxNum()> schedMode;

    // worker escape
    EscapeConfig escapeConfig;
    std::atomic<bool> submittedDelayedTask_[QoS::MaxNum()] = {0};
    WaitUntilEntry *we_[QoS::MaxNum()] = {nullptr};
    virtual void ExecuteEscape(int qos) = 0;

    inline uint64_t CalEscapeInterval(uint64_t totalWorkerNum)
    {
        if (totalWorkerNum < escapeConfig.oneStageWorkerNum_) {
            return escapeConfig.oneStageIntervalMs_;
        } else if (totalWorkerNum >= escapeConfig.oneStageWorkerNum_ &&
            totalWorkerNum < escapeConfig.twoStageWorkerNum_) {
            return escapeConfig.twoStageIntervalMs_;
        } else {
            return escapeConfig.threeStageIntervalMs_;
        }
    }

#ifdef FFRT_WORKERS_DYNAMIC_SCALING
    bool IsBlockAwareInit(void);
#endif
};
} // namespace ffrt
#endif
