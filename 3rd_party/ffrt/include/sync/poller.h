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

#ifndef HICORO_POLLER_H
#define HICORO_POLLER_H
#ifndef _MSC_VER
#include <sys/epoll.h>
#include <sys/eventfd.h>
#endif
#include <list>
#include <map>
#include <unordered_map>
#include <array>
#ifdef USE_OHOS_QOS
#include "qos.h"
#else
#include "staging_qos/sched/qos.h"
#endif
#include "sync/sync.h"
#include "tm/task_base.h"
#include "internal_inc/non_copyable.h"
#include "c/executor_task.h"
#include "c/timer.h"
#ifdef FFRT_ENABLE_HITRACE_CHAIN
#include "dfx/trace/ffrt_trace_chain.h"
#endif

namespace ffrt {
enum class PollerRet {
    RET_NULL,
    RET_EPOLL,
    RET_TIMER,
};

enum class EpollStatus {
    WAIT,
    WAKE,
    TEARDOWN,
};

enum class TimerStatus {
    EXECUTING,
    EXECUTED,
};

constexpr int EPOLL_EVENT_SIZE = 1024;

struct WakeDataWithCb {
    WakeDataWithCb() {}
    WakeDataWithCb(int fdVal, void *dataVal, std::function<void(void *, uint32_t)> cbVal, CoTask *taskVal)
        : fd(fdVal), data(dataVal), cb(cbVal), task(taskVal)
    {
        if (cb != nullptr) {
#ifdef FFRT_ENABLE_HITRACE_CHAIN
            if (TraceChainAdapter::Instance().HiTraceChainGetId().valid == HITRACE_ID_VALID) {
                traceId = TraceChainAdapter::Instance().HiTraceChainCreateSpan();
            };
#endif
        }
    }

    int fd = 0;
    void* data = nullptr;
    std::function<void(void*, uint32_t)> cb = nullptr;
    CoTask* task = nullptr;
    uint32_t monitorEvents = 0;
    HiTraceIdStruct traceId;
};

struct TimerDataWithCb {
    TimerDataWithCb() {}
    TimerDataWithCb(void *dataVal, std::function<void(void *)> cbVal, CoTask *taskVal, bool repeat, uint64_t timeout)
        : data(dataVal), cb(cbVal), task(taskVal), repeat(repeat), timeout(timeout)
    {
        if (cb != nullptr) {
#ifdef FFRT_ENABLE_HITRACE_CHAIN
            if (TraceChainAdapter::Instance().HiTraceChainGetId().valid == HITRACE_ID_VALID) {
                traceId = TraceChainAdapter::Instance().HiTraceChainCreateSpan();
            };
#endif
        }
    }

    void* data = nullptr;
    std::function<void(void*)> cb = nullptr;
    int handle = -1;
    CoTask* task = nullptr;
    bool repeat = false;
    uint64_t timeout = 0;
    HiTraceIdStruct traceId;
};

struct SyncData {
    SyncData() {}
    SyncData(void *eventsPtr, int maxEvents, int *nfdsPtr, TimePoint waitTP)
        : eventsPtr(eventsPtr), maxEvents(maxEvents), nfdsPtr(nfdsPtr), waitTP(waitTP)
    {}

    void* eventsPtr = nullptr;
    int maxEvents = 0;
    int* nfdsPtr = nullptr;
    TimePoint waitTP;
    int timerHandle = -1;
};

using EventVec = typename std::vector<epoll_event>;
class Poller : private NonCopyable {
    using WakeDataList = typename std::list<std::unique_ptr<struct WakeDataWithCb>>;
public:
    Poller() noexcept;
    ~Poller() noexcept;

    int AddFdEvent(int op, uint32_t events, int fd, void* data, ffrt_poller_cb cb) noexcept;
    int DelFdEvent(int fd) noexcept;
    int WaitFdEvent(struct epoll_event *eventsVec, int maxevents, int timeout) noexcept;

    PollerRet PollOnce(int timeout = -1) noexcept;
    void WakeUp() noexcept;

    int RegisterTimer(uint64_t timeout, void* data, ffrt_timer_cb cb, bool repeat = false) noexcept;
    int UnregisterTimer(int handle) noexcept;
    ffrt_timer_query_t GetTimerStatus(int handle) noexcept;

    uint64_t GetPollCount() noexcept;

    uint64_t GetTaskWaitTime(CoTask* task) noexcept;

    bool DetermineEmptyMap() noexcept;
    bool DeterminePollerReady() noexcept;

    void ClearCachedEvents(CoTask* task) noexcept;

private:
    void ReleaseFdWakeData() noexcept;
    void WakeSyncTask(std::unordered_map<CoTask*, EventVec>& syncTaskEvents) noexcept;
    void ProcessWaitedFds(int nfds, std::unordered_map<CoTask*, EventVec>& syncTaskEvents,
                          std::array<epoll_event, EPOLL_EVENT_SIZE>& waitedEvents) noexcept;

    void ExecuteTimerCb(TimePoint timer) noexcept;
    void ProcessTimerDataCb(CoTask* task) noexcept;
    void RegisterTimerImpl(const TimerDataWithCb& data) noexcept;

    void CacheEventsAndDoMask(CoTask* task, EventVec& eventVec) noexcept;
    int FetchCachedEventAndDoUnmask(CoTask* task, struct epoll_event* eventsVec) noexcept;
    int FetchCachedEventAndDoUnmask(EventVec& cachedEventsVec, struct epoll_event* eventsVec) noexcept;

    inline void CacheDelFd(int fd, CoTask *task) noexcept
    {
        m_delFdCacheMap.emplace(fd, task);
    }

    inline void CacheMaskWakeData(CoTask* task, std::unique_ptr<struct WakeDataWithCb>& maskWakeData) noexcept
    {
        m_maskWakeDataWithCbMap[task].emplace_back(std::move(maskWakeData));
    }

    void CacheMaskFdAndEpollDel(int fd, CoTask *task) noexcept;
    int ClearMaskWakeDataWithCbCache(CoTask *task) noexcept;
    int ClearMaskWakeDataWithCbCacheWithFd(CoTask *task, int fd) noexcept;
    int ClearDelFdCache(int fd) noexcept;

    bool IsFdExist() noexcept;
    bool IsTimerReady() noexcept;

    int m_epFd;
    std::atomic<uint64_t> pollerCount_ = 0;
    int timerHandle_ = -1;
    std::atomic<EpollStatus> flag_ = EpollStatus::WAKE;
    struct WakeDataWithCb m_wakeData;
    std::unordered_map<int, WakeDataList> m_wakeDataMap;
    std::unordered_map<int, int> m_delCntMap;
    std::unordered_map<CoTask*, SyncData> m_waitTaskMap;
    std::unordered_map<CoTask*, EventVec> m_cachedTaskEvents;

    std::unordered_map<int, CoTask*> m_delFdCacheMap;
    std::unordered_map<CoTask*, WakeDataList> m_maskWakeDataWithCbMap;

    std::unordered_map<int, TimerStatus> executedHandle_;
    std::multimap<TimePoint, TimerDataWithCb> timerMap_;
    std::atomic_bool fdEmpty_ {true};
    std::atomic_bool timerEmpty_ {true};
    mutable spin_mutex m_mapMutex;
    mutable spin_mutex timerMutex_;
};

struct PollerProxy {
public:
    static PollerProxy& Instance();

    Poller& GetPoller(const QoS& qos = QoS(ffrt_qos_default))
    {
        return qosPollers[static_cast<size_t>(qos())];
    }

private:
    std::array<Poller, QoS::MaxNum()> qosPollers;
};
} // namespace ffrt
#endif