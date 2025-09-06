/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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

#ifndef FFRT_POLLER_MANAGER_H
#define FFRT_POLLER_MANAGER_H

#ifndef _MSC_VER
#include <sys/epoll.h>
#include <sys/eventfd.h>
#endif
#include <list>
#include <thread>
#include <unordered_map>
#include <array>
#ifdef USE_OHOS_QOS
#include "qos.h"
#else
#include "staging_qos/sched/qos.h"
#endif
#include "sync/sync.h"
#include "internal_inc/non_copyable.h"
#include "c/executor_task.h"
#include "sync/poller.h"
#include "tm/task_base.h"
#ifdef FFRT_ENABLE_HITRACE_CHAIN
#include "dfx/trace/ffrt_trace_chain.h"
#endif

namespace ffrt {
enum class PollerState {
    HANDLING, // worker执行事件回调（如果是同步回调函数执行有可能阻塞worker）
    POLLING, // worker处于epoll_wait睡眠（事件响应）
    EXITED, // worker没有事件时销毁线程（重新注册时触发创建线程）
};

// 根据历史继承的能力
enum class PollerType {
    WAKEUP,
    SYNC_IO,
    ASYNC_CB,
    ASYNC_IO,
};

struct WakeData {
    WakeData() {}
    WakeData(int fdVal, CoTask *taskVal) : fd(fdVal), task(taskVal)
    {
        mode = PollerType::SYNC_IO;
    }
    WakeData(int fdVal, void *dataVal, std::function<void(void *, uint32_t)> cbVal, CoTask *taskVal)
        : fd(fdVal), data(dataVal), cb(cbVal), task(taskVal)
    {
        if (cb == nullptr) {
            mode = PollerType::ASYNC_IO;
        } else {
            mode = PollerType::ASYNC_CB;
#ifdef FFRT_ENABLE_HITRACE_CHAIN
            if (TraceChainAdapter::Instance().HiTraceChainGetId().valid == HITRACE_ID_VALID) {
                traceId = TraceChainAdapter::Instance().HiTraceChainCreateSpan();
            };
#endif
        }
    }

    PollerType mode;
    int fd = 0;
    void* data = nullptr;
    std::function<void(void*, uint32_t)> cb = nullptr;
    CoTask* task = nullptr;
    uint32_t monitorEvents = 0;
    HiTraceIdStruct traceId;
};

struct TimeOutReport {
    TimeOutReport() {}
    std::atomic<uint64_t> cbStartTime = 0; // block info report
    uint64_t reportCount = 0;
};

using EventVec = typename std::vector<epoll_event>;
class IOPoller : private NonCopyable {
    static constexpr int EPOLL_EVENT_SIZE = 1024;
    using WakeDataList = typename std::list<std::unique_ptr<struct WakeData>>;
public:
    static IOPoller& Instance();
    ~IOPoller() noexcept;

    int AddFdEvent(int op, uint32_t events, int fd, void* data, ffrt_poller_cb cb) noexcept;
    int DelFdEvent(int fd) noexcept;
    int WaitFdEvent(struct epoll_event *eventsVec, int maxevents, int timeout) noexcept;
    void WaitFdEvent(int fd) noexcept;

    inline uint64_t GetPollCount() noexcept
    {
        return pollerCount_;
    }

    inline uint64_t GetTaskWaitTime(CoTask* task) noexcept
    {
        std::lock_guard lock(m_mapMutex);
        auto iter = m_waitTaskMap.find(task);
        if (iter == m_waitTaskMap.end()) {
            return 0;
        }
        return std::chrono::duration_cast<std::chrono::seconds>(
            iter->second.waitTP.time_since_epoch()).count();
    }

    inline void ClearCachedEvents(CoTask* task) noexcept
    {
        std::lock_guard lock(m_mapMutex);
        auto iter = m_cachedTaskEvents.find(task);
        if (iter == m_cachedTaskEvents.end()) {
            return;
        }
        m_cachedTaskEvents.erase(iter);
        ClearMaskWakeDataCache(task);
    }

    void WakeUp() noexcept;
    void WakeTimeoutTask(CoTask* task) noexcept;
    void MonitTimeOut();

private:
    IOPoller() noexcept;

    void ThreadInit();
    void Run();
    int PollOnce(int timeout = -1) noexcept;

    void ReleaseFdWakeData() noexcept;
    void WakeSyncTask(std::unordered_map<CoTask*, EventVec>& syncTaskEvents) noexcept;

    void CacheEventsAndDoMask(CoTask* task, EventVec& eventVec) noexcept;
    int FetchCachedEventAndDoUnmask(CoTask* task, struct epoll_event* eventsVec) noexcept;
    int FetchCachedEventAndDoUnmask(EventVec& cachedEventsVec, struct epoll_event* eventsVec) noexcept;
    void CacheMaskFdAndEpollDel(int fd, CoTask *task) noexcept;
    int ClearMaskWakeDataCache(CoTask *task) noexcept;
    int ClearMaskWakeDataCacheWithFd(CoTask *task, int fd) noexcept;
    int ClearDelFdCache(int fd) noexcept;

    int m_epFd; // epoll fd
    struct WakeData m_wakeData; // self wakeup fd
    mutable spin_mutex m_mapMutex;
    struct TimeOutReport timeOutReport;

    std::atomic_uint64_t m_syncFdCnt { 0 }; // record sync fd counts
    // record async fd and events
    std::unordered_map<int, WakeDataList> m_wakeDataMap;
    std::unordered_map<int, int> m_delCntMap;
    std::unordered_map<CoTask*, SyncData> m_waitTaskMap;
    std::unordered_map<CoTask*, EventVec> m_cachedTaskEvents;
    std::unordered_map<int, CoTask*> m_delFdCacheMap;
    std::unordered_map<CoTask*, WakeDataList> m_maskWakeDataMap;

    std::unique_ptr<std::thread> m_runner { nullptr }; // ffrt_io_poller thread
    bool m_exitFlag { true }; // thread exit
    bool m_teardown { false }; // process teardown
    std::atomic<uint64_t> pollerCount_ { 0 };
    std::atomic<PollerState> m_state { PollerState::EXITED }; // worker state

    std::array<queue*, QoS::MaxNum()> workQue; // queue(per qos) for execute async cb
};
}
#endif
