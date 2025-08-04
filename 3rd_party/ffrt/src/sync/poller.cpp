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

#include "sync/poller.h"
#include <securec.h>
#include "sched/execute_ctx.h"
#include "tm/scpu_task.h"
#include "dfx/log/ffrt_log_api.h"
#ifdef FFRT_ENABLE_HITRACE_CHAIN
#include "dfx/trace/ffrt_trace_chain.h"
#endif

constexpr uint64_t MAX_TIMER_MS_COUNT = 1000ULL * 100 * 60 * 60 * 24 * 365; // 100 year

namespace ffrt {
Poller::Poller() noexcept: m_epFd { ::epoll_create1(EPOLL_CLOEXEC) }
{
    if (m_epFd < 0) {
        FFRT_LOGE("epoll_create1 failed: errno=%d", errno);
    }
#ifdef OHOS_STANDARD_SYSTEM
    fdsan_exchange_owner_tag(m_epFd, 0, fdsan_create_owner_tag(FDSAN_OWNER_TYPE_FILE, static_cast<uint64_t>(m_epFd)));
#endif
    m_wakeData.cb = nullptr;
    m_wakeData.fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (m_wakeData.fd < 0) {
        FFRT_LOGE("eventfd failed: errno=%d", errno);
    }
#ifdef OHOS_STANDARD_SYSTEM
    fdsan_exchange_owner_tag(m_wakeData.fd, 0, fdsan_create_owner_tag(FDSAN_OWNER_TYPE_FILE,
        static_cast<uint64_t>(m_wakeData.fd)));
#endif
    epoll_event ev { .events = EPOLLIN, .data = { .ptr = static_cast<void*>(&m_wakeData) } };
    FFRT_COND_TERMINATE((epoll_ctl(m_epFd, EPOLL_CTL_ADD, m_wakeData.fd, &ev) < 0),
        "epoll_ctl add fd error: efd=%d, fd=%d, errorno=%d", m_epFd, m_wakeData.fd, errno);
}

Poller::~Poller() noexcept
{
#ifdef OHOS_STANDARD_SYSTEM
    fdsan_close_with_tag(m_wakeData.fd, fdsan_create_owner_tag(FDSAN_OWNER_TYPE_FILE,
        static_cast<uint64_t>(m_wakeData.fd)));
    fdsan_close_with_tag(m_epFd, fdsan_create_owner_tag(FDSAN_OWNER_TYPE_FILE, static_cast<uint64_t>(m_epFd)));
#else
    ::close(m_wakeData.fd);
    ::close(m_epFd);
#endif
    timerHandle_ = -1;
    {
        std::lock_guard lg(m_mapMutex);
        m_wakeDataMap.clear();
        m_delCntMap.clear();
        m_waitTaskMap.clear();
        m_cachedTaskEvents.clear();
    }
    {
        std::lock_guard lg(timerMutex_);
        timerMap_.clear();
        executedHandle_.clear();
    }
    flag_ = EpollStatus::TEARDOWN;
}

PollerProxy& PollerProxy::Instance()
{
    static PollerProxy pollerInstance;
    return pollerInstance;
}

int Poller::AddFdEvent(int op, uint32_t events, int fd, void* data, ffrt_poller_cb cb) noexcept
{
    CoTask* task = IsCoTask(ExecuteCtx::Cur()->task) ? static_cast<CoTask*>(ExecuteCtx::Cur()->task) : nullptr;
    auto wakeData = std::make_unique<WakeDataWithCb>(fd, data, cb, task);
    if (task) {
        task->pollerEnable = true;
    }
    void* ptr = static_cast<void*>(wakeData.get());
    if (ptr == nullptr || wakeData == nullptr) {
        FFRT_SYSEVENT_LOGE("Construct WakeDataWithCb instance failed! or wakeData is nullptr");
        return -1;
    }
    wakeData->monitorEvents = events;

    epoll_event ev = { .events = events, .data = { .ptr = ptr } };
    std::lock_guard lg(m_mapMutex);
    if (epoll_ctl(m_epFd, op, fd, &ev) != 0) {
        FFRT_SYSEVENT_LOGE("epoll_ctl add fd error: efd=%d, fd=%d, errorno=%d", m_epFd, fd, errno);
        return -1;
    }

    if (op == EPOLL_CTL_ADD) {
        m_wakeDataMap[fd].emplace_back(std::move(wakeData));
        fdEmpty_.store(false);
    } else if (op == EPOLL_CTL_MOD) {
        auto iter = m_wakeDataMap.find(fd);
        FFRT_COND_RETURN_ERROR(iter == m_wakeDataMap.end(), -1, "fd %d does not exist in wakeDataMap", fd);
        if (iter->second.size() != 1) {
            FFRT_SYSEVENT_LOGE("epoll_ctl mod fd wakedata num invalid");
            return -1;
        }
        iter->second.pop_back();
        iter->second.emplace_back(std::move(wakeData));
    }
    return 0;
}

void Poller::CacheMaskFdAndEpollDel(int fd, CoTask *task) noexcept
{
    auto maskWakeDataWithCb = m_maskWakeDataWithCbMap.find(task);
    if (maskWakeDataWithCb != m_maskWakeDataWithCbMap.end()) {
        if (epoll_ctl(m_epFd, EPOLL_CTL_DEL, fd, nullptr) != 0) {
            FFRT_SYSEVENT_LOGE("fd[%d] ffrt epoll ctl del fail errorno=%d", fd, errno);
        }
        CacheDelFd(fd, task);
    }
}

int Poller::ClearMaskWakeDataWithCbCache(CoTask *task) noexcept
{
    auto maskWakeDataWithCbIter = m_maskWakeDataWithCbMap.find(task);
    if (maskWakeDataWithCbIter != m_maskWakeDataWithCbMap.end()) {
        WakeDataList& wakeDataList = maskWakeDataWithCbIter->second;
        for (auto iter = wakeDataList.begin(); iter != wakeDataList.end(); ++iter) {
            WakeDataWithCb* ptr = iter->get();
            m_delFdCacheMap.erase(ptr->fd);
        }
        m_maskWakeDataWithCbMap.erase(maskWakeDataWithCbIter);
    }
    return 0;
}

int Poller::ClearMaskWakeDataWithCbCacheWithFd(CoTask *task, int fd) noexcept
{
    auto maskWakeDataWithCbIter = m_maskWakeDataWithCbMap.find(task);
    if (maskWakeDataWithCbIter != m_maskWakeDataWithCbMap.end()) {
        WakeDataList& wakeDataList = maskWakeDataWithCbIter->second;
        auto pred = [fd](auto& value) { return value->fd == fd; };
        wakeDataList.remove_if(pred);
        if (wakeDataList.size() == 0) {
            m_maskWakeDataWithCbMap.erase(maskWakeDataWithCbIter);
        }
    }
    return 0;
}

int Poller::ClearDelFdCache(int fd) noexcept
{
    auto fdDelCacheIter = m_delFdCacheMap.find(fd);
    if (fdDelCacheIter != m_delFdCacheMap.end()) {
        CoTask *task = fdDelCacheIter->second;
        ClearMaskWakeDataWithCbCacheWithFd(task, fd);
        m_delFdCacheMap.erase(fdDelCacheIter);
    }
    return 0;
}

int Poller::DelFdEvent(int fd) noexcept
{
    std::lock_guard lg(m_mapMutex);
    ClearDelFdCache(fd);
    auto wakeDataIter = m_wakeDataMap.find(fd);
    if (wakeDataIter == m_wakeDataMap.end() || wakeDataIter->second.size() == 0) {
        FFRT_SYSEVENT_LOGW("fd[%d] has not been added to epoll, ignore", fd);
        return -1;
    }
    auto delCntIter = m_delCntMap.find(fd);
    if (delCntIter != m_delCntMap.end()) {
        int diff = static_cast<int>(wakeDataIter->second.size()) - delCntIter->second;
        if (diff == 0) {
            FFRT_SYSEVENT_LOGW("fd:%d, addCnt:%d, delCnt:%d has not been added to epoll, ignore", fd,
                wakeDataIter->second.size(), delCntIter->second);
            return -1;
        }
    }

    if (epoll_ctl(m_epFd, EPOLL_CTL_DEL, fd, nullptr) != 0) {
        FFRT_SYSEVENT_LOGE("epoll_ctl del fd error: efd=%d, fd=%d, errorno=%d", m_epFd, fd, errno);
        return -1;
    }

    for (auto it = m_cachedTaskEvents.begin(); it != m_cachedTaskEvents.end();) {
        auto& events = it->second;
        events.erase(std::remove_if(events.begin(), events.end(),
            [fd](const epoll_event& event) {
                return event.data.fd == fd;
            }), events.end());

        if (events.empty()) {
            it = m_cachedTaskEvents.erase(it);
        } else {
            ++it;
        }
    }

    m_delCntMap[fd]++;
    WakeUp();
    return 0;
}

void Poller::ClearCachedEvents(CoTask* task) noexcept
{
    std::lock_guard lg(m_mapMutex);
    auto iter = m_cachedTaskEvents.find(task);
    if (iter == m_cachedTaskEvents.end()) {
        return;
    }
    m_cachedTaskEvents.erase(iter);
    ClearMaskWakeDataWithCbCache(task);
}

int Poller::FetchCachedEventAndDoUnmask(EventVec& cachedEventsVec, struct epoll_event* eventsVec) noexcept
{
    std::unordered_map<int, int> seenFd;
    int fdCnt = 0;
    for (size_t i = 0; i < cachedEventsVec.size(); i++) {
        auto eventInfo = cachedEventsVec[i];
        int currFd = eventInfo.data.fd;
        // check if seen
        auto iter = seenFd.find(currFd);
        if (iter == seenFd.end()) {
            // if not seen, copy cached events and record idx
            eventsVec[fdCnt].data.fd = currFd;
            eventsVec[fdCnt].events = eventInfo.events;
            seenFd[currFd] = fdCnt;
            fdCnt++;
        } else {
            // if seen, update event to newest
            eventsVec[iter->second].events |= eventInfo.events;
            FFRT_LOGD("fd[%d] has mutilple cached events", currFd);
            continue;
        }

        // Unmask to origin events
        auto wakeDataIter = m_wakeDataMap.find(currFd);
        if (wakeDataIter == m_wakeDataMap.end() || wakeDataIter->second.size() == 0) {
            FFRT_LOGD("fd[%d] may be deleted", currFd);
            continue;
        }

        auto& wakeData = wakeDataIter->second.back();
        epoll_event ev = { .events = wakeData->monitorEvents, .data = { .ptr = static_cast<void*>(wakeData.get()) } };
        auto fdDelCacheIter = m_delFdCacheMap.find(currFd);
        if (fdDelCacheIter != m_delFdCacheMap.end()) {
            ClearDelFdCache(currFd);
            if (epoll_ctl(m_epFd, EPOLL_CTL_ADD, currFd, &ev) != 0) {
                FFRT_SYSEVENT_LOGE("fd[%d] epoll ctl add fail, errorno=%d", currFd, errno);
                continue;
            }
        } else {
            if (epoll_ctl(m_epFd, EPOLL_CTL_MOD, currFd, &ev) != 0) {
                FFRT_SYSEVENT_LOGE("fd[%d] epoll ctl mod fail, errorno=%d", currFd, errno);
                continue;
            }
        }
    }
    return fdCnt;
}

int Poller::FetchCachedEventAndDoUnmask(CoTask* task, struct epoll_event* eventsVec) noexcept
{
    // should used in lock
    auto syncTaskIter = m_cachedTaskEvents.find(task);
    if (syncTaskIter == m_cachedTaskEvents.end() || syncTaskIter->second.size() == 0) {
        return 0;
    }

    int nfds = FetchCachedEventAndDoUnmask(syncTaskIter->second, eventsVec);
    m_cachedTaskEvents.erase(syncTaskIter);
    ClearMaskWakeDataWithCbCache(task);
    return nfds;
}

int Poller::WaitFdEvent(struct epoll_event* eventsVec, int maxevents, int timeout) noexcept
{
    FFRT_COND_DO_ERR((eventsVec == nullptr), return -1, "eventsVec cannot be null");

    CoTask* task = IsCoTask(ExecuteCtx::Cur()->task) ? static_cast<CoTask*>(ExecuteCtx::Cur()->task) : nullptr;
    if (!task) {
        FFRT_SYSEVENT_LOGE("nonworker shall not call this fun.");
        return -1;
    }

    FFRT_COND_DO_ERR((maxevents < EPOLL_EVENT_SIZE), return -1, "maxEvents:%d cannot be less than 1024", maxevents);

    int nfds = 0;
    if (task->Block() == BlockType::BLOCK_THREAD) {
        std::unique_lock<std::mutex> lck(task->mutex_);
        std::unique_lock mapLock(m_mapMutex);
        int cachedNfds = FetchCachedEventAndDoUnmask(task, eventsVec);
        if (cachedNfds > 0) {
            mapLock.unlock();
            FFRT_LOGD("task[%s] id[%d] has [%d] cached events, return directly",
                task->GetLabel().c_str(), task->gid, cachedNfds);
            task->Wake();
            return cachedNfds;
        }

        if (m_waitTaskMap.find(task) != m_waitTaskMap.end()) {
            FFRT_SYSEVENT_LOGE("task has waited before");
            mapLock.unlock();
            task->Wake();
            return 0;
        }
        auto currTime = std::chrono::steady_clock::now();
        m_waitTaskMap[task] = {static_cast<void*>(eventsVec), maxevents, &nfds, currTime};
        if (timeout > -1) {
            FFRT_LOGD("poller meet timeout={%d}", timeout);
            m_waitTaskMap[task].timerHandle = RegisterTimer(timeout, nullptr, nullptr);
        }
        mapLock.unlock();
        task->waitCond_.wait(lck);
        FFRT_LOGD("task[%s] id[%d] has [%d] events", task->GetLabel().c_str(), task->gid, nfds);
        task->Wake();
        return nfds;
    }

    CoWait([&](CoTask *task)->bool {
        std::unique_lock mapLock(m_mapMutex);
        int cachedNfds = FetchCachedEventAndDoUnmask(task, eventsVec);
        if (cachedNfds > 0) {
            mapLock.unlock();
            FFRT_LOGD("task[%s] id[%d] has [%d] cached events, return directly",
                task->GetLabel().c_str(), task->gid, cachedNfds);
            nfds = cachedNfds;
            return false;
        }

        if (m_waitTaskMap.find(task) != m_waitTaskMap.end()) {
            FFRT_SYSEVENT_LOGE("task has waited before");
            return false;
        }
        auto currTime = std::chrono::steady_clock::now();
        m_waitTaskMap[task] = {static_cast<void*>(eventsVec), maxevents, &nfds, currTime};
        if (timeout > -1) {
            FFRT_LOGD("poller meet timeout={%d}", timeout);
            m_waitTaskMap[task].timerHandle = RegisterTimer(timeout, nullptr, nullptr);
        }
        // The ownership of the task belongs to m_waitTaskMap, and the task cannot be accessed any more.
        return true;
    });
    FFRT_LOGD("task[%s] id[%d] has [%d] events", task->GetLabel().c_str(), task->gid, nfds);
    return nfds;
}

void Poller::WakeUp() noexcept
{
    uint64_t one = 1;
    (void)::write(m_wakeData.fd, &one, sizeof one);
}

void Poller::ProcessWaitedFds(int nfds, std::unordered_map<CoTask*, EventVec>& syncTaskEvents,
                              std::array<epoll_event, EPOLL_EVENT_SIZE>& waitedEvents) noexcept
{
    for (unsigned int i = 0; i < static_cast<unsigned int>(nfds); ++i) {
        struct WakeDataWithCb *data = reinterpret_cast<struct WakeDataWithCb *>(waitedEvents[i].data.ptr);
        int currFd = data->fd;
        if (currFd == m_wakeData.fd) {
            uint64_t one = 1;
            (void)::read(m_wakeData.fd, &one, sizeof one);
            continue;
        }

        if (data->cb != nullptr) {
#ifdef FFRT_ENABLE_HITRACE_CHAIN
            if (data->traceId.valid == HITRACE_ID_VALID) {
                TraceChainAdapter::Instance().HiTraceChainRestoreId(&data->traceId);
            }
#endif
            data->cb(data->data, waitedEvents[i].events);
#ifdef FFRT_ENABLE_HITRACE_CHAIN
            if (data->traceId.valid == HITRACE_ID_VALID) {
                TraceChainAdapter::Instance().HiTraceChainClearId();
            }
#endif
            continue;
        }

        if (data->task != nullptr) {
            epoll_event ev = { .events = waitedEvents[i].events, .data = {.fd = currFd} };
            syncTaskEvents[data->task].push_back(ev);
            if (waitedEvents[i].events & (EPOLLHUP | EPOLLERR)) {
                std::lock_guard lg(m_mapMutex);
                CacheMaskFdAndEpollDel(currFd, data->task);
            }
        }
    }
}

namespace {
void WakeTask(CoTask* task)
{
    if (task->GetBlockType() == BlockType::BLOCK_THREAD) {
        std::lock_guard<std::mutex> lg(task->mutex_);
        task->waitCond_.notify_one();
    } else {
        CoRoutineFactory::CoWakeFunc(task, CoWakeType::NO_TIMEOUT_WAKE);
    }
}

int CopyEventsToConsumer(EventVec& cachedEventsVec, struct epoll_event* eventsVec) noexcept
{
    int nfds = cachedEventsVec.size();
    for (int i = 0; i < nfds; i++) {
        eventsVec[i].events = cachedEventsVec[i].events;
        eventsVec[i].data.fd = cachedEventsVec[i].data.fd;
    }
    return nfds;
}

void CopyEventsInfoToConsumer(SyncData& taskInfo, EventVec& cachedEventsVec)
{
    epoll_event* eventsPtr = (epoll_event*)taskInfo.eventsPtr;
    int* nfdsPtr = taskInfo.nfdsPtr;
    if (eventsPtr == nullptr || nfdsPtr == nullptr) {
        FFRT_LOGE("usr ptr is nullptr");
        return;
    }
    *nfdsPtr = CopyEventsToConsumer(cachedEventsVec, eventsPtr);
}
} // namespace

void Poller::CacheEventsAndDoMask(CoTask* task, EventVec& eventVec) noexcept
{
    auto& syncTaskEvents = m_cachedTaskEvents[task];
    for (size_t i = 0; i < eventVec.size(); i++) {
        int currFd = eventVec[i].data.fd;
        auto wakeDataIter = m_wakeDataMap.find(currFd);
        if (wakeDataIter == m_wakeDataMap.end() ||
            wakeDataIter->second.size() == 0 ||
            wakeDataIter->second.back()->task != task) {
            FFRT_LOGD("fd[%d] may be deleted", currFd);
            continue;
        }

        auto delIter = m_delCntMap.find(currFd);
        if (delIter != m_delCntMap.end() && wakeDataIter->second.size() == static_cast<size_t>(delIter->second)) {
            FFRT_LOGD("fd[%d] may be deleted", currFd);
            continue;
        }

        struct epoll_event maskEv;
        maskEv.events = 0;
        auto& wakeData = wakeDataIter->second.back();
        std::unique_ptr<struct WakeDataWithCb> maskWakeData = std::make_unique<WakeDataWithCb>(currFd,
            wakeData->data, wakeData->cb, wakeData->task);
        void* ptr = static_cast<void*>(maskWakeData.get());
        if (ptr == nullptr || maskWakeData == nullptr) {
            FFRT_SYSEVENT_LOGE("CacheEventsAndDoMask Construct WakeDataWithCb instance failed! or wakeData is nullptr");
            continue;
        }
        maskWakeData->monitorEvents = 0;
        CacheMaskWakeData(task, maskWakeData);
        maskEv.data = {.ptr = ptr};
        if (epoll_ctl(m_epFd, EPOLL_CTL_MOD, currFd, &maskEv) != 0 && errno != ENOENT) {
            // ENOENT indicate fd is not in epfd, may be deleted
            FFRT_SYSEVENT_LOGW("epoll_ctl mod fd error: efd=%d, fd=%d, errorno=%d", m_epFd, currFd, errno);
        }
        FFRT_LOGD("fd[%d] event has no consumer, so cache it", currFd);
        syncTaskEvents.push_back(eventVec[i]);
    }
}

void Poller::WakeSyncTask(std::unordered_map<CoTask*, EventVec>& syncTaskEvents) noexcept
{
    if (syncTaskEvents.empty()) {
        return;
    }

    std::unordered_set<int> timerHandlesToRemove;
    std::unordered_set<CoTask*> tasksToWake;
    {
        std::lock_guard lg(m_mapMutex);
        for (auto& taskEventPair : syncTaskEvents) {
            CoTask* currTask = taskEventPair.first;
            auto iter = m_waitTaskMap.find(currTask);
            if (iter == m_waitTaskMap.end()) {
                CacheEventsAndDoMask(currTask, taskEventPair.second);
                continue;
            }
            CopyEventsInfoToConsumer(iter->second, taskEventPair.second);
            auto timerHandle = iter->second.timerHandle;
            if (timerHandle > -1) {
                timerHandlesToRemove.insert(timerHandle);
            }
            tasksToWake.insert(currTask);
            m_waitTaskMap.erase(iter);
        }
    }
    if (timerHandlesToRemove.size() > 0) {
        std::lock_guard lock(timerMutex_);
        for (auto it = timerMap_.begin(); it != timerMap_.end();) {
            if (timerHandlesToRemove.find(it->second.handle) != timerHandlesToRemove.end()) {
                it = timerMap_.erase(it);
            } else {
                ++it;
            }
        }
        timerEmpty_.store(timerMap_.empty());
    }

    for (auto task : tasksToWake) {
        WakeTask(task);
    }
}

uint64_t Poller::GetTaskWaitTime(CoTask* task) noexcept
{
    std::lock_guard lg(m_mapMutex);
    auto iter = m_waitTaskMap.find(task);
    if (iter == m_waitTaskMap.end()) {
        return 0;
    }

    return std::chrono::duration_cast<std::chrono::seconds>(
        iter->second.waitTP.time_since_epoch()).count();
}

PollerRet Poller::PollOnce(int timeout) noexcept
{
    int realTimeout = timeout;
    int timerHandle = -1;
    {
        std::lock_guard lg(timerMutex_);
        if (!timerMap_.empty()) {
            auto cur = timerMap_.begin();
            timerHandle = cur->second.handle;
            TimePoint now = std::chrono::steady_clock::now();
            realTimeout = std::chrono::duration_cast<std::chrono::milliseconds>(
                cur->first - now).count();
            if (realTimeout <= 0) {
                ExecuteTimerCb(now);
                return PollerRet::RET_TIMER;
            }

            if (timeout != -1 && realTimeout > timeout) {
                timerHandle = -1;
                realTimeout = timeout;
            }

            flag_ = EpollStatus::WAIT;
        }
    }

    pollerCount_++;
    std::array<epoll_event, EPOLL_EVENT_SIZE> waitedEvents;
    int nfds = epoll_wait(m_epFd, waitedEvents.data(), waitedEvents.size(), realTimeout);
    flag_ = EpollStatus::WAKE;
    if (nfds < 0) {
        if (errno != EINTR) {
            FFRT_SYSEVENT_LOGE("epoll_wait error, errorno= %d.", errno);
        }
        return PollerRet::RET_NULL;
    }
    if (nfds == 0) {
        if (timerHandle != -1) {
            std::lock_guard lg(timerMutex_);
            for (auto it = timerMap_.begin(); it != timerMap_.end(); it++) {
                if (it->second.handle == timerHandle) {
                    ExecuteTimerCb(it->first);
                    return PollerRet::RET_TIMER;
                }
            }
        }
        return PollerRet::RET_NULL;
    }

    std::unordered_map<CoTask*, EventVec> syncTaskEvents;
    ProcessWaitedFds(nfds, syncTaskEvents, waitedEvents);
    WakeSyncTask(syncTaskEvents);
    ReleaseFdWakeData();
    return PollerRet::RET_EPOLL;
}

void Poller::ReleaseFdWakeData() noexcept
{
    std::lock_guard lg(m_mapMutex);
    for (auto delIter = m_delCntMap.begin(); delIter != m_delCntMap.end();) {
        int delFd = delIter->first;
        unsigned int delCnt = static_cast<unsigned int>(delIter->second);
        auto& wakeDataList = m_wakeDataMap[delFd];
        int diff = wakeDataList.size() - delCnt;
        if (diff == 0) {
            m_wakeDataMap.erase(delFd);
            m_delCntMap.erase(delIter++);
            continue;
        } else if (diff == 1) {
            for (unsigned int i = 0; i < delCnt - 1; i++) {
                wakeDataList.pop_front();
            }
            m_delCntMap[delFd] = 1;
        } else {
            FFRT_SYSEVENT_LOGE("fd=%d count unexpected, added num=%d, del num=%d", delFd, wakeDataList.size(), delCnt);
        }
        delIter++;
    }

    fdEmpty_.store(m_wakeDataMap.empty());
}

void Poller::ProcessTimerDataCb(CoTask* task) noexcept
{
    std::lock_guard lg(m_mapMutex);
    auto iter = m_waitTaskMap.find(task);
    if (iter != m_waitTaskMap.end()) {
        WakeTask(task);
        m_waitTaskMap.erase(iter);
    }
}

void Poller::ExecuteTimerCb(TimePoint timer) noexcept
{
    while (!timerMap_.empty()) {
        auto iter = timerMap_.begin();
        if (iter->first > timer) {
            break;
        }

        TimerDataWithCb data = iter->second;
        if (data.cb != nullptr) {
            executedHandle_[data.handle] = TimerStatus::EXECUTING;
        }

        timerMap_.erase(iter);
        timerEmpty_.store(timerMap_.empty());

        if (data.cb != nullptr) {
            timerMutex_.unlock();
#ifdef FFRT_ENABLE_HITRACE_CHAIN
            if (data.traceId.valid == HITRACE_ID_VALID) {
                TraceChainAdapter::Instance().HiTraceChainRestoreId(&data.traceId);
            }
#endif
            data.cb(data.data);
#ifdef FFRT_ENABLE_HITRACE_CHAIN
            if (data.traceId.valid == HITRACE_ID_VALID) {
                TraceChainAdapter::Instance().HiTraceChainClearId();
            }
#endif
            timerMutex_.lock();
            executedHandle_[data.handle] = TimerStatus::EXECUTED;
        } else if (data.task != nullptr) {
            timerMutex_.unlock();
            ProcessTimerDataCb(data.task);
            timerMutex_.lock();
        }

        if (data.repeat && (executedHandle_.find(data.handle) != executedHandle_.end())) {
            executedHandle_.erase(data.handle);
            RegisterTimerImpl(data);
        }
    }
    timerMutex_.unlock();
}

void Poller::RegisterTimerImpl(const TimerDataWithCb& data) noexcept
{
    if (flag_ == EpollStatus::TEARDOWN) {
        return;
    }

    TimePoint absoluteTime = std::chrono::steady_clock::now() + std::chrono::milliseconds(data.timeout);
    bool wake = timerMap_.empty() || (absoluteTime < timerMap_.begin()->first && flag_ == EpollStatus::WAIT);

    timerMap_.emplace(absoluteTime, data);
    timerEmpty_.store(false);

    if (wake) {
        WakeUp();
    }
}

int Poller::RegisterTimer(uint64_t timeout, void* data, ffrt_timer_cb cb, bool repeat) noexcept
{
    if (flag_ == EpollStatus::TEARDOWN) {
        return -1;
    }

    std::lock_guard lock(timerMutex_);
    timerHandle_ += 1;

    CoTask* task = IsCoTask(ExecuteCtx::Cur()->task) ? static_cast<CoTask*>(ExecuteCtx::Cur()->task) : nullptr;
    if (timeout > MAX_TIMER_MS_COUNT) {
        FFRT_LOGW("timeout exceeds maximum allowed value %llu ms. Clamping to %llu ms.", timeout, MAX_TIMER_MS_COUNT);
        timeout = MAX_TIMER_MS_COUNT;
    }
    TimerDataWithCb timerMapValue(data, cb, task, repeat, timeout);
    timerMapValue.handle = timerHandle_;
    RegisterTimerImpl(timerMapValue);

    return timerHandle_;
}

int Poller::UnregisterTimer(int handle) noexcept
{
    if (flag_ == EpollStatus::TEARDOWN) {
        return -1;
    }

    std::lock_guard lock(timerMutex_);
    auto it = executedHandle_.find(handle);
    if (it != executedHandle_.end()) {
        while (it->second == TimerStatus::EXECUTING) {
            timerMutex_.unlock();
            std::this_thread::yield();
            timerMutex_.lock();
            it = executedHandle_.find(handle);
            if (it == executedHandle_.end()) {
                break;
            }
        }

        if (it != executedHandle_.end()) {
            executedHandle_.erase(it);
            return 0;
        }
    }

    bool wake = false;
    int ret = -1;
    for (auto cur = timerMap_.begin(); cur != timerMap_.end(); cur++) {
        if (cur->second.handle == handle) {
            if (cur == timerMap_.begin() && flag_ == EpollStatus::WAIT) {
                wake = true;
            }
            timerMap_.erase(cur);
            ret = 0;
            break;
        }
    }

    timerEmpty_.store(timerMap_.empty());

    if (wake) {
        WakeUp();
    }
    return ret;
}

bool Poller::DetermineEmptyMap() noexcept
{
    return fdEmpty_ && timerEmpty_;
}

bool Poller::DeterminePollerReady() noexcept
{
    return IsFdExist() || IsTimerReady();
}

bool Poller::IsFdExist() noexcept
{
    return !fdEmpty_;
}

bool Poller::IsTimerReady() noexcept
{
    TimePoint now = std::chrono::steady_clock::now();
    std::lock_guard lock(timerMutex_);
    if (timerMap_.empty()) {
        return false;
    }

    if (now >= timerMap_.begin()->first) {
        return true;
    }
    return false;
}

ffrt_timer_query_t Poller::GetTimerStatus(int handle) noexcept
{
    if (flag_ == EpollStatus::TEARDOWN) {
        return ffrt_timer_notfound;
    }

    std::lock_guard lock(timerMutex_);
    for (auto cur = timerMap_.begin(); cur != timerMap_.end(); cur++) {
        if (cur->second.handle == handle) {
            return ffrt_timer_not_executed;
        }
    }

    auto it = executedHandle_.find(handle);
    if (it != executedHandle_.end()) {
        while (it->second == TimerStatus::EXECUTING) {
            timerMutex_.unlock();
            std::this_thread::yield();
            timerMutex_.lock();
            it = executedHandle_.find(handle);
            if (it == executedHandle_.end()) {
                break;
            }
        }
        return ffrt_timer_executed;
    }

    return ffrt_timer_notfound;
}

uint64_t Poller::GetPollCount() noexcept
{
    return pollerCount_;
}
}
