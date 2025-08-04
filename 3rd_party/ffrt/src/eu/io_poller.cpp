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

#include "eu/io_poller.h"
#include <securec.h>
#include <sys/prctl.h>
#include "eu/blockaware.h"
#include "eu/execute_unit.h"
#include "sched/execute_ctx.h"
#include "tm/scpu_task.h"
#include "dfx/log/ffrt_log_api.h"
#include "util/ffrt_facade.h"
#include "util/time_format.h"
#include "util/name_manager.h"
#include "sync/timer_manager.h"
#ifdef FFRT_OH_TRACE_ENABLE
#include "backtrace_local.h"
#endif

namespace {
const std::vector<uint64_t> TIMEOUT_RECORD_CYCLE_LIST = { 1, 3, 5, 10, 30, 60, 10 * 60, 30 * 60 };
}
namespace ffrt {
namespace {
static void TimeoutProc(void* task)
{
    IOPoller& ins = IOPoller::Instance();
    ins.WakeTimeoutTask(reinterpret_cast<CoTask*>(task));
}

void WakeTask(CoTask* task)
{
    std::unique_lock<std::mutex> lck(task->mutex_);
    if (task->GetBlockType() == BlockType::BLOCK_THREAD) {
        task->waitCond_.notify_one();
    } else {
        lck.unlock();
        CoRoutineFactory::CoWakeFunc(task, CoWakeType::NO_TIMEOUT_WAKE);
    }
}

int CopyEventsToConsumer(EventVec& cachedEventsVec, struct epoll_event* eventsVec) noexcept
{
    int nfds = static_cast<int>(cachedEventsVec.size());
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

IOPoller& IOPoller::Instance()
{
    static IOPoller ins;
    return ins;
}

IOPoller::IOPoller() noexcept: m_epFd { ::epoll_create1(EPOLL_CLOEXEC) }
{
#ifdef OHOS_STANDARD_SYSTEM
    fdsan_exchange_owner_tag(m_epFd, 0, fdsan_create_owner_tag(FDSAN_OWNER_TYPE_FILE, static_cast<uint64_t>(m_epFd)));
#endif
    m_wakeData.mode = PollerType::WAKEUP;
    m_wakeData.fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
#ifdef OHOS_STANDARD_SYSTEM
    fdsan_exchange_owner_tag(m_wakeData.fd, 0, fdsan_create_owner_tag(FDSAN_OWNER_TYPE_FILE,
        static_cast<uint64_t>(m_wakeData.fd)));
#endif
    epoll_event ev { .events = EPOLLIN, .data = { .ptr = static_cast<void*>(&m_wakeData) } };
    FFRT_COND_TERMINATE((epoll_ctl(m_epFd, EPOLL_CTL_ADD, m_wakeData.fd, &ev) < 0),
        "epoll_ctl add fd error: efd=%d, fd=%d, errorno=%d", m_epFd, m_wakeData.fd, errno);
}

IOPoller::~IOPoller() noexcept
{
    {
        std::lock_guard lock(m_mapMutex);
        m_teardown = true;
        WakeUp();
    }
    if (m_runner != nullptr && m_runner->joinable()) {
        m_runner->join();
    }
#ifdef OHOS_STANDARD_SYSTEM
    fdsan_close_with_tag(m_wakeData.fd, fdsan_create_owner_tag(FDSAN_OWNER_TYPE_FILE,
        static_cast<uint64_t>(m_wakeData.fd)));
    fdsan_close_with_tag(m_epFd, fdsan_create_owner_tag(FDSAN_OWNER_TYPE_FILE, static_cast<uint64_t>(m_epFd)));
#else
    ::close(m_wakeData.fd);
    ::close(m_epFd);
#endif
}

void IOPoller::ThreadInit()
{
    if (m_runner != nullptr && m_runner->joinable()) {
        m_runner->join();
    }
    m_runner = std::make_unique<std::thread>([this] { Run(); });
}

void IOPoller::Run()
{
    struct sched_param param;
    param.sched_priority = 1;
    int ret = pthread_setschedparam(pthread_self(), SCHED_RR, &param);
    if (ret != 0) {
        FFRT_LOGW("[%d] set priority warn ret[%d] eno[%d]\n", pthread_self(), ret, errno);
    }
    prctl(PR_SET_NAME, IO_POLLER_NAME);
    while (1) {
        ret = PollOnce(30000);
        std::lock_guard lock(m_mapMutex);
        if (m_teardown) {
            // teardown
            m_exitFlag = true;
            return;
        }
        if (ret == 0 && m_wakeDataMap.empty() && m_syncFdCnt.load() == 0) {
            // timeout 30s and no fd added
            m_exitFlag = true;
            return;
        }
    }
}

void IOPoller::WakeUp() noexcept
{
    uint64_t one = 1;
    (void)::write(m_wakeData.fd, &one, sizeof one);
}

int IOPoller::PollOnce(int timeout) noexcept
{
    pollerCount_++;
    std::array<epoll_event, EPOLL_EVENT_SIZE> waitedEvents;
    int nfds = epoll_wait(m_epFd, waitedEvents.data(), waitedEvents.size(), timeout);
    if (nfds < 0) {
        if (errno != EINTR) {
            FFRT_SYSEVENT_LOGE("epoll_wait error, errorno= %d.", errno);
        }
        return -1;
    }
    if (nfds == 0) {
        return 0;
    }

    std::unordered_map<CoTask*, EventVec> syncTaskEvents;
    for (unsigned int i = 0; i < static_cast<unsigned int>(nfds); ++i) {
        struct WakeData *data = reinterpret_cast<struct WakeData *>(waitedEvents[i].data.ptr);
        if (data->mode == PollerType::WAKEUP) {
            // self wakeup
            uint64_t one = 1;
            (void)::read(m_wakeData.fd, &one, sizeof one);
            continue;
        }

        if (data->mode == PollerType::SYNC_IO) {
            // sync io wait fd, del fd when waked up
            if (epoll_ctl(m_epFd, EPOLL_CTL_DEL, data->fd, nullptr) != 0) {
                FFRT_SYSEVENT_LOGE("epoll_ctl del fd error: fd=%d, errorno=%d", data->fd, errno);
                continue;
            }
            m_syncFdCnt--;
            WakeTask(data->task);
            continue;
        }

        if (data->mode == PollerType::ASYNC_CB) {
            // async io callback
            timeOutReport.cbStartTime.store(TimeStamp(), std::memory_order_relaxed);
            timeOutReport.reportCount = 0;
#ifdef FFRT_ENABLE_HITRACE_CHAIN
            if (data->traceId.valid == HITRACE_ID_VALID) {
                TraceChainAdapter::Instance().HiTraceChainRestoreId(&data->traceId);
            }
#endif
            data->cb(data->data, waitedEvents[i].events);
            timeOutReport.cbStartTime.store(0, std::memory_order_relaxed);
#ifdef FFRT_ENABLE_HITRACE_CHAIN
            if (data->traceId.valid == HITRACE_ID_VALID) {
                TraceChainAdapter::Instance().HiTraceChainClearId();
            }
#endif
            timeOutReport.cbStartTime = 0;
            continue;
        }

        if (data->mode == PollerType::ASYNC_IO) {
            // async io task wait fd
            epoll_event ev = { .events = waitedEvents[i].events, .data = {.fd = data->fd} };
            syncTaskEvents[data->task].push_back(ev);
            if ((waitedEvents[i].events & (EPOLLHUP | EPOLLERR)) != 0) {
                std::lock_guard lock(m_mapMutex);
                CacheMaskFdAndEpollDel(data->fd, data->task);
            }
        }
    }

    WakeSyncTask(syncTaskEvents);
    ReleaseFdWakeData();
    return 1;
}

// mode ASYNC_CB/ASYNC_IO
int IOPoller::AddFdEvent(int op, uint32_t events, int fd, void* data, ffrt_poller_cb cb) noexcept
{
    CoTask* task = IsCoTask(ExecuteCtx::Cur()->task) ? static_cast<CoTask*>(ExecuteCtx::Cur()->task) : nullptr;
    auto wakeData = std::make_unique<WakeData>(fd, data, cb, task);
    if (task) {
        task->pollerEnable = true;
    }
    void* ptr = static_cast<void*>(wakeData.get());
    if (ptr == nullptr || wakeData == nullptr) {
        FFRT_SYSEVENT_LOGE("Construct WakeData instance failed! or wakeData is nullptr");
        return -1;
    }
    wakeData->monitorEvents = events;

    epoll_event ev = { .events = events, .data = { .ptr = ptr } };
    std::lock_guard lock(m_mapMutex);
    if (m_teardown) {
        return -1;
    }

    if (m_exitFlag) {
        ThreadInit();
        m_exitFlag = false;
    }

    if (epoll_ctl(m_epFd, op, fd, &ev) != 0) {
        FFRT_SYSEVENT_LOGE("epoll_ctl add fd error: efd=%d, fd=%d, errorno=%d", m_epFd, fd, errno);
        return -1;
    }

    if (op == EPOLL_CTL_ADD) {
        m_wakeDataMap[fd].emplace_back(std::move(wakeData));
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

int IOPoller::DelFdEvent(int fd) noexcept
{
    std::lock_guard lock(m_mapMutex);
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

// mode ASYNC_IO
int IOPoller::WaitFdEvent(struct epoll_event* eventsVec, int maxevents, int timeout) noexcept
{
    FFRT_COND_DO_ERR((eventsVec == nullptr), return -1, "eventsVec cannot be null");

    CoTask* task = IsCoTask(ExecuteCtx::Cur()->task) ? static_cast<CoTask*>(ExecuteCtx::Cur()->task) : nullptr;
    if (!task) {
        FFRT_SYSEVENT_LOGE("nonworker shall not call this fun.");
        return -1;
    }

    FFRT_COND_DO_ERR((maxevents < EPOLL_EVENT_SIZE), return -1, "maxEvents:%d cannot be less than 1024", maxevents);

    int nfds = 0;
    std::unique_lock<std::mutex> lck(task->mutex_);
    if (task->Block() == BlockType::BLOCK_THREAD) {
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
            m_waitTaskMap[task].timerHandle = FFRTFacade::GetTMInstance().RegisterTimer(task->qos_, timeout,
                reinterpret_cast<void*>(task), TimeoutProc);
        }
        mapLock.unlock();
        task->waitCond_.wait(lck);
        FFRT_LOGD("task[%s] id[%d] has [%d] events", task->GetLabel().c_str(), task->gid, nfds);
        task->Wake();
        return nfds;
    }
    lck.unlock();

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
            m_waitTaskMap[task].timerHandle = FFRTFacade::GetTMInstance().RegisterTimer(task->qos_, timeout,
                reinterpret_cast<void*>(task), TimeoutProc);
        }
        // The ownership of the task belongs to m_waitTaskMap, and the task cannot be accessed any more.
        return true;
    });
    FFRT_LOGD("task[%s] id[%d] has [%d] events", task->GetLabel().c_str(), task->gid, nfds);
    return nfds;
}

void IOPoller::WakeTimeoutTask(CoTask* task) noexcept
{
    std::unique_lock mapLock(m_mapMutex);
    auto iter = m_waitTaskMap.find(task);
    if (iter != m_waitTaskMap.end()) {
        // wake task, erase from wait map
        m_waitTaskMap.erase(iter);
        mapLock.unlock();
        WakeTask(task);
    }
}

void IOPoller::WakeSyncTask(std::unordered_map<CoTask*, EventVec>& syncTaskEvents) noexcept
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
            if (iter == m_waitTaskMap.end()) { // task not in wait map
                CacheEventsAndDoMask(currTask, taskEventPair.second);
                continue;
            }
            CopyEventsInfoToConsumer(iter->second, taskEventPair.second);
            // remove timer, wake task, erase from wait map
            auto timerHandle = iter->second.timerHandle;
            if (timerHandle > -1) {
                timerHandlesToRemove.insert(timerHandle);
            }
            tasksToWake.insert(currTask);
            m_waitTaskMap.erase(iter);
        }
    }
    for (auto timerHandle : timerHandlesToRemove) {
        FFRTFacade::GetTMInstance().UnregisterTimer(timerHandle);
    }
    for (auto task : tasksToWake) {
        WakeTask(task);
    }
}

// mode SYNC_IO
void IOPoller::WaitFdEvent(int fd) noexcept
{
    CoTask* task = IsCoTask(ExecuteCtx::Cur()->task) ? static_cast<CoTask*>(ExecuteCtx::Cur()->task) : nullptr;
    if (!task) {
        FFRT_LOGI("nonworker shall not call this fun.");
        return;
    }

    struct WakeData data(fd, task);
    epoll_event ev = { .events = EPOLLIN, .data = {.ptr = static_cast<void*>(&data)} };
    {
        std::lock_guard lock(m_mapMutex);
        if (m_teardown) {
            return;
        }

        if (m_exitFlag) {
            ThreadInit();
            m_exitFlag = false;
        }

        m_syncFdCnt++;
    }

    FFRT_BLOCK_TRACER(task->gid, fd);
    if (task->Block() == BlockType::BLOCK_THREAD) {
        std::unique_lock<std::mutex> lck(task->mutex_);
        if (epoll_ctl(m_epFd, EPOLL_CTL_ADD, fd, &ev) == 0) {
            task->waitCond_.wait(lck);
        }
        task->Wake();
        m_syncFdCnt--;
        return;
    }

    CoWait([&](CoTask *task)->bool {
        (void)task;
        if (epoll_ctl(m_epFd, EPOLL_CTL_ADD, fd, &ev) == 0) {
            return true;
        }
        // The ownership of the task belongs to epoll, and the task cannot be accessed any more.
        FFRT_LOGI("epoll_ctl add err:efd:=%d, fd=%d errorno = %d", m_epFd, fd, errno);
        m_syncFdCnt--;
        return false;
    });
}

void IOPoller::ReleaseFdWakeData() noexcept
{
    std::lock_guard lock(m_mapMutex);
    for (auto delIter = m_delCntMap.begin(); delIter != m_delCntMap.end();) {
        int delFd = delIter->first;
        unsigned int delCnt = static_cast<unsigned int>(delIter->second);
        auto& wakeDataList = m_wakeDataMap[delFd];
        unsigned int diff = wakeDataList.size() - delCnt;
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
}

void IOPoller::CacheMaskFdAndEpollDel(int fd, CoTask *task) noexcept
{
    auto maskWakeData = m_maskWakeDataMap.find(task);
    if (maskWakeData != m_maskWakeDataMap.end()) {
        if (epoll_ctl(m_epFd, EPOLL_CTL_DEL, fd, nullptr) != 0) {
            FFRT_SYSEVENT_LOGE("fd[%d] ffrt epoll ctl del fail errorno=%d", fd, errno);
        }
        m_delFdCacheMap.emplace(fd, task);
    }
}

int IOPoller::ClearMaskWakeDataCache(CoTask *task) noexcept
{
    auto maskWakeDataIter = m_maskWakeDataMap.find(task);
    if (maskWakeDataIter != m_maskWakeDataMap.end()) {
        WakeDataList& wakeDataList = maskWakeDataIter->second;
        for (auto iter = wakeDataList.begin(); iter != wakeDataList.end(); ++iter) {
            WakeData* ptr = iter->get();
            m_delFdCacheMap.erase(ptr->fd);
        }
        m_maskWakeDataMap.erase(maskWakeDataIter);
    }
    return 0;
}

int IOPoller::ClearMaskWakeDataCacheWithFd(CoTask *task, int fd) noexcept
{
    auto maskWakeDataIter = m_maskWakeDataMap.find(task);
    if (maskWakeDataIter != m_maskWakeDataMap.end()) {
        WakeDataList& wakeDataList = maskWakeDataIter->second;
        auto pred = [fd](auto& value) { return value->fd == fd; };
        wakeDataList.remove_if(pred);
        if (wakeDataList.size() == 0) {
            m_maskWakeDataMap.erase(maskWakeDataIter);
        }
    }
    return 0;
}

int IOPoller::ClearDelFdCache(int fd) noexcept
{
    auto fdDelCacheIter = m_delFdCacheMap.find(fd);
    if (fdDelCacheIter != m_delFdCacheMap.end()) {
        CoTask *task = fdDelCacheIter->second;
        ClearMaskWakeDataCacheWithFd(task, fd);
        m_delFdCacheMap.erase(fdDelCacheIter);
    }
    return 0;
}

int IOPoller::FetchCachedEventAndDoUnmask(EventVec& cachedEventsVec, struct epoll_event* eventsVec) noexcept
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

int IOPoller::FetchCachedEventAndDoUnmask(CoTask* task, struct epoll_event* eventsVec) noexcept
{
    // should used in lock
    auto syncTaskIter = m_cachedTaskEvents.find(task);
    if (syncTaskIter == m_cachedTaskEvents.end() || syncTaskIter->second.size() == 0) {
        return 0;
    }

    int nfds = FetchCachedEventAndDoUnmask(syncTaskIter->second, eventsVec);
    m_cachedTaskEvents.erase(syncTaskIter);
    ClearMaskWakeDataCache(task);
    return nfds;
}

void IOPoller::CacheEventsAndDoMask(CoTask* task, EventVec& eventVec) noexcept
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
        std::unique_ptr<struct WakeData> maskWakeData = std::make_unique<WakeData>(currFd,
            wakeData->data, wakeData->cb, wakeData->task);
        void* ptr = static_cast<void*>(maskWakeData.get());
        if (ptr == nullptr || maskWakeData == nullptr) {
            FFRT_SYSEVENT_LOGE("CacheEventsAndDoMask Construct WakeData instance failed! or wakeData is nullptr");
            continue;
        }
        maskWakeData->monitorEvents = 0;
        m_maskWakeDataMap[task].emplace_back(std::move(maskWakeData));

        maskEv.data = {.ptr = ptr};
        if (epoll_ctl(m_epFd, EPOLL_CTL_MOD, currFd, &maskEv) != 0 && errno != ENOENT) {
            // ENOENT indicate fd is not in epfd, may be deleted
            FFRT_SYSEVENT_LOGW("epoll_ctl mod fd error: efd=%d, fd=%d, errorno=%d", m_epFd, currFd, errno);
        }
        FFRT_LOGD("fd[%d] event has no consumer, so cache it", currFd);
        syncTaskEvents.push_back(eventVec[i]);
    }
}

void IOPoller::MonitTimeOut()
{
    if (m_teardown) {
        return;
    }

    if (timeOutReport.cbStartTime == 0) {
        return;
    }
    uint64_t now = TimeStamp();
    static const uint64_t freq = [] {
        uint64_t f = Arm64CntFrq();
        return (f == 1) ? 1000000 : f;
    } ();
    uint64_t diff = (now - timeOutReport.cbStartTime) / freq;
    if (timeOutReport.reportCount < TIMEOUT_RECORD_CYCLE_LIST.size() &&
        diff >= TIMEOUT_RECORD_CYCLE_LIST[timeOutReport.reportCount]) {
#ifdef FFRT_OH_TRACE_ENABLE
        std::string dumpInfo;
        static pid_t pid = syscall(SYS_gettid);
        if (OHOS::HiviewDFX::GetBacktraceStringByTid(dumpInfo, pid, 0, false)) {
            FFRT_LOGW("IO_Poller Backtrace Info:\n%s", dumpInfo.c_str());
        }
#endif
        timeOutReport.reportCount++;
    }
}
}
