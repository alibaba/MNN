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

#ifndef _MUTEX_PRIVATE_H_
#define _MUTEX_PRIVATE_H_

#include "sync/sync.h"

#ifdef FFRT_MUTEX_DEADLOCK_CHECK
#include "util/graph_check.h"
#include "dfx/log/ffrt_log_api.h"
#include "tm/cpu_task.h"

#define TID_MAX (0x400000)
#endif
#ifdef FFRT_OH_EVENT_RECORD
#include "hisysevent.h"
#endif
namespace ffrt {
#ifdef FFRT_MUTEX_DEADLOCK_CHECK
class MutexGraph {
    std::mutex mutex;
    GraphCheckCyclic graph;

public:
    static MutexGraph& Instance()
    {
        static MutexGraph mgraph;
        return mgraph;
    }

    void SendEvent(const std::string &msg, const std::string& eventName)
    {
#ifdef FFRT_OH_EVENT_RECORD
        int32_t pid = getpid();
        int32_t gid = getgid();
        int32_t uid = getuid();
        time_t cur_time = time(nullptr);
        std::string sendMsg = std::string((ctime(&cur_time) == nullptr) ? "" : ctime(&cur_time)) +
                              "\n" + msg + "\n";

        HiSysEventWrite(OHOS::HiviewDFX::HiSysEvent::Domain::FFRT, eventName,
                        OHOS::HiviewDFX::HiSysEvent::EventType::FAULT,
                        "PID", pid,
                        "TGID", gid,
                        "UID", uid,
                        "MODULE_NAME", "ffrt",
                        "PROCESS_NAME", "ffrt",
                        "MSG", sendMsg);
        FFRT_LOGE("send event [FRAMEWORK,%s], msg=%s", eventName.c_str(), msg.c_str());
#endif
    }

    void AddNode(uint64_t task, uint64_t ownerTask, bool edge)
    {
        std::lock_guard<std::mutex> lg{mutex};
        graph.AddVetexByLabel(task);
        if (edge) {
            graph.AddEdgeByLabel(ownerTask, task);
            if (graph.IsCyclic()) {
                std::string dlockInfo = "A possible mutex deadlock detected!\n";
                for (uint64_t taskId : {ownerTask, task}) {
                    CPUEUTask* taskCtx = nullptr;
                    if (taskId >= TID_MAX) {
                        taskCtx = reinterpret_cast<CPUEUTask*>(taskId);
#ifdef FFRT_CO_BACKTRACE_OH_ENABLE
                        std::string dumpInfo;
                        CPUEUTask::DumpTask(taskCtx, dumpInfo, 1);
                        dlockInfo += dumpInfo;
#endif
                    } else {
                        std::string threadInfo = "The linux thread id is ";
                        threadInfo += std::to_string(taskId);
                        threadInfo.append("\n");
                        dlockInfo += threadInfo;
                    }
                }
                SendEvent(dlockInfo, "TASK_DEADLOCK");
            }
        }
    }

    void RemoveNode(uint64_t task)
    {
        std::lock_guard<std::mutex> lg{mutex};
        graph.RemoveEdgeByLabel(task);
    }
};
#endif

class mutexBase {
public:
    mutexBase() = default;
    virtual ~mutexBase() = default;
    virtual void lock() {}
    virtual void unlock() {}
    virtual bool try_lock() {return false;}
};

class mutexPrivate : public mutexBase {
    std::atomic<int> l;
#ifdef FFRT_MUTEX_DEADLOCK_CHECK
    std::atomic<uintptr_t> owner;
#endif
    fast_mutex wlock;
    LinkedList list;

    void wait();
    void wake();

public:
#ifdef FFRT_MUTEX_DEADLOCK_CHECK
    mutexPrivate() : l(sync_detail::UNLOCK), owner(0) {}
#else
    mutexPrivate() : l(sync_detail::UNLOCK) {}
#endif
    mutexPrivate(mutexPrivate const &) = delete;
    void operator = (mutexPrivate const &) = delete;

    bool try_lock() override;
    void lock() override;
    void unlock() override;
};

class RecursiveMutexPrivate : public mutexBase {
public:
    void lock() override;
    void unlock() override;
    bool try_lock() override;

    RecursiveMutexPrivate() = default;
    ~RecursiveMutexPrivate() = default;
    RecursiveMutexPrivate(RecursiveMutexPrivate const&) = delete;
    void operator = (RecursiveMutexPrivate const&) = delete;

private:
    std::pair<uint64_t, uint32_t> taskLockNums = std::make_pair(UINT64_MAX, 0);
    fast_mutex fMutex;
    mutexPrivate mt;
};

} // namespace ffrt

#endif
