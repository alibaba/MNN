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
#ifndef UTIL_FFRTFACADE_HPP
#define UTIL_FFRTFACADE_HPP
#include "sched/scheduler.h"
#include "eu/co_routine.h"
#include "eu/execute_unit.h"
#include "dm/dependence_manager.h"
#include "queue/queue_monitor.h"
#include "sync/delayed_worker.h"
#include "eu/io_poller.h"
#include "sync/timer_manager.h"
#include "util/worker_monitor.h"

namespace ffrt {
bool GetExitFlag();
std::shared_mutex& GetExitMtx();
const char* GetCurrentProcessName();

class FFRTFacade {
public:
    static inline ExecuteUnit& GetEUInstance()
    {
        return Instance().GetEUInstanceImpl();
    }

    static inline DependenceManager& GetDMInstance()
    {
        return Instance().GetDMInstanceImpl();
    }

    static inline IOPoller& GetPPInstance()
    {
        return Instance().GetPPInstanceImpl();
    }

    static inline TimerManager& GetTMInstance()
    {
        return Instance().GetTMInstanceImpl();
    }

    static inline DelayedWorker& GetDWInstance()
    {
        return Instance().GetDWInstanceImpl();
    }

    static inline Scheduler* GetSchedInstance()
    {
        return Instance().GetSchedInstanceImpl();
    }

    static inline CoStackAttr* GetCSAInstance()
    {
        return Instance().GetCSAInstanceImpl();
    }

    static inline QueueMonitor& GetQMInstance()
    {
        return Instance().GetQMInstanceImpl();
    }

    static inline WorkerMonitor& GetWMInstance()
    {
        return Instance().GetWMInstanceImpl();
    }

private:
    FFRTFacade();

    static FFRTFacade& Instance()
    {
        static FFRTFacade facade;
        return facade;
    }

    inline ExecuteUnit& GetEUInstanceImpl()
    {
        return ExecuteUnit::Instance();
    }

    inline DependenceManager& GetDMInstanceImpl()
    {
        return DependenceManager::Instance();
    }

    inline IOPoller& GetPPInstanceImpl()
    {
        return IOPoller::Instance();
    }

    inline TimerManager& GetTMInstanceImpl()
    {
        return TimerManager::Instance();
    }

    inline DelayedWorker& GetDWInstanceImpl()
    {
        return DelayedWorker::GetInstance();
    }

    inline Scheduler* GetSchedInstanceImpl()
    {
        return Scheduler::Instance();
    }

    inline CoStackAttr* GetCSAInstanceImpl()
    {
        return CoStackAttr::Instance();
    }

    inline QueueMonitor& GetQMInstanceImpl()
    {
        return QueueMonitor::GetInstance();
    }

    inline WorkerMonitor& GetWMInstanceImpl()
    {
        return WorkerMonitor::GetInstance();
    }
};

} // namespace FFRT
#endif