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

#ifndef _DELAYED_WORKER_H_
#define _DELAYED_WORKER_H_

#include <map>
#include <functional>
#include <thread>
#include "cpp/sleep.h"
#include "sched/execute_ctx.h"
namespace ffrt {
using TimePoint = std::chrono::steady_clock::time_point;

struct DelayedWork {
    WaitEntry* we;
    const std::function<void(WaitEntry*)>* cb;
};

class DelayedWorker {
    std::multimap<TimePoint, DelayedWork> map;
    std::mutex lock;
    std::atomic_bool toExit = false;
    std::unique_ptr<std::thread> delayedWorker = nullptr;
    int noTaskDelayCount_{0};
    bool exited_ = true;
    int epollfd_{-1};
    int timerfd_{-1};
#ifdef FFRT_WORKERS_DYNAMIC_SCALING
    int monitorfd_{-1};
#endif
    std::atomic<int> asyncTaskCnt_ {0};
    int HandleWork(void);
    void ThreadInit();

public:
    static DelayedWorker &GetInstance();
    static void ThreadEnvCreate();
    static bool IsDelayerWorkerThread();

    DelayedWorker(DelayedWorker const&) = delete;
    void operator=(DelayedWorker const&) = delete;

    bool dispatch(const TimePoint& to, WaitEntry* we, const std::function<void(WaitEntry*)>& wakeup,
        bool skipTimeCheck = false);
    bool remove(const TimePoint& to, WaitEntry* we);
    void SubmitAsyncTask(std::function<void()>&& func);
    void Terminate();

private:
    DelayedWorker();
    void DumpMap();
    ~DelayedWorker();
};
} // namespace ffrt
#endif
