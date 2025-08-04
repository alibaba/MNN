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

#ifndef FFRT_SEXECUTE_UNIT_HPP
#define FFRT_SEXECUTE_UNIT_HPP

#include "execute_unit.h"
#include "util/ffrt_facade.h"

namespace ffrt {
class SExecuteUnit : public ExecuteUnit {
public:
    static SExecuteUnit& Instance()
    {
        static SExecuteUnit ins;
        return ins;
    }

    void WorkerInit() override {}

    void WorkerPrepare(CPUWorker* thread) override
    {
        WorkerJoinTg(thread->GetQos(), thread->Id());
    }

    WorkerAction WorkerIdleAction(CPUWorker* thread) override;

    void WakeupWorkers(const QoS& qos) override;

    void IntoSleep(const QoS& qos) override
    {
        CPUWorkerGroup& group = workerGroup[qos];
        std::lock_guard lg(group.lock);
        group.sleepingNum++;
        group.executingNum--;
    }

    /* strategy options for handling task notify events */
    static void HandleTaskNotifyDefault(SExecuteUnit* manager, const QoS& qos, TaskNotifyType notifyType);
    static void HandleTaskNotifyConservative(SExecuteUnit* manager, const QoS& qos, TaskNotifyType notifyType);
    static void HandleTaskNotifyUltraConservative(SExecuteUnit* manager, const QoS& qos, TaskNotifyType notifyType);
private:
    SExecuteUnit();
    ~SExecuteUnit() override;

    void PokeAdd(const QoS& qos) override
    {
        handleTaskNotify(this, qos, TaskNotifyType::TASK_ADDED);
    }
    void PokePick(const QoS& qos) override
    {
        handleTaskNotify(this, qos, TaskNotifyType::TASK_PICKED);
    }
    void PokeLocal(const QoS& qos) override
    {
        if (FFRTFacade::GetSchedInstance()->GetScheduler(qos).stealWorkers.load(std::memory_order_relaxed) == 0) {
            handleTaskNotify(this, qos, TaskNotifyType::TASK_LOCAL);
        }
    }
    void PokeEscape(const QoS& qos, bool isPollWait) override
    {
        handleTaskNotify(this, qos, TaskNotifyType::TASK_ESCAPED);
    }

    void PokeAddRtq(const QoS &qos, bool isRisingEdge) override
    {
        (void)isRisingEdge; // deprecated
        handleTaskNotify(this, qos, TaskNotifyType::TASK_ADDED);
    }

    void PokeImpl(const QoS& qos, uint32_t taskCount, TaskNotifyType notifyType);
    void ExecuteEscape(int qos) override;

    std::function<void (SExecuteUnit*, const QoS&, TaskNotifyType)> handleTaskNotify { nullptr };
};
} // namespace ffrt
#endif
