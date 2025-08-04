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

#ifndef FFRT_INTERVAL_HPP
#define FFRT_INTERVAL_HPP

#include <deque>
#ifdef USE_OHOS_QOS
#include "qos.h"
#else
#include "staging_qos/sched/qos.h"
#endif
#include "sched/load_predictor.h"
#include "sched/load_tracking.h"
#include "eu/thread_group.h"
#include "sync/sync.h"

namespace ffrt {
class Interval;
constexpr uint64_t NS_PER_US = 1000;
constexpr uint64_t NS_PER_MS = 1000000;
class Deadline {
public:
    Deadline(uint64_t deadlineUs)
    {
        Update(deadlineUs);
    }

    uint64_t ToNs() const
    {
        return deadlineNs;
    }

    uint64_t ToUs() const
    {
        uint64_t us = deadlineNs / NS_PER_US;
        return us > 0 ? us : 1;
    }

    uint64_t ToMs() const
    {
        uint64_t ms = deadlineNs / NS_PER_MS;
        return ms > 0 ? ms : 1;
    }

    uint64_t LeftNs() const
    {
        uint64_t nowNs = AbsNowNs();
        uint64_t left = (absDeadlineNs > nowNs) ? (absDeadlineNs - nowNs) : 1;
        return left;
    }

    void Update(uint64_t deadlineUs);

private:
    static uint64_t AbsNowNs();

    uint64_t deadlineNs = 0;
    uint64_t absDeadlineNs = 0;
};

class PerfCtrl {
public:
    PerfCtrl(const QoS& qos);
    ~PerfCtrl();

    const QoS& Qos() const
    {
        return qos;
    }

    ThreadGroup* TG()
    {
        return tg;
    }

    bool isBusy()
    {
        if (tg) {
            return tg->isBegin();
        }

        return false;
    }

    void Begin()
    {
        if (tg) {
            tg->Begin();
        }
    }

    void End()
    {
        if (tg) {
            tg->End();
        }
    }

    uint64_t GetLoad()
    {
        return tg ? tg->GetLoad().load : 0;
    }

    void SetWindowSize(uint64_t size)
    {
        if (tg) {
            tg->SetWindowSize(size);
        }
    }

    void SetInvalidInterval(uint64_t interval)
    {
        if (tg) {
            tg->SetInvalidInterval(interval);
        }
    }

    bool Join()
    {
        return tg ? tg->Join() : false;
    }

    bool Leave()
    {
        return tg ? tg->Leave() : false;
    }

    void Clear()
    {
        predUtil = 0;
        curUtil = 0;
    }

    void Update(bool force = false);
    void Update(uint64_t deadlineNs, uint64_t load, bool force = false);

private:
    static constexpr int SCHED_CAPACITY_SHIFT = 10;
    static constexpr int SCHED_MAX_CAPACITY = 1 << SCHED_CAPACITY_SHIFT;

    QoS qos;
    ThreadGroup* tg = nullptr;

    uint64_t predUtil = 0;
    uint64_t curUtil = 0;
};

class IntervalLoadPredictor {
public:
    IntervalLoadPredictor()
    {
        cpLoad.resize(1);
    }

    void ResetCPIndex()
    {
        cpLoadIndex = 0;
    }

    void UpdateTotalLoad(uint64_t load);
    void UpdateCPLoad(uint64_t load);

    uint64_t GetTotalLoad();
    uint64_t GetCPLoad();

private:
    SimpleLoadPredictor totalLoad;
    std::deque<SimpleLoadPredictor> cpLoad;
    uint32_t cpLoadIndex = 0;
};

class Interval {
public:

    Interval(uint64_t deadlineUs, const QoS& qos) : dl(deadlineUs)
    {
        (void)qos;
    }
    virtual ~Interval() = default;

    virtual int Begin() = 0;

    virtual void Update(uint64_t deadlineUs) = 0;

    virtual void End() = 0;

    virtual void CheckPoint() = 0;

    virtual void Join() = 0;

    virtual void Leave() = 0;

    virtual const QoS& Qos() const = 0;

    virtual void UpdateTaskSwitch(TaskSwitchState state)
    {
    }

    Deadline& Ddl()
    {
        return dl;
    }
private:
    Deadline dl;
};

class DefaultInterval : public Interval {
public:
    DefaultInterval(uint64_t deadlineUs, const QoS& qos);
    ~DefaultInterval() override;

    bool Enabled() const
    {
        return enabled;
    }

    const QoS& Qos() const override
    {
        return ctrl.Qos();
    }

    PerfCtrl& Ctrl()
    {
        return ctrl;
    }

    int Begin() override;

    void Update(uint64_t deadlineUs) override;

    void End() override;

    void CheckPoint() override;

    void Join() override;

    void Leave() override;

    void UpdateTaskSwitch(TaskSwitchState state) override;

private:
    bool enabled = false;

    KernelLoadTracking lt;
    IntervalLoadPredictor lp;
    PerfCtrl ctrl;

    fast_mutex mutex;
};
} // namespace ffrt

#endif
