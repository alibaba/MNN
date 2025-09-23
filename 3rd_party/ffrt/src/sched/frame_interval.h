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

#ifndef FFRT_FRAME_INTERVAL
#define FFRT_FRAME_INTERVAL

#include "sched/interval.h"
#include "sched/workgroup_internal.h"

namespace ffrt {
enum class IntervalState {
    DEADLINE_BEGIN,
    DEADLINE_END
};

class FrameInterval : public Interval {
public:
    FrameInterval(uint64_t deadline, const QoS& qos);
    ~FrameInterval() override;

    const QoS& Qos() const override
    {
        return qos;
    }

    int Begin() override;

    void End() override;

    void Join() override;

    void Leave() override
    {
    }

    void Update(uint64_t deadlineUs) override
    {
    }

    void CheckPoint() override
    {
    }

private:
    struct WorkGroup* wg = nullptr;
    bool isBegin = false;
    QoS qos;

    void OnQoSIntervals(IntervalState state);
};
} // namespace ffrt

#endif
