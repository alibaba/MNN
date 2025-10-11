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

#include "sched/frame_interval.h"
#include "dfx/log/ffrt_log_api.h"
#include "sched/workgroup_internal.h"
#include "util/ffrt_facade.h"

#define GET_TID() syscall(SYS_gettid)

namespace ffrt {
FrameInterval::FrameInterval(uint64_t deadline, const QoS& qos) : Interval(deadline, qos), qos(qos)
{
    wg = nullptr;
    isBegin = false;
    wg = WorkgroupCreate(deadline, qos);
    if (wg == nullptr) {
        FFRT_LOGE("[WorkGroup][Interface] Create WorkGroup Failed");
    } else {
        FFRTFacade::GetEUInstance().BindWG(this->qos);
    }
}

FrameInterval::~FrameInterval()
{
    if (wg == nullptr) {
        FFRT_LOGE("[Error] WorkGroup is nullptr");
    } else {
        WorkgroupClear(wg);
    }
}

void FrameInterval::OnQoSIntervals(IntervalState state)
{
    if (wg == nullptr) {
        FFRT_LOGE("[Error] Interval's workgroup is null in %s", __func__);
        return;
    }
    if (state == IntervalState::DEADLINE_BEGIN) {
        WorkgroupStartInterval(wg);
    } else if (state == IntervalState::DEADLINE_END) {
        WorkgroupStopInterval(wg);
    }
}

int FrameInterval::Begin()
{
    if (isBegin) {
        FFRT_LOGD("[Error] Interval is already begun");
        return -1;
    }
    isBegin = true;
    OnQoSIntervals(ffrt::IntervalState::DEADLINE_BEGIN);

    return 0;
}

void FrameInterval::End()
{
    if (!isBegin) {
        FFRT_LOGD("[Error] Interval is already end");
        return;
    }
    isBegin = false;
    OnQoSIntervals(ffrt::IntervalState::DEADLINE_END);
}

void FrameInterval::Join()
{
    if (wg == nullptr) {
        FFRT_LOGD("[Error] Interval has no workgroup");
        return;
    }
    int tid = GET_TID();
    WorkgroupJoin(wg, tid);
}
}
