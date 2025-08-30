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

#ifndef __FFRT_PERF_H__
#define __FFRT_PERF_H__
#include "dfx/trace/ffrt_trace.h"

// default disabled for ffrt, enable it for debugging or playback
#ifdef FFRT_PERF_ENABLE
struct PerfEU {
    static constexpr size_t qos_max = 15;
    static inline std::atomic_int worker_num[qos_max] = {0};
    static inline const char* worker_num_tag[qos_max] = {
        "qos0_wrk",
        "qos1_wrk",
        "qos2_wrk",
        "qos3_wrk",
        "qos4_wrk",
        "qos5_wrk",
        "qos6_wrk",
        "qos7_wrk",
        "qos8_wrk",
        "qos9_wrk",
        "qos10_wrk",
        "qos11_wrk",
        "qos12_wrk",
        "qos13_wrk",
        "qos14_wrk"
    };

    static inline const char* task_num_tag[qos_max] = {
        "qos0_tsk",
        "qos1_tsk",
        "qos2_tsk",
        "qos3_tsk",
        "qos4_tsk",
        "qos5_tsk",
        "qos6_tsk",
        "qos7_tsk",
        "qos8_tsk",
        "qos9_tsk",
        "qos10_tsk",
        "qos11_tsk",
        "qos12_tsk",
        "qos13_tsk",
        "qos14_tsk"
    };
};

inline void FFRTPerfWorkerIdle(int qos)
{
    if (qos >= 0 && qos < static_cast<int>(PerfEU::qos_max)) {
        FFRT_TRACE_COUNT(PerfEU::worker_num_tag[qos],
            PerfEU::worker_num[qos].fetch_sub(1, std::memory_order_relaxed) - 1);
    }
}

inline void FFRTPerfWorkerAwake(int qos)
{
    if (qos >= 0 && qos < static_cast<int>(PerfEU::qos_max)) {
        FFRT_TRACE_COUNT(PerfEU::worker_num_tag[qos],
            PerfEU::worker_num[qos].fetch_add(1, std::memory_order_relaxed) + 1);
    }
}

inline void FFRTPerfTaskNum(int qos, int taskn)
{
    if (qos >= 0 && qos < static_cast<int>(PerfEU::qos_max)) {
        FFRT_TRACE_COUNT(PerfEU::task_num_tag[qos], taskn);
    }
}

#define FFRT_PERF_WORKER_IDLE(qos) FFRTPerfWorkerIdle(qos)
#define FFRT_PERF_WORKER_AWAKE(qos) FFRTPerfWorkerAwake(qos)
#define FFRT_PERF_TASK_NUM(qos, taskn) FFRTPerfTaskNum(qos, taskn)
#else
#define FFRT_PERF_WORKER_IDLE(qos)
#define FFRT_PERF_WORKER_AWAKE(qos)
#define FFRT_PERF_TASK_NUM(qos, taskn)
#endif

#endif // __FFRT_PERF_H__