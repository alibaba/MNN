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

#ifndef SCHED_RTG_H
#define SCHED_RTG_H

#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEFAULT_RT_FRAME_ID 8
#define FRAME_START (1 << 0)
#define FRAME_END (1 << 1)

void SetTaskRtg(pid_t tid, unsigned int grpId);
void SetRtgStatus(unsigned long long status);
void SetRtgQos(int qos);
void SetRtgLoadMode(unsigned int grpId, bool utilEnabled, bool freqEnabled);
void set_task_min_util(pid_t tid, unsigned int util);

/* inner use */

struct RtgGroupTask {
    pid_t pid;
    unsigned int grpId;
    bool pmu_sample_enabled;
};

struct RtgLoadMode {
    unsigned int grpId;
    unsigned int freqEnabled;
    unsigned int utilEnabled;
};

struct TaskConfig {
    pid_t pid;
    unsigned int value;
};

#define SET_TASK_RTG 11
#define SET_FRAME_STATUS 10
#define SET_FRAME_RATE 8
#define SET_RTG_LOAD_MODE 22
#define SET_TASK_MIN_UTIL 28

#define PERF_CTRL_MAGIC 'x'
#define PERF_CTRL_SET_TASK_RTG _IOWR(PERF_CTRL_MAGIC, SET_TASK_RTG, struct RtgGroupTask)
#define PERF_CTRL_SET_FRAME_STATUS _IOWR(PERF_CTRL_MAGIC, SET_FRAME_STATUS, unsigned long long)
#define PERF_CTRL_SET_FRAME_RATE _IOWR(PERF_CTRL_MAGIC, SET_FRAME_RATE, int)
#define PERF_CTRL_SET_RTG_LOAD_MODE _IOW(PERF_CTRL_MAGIC, SET_RTG_LOAD_MODE, struct RtgLoadMode)
#define PERF_CTRL_SET_TASK_MIN_UTIL _IOW(PERF_CTRL_MAGIC, SET_TASK_MIN_UTIL, struct TaskConfig)

#ifdef __cplusplus
}
#endif

#endif /* SCHED_RTG_H */
