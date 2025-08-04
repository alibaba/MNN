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

#ifndef __PERF_COUNTER_H__
#define __PERF_COUNTER_H__
#include <cstdlib>
#include <unistd.h>

constexpr int RECORD_NUM = 1024 * 32 * 16;
constexpr int NAME_LEN = 32;

constexpr int TASK_NUM = 1;

constexpr int MAX_COUNTERS = 7;

struct Counters {
    unsigned long nr;
    unsigned long vals[MAX_COUNTERS];
};

struct PerfRecord {
    char name[NAME_LEN];
    short beginFlag;
    short endFlag;
    struct Counters countersBegin;
    struct Counters countersEnd;
};

struct PerfTask {
    unsigned long rd;

    struct PerfRecord perfRecord[RECORD_NUM];
};

struct PerfStat {
    int perfFD;
    int nCounters;
    pid_t pid;
    int rsvd;
    struct PerfTask perfTask[TASK_NUM];
};

#define CPU_CYCLES 0x11
#define INST_RETIRED 0x08
#define INST_SPEC 0x1b
#define BR_RETIRED 0x21
#define BR_MIS_PRED_RETIRED 0x22
#define STALL_FRONTEND 0x23
#define STALL_BACKEND 0x24

#endif