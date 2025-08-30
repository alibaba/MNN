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

#include "perf_counter.h"

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif
#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif

#ifdef PERF_MONITOR

static const char* config_file = "perf_config.txt";
static const char* output_file = "perf_result.txt";
static pthread_mutex_t __pw_mutex = PTHREAD_MUTEX_INITIALIZER;
static int __perf_write = 0;
static std::atomic<int> __perf_init = 0;

static pthread_mutex_t __g_stat_mutex = PTHREAD_MUTEX_INITIALIZER;
static std::vector<struct PerfStat*>g_perfstat;

static int n_event = 5;
static int even_type = PERF_TYPE_SOFTWARE;
static int pmu_event[MAX_COUNTERS] = {PERF_COUNT_SW_CPU_CLOCK, PERF_COUNT_SW_TASK_CLOCK, PERF_COUNT_SW_PAGE_FAULTS,
    PERF_COUNT_SW_CONTEXT_SWITCHES, PERF_COUNT_SW_CPU_MIGRATIONS};

static __thread struct PerfStat* t_perfStat = NULL;

static inline pid_t __gettid(void)
{
    return syscall(__NR_gettid);
}

static inline int perf_event_open(struct perf_event_attr* attr, pid_t pid, int cpu, int group_fd, unsigned long flags)
{
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static void perf_open(struct PerfStat* pf, int event)
{
    struct perf_event_attr attr = {0};

    attr.size = sizeof(struct perf_event_attr);
    attr.type = even_type;
    attr.config = event;
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.read_format = PERF_FORMAT_GROUP;

    // calling process/thread on any CPU
    /********************************************************************************************************
      detail in https://man7.org/linux/man-pages/man2/perf_event_open.2.html

       int perf_event_open(struct perf_event_attr *attr, pid_t pid, int cpu, int group_fd, unsigned long flags);
       pid == 0 and cpu == -1
              This measures the calling process/thread on any CPU.

       pid == 0 and cpu >= 0
              This measures the calling process/thread only when running
              on the specified CPU.

       pid > 0 and cpu == -1
              This measures the specified process/thread on any CPU.

       pid > 0 and cpu >= 0
              This measures the specified process/thread only when
              running on the specified CPU.

       pid == -1 and cpu >= 0
              This measures all processes/threads on the specified CPU.
              This requires CAP_PERFMON (since Linux 5.8) or
              CAP_SYS_ADMIN capability or a
              /proc/sys/kernel/perf_event_paranoid value of less than 1.

       pid == -1 and cpu == -1
              This setting is invalid and will return an error.

    *********************************************************************************************************/
    int ret = perf_event_open(&attr, pf->pid, -1, pf->perfFD, 0);
    if (ret < 0) {
        return;
    }

    if (pf->perfFD == -1) {
        pf->perfFD = ret;
    }
    pf->nCounters++;
}

static void perf_init(void)
{
    std::fstream file(config_file, std::ios::in);
    if (!file) {
        printf("perf_config.txt not exist.\n");
        return;
    }

    if (!(file >> even_type)) {
        printf("perf event type not exist.\n");
        file.close();
        return;
    }

    if (!(file >> n_event)) {
        printf("perf event num not exist.\n");
        file.close();
        return;
    }

    if ((n_event > MAX_COUNTERS) || (even_type > PERF_TYPE_MAX)) {
        printf("pmu config err type:%d, num:%d.\n", even_type, n_event);
        file.close();
        return;
    }

    for (int i = 0; i < n_event; i++) {
        if (!(file >> pmu_event[i]))
            break;
        printf("pmu event id:%d.\n", pmu_event[i]);
    }

    file.close();
}

struct perf_ignore {
    const char* ig_task;
    int ignore_count;
};

static struct perf_ignore __g_ignore[] = {
    {"cpu_work", 2},
    {"task_build", 2},
};

static struct perf_ignore* find_ignore(struct PerfRecord* record)
{
    int i;
    int size = sizeof(__g_ignore) / sizeof(struct perf_ignore);
    for (i = 0; i < size; i++) {
        if (strncmp(__g_ignore[i].ig_task, record->name, NAME_LEN) == 0) {
            return &__g_ignore[i];
        }
    }

    return nullptr;
}

static bool perf_ignore(struct PerfRecord* record)
{
    struct perf_ignore* p = find_ignore(record);
    if ((!p) || (!p->ignore_count)) {
        return false;
    }
    p->ignore_count--;
    return true;
}

static void perf_counter_output(struct PerfStat* stat)
{
    if (!stat) {
        printf("no perf stat,tid:%d\n", __gettid());
        return;
    }
    std::map<std::string, Counters> m_counters;

    pthread_mutex_lock(&__pw_mutex);

    FILE* fd = fopen(output_file, __perf_write == 0 ? (__perf_write = 1, "wt") : "a");
    if (!fd) {
        printf("perf_result.txt creat err.\n");
        pthread_mutex_unlock(&__pw_mutex);
        return;
    }

    auto doRecord = [&](struct PerfRecord* record) {
        auto it = m_counters.find(record->name);
        if (it != m_counters.end()) {
            it->second.nr++;
            for (int k = 0; k < MAX_COUNTERS; k++) {
                it->second.vals[k] += (record->countersEnd.vals[k] - record->countersBegin.vals[k]);
            }
        } else {
            Counters new_;
            new_.nr = 1;
            for (int k = 0; k < MAX_COUNTERS; k++) {
                new_.vals[k] = (record->countersEnd.vals[k] - record->countersBegin.vals[k]);
            }
            m_counters.insert(std::pair<std::string, Counters>(record->name, new_));
        }
    };

    for (int i = 0; i < TASK_NUM; i++) {
        struct PerfTask* task = &stat->perfTask[i];
        int max_rd = (task->rd > RECORD_NUM) ? RECORD_NUM : task->rd;
        for (int j = 0; j < max_rd; j++) {
            struct PerfRecord* record = &task->perfRecord[j];
            if (!(record->beginFlag) || !(record->endFlag)) {
                continue;
            }
            if (perf_ignore(record)) {
                continue;
            }

            doRecord(record);
        }
    }

    for (auto iter = m_counters.begin(); iter != m_counters.end(); iter++) {
        fprintf_s(fd,
            "pid:%d, taskname:%s, taskid:%d, recordid:%d, evt_num:%d, pmu_%x:%lu, pmu_%x:%lu, pmu_%x:%lu, pmu_%x:%lu, "
            "pmu_%x:%lu, pmu_%x:%lu, pmu_%x:%lu, nr:%lu.\n",
            stat->pid, iter->first.c_str(), 0, 1, stat->nCounters, pmu_event[0], iter->second.vals[0], pmu_event[1],
            iter->second.vals[1], pmu_event[2], iter->second.vals[2], pmu_event[3], iter->second.vals[3], pmu_event[4],
            iter->second.vals[4], pmu_event[5], iter->second.vals[5], pmu_event[6], iter->second.vals[6],
            iter->second.nr);
    }

    fclose(fd);
    pthread_mutex_unlock(&__pw_mutex);
    m_counters.clear();
}
#endif
