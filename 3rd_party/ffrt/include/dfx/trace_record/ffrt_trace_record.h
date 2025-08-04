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
#ifndef FFRT_TRACE_RECORD_H
#define FFRT_TRACE_RECORD_H

#include "util/ffrt_ring_buffer.h"
#include "dfx/log/ffrt_log_api.h"
#include "internal_inc/osal.h"
#include "tm/task_base.h"
#include "tm/cpu_task.h"
#include "tm/queue_task.h"

#define FFRT_TRACE_RECORD_LEVEL_1 (1)
#define FFRT_TRACE_RECORD_LEVEL_2 (2)
#define FFRT_TRACE_RECORD_LEVEL_3 (3)

namespace ffrt {
typedef struct ffrt_record_task_counter {
    alignas(cacheline_size) std::atomic<unsigned int> submitCounter{0};
    alignas(cacheline_size) std::atomic<unsigned int> enqueueCounter{0};
    alignas(cacheline_size) std::atomic<unsigned int> coSwitchCounter{0};
    alignas(cacheline_size) std::atomic<unsigned int> runCounter{0};
    alignas(cacheline_size) std::atomic<unsigned int> doneCounter{0};
    alignas(cacheline_size) std::atomic<unsigned int> cancelCounter{0};
} ffrt_record_task_counter_t;

typedef struct ffrt_record_task_time {
    std::atomic<uint64_t> waitTime{0};
    std::atomic<uint64_t> runDuration{0};
    std::atomic<uint64_t> executeTime{0};
    uint64_t maxWaitTime{0};
    uint64_t maxRunDuration{0};
} ffrt_record_task_time_t;

class FFRTTraceRecord {
public:
    static const int TASK_TYPE_NUM = ffrt_queue_task + 1;
    static std::atomic<bool> ffrt_be_used_;
    static bool stat_enable_;
    static int g_recordMaxWorkerNumber_[QoS::MaxNum()];
    static ffrt_record_task_counter_t g_recordTaskCounter_[TASK_TYPE_NUM][QoS::MaxNum()];
    static ffrt_record_task_time_t g_recordTaskTime_[TASK_TYPE_NUM][QoS::MaxNum()];

public:
    FFRTTraceRecord() = default;
    ~FFRTTraceRecord() = default;

    static inline bool FfrtBeUsed()
    {
        return ffrt_be_used_;
    }

    static int StatEnable(char *buf, uint32_t len)
    {
        if (len < TASK_STAT_LENGTH) {
            FFRT_LOGE("Buffer size is not enough, len = %u", len);
            return -1;
        }
        if (stat_enable_) {
            FFRT_LOGW("Statistics are already enabled.");
            return -1;
        }
        if (ringBuffer_ != nullptr) {
            FFRT_LOGW(
                "Buffer has been set, do not set again. "
                "Previous buffer start addr = %p, size = %u",
                ringBuffer_->GetBuffer(), ringBuffer_->GetBufferSize());
            return -1;
        }
        ringBuffer_ = std::make_unique<FFRTRingBuffer>(buf, len);
        stat_enable_ = true;
        return 0;
    }

    static int StatsDisable(char *buf)
    {
        FFRT_LOGW("StatsDisable");
        if (ringBuffer_ == nullptr) {
            FFRT_LOGW("StatsDisable: ringBuffer_ is already nullptr");
            return -1;
        }

        if (buf != ringBuffer_->GetBuffer()) {
            FFRT_LOGW(
                "Buffer addr invalid "
                "Previous buffer addr = %p, now is = %p",
                ringBuffer_->GetBuffer(), buf);
            return -1;
        }

        stat_enable_ = false;
        ringBuffer_.reset();
        return 0;
    }

    static inline void UseFfrt()
    {
        if (unlikely(!ffrt_be_used_)) {
            ffrt_be_used_ = true;
        }
    }

    template<ffrt_executor_task_type_t taskType>
    static inline void TaskSubmit(int qos)
    {
#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_2)
        g_recordTaskCounter_[taskType][qos].submitCounter.fetch_add(1, std::memory_order_relaxed);
#endif
    }

    static inline void TaskSubmit(uint64_t* createTime, int32_t* fromTid)
    {
        *createTime = TimeStamp();
        *fromTid = ExecuteCtx::Cur()->tid;
    }

    template<ffrt_executor_task_type_t taskType>
    static inline void TaskSubmit(int qos, uint64_t* createTime, int32_t* fromTid)
    {
#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_2)
        g_recordTaskCounter_[taskType][qos].submitCounter.fetch_add(1, std::memory_order_relaxed);
#endif
        *createTime = TimeStamp();
        *fromTid = ExecuteCtx::Cur()->tid;
    }

    static inline void TaskExecute(uint64_t* executeTime)
    {
        *executeTime = TimeStamp();
    }

    template<ffrt_executor_task_type_t taskType>
    static inline void TaskExecute(int qos)
    {
#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_2)
        g_recordTaskCounter_[taskType][qos].runCounter.fetch_add(1, std::memory_order_relaxed);
#endif
    }

    template<ffrt_executor_task_type_t taskType>
    static inline void TaskDone(int qos)
    {
#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_2)
        g_recordTaskCounter_[taskType][qos].doneCounter.fetch_add(1, std::memory_order_relaxed);
#endif
    }

    template<ffrt_executor_task_type_t taskType>
    static void TaskDone(int qos, TaskBase* task)
    {
        if (stat_enable_) {
            uint64_t endTime = TimeStamp();
            ffrt_stat stat;
            errno_t ret;
            if (taskType == ffrt_normal_task) {
                auto baseTask = reinterpret_cast<CPUEUTask*>(task);
                auto f = reinterpret_cast<ffrt_function_header_t*>(baseTask->func_storage);
                baseTask->label.resize(MAX_TASK_NAME_LENGTH - 1, ' ');
                ret = strcpy_s(stat.taskName, MAX_TASK_NAME_LENGTH, baseTask->GetLabel().c_str());
                stat.funcPtr = reinterpret_cast<uint64_t>(f->exec);
            }
            if (taskType == ffrt_queue_task) {
                auto baseTask = reinterpret_cast<QueueTask*>(task);
                auto f = reinterpret_cast<ffrt_function_header_t*>(baseTask->func_storage);
                baseTask->label.resize(MAX_TASK_NAME_LENGTH - 1, ' ');
                ret = strcpy_s(stat.taskName, MAX_TASK_NAME_LENGTH, baseTask->GetLabel().c_str());
                stat.funcPtr = reinterpret_cast<uint64_t>(f->exec);
            }
            if (ret != EOK) {
                FFRT_LOGE("strcpy_s failed");
                return;
            }
            stat.startTime = task->createTime;
            stat.endTime = endTime;
            ringBuffer_->Write(stat);
        }

#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_3)
        auto runDuration = TimeStamp() - task->executeTime;
        g_recordTaskTime_[taskType][qos].runDuration += runDuration;

        if (g_recordTaskTime_[taskType][qos].maxRunDuration < runDuration) {
            g_recordTaskTime_[taskType][qos].maxRunDuration = runDuration;
        }

        auto waitTime = task->executeTime - task->createTime;
        g_recordTaskTime_[taskType][qos].waitTime += waitTime;
        if (g_recordTaskTime_[taskType][qos].maxWaitTime < waitTime) {
            g_recordTaskTime_[taskType][qos].maxWaitTime = waitTime;
        }
#endif
    }

    template<ffrt_executor_task_type_t taskType>
    static inline void TaskEnqueue(int qos)
    {
#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_2)
        g_recordTaskCounter_[taskType][qos].enqueueCounter.fetch_add(1, std::memory_order_relaxed);
#endif
    }

    template<ffrt_executor_task_type_t taskType>
    static inline void TaskCancel(int qos)
    {
#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_2)
        g_recordTaskCounter_[taskType][qos].cancelCounter.fetch_add(1, std::memory_order_relaxed);
#endif
    }

    static inline void TaskRun(int qos, TaskBase* task)
    {
#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_2)
        g_recordTaskCounter_[task->type][qos].runCounter.fetch_add(1, std::memory_order_relaxed);
#endif
    }

    static inline void TaskCoSwitchOut(TaskBase* task)
    {
#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_2)
        g_recordTaskCounter_[task->type][task->GetQos()].coSwitchCounter.fetch_add(1, std::memory_order_relaxed);
#endif
    }

    static inline void WorkRecord(int qos, int workerNum)
    {
#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_3)
        if (g_recordMaxWorkerNumber_[qos] < workerNum) {
            g_recordMaxWorkerNumber_[qos] = workerNum;
        }
#endif
    }

#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_2)
    static int StatisticInfoDump(char* buf, uint32_t len);
    static void DumpNormalTaskStatisticInfo(std::ostringstream& oss);
    static void DumpQueueTaskStatisticInfo(std::ostringstream& oss);
    static void DumpUVTaskStatisticInfo(std::ostringstream& oss);

    static unsigned int GetSubmitCount();
    static unsigned int GetEnqueueCount();
    static unsigned int GetRunCount();
    static unsigned int GetDoneCount();
    static unsigned int GetCoSwitchCount();
    static unsigned int GetFinishCount();
#endif

private:
    static std::unique_ptr<FFRTRingBuffer> ringBuffer_;
};
}
#endif // FFRT_TRACE_RECORD_H
