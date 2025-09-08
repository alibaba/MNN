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
#include <securec.h>
#include <sstream>
#include <fcntl.h>
#include <iomanip>
#include <sys/syscall.h>
#include <unistd.h>
#include "dfx/bbox/bbox.h"
#include "dfx/trace_record/ffrt_trace_record.h"

namespace ffrt {
const int COLUMN_WIDTH_3 = 3;
const int COLUMN_WIDTH_9 = 9;
const int COLUMN_WIDTH_10 = 10;
const int COLUMN_WIDTH_12 = 12;
const int COLUMN_WIDTH_13 = 13;
const int COLUMN_WIDTH_16 = 16;
const int COLUMN_WIDTH_18 = 18;
const int COLUMN_WIDTH_19 = 19;
const int COLUMN_WIDTH_22 = 22;
std::atomic<bool> FFRTTraceRecord::ffrt_be_used_ = false;
bool FFRTTraceRecord::stat_enable_ = false;
std::unique_ptr<FFRTRingBuffer> FFRTTraceRecord::ringBuffer_ = nullptr;
int FFRTTraceRecord::g_recordMaxWorkerNumber_[QoS::MaxNum()] = {};
ffrt_record_task_counter_t FFRTTraceRecord::g_recordTaskCounter_[FFRTTraceRecord::TASK_TYPE_NUM][QoS::MaxNum()] = {};
ffrt_record_task_time_t FFRTTraceRecord::g_recordTaskTime_[FFRTTraceRecord::TASK_TYPE_NUM][QoS::MaxNum()] = {};

#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_2)
void FFRTTraceRecord::DumpNormalTaskStatisticInfo(std::ostringstream& oss)
{
    for (size_t i = 0; i < QoS::MaxNum(); i++) {
        if (g_recordTaskCounter_[ffrt_normal_task][i].submitCounter <= 0) {
            continue;
        }
        oss << std::setw(COLUMN_WIDTH_3) << i
            << std::setw(COLUMN_WIDTH_9) << "normal"
            << std::setw(COLUMN_WIDTH_10) << g_recordTaskCounter_[ffrt_normal_task][i].submitCounter
            << std::setw(COLUMN_WIDTH_10) << g_recordTaskCounter_[ffrt_normal_task][i].enqueueCounter
            << std::setw(COLUMN_WIDTH_12) << g_recordTaskCounter_[ffrt_normal_task][i].coSwitchCounter
            << std::setw(COLUMN_WIDTH_10) << g_recordTaskCounter_[ffrt_normal_task][i].doneCounter
            << std::setw(COLUMN_WIDTH_10) << g_recordTaskCounter_[ffrt_normal_task][i].doneCounter;
#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_3)
        unsigned int doneCount = g_recordTaskCounter_[ffrt_normal_task][i].doneCounter.load();
        if (doneCount > 0) {
            oss << std::setw(COLUMN_WIDTH_13) << g_recordMaxWorkerNumber_[i]
                << std::setw(COLUMN_WIDTH_16) << g_recordTaskTime_[ffrt_normal_task][i].maxWaitTime
                << std::setw(COLUMN_WIDTH_19) << g_recordTaskTime_[ffrt_normal_task][i].maxRunDuration
                << std::setw(COLUMN_WIDTH_16) << (g_recordTaskTime_[ffrt_normal_task][i].waitTime/doneCount)
                << std::setw(COLUMN_WIDTH_19) << (g_recordTaskTime_[ffrt_normal_task][i].runDuration/doneCount)
                << std::setw(COLUMN_WIDTH_18) << g_recordTaskTime_[ffrt_normal_task][i].waitTime
                << std::setw(COLUMN_WIDTH_22) << g_recordTaskTime_[ffrt_normal_task][i].runDuration;
        }
#endif
        oss << "\n";
    }
}

void FFRTTraceRecord::DumpQueueTaskStatisticInfo(std::ostringstream& oss)
{
    for (size_t i = 0; i < QoS::MaxNum(); i++) {
        if (g_recordTaskCounter_[ffrt_queue_task][i].submitCounter <= 0) {
            continue;
        }
        oss << std::setw(COLUMN_WIDTH_3) << i
            << std::setw(COLUMN_WIDTH_9) << "queue"
            << std::setw(COLUMN_WIDTH_10) << g_recordTaskCounter_[ffrt_queue_task][i].submitCounter
            << std::setw(COLUMN_WIDTH_10) << g_recordTaskCounter_[ffrt_queue_task][i].submitCounter
            << std::setw(COLUMN_WIDTH_12) << g_recordTaskCounter_[ffrt_queue_task][i].coSwitchCounter
            << std::setw(COLUMN_WIDTH_10) << g_recordTaskCounter_[ffrt_queue_task][i].doneCounter
            << std::setw(COLUMN_WIDTH_10) << (g_recordTaskCounter_[ffrt_queue_task][i].doneCounter +
                g_recordTaskCounter_[ffrt_queue_task][i].cancelCounter);
#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_3)
        unsigned int doneCount = g_recordTaskCounter_[ffrt_queue_task][i].doneCounter.load();
        if (doneCount > 0) {
            oss << std::setw(COLUMN_WIDTH_13) << g_recordMaxWorkerNumber_[i]
                << std::setw(COLUMN_WIDTH_16) << g_recordTaskTime_[ffrt_queue_task][i].maxWaitTime
                << std::setw(COLUMN_WIDTH_19) << g_recordTaskTime_[ffrt_queue_task][i].maxRunDuration
                << std::setw(COLUMN_WIDTH_16) << (g_recordTaskTime_[ffrt_queue_task][i].waitTime/doneCount)
                << std::setw(COLUMN_WIDTH_19) << (g_recordTaskTime_[ffrt_queue_task][i].runDuration/doneCount)
                << std::setw(COLUMN_WIDTH_18) << g_recordTaskTime_[ffrt_queue_task][i].waitTime
                << std::setw(COLUMN_WIDTH_22) << g_recordTaskTime_[ffrt_queue_task][i].runDuration;
        }
#endif
        oss << "\n";
    }
}

void FFRTTraceRecord::DumpUVTaskStatisticInfo(std::ostringstream& oss)
{
    for (size_t i = 0; i < QoS::MaxNum(); i++) {
        if (g_recordTaskCounter_[ffrt_uv_task][i].submitCounter <= 0) {
            continue;
        }
        oss << std::setw(COLUMN_WIDTH_3) << i
            << std::setw(COLUMN_WIDTH_9) << "uv"
            << std::setw(COLUMN_WIDTH_10) << g_recordTaskCounter_[ffrt_uv_task][i].submitCounter
            << std::setw(COLUMN_WIDTH_10) << g_recordTaskCounter_[ffrt_uv_task][i].enqueueCounter
            << std::setw(COLUMN_WIDTH_12) << 0
            << std::setw(COLUMN_WIDTH_10) << g_recordTaskCounter_[ffrt_uv_task][i].doneCounter
            << std::setw(COLUMN_WIDTH_10) << (g_recordTaskCounter_[ffrt_uv_task][i].doneCounter +
                g_recordTaskCounter_[ffrt_uv_task][i].cancelCounter);
        oss << "\n";
    }
}

int FFRTTraceRecord::StatisticInfoDump(char* buf, uint32_t len)
{
    std::ostringstream oss;
    oss << "---\n" << "Qos TaskType SubmitNum EnueueNum CoSwitchNum   DoneNum FinishNum";
#if (FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_3)
    oss << " MaxWorkerNum MaxWaitTime(us) MaxRunDuration(us) AvgWaitTime(us) AvgRunDuration(us) TotalWaitTime(us)"
        << "  TotalRunDuration(us)";
#endif
    oss << "\n";
    DumpNormalTaskStatisticInfo(oss);
    DumpQueueTaskStatisticInfo(oss);
    DumpUVTaskStatisticInfo(oss);
    oss << "---\n";
    return snprintf_s(buf, len, len - 1, "%s", oss.str().c_str());
}

unsigned int FFRTTraceRecord::GetSubmitCount()
{
    int maxQos = QoS::MaxNum();
    unsigned int totalCount = 0;
    for (int i = 0; i < maxQos; i++) {
        totalCount += g_recordTaskCounter_[ffrt_normal_task][i].submitCounter;
        totalCount += g_recordTaskCounter_[ffrt_uv_task][i].submitCounter;
        totalCount += g_recordTaskCounter_[ffrt_queue_task][i].submitCounter;
    }
    return totalCount;
}

unsigned int FFRTTraceRecord::GetEnqueueCount()
{
    int maxQos = QoS::MaxNum();
    unsigned int totalCount = 0;
    for (int i = 0; i < maxQos; i++) {
        totalCount += g_recordTaskCounter_[ffrt_normal_task][i].enqueueCounter;
        totalCount += g_recordTaskCounter_[ffrt_uv_task][i].enqueueCounter;
        totalCount += g_recordTaskCounter_[ffrt_queue_task][i].enqueueCounter;
    }
    return totalCount;
}

unsigned int FFRTTraceRecord::GetRunCount()
{
    int maxQos = QoS::MaxNum();
    unsigned int totalCount = 0;
    for (int i = 0; i < maxQos; i++) {
        totalCount += g_recordTaskCounter_[ffrt_normal_task][i].runCounter;
        totalCount += g_recordTaskCounter_[ffrt_uv_task][i].runCounter;
        totalCount += g_recordTaskCounter_[ffrt_queue_task][i].runCounter;
    }
    return totalCount;
}

unsigned int FFRTTraceRecord::GetDoneCount()
{
    int maxQos = QoS::MaxNum();
    unsigned int totalCount = 0;
    for (int i = 0; i < maxQos; i++) {
        totalCount += g_recordTaskCounter_[ffrt_normal_task][i].doneCounter;
        totalCount += g_recordTaskCounter_[ffrt_uv_task][i].doneCounter;
        totalCount += g_recordTaskCounter_[ffrt_queue_task][i].doneCounter;
    }
    return totalCount;
}

unsigned int FFRTTraceRecord::GetCoSwitchCount()
{
    int maxQos = QoS::MaxNum();
    unsigned int totalCount = 0;
    for (int i = 0; i < maxQos; i++) {
        totalCount += g_recordTaskCounter_[ffrt_normal_task][i].coSwitchCounter;
        totalCount += g_recordTaskCounter_[ffrt_queue_task][i].coSwitchCounter;
    }
    return totalCount;
}

unsigned int FFRTTraceRecord::GetFinishCount()
{
    int maxQos = QoS::MaxNum();
    unsigned int totalCount = 0;
    for (int i = 0; i < maxQos; i++) {
        totalCount += g_recordTaskCounter_[ffrt_normal_task][i].doneCounter;
        totalCount += g_recordTaskCounter_[ffrt_normal_task][i].cancelCounter;
        totalCount += g_recordTaskCounter_[ffrt_uv_task][i].doneCounter;
        totalCount += g_recordTaskCounter_[ffrt_uv_task][i].cancelCounter;
        totalCount += g_recordTaskCounter_[ffrt_queue_task][i].doneCounter;
        totalCount += g_recordTaskCounter_[ffrt_queue_task][i].cancelCounter;
    }
    return totalCount;
}
#endif // FFRT_TRACE_RECORD_LEVEL >= FFRT_TRACE_RECORD_LEVEL_2
}
