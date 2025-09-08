/*
 * Copyright (c) 2024 Huawei Device Co., Ltd.
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

#include "queue/traffic_record.h"
#include <future>
#include <sstream>
#include "queue/queue_handler.h"
#include "dfx/sysevent/sysevent.h"
#include "util/time_format.h"

namespace {
constexpr uint64_t DEFAULT_TIME_INTERVAL = 6000000;
constexpr uint32_t REPORT_INTERVAL = 60000000; // 60s
constexpr uint32_t MIN_OVERLOAD_INTERVAL = 15;
constexpr uint32_t INITIAL_RECORD_LIMIT = 16;
constexpr uint32_t MAX_RECORD_LIMIT = 128;
constexpr uint32_t US_TO_MS = 1000;
}

namespace ffrt {
std::vector<std::pair<uint64_t, std::string>> TrafficRecord::trafficRecordInfo =
    std::async(std::launch::deferred, []() {
        std::vector<std::pair<uint64_t, std::string>> trafficInfo;
        trafficInfo.reserve(INITIAL_RECORD_LIMIT);
        return trafficInfo;
    }).get();
std::mutex TrafficRecord::mtx_;

TrafficRecord::TrafficRecord() {}

#ifdef FFRT_ENABLE_TRAFFIC_MONITOR
void TrafficRecord::SetTimeInterval(const uint64_t timeInterval)
{
    const uint64_t& time = TimeStampCntvct();
    timeInterval_ = timeInterval;
    nextUpdateTime_ = time + timeInterval_;
}

void TrafficRecord::SubmitTraffic(QueueHandler* handler)
{
    submitCnt_.fetch_add(1, std::memory_order_relaxed);

    if (detectCnt_.fetch_add(1, std::memory_order_relaxed) < MIN_OVERLOAD_INTERVAL) {
        return;
    }
    detectCnt_ = 0;
    const uint64_t& time = TimeStampCntvct();
    uint64_t oldVal = nextUpdateTime_.load();
    if (likely(time < oldVal) || timeInterval_ == 0) {
        return;
    }

    uint64_t nextTime = time + timeInterval_;

    if (nextUpdateTime_.compare_exchange_strong(oldVal, nextTime)) {
        CalculateTraffic(handler, time);
    }
}

void TrafficRecord::DoneTraffic()
{
    doneCnt_.fetch_add(1, std::memory_order_relaxed);
}

void TrafficRecord::DoneTraffic(uint32_t count)
{
    doneCnt_.fetch_add(count, std::memory_order_relaxed);
}
#else
void TrafficRecord::SetTimeInterval(const uint64_t timeInterval) {}
void TrafficRecord::SubmitTraffic(QueueHandler* handler) {}
void TrafficRecord::DoneTraffic() {}
void TrafficRecord::DoneTraffic(uint32_t count) {}
#endif

void TrafficRecord::CalculateTraffic(QueueHandler* handler, const uint64_t& time)
{
    uint32_t inflows = submitCnt_ - submitCntOld_;
    uint32_t outflows = doneCnt_ - doneCntOld_;
    DetectCountThreshold();
    submitCntOld_.store(submitCnt_.load());
    doneCntOld_.store(doneCnt_.load());

    uint64_t reportInterval = lastCheckTime_ != 0 ? time - lastCheckTime_ : timeInterval_;
    lastCheckTime_ = time;

    if (inflows > outflows && (submitCnt_ - doneCnt_ > MIN_OVERLOAD_INTERVAL * (reportInterval / timeInterval_))
        && handler->GetTaskCnt() > MIN_OVERLOAD_INTERVAL) {
        std::stringstream ss;
        ss << "[" << handler->GetName().c_str() << "]overload over[" << reportInterval / US_TO_MS <<
            "]ms, in[" << inflows << "], out[" << outflows << "], subcnt[" << submitCnt_.load() << "], doncnt[" <<
            doneCnt_.load() << "]";
        FFRT_LOGW("%s", ss.str().c_str());

        {
            std::lock_guard lock(mtx_);
            if (trafficRecordInfo.size() > MAX_RECORD_LIMIT) {
                trafficRecordInfo.erase(trafficRecordInfo.begin());
            }
            trafficRecordInfo.emplace_back(std::make_pair(time, ss.str()));
        }

        if (submitCnt_ - doneCnt_ > outflows && handler->GetQueueDueCount() > outflows) {
            if (time - lastReportTime_ > REPORT_INTERVAL) {
                ReportOverload(ss);
                lastReportTime_ = time;
            }
            FFRT_LOGE("Queue overload syswarning is triggerred, duecnt %llu", handler->GetQueueDueCount());
        }
    }
}

void TrafficRecord::DetectCountThreshold()
{
    if (doneCnt_.load() > 0x00FFFFFF) {
        FFRT_LOGW("Traffic Record exceed threshold, trigger remove");
        uint32_t oldTemp = doneCnt_.load();
        doneCnt_.fetch_sub(oldTemp);
        submitCnt_.fetch_sub(oldTemp);
    }
}

void TrafficRecord::ReportOverload(std::stringstream& ss)
{
#ifdef FFRT_SEND_EVENT
    std::string senarioName = "QUEUE_TASK_OVERLOAD";
    TrafficOverloadReport(ss, senarioName);
#endif
}

std::string TrafficRecord::DumpTrafficInfo(bool withLock)
{
    std::stringstream ss;
    if (withLock) {
        mtx_.lock();
    }
    if (trafficRecordInfo.size() != 0) {
        for (auto it = trafficRecordInfo.rbegin(); it != trafficRecordInfo.rend(); ++it) {
            auto& record = *it;
            ss << "{" << FormatDateString4SteadyClock(record.first) << ", " << record.second << "} \n";
        }
    } else {
        ss << "Queue Traffic Record Empty";
    }
    if (withLock) {
        mtx_.unlock();
    }
    return ss.str();
}
} // namespace ffrt
