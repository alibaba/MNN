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

#ifndef FFRT_QUEUE_TRAFFIC_RECORD_H
#define FFRT_QUEUE_TRAFFIC_RECORD_H

#include <vector>
#include "queue/serial_queue.h"

namespace ffrt {
constexpr uint64_t DEFAULT_TRAFFIC_INTERVAL = 6000000;

class QueueHandler;
class TrafficRecord {
public:
    explicit TrafficRecord();
    void SetTimeInterval(const uint64_t timeInterval);
    void SubmitTraffic(QueueHandler* handler);
    void DoneTraffic();
    void DoneTraffic(uint32_t count);
    static std::string DumpTrafficInfo(bool withLock = true);

    static std::vector<std::pair<uint64_t, std::string>> trafficRecordInfo;
private:
    void CalculateTraffic(QueueHandler* handler, const uint64_t& time);
    void ReportOverload(std::stringstream& ss);
    void DetectCountThreshold();

    uint64_t timeInterval_ = DEFAULT_TRAFFIC_INTERVAL;
    uint64_t lastCheckTime_ = 0;
    uint64_t lastReportTime_ = 0;
    std::atomic_uint32_t detectCnt_ = 0;
    std::atomic_uint64_t nextUpdateTime_ = 0;

    std::atomic_uint32_t submitCnt_ = 0;
    std::atomic_uint32_t doneCnt_ = 0;

    std::atomic_uint32_t doneCntOld_ = 0;
    std::atomic_uint32_t submitCntOld_ = 0;

    static std::mutex mtx_;
    int recordIndex_ = 0;
};
} // namespace ffrt

#endif // FFRT_QUEUE_TRAFFIC_RECORD_H
