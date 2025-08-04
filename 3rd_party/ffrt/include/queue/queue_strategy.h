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
#ifndef FFRT_QUEUE_STRATEGY_H
#define FFRT_QUEUE_STRATEGY_H

#include <map>
#include <vector>
#include <algorithm>
#include "c/type_def.h"
#include "c/queue_ext.h"
#include "dfx/log/ffrt_log_api.h"

namespace ffrt {
template<typename T>
class QueueStrategy {
public:
    using DequeFunc = T*(*)(const uint32_t, const uint64_t, std::multimap<uint64_t, T*>*, void*);

    static T* DequeBatch(const uint32_t queueId, const uint64_t now,
        std::multimap<uint64_t, T*>* whenMapIn, void* args)
    {
        (void)args;
        auto& whenMap = *whenMapIn;
        // dequeue due tasks in batch
        T* head = whenMap.begin()->second;
        whenMap.erase(whenMap.begin());
        head->Dequeue();

        T* node = head;
        while (!whenMap.empty() && whenMap.begin()->first < now) {
            auto next = whenMap.begin()->second;
            if (next->GetQos() != head->GetQos()) {
                break;
            }
            node->SetNextTask(next);
            whenMap.erase(whenMap.begin());
            next->Dequeue();
            node = next;
        }
        FFRT_LOGD("dequeue [gid=%llu -> gid=%llu], %u other tasks in [queueId=%u] ",
            head->gid, node->gid, whenMap.size(), queueId);
        return head;
    }

    static T* DequeSingleByPriority(const uint32_t queueId,
        const uint64_t now, std::multimap<uint64_t, T*>* whenMapVec, void* args)
    {
        (void)args;
        // dequeue next expired task by priority
        int iterIndex = ffrt_queue_priority_idle;
        auto iterTarget = whenMapVec[iterIndex].cbegin();
        for (int idx = ffrt_queue_priority_immediate; idx <= ffrt_queue_priority_idle; idx++) {
            const auto& currentMap = whenMapVec[idx];
            if (currentMap.empty()) {
                continue;
            }
            if (whenMapVec[iterIndex].empty() || iterTarget->first > currentMap.cbegin()->first) {
                iterIndex = idx;
                iterTarget = currentMap.cbegin();
            }
        }

        for (int idx = ffrt_queue_priority_immediate; idx <= ffrt_queue_priority_idle; idx++) {
            const auto& currentMap = whenMapVec[idx];
            if (!currentMap.empty() && currentMap.cbegin()->first < now) {
                iterTarget = currentMap.cbegin();
                iterIndex = idx;
                break;
            }
        }
        T* head = iterTarget->second;
        whenMapVec[iterIndex].erase(iterTarget);
        head->Dequeue();

        size_t mapCount = 0;
        for (int idx = ffrt_queue_priority_immediate; idx <= ffrt_queue_priority_idle; idx++) {
            auto& currentMap = whenMapVec[idx];
            mapCount += currentMap.size();
        }
        FFRT_LOGD("dequeue [gid=%llu], %u other tasks in [queueId=%u] ", head->gid, mapCount, queueId);
        return head;
    }

    static T* DequeSingleAgainstStarvation(const uint32_t queueId,
        const uint64_t now, std::multimap<uint64_t, T*>* whenMapVec, void* args)
    {
        // dequeue in descending order of priority
        // a low-priority task is dequeued every time five high-priority tasks are dequeued
        constexpr int maxPullTaskCount = 5;
        std::vector<int>* pulledTaskCount = static_cast<std::vector<int>*>(args);

        int iterIndex = ffrt_inner_queue_priority_idle;
        auto iterTarget = whenMapVec[iterIndex].cbegin();
        for (int idx = 0; idx < ffrt_inner_queue_priority_idle; idx++) {
            const auto& currentMap = whenMapVec[idx];
            if (currentMap.empty()) {
                continue;
            }
            if (whenMapVec[iterIndex].empty() || iterTarget->first > currentMap.cbegin()->first) {
                iterIndex = idx;
                iterTarget = currentMap.cbegin();
            }
        }

        for (int idx = 0; idx < ffrt_inner_queue_priority_idle; idx++) {
            if ((*pulledTaskCount)[idx] >= maxPullTaskCount) {
                continue;
            }

            const auto& currentMap = whenMapVec[idx];
            if (!currentMap.empty() && currentMap.cbegin()->first < now) {
                iterTarget = currentMap.cbegin();
                iterIndex = idx;
                break;
            }
        }

        T* head = iterTarget->second;
        (*pulledTaskCount)[iterIndex]++;

        for (int idx = 0; idx < iterIndex; idx++) {
            (*pulledTaskCount)[idx] = 0;
        }
        whenMapVec[iterIndex].erase(iterTarget);
        head->Dequeue();

        size_t mapCount = 0;
        for (int idx = 0; idx <= ffrt_inner_queue_priority_idle; idx++) {
            auto& currentMap = whenMapVec[idx];
            mapCount += currentMap.size();
        }
        FFRT_LOGD("dequeue [gid=%llu], prio %d, %u other tasks in [queueId=%u] ",
            head->gid, head->GetPriority(), mapCount, queueId);

        return head;
    }
};
} // namespace ffrt

#endif // FFRT_QUEUE_STRATEGY_H