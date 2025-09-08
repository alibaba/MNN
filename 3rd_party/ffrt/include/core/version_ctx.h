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

#ifndef FFRT_VERSION_CTX_H
#define FFRT_VERSION_CTX_H
#include <unordered_set>
#include <vector>
#include <string>

#include "internal_inc/types.h"
#include "tm/scpu_task.h"
namespace ffrt {
/* The relationship of VersionCtx is implemented using a doubly linked list：
 * 0、data represents the root node of this data signature
 * 1、Non-nested scenes: v1<—>v2<—>v3<—>data
 * 2、Nested scenes: v1<—>v2.1<—>v2.2<—>v2<—>v3<—>data
 */

struct VersionCtx : private NonCopyable {
    VersionCtx(const void* signature, VersionCtx* next, VersionCtx* last)
        : signature(signature), next(next), last(last) {};
    // Unique identifier for the data, taking the memory address of the actual data
    const void* signature;

    // Nested scenes, is next version, in non-nested scenes, is the next sub version's parent version
    VersionCtx* next {nullptr};
    // Non-nested scenes, is last version, in nested scenes, is the parent's last sub version
    VersionCtx* last {nullptr};

    // Current version's consumers, notify all when ready
    std::unordered_set<SCPUEUTask*> consumers;
    // Current version's producer
    SCPUEUTask* myProducer {nullptr};
    // Next version's producer, notify when consumed
    SCPUEUTask* nextProducer {nullptr};

    DataStatus status {DataStatus::IDLE};
    std::vector<SCPUEUTask*> dataWaitTaskByThis;

    void AddConsumer(SCPUEUTask* consumer, NestType nestType);
    void AddProducer(SCPUEUTask* producer);
    inline void AddDataWaitTaskByThis(SCPUEUTask* dataWaitTask)
    {
        if (last != nullptr && last->status == DataStatus::IDLE) {
            auto waitVersion = last;
            waitVersion->dataWaitTaskByThis.push_back(dataWaitTask);
            dataWaitTask->IncWaitDataRef();
        }
    }
    void onProduced();
    void onConsumed(SCPUEUTask* consumer);
protected:
    void CreateChildVersion(SCPUEUTask* task, DataStatus dataStatus);
    void MergeChildVersion();
    inline void NotifyDataWaitTask()
    {
        for (auto& dataWaitTask : std::as_const(dataWaitTaskByThis)) {
            dataWaitTask->DecWaitDataRef();
        }
        dataWaitTaskByThis.clear();
    }

    inline void NotifyConsumers()
    {
        for (auto consumer : std::as_const(consumers)) {
            consumer->DecDepRef();
        }
    }

    inline void NotifyNextProducer()
    {
        if (nextProducer != nullptr) {
            nextProducer->DecDepRef();
            nextProducer = nullptr;
        }
    }

    inline void MergeConsumerInDep(VersionCtx* v)
    {
        for (const auto& consumer : std::as_const(v->consumers)) {
            consumer->ins.insert(this);
            consumer->ins.erase(consumer->ins.find(v));
        }
    }

    inline void MergeProducerOutDep(VersionCtx* v)
    {
        v->myProducer->outs.insert(this);
        v->myProducer->outs.erase(v->myProducer->outs.find(v));
    }
};
} /* namespace ffrt */
#endif
