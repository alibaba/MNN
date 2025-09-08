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

#include "core/version_ctx.h"
#include "core/entity.h"
#include "util/slab.h"
#include "dfx/trace/ffrt_trace.h"

namespace ffrt {
static inline void BuildConsumeRelationship(VersionCtx* version, SCPUEUTask* consumer)
{
    if (version->status == DataStatus::IDLE) {
        consumer->IncDepRef();
    }
    version->consumers.insert(consumer);
    if (version->status == DataStatus::CONSUMED) {
        version->status = DataStatus::READY;
    }
}

static inline void BuildProducerProducerRelationship(VersionCtx* preVersion, SCPUEUTask* nextProducer)
{
    if (preVersion->status != DataStatus::CONSUMED) {
        preVersion->nextProducer = nextProducer;
        nextProducer->IncDepRef();
    }
}

void VersionCtx::AddConsumer(SCPUEUTask* consumer, NestType nestType)
{
    FFRT_TRACE_SCOPE(2, AddConsumer);
    // Parent's VersionCtx
    VersionCtx* beConsumeVersion = this;
    if (nestType == NestType::PARENTOUT || nestType == NestType::DEFAULT) {
        // Create READY version when last is nullptr
        if (last == nullptr) {
            CreateChildVersion(consumer, DataStatus::READY);
        }
        beConsumeVersion = last;
    }
    BuildConsumeRelationship(beConsumeVersion, consumer);
    consumer->ins.insert(beConsumeVersion);
}

void VersionCtx::AddProducer(SCPUEUTask* producer)
{
    FFRT_TRACE_SCOPE(2, AddAddProducer);
    // Parent's VersionCtx
    auto parentVersion = this;
    if (parentVersion->last != nullptr) {
        VersionCtx* preVersion = parentVersion->last;
        BuildProducerProducerRelationship(preVersion, producer);
    }
    parentVersion->CreateChildVersion(producer, DataStatus::IDLE);
    producer->outs.insert(parentVersion->last);
    parentVersion->last->myProducer = producer;
}

void VersionCtx::onProduced()
{
    /* No merge operation, merge operation can only occur when the parent version is produced and
     * the last child version is not in the CONSUMED state
     */
    if (last == nullptr || last->status == DataStatus::CONSUMED) {
        // No consumers, directly into CONSUMED after being produced
        if (consumers.empty()) {
            status = DataStatus::CONSUMED;
            NotifyNextProducer();
            Entity::Instance()->versionTrashcan.push_back(this);
        } else { // if have consumers,notify them
            status = DataStatus::READY;
            NotifyConsumers();
        }
        NotifyDataWaitTask();
    } else { // Merge previous VersionCtx
        MergeChildVersion();
    }
}

void VersionCtx::onConsumed(SCPUEUTask* consumer)
{
    auto it = std::as_const(consumers).find(consumer);
    if (it != consumers.end()) {
        consumers.erase(it);
    }
    if (consumers.empty()) {
        status = DataStatus::CONSUMED;
        NotifyNextProducer();
        Entity::Instance()->versionTrashcan.push_back(this);
    }
}

void VersionCtx::CreateChildVersion(SCPUEUTask* task __attribute__((unused)), DataStatus dataStatus)
{
    // Add VersionCtx
    auto prev = last;
    last = new (SimpleAllocator<VersionCtx>::AllocMem()) VersionCtx(this->signature, this, prev);
    last->status = dataStatus;
    if (prev != nullptr) {
        prev->next = last;
    }
}

void VersionCtx::MergeChildVersion()
{
    // Merge VersionCtx
    auto versionToMerge = last;
    status = versionToMerge->status;
    if (status == DataStatus::READY) {
        NotifyConsumers();
        NotifyDataWaitTask();
    }
    MergeConsumerInDep(versionToMerge);
    if (status == DataStatus::IDLE) {
        consumers.insert(versionToMerge->consumers.cbegin(), versionToMerge->consumers.cend());
        MergeProducerOutDep(versionToMerge);
        myProducer = versionToMerge->myProducer;
    }
    Entity::Instance()->versionTrashcan.push_back(versionToMerge);
}
} /* namespace ffrt */