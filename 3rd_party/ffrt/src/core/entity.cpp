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

#include "core/entity.h"
#include "core/version_ctx.h"
#include "util/slab.h"

namespace ffrt {
VersionCtx* Entity::VA2Ctx(const void* p, SCPUEUTask* task __attribute__((unused)))
{
    auto it = std::as_const(vaMap).find(p);
    if (it != vaMap.end()) {
        return it->second;
    }
    auto version = new (SimpleAllocator<VersionCtx>::AllocMem()) VersionCtx(p, nullptr, nullptr);
    vaMap[p] = version;
    return version;
}

void Entity::RecycleVersion()
{
    for (auto it = Entity::Instance()->versionTrashcan.cbegin(); it != Entity::Instance()->versionTrashcan.cend();) {
        VersionCtx* cur = *it;
        VersionCtx* next = cur->next;
        // VersionCtx list delete
        next->last = cur->last;
        if (cur->last != nullptr) {
            cur->last->next = next;
        }
        SimpleAllocator<VersionCtx>::FreeMem(cur);
        if (next->next == nullptr) {
            // Delete root version
            auto data = std::as_const(Entity::Instance()->vaMap).find(next->signature);
            if (data != Entity::Instance()->vaMap.end()) {
                Entity::Instance()->vaMap.erase(data);
            }
            SimpleAllocator<VersionCtx>::FreeMem(next);
        }
        Entity::Instance()->versionTrashcan.erase(it++);
    }
}
} /* namespace ffrt */