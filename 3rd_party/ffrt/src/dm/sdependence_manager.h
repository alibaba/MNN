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

#ifndef FFRT_SDEPENDENCE_MANAGER_H
#define FFRT_SDEPENDENCE_MANAGER_H

#include "dm/dependence_manager.h"

namespace ffrt {
class SDependenceManager : public DependenceManager {
public:
    static SDependenceManager& Instance()
    {
        static SDependenceManager ins;
        return ins;
    }

    void onSubmit(bool has_handle, ffrt_task_handle_t &handle, ffrt_function_header_t *f, const ffrt_deps_t *ins,
        const ffrt_deps_t *outs, const task_attr_private *attr) override;

    void onWait() override;

    void onWait(const ffrt_deps_t* deps) override;

    int onExecResults(ffrt_task_handle_t handle) override;

    void onTaskDone(CPUEUTask* task) override;

private:
    SDependenceManager();
    ~SDependenceManager() override;

    void RemoveRepeatedDeps(std::vector<CPUEUTask*>& in_handles, const ffrt_deps_t* ins, const ffrt_deps_t* outs,
        std::vector<const void *>& insNoDup, std::vector<const void *>& outsNoDup);
    void MapSignature2Deps(SCPUEUTask* task, const std::vector<const void*>& inDeps,
        const std::vector<const void*>& outDeps, std::vector<std::pair<VersionCtx*, NestType>>& inVersions,
        std::vector<std::pair<VersionCtx*, NestType>>& outVersions);

    fast_mutex& criticalMutex_;
};
} // namespace ffrt
#endif