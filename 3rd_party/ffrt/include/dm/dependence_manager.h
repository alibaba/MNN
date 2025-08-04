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

#ifndef FFRT_DEPENDENCE_MANAGER_H
#define FFRT_DEPENDENCE_MANAGER_H
#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>
#include <shared_mutex>
#include "internal_inc/types.h"
#include "internal_inc/osal.h"
#include "core/version_ctx.h"
#include "sched/execute_ctx.h"
#ifdef USE_OHOS_QOS
#include "qos.h"
#else
#include "staging_qos/sched/qos.h"
#endif
#include "dfx/trace/ffrt_trace.h"
#include "sched/scheduler.h"
#include "eu/execute_unit.h"
#include "core/entity.h"
#include "dfx/watchdog/watchdog_util.h"
#include "dfx/trace_record/ffrt_trace_record.h"
#include "tm/cpu_task.h"

namespace ffrt {
#define OFFSETOF(TYPE, MEMBER) (reinterpret_cast<size_t>(&((reinterpret_cast<TYPE *>(0))->MEMBER)))

inline bool CheckOutsHandle(const ffrt_deps_t* outs)
{
    if (outs == nullptr) {
        return true;
    }
    for (uint32_t i = 0; i < outs->len; i++) {
        if ((outs->items[i].type) == ffrt_dependence_task) {
            FFRT_LOGE("handle can't be used as out dependence");
            return false;
        }
    }
    return true;
}
inline void OutsDedup(std::vector<const void *>& outsNoDup, const ffrt_deps_t* outs)
{
    for (uint32_t i = 0; i < outs->len; i++) {
        if (std::find(outsNoDup.begin(), outsNoDup.end(), outs->items[i].ptr) == outsNoDup.end()) {
            outsNoDup.push_back(outs->items[i].ptr);
        }
    }
}

inline void InsDedup(std::vector<CPUEUTask*> &in_handles, std::vector<const void *> &insNoDup,
    std::vector<const void *> &outsNoDup, const ffrt_deps_t *ins)
{
    for (uint32_t i = 0; i < ins->len; i++) {
        if (std::find(outsNoDup.begin(), outsNoDup.end(), ins->items[i].ptr) == outsNoDup.end()) {
            if ((ins->items[i].type == ffrt_dependence_task) && (ins->items[i].ptr != nullptr)) {
                static_cast<ffrt::CPUEUTask*>(const_cast<void*>(ins->items[i].ptr))->IncDeleteRef();
                in_handles.emplace_back(static_cast<ffrt::CPUEUTask*>(const_cast<void*>(ins->items[i].ptr)));
            }
            insNoDup.push_back(ins->items[i].ptr);
        }
    }
}

class DependenceManager : public NonCopyable {
public:
    static DependenceManager& Instance();

    static void RegistInsCb(SingleInsCB<DependenceManager>::Instance &&cb);

    virtual void onSubmit(bool has_handle, ffrt_task_handle_t &handle, ffrt_function_header_t *f,
        const ffrt_deps_t *ins, const ffrt_deps_t *outs, const task_attr_private *attr) = 0;

    void onSubmitUV(ffrt_executor_task_t *task, const task_attr_private *attr);

    void onSubmitIO(const ffrt_io_callable_t &work, const task_attr_private *attr);

    virtual void onWait() = 0;

    virtual void onWait(const ffrt_deps_t* deps) = 0;

    virtual int onExecResults(ffrt_task_handle_t handle) = 0;

    virtual void onTaskDone(CPUEUTask* task) = 0;

    virtual int onSkip(ffrt_task_handle_t handle);

    static inline CPUEUTask* Root()
    {
        // Within an ffrt process, different threads may have different QoS interval
        thread_local static RootTaskCtxWrapper root_wraper;
        return root_wraper.Root();
    }

protected:
    DependenceManager() {}
    virtual ~DependenceManager() {}
};

} // namespace ffrt
#endif
