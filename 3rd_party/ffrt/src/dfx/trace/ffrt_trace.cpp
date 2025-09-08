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

#include "dfx/trace/ffrt_trace.h"
#include "internal_inc/osal.h"

namespace ffrt {
TraceLevelManager::TraceLevelManager()
{
    traceLevel_ = FFRT_TRACE_LEVEL;
    std::string trace = GetEnv("FFRT_TRACE_LEVEL");
    if (trace.size() == 0) {
        return;
    }

    int traceLevel = std::stoi(trace);
    if (traceLevel >= TRACE_LEVEL_MAX || traceLevel < TRACE_LEVEL0) {
        FFRT_LOGE("get invalid trace level, %d", traceLevel);
        return;
    }
    traceLevel_ = static_cast<uint8_t>(traceLevel);
    FFRT_LOGW("set trace level %d", traceLevel_);
}

// make sure TraceLevelManager last free
static __attribute__((constructor)) void TraceInit(void)
{
    ffrt::TraceLevelManager::Instance();
}

ScopedTrace::ScopedTrace(uint64_t level, const char* name)
    : isTraceEnable_(false)
{
    if (level <= TraceLevelManager::Instance()->GetTraceLevel()) {
        isTraceEnable_ = true;
        FFRT_TRACE_BEGIN(name);
    }
}

ScopedTrace::~ScopedTrace()
{
    if (!isTraceEnable_) {
        return;
    }

    FFRT_TRACE_END();
}
} // namespace ffrt