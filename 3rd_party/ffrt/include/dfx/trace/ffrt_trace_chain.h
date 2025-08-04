/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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

#ifndef __FFRT_TRACE_CHAIN_H__
#define __FFRT_TRACE_CHAIN_H__

#include "tm/task_base.h"
#include "dfx/log/ffrt_log_api.h"

namespace ffrt {

class TraceChainAdapter {
public:
    TraceChainAdapter();
    ~TraceChainAdapter();

    static TraceChainAdapter& Instance();

    HiTraceIdStruct HiTraceChainGetId();
    void HiTraceChainClearId();
    void HiTraceChainRestoreId(const HiTraceIdStruct* pId);
    HiTraceIdStruct HiTraceChainCreateSpan();
    HiTraceIdStruct HiTraceChainBegin(const char* name, int flags);
    void HiTraceChainEnd(const HiTraceIdStruct* pId);

private:
    void Load();
    void UnLoad();

    void* handle_ = nullptr;

    using HiTraceChainGetIdFunc = HiTraceIdStruct (*)();
    using HiTraceChainClearIdFunc = void (*)();
    using HiTraceChainRestoreIdFunc = void (*)(const HiTraceIdStruct*);
    using HiTraceChainCreateSpanFunc = HiTraceIdStruct (*)();
    using HiTraceChainBeginFunc = HiTraceIdStruct (*)(const char*, int);
    using HiTraceChainEndFunc = void (*)(const HiTraceIdStruct*);

    HiTraceChainGetIdFunc getIdFunc_ = nullptr;
    HiTraceChainClearIdFunc clearIdFunc_ = nullptr;
    HiTraceChainRestoreIdFunc restoreIdFunc_ = nullptr;
    HiTraceChainCreateSpanFunc createSpanFunc_ = nullptr;
    HiTraceChainBeginFunc beginChainFunc_ = nullptr;
    HiTraceChainEndFunc endChainFunc_ = nullptr;
};
} // namespace ffrt

#endif // __FFRT_TRACE_CHAIN_H__