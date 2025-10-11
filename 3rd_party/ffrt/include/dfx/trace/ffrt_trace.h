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

#ifndef __FFRT_TRACE_H__
#define __FFRT_TRACE_H__

#include <atomic>
#include <chrono>
#include "internal_inc/osal.h"
#include "dfx/log/ffrt_log_api.h"

#ifdef FFRT_OH_TRACE_ENABLE
#include <dlfcn.h>
#endif

namespace ffrt {
enum TraceLevel {
    TRACE_LEVEL0 = 0,
    TRACE_LEVEL1,
    TRACE_LEVEL2,
    TRACE_LEVEL3, // lowest level, trace all
    TRACE_LEVEL_MAX,
};

class TraceLevelManager {
public:
    TraceLevelManager();
    ~TraceLevelManager() = default;

    uint64_t GetTraceLevel() const
    {
        return traceLevel_;
    }

    static inline TraceLevelManager* Instance()
    {
        static TraceLevelManager ins;
        return &ins;
    }

private:
    uint8_t traceLevel_;
};

class ScopedTrace {
public:
    ScopedTrace(uint64_t level, const char* name);
    ~ScopedTrace();

private:
    std::atomic<bool> isTraceEnable_;
};
} // namespace ffrt

#ifdef FFRT_OH_TRACE_ENABLE
constexpr uint64_t HITRACE_TAG_FFRT = (1ULL << 13); // ffrt tasks.
bool IsTagEnabled(uint64_t tag);
void StartTrace(uint64_t label, const std::string& value, float limit = -1);
void FinishTrace(uint64_t label);
void StartAsyncTrace(uint64_t label, const std::string& value, int32_t taskId, float limit = -1);
void FinishAsyncTrace(uint64_t label, const std::string& value, int32_t taskId);
void CountTrace(uint64_t label, const std::string& name, int64_t count);
#ifdef APP_USE_ARM
constexpr const char* TRACE_LIB_PATH = "libhitrace_meter.so";
#else
constexpr const char* TRACE_LIB_PATH = "libhitrace_meter.so";
#endif
class TraceAdapter {
public:
    TraceAdapter()
    {
        Load();
    }

    ~TraceAdapter()
    {
        UnLoad();
    }

    static TraceAdapter* Instance()
    {
        static TraceAdapter instance;
        return &instance;
    }

#define REG_FUNC(func) using func##Type = decltype(func)*; func##Type func = nullptr
    REG_FUNC(IsTagEnabled);
    REG_FUNC(StartTrace);
    REG_FUNC(FinishTrace);
    REG_FUNC(StartAsyncTrace);
    REG_FUNC(FinishAsyncTrace);
    REG_FUNC(CountTrace);
#undef REG_FUNC

private:
    bool Load()
    {
        if (handle != nullptr) {
            FFRT_LOGD("handle exits");
            return true;
        }

        handle = dlopen(TRACE_LIB_PATH, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
        if (handle == nullptr) {
            FFRT_LOGE("load so[%s] fail", TRACE_LIB_PATH);
            return false;
        }

#define LOAD_FUNC(x) x = reinterpret_cast<x##Type>(dlsym(handle, #x));                        \
        if (x == nullptr)                                                                     \
        {                                                                                     \
            FFRT_LOGE("load func %s from %s failed", #x, TRACE_LIB_PATH);             \
            return false;                                                                     \
        }
            LOAD_FUNC(IsTagEnabled);
            LOAD_FUNC(StartTrace);
            LOAD_FUNC(FinishTrace);
            LOAD_FUNC(StartAsyncTrace);
            LOAD_FUNC(FinishAsyncTrace);
            LOAD_FUNC(CountTrace);
#undef LOAD_FUNC
        return true;
    }

    bool UnLoad()
    {
        if (handle != nullptr) {
            if (dlclose(handle) != 0) {
                return false;
            }
            handle = nullptr;
            return true;
        }
        return true;
    }

    void* handle = nullptr;
};

#define GET_TRACE_FUNC(x) (TraceAdapter::Instance()->x)

static bool _IsTagEnabled(uint64_t label)
{
    auto func = GET_TRACE_FUNC(IsTagEnabled);
    if (func != nullptr) {
        return func(label);
    }
    return false;
}
#define _StartTrace(label, tag, limit) \
    do { \
        auto func = GET_TRACE_FUNC(StartTrace); \
        if (func != nullptr) { \
            func(label, tag, limit); \
        } \
    } while (0)
#define _FinishTrace(label) \
    do { \
        auto func = GET_TRACE_FUNC(FinishTrace); \
        if (func != nullptr) { \
            func(label); \
        } \
    } while (0)
#define _StartAsyncTrace(label, tag, tid, limit) \
    do { \
        auto func = GET_TRACE_FUNC(StartAsyncTrace); \
        if (func != nullptr) { \
            func(label, tag, tid, limit); \
        } \
    } while (0)
#define _FinishAsyncTrace(label, tag, tid) \
    do { \
        auto func = GET_TRACE_FUNC(FinishAsyncTrace); \
        if (func != nullptr) { \
            func(label, tag, tid); \
        } \
    } while (0)
#define _TraceCount(label, tag, value) \
    do { \
        auto func = GET_TRACE_FUNC(CountTrace); \
        if (func != nullptr) { \
            func(label, tag, value); \
        } \
    } while (0)

#define FFRT_TRACE_BEGIN(tag) \
    do { \
        if (__builtin_expect(!!(_IsTagEnabled(HITRACE_TAG_FFRT)), 0)) \
            _StartTrace(HITRACE_TAG_FFRT, tag, -1); \
    } while (false)
#define FFRT_TRACE_END() \
    do { \
        if (__builtin_expect(!!(_IsTagEnabled(HITRACE_TAG_FFRT)), 0)) \
            _FinishTrace(HITRACE_TAG_FFRT); \
    } while (false)
#define FFRT_TRACE_ASYNC_BEGIN(tag, tid) \
    do { \
        if (__builtin_expect(!!(_IsTagEnabled(HITRACE_TAG_FFRT)), 0)) \
            _StartAsyncTrace(HITRACE_TAG_FFRT, tag, tid, -1); \
    } while (false)
#define FFRT_TRACE_ASYNC_END(tag, tid) \
    do { \
        if (__builtin_expect(!!(_IsTagEnabled(HITRACE_TAG_FFRT)), 0)) \
            _FinishAsyncTrace(HITRACE_TAG_FFRT, tag, tid); \
    } while (false)
#define FFRT_TRACE_COUNT(tag, value) \
    do { \
        if (__builtin_expect(!!(_IsTagEnabled(HITRACE_TAG_FFRT)), 0)) \
            _TraceCount(HITRACE_TAG_FFRT, tag, value); \
    } while (false)
#define FFRT_SUBMIT_MARKER(gid) \
    do { \
        if (__builtin_expect(!!(_IsTagEnabled(HITRACE_TAG_FFRT)), 0)) { \
            _StartTrace(HITRACE_TAG_FFRT, ("P " + std::to_string(gid)).c_str(), -1); \
        } \
    } while (false)
#define FFRT_TASK_BEGIN(tag, gid) \
    do { \
        if (__builtin_expect(!!(_IsTagEnabled(HITRACE_TAG_FFRT)), 0)) { \
            _StartTrace(HITRACE_TAG_FFRT, ("FFRT" + (tag) + "|" + std::to_string(gid)).c_str(), -1); \
        } \
    } while (false)
#define FFRT_BLOCK_TRACER(gid, tag) \
    do { \
        if (__builtin_expect(!!(_IsTagEnabled(HITRACE_TAG_FFRT)), 0)) \
            _StartTrace(HITRACE_TAG_FFRT, ("FFBK" #tag "|" + std::to_string(gid)).c_str(), -1); \
        FFRT_TRACE_END(); \
    } while (false)
#define FFRT_WAKE_TRACER(gid) \
    do { \
        if (__builtin_expect(!!(_IsTagEnabled(HITRACE_TAG_FFRT)), 0)) \
            _StartTrace(HITRACE_TAG_FFRT, ("FFWK|" + std::to_string(gid)).c_str(), -1); \
        FFRT_TRACE_END(); \
    } while (false)
#define FFRT_EXECUTOR_TASK_BEGIN(gid) \
    do { \
        if (__builtin_expect(!!(_IsTagEnabled(HITRACE_TAG_FFRT)), 0)) \
            _StartTrace(HITRACE_TAG_FFRT, ("FFRTex_task|" + std::to_string(gid)).c_str(), -1); \
    } while (false)
#define FFRT_SERIAL_QUEUE_TASK_SUBMIT_MARKER(qid, gid) \
    do { \
        if (__builtin_expect(!!(_IsTagEnabled(HITRACE_TAG_FFRT)), 0)) \
            _StartTrace(HITRACE_TAG_FFRT, ("P[sq_" + \
                std::to_string(qid) + "]|" + std::to_string(gid)).c_str(), -1); \
        FFRT_TRACE_END(); \
    } while (false)
#define FFRT_TRACE_SCOPE(level, tag) ffrt::ScopedTrace ___tracer##tag(level, #tag)
#else
#define FFRT_TRACE_BEGIN(tag)
#define FFRT_TRACE_END()
#define FFRT_TRACE_ASYNC_BEGIN(tag, tid)
#define FFRT_TRACE_ASYNC_END(tag, tid)
#define FFRT_TRACE_COUNT(tag, value)
#define FFRT_TRACE_SCOPE(level, tag)
#define FFRT_SUBMIT_MARKER(gid)
#define FFRT_TASK_BEGIN(tag, gid)
#define FFRT_BLOCK_TRACER(gid, tag)
#define FFRT_WAKE_TRACER(gid)
#define FFRT_EXECUTOR_TASK_BEGIN(gid)
#define FFRT_SERIAL_QUEUE_TASK_SUBMIT_MARKER(qid, gid)
#endif

// DFX Trace for FFRT Normal Task
#define FFRT_WORKER_IDLE_BEGIN_MARKER()
#define FFRT_WORKER_IDLE_END_MARKER()
#define FFRT_READY_MARKER(gid) \
    { \
        FFRT_TRACE_ASYNC_END("R", gid); \
    }
#define FFRT_BLOCK_MARKER(gid) \
    { \
        FFRT_TRACE_ASYNC_END("B", gid); \
    }
#define FFRT_TASKDONE_MARKER(gid) \
    { \
        FFRT_TRACE_ASYNC_END("F", gid); \
    }
#define FFRT_TASK_END() \
    { \
        FFRT_TRACE_END(); \
    }
// DFX Trace for FFRT Executor Task
#define FFRT_EXECUTOR_TASK_SUBMIT_MARKER(gid) \
    { \
        FFRT_TRACE_ASYNC_END("P", gid); \
    }
#endif