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

#ifndef FFRT_CO_ROUTINE_HPP
#define FFRT_CO_ROUTINE_HPP
#include <atomic>
#include <functional>
#include <thread>
#include <pthread.h>
#include "co2_context.h"

#if defined(__aarch64__)
constexpr size_t STACK_MAGIC = 0x7BCDABCDABCDABCD;
#elif defined(__arm__)
constexpr size_t STACK_MAGIC = 0x7BCDABCD;
#elif defined(__x86_64__)
constexpr size_t STACK_MAGIC = 0x7BCDABCDABCDABCD;
#endif

#ifndef FFRT_STACK_SIZE
#define FFRT_STACK_SIZE (1 << 20)
#endif

#ifdef ASAN_MODE
extern "C" void __sanitizer_start_switch_fiber(void **fake_stack_save, const void *bottom, size_t size);
extern "C" void __sanitizer_finish_switch_fiber(void *fake_stack_save, const void **bottom_old, size_t *size_old);
extern "C" void __asan_handle_no_return();
#endif

namespace ffrt {
class CoTask;
class CPUEUTask;
struct WaitEntry;
} // namespace ffrt
struct CoRoutine;

enum class CoStatus {
    CO_UNINITIALIZED,
    CO_NOT_FINISH,
    CO_RUNNING,
};

enum class CoStackProtectType {
    CO_STACK_WEAK_PROTECT,
    CO_STACK_STRONG_PROTECT
};

enum class CoWakeType {
    TIMEOUT_WAKE,
    NO_TIMEOUT_WAKE
};

constexpr uint64_t STACK_SIZE = FFRT_STACK_SIZE;
constexpr uint64_t MIN_STACK_SIZE = 32 * 1024;
constexpr uint64_t STACK_MEM_SIZE = 8;

using CoCtx = ffrt_fiber_t;

struct CoRoutineEnv {
    // when task is running, runningCo same with task->co
    // if task switch out, set to null. if task complete, be used as co cache for next task.
    CoRoutine* runningCo = nullptr;
    CoCtx schCtx;
    const std::function<bool(ffrt::CoTask*)>* pending = nullptr;
};

struct StackMem {
    uint64_t size;
    size_t magic;
    uint8_t stk[STACK_MEM_SIZE];
};

struct CoRoutine {
    std::atomic_int status {static_cast<int>(CoStatus::CO_UNINITIALIZED)};
    CoRoutineEnv* thEnv;
    ffrt::CoTask* task;
#ifdef ASAN_MODE
    void *asanFakeStack = nullptr;  // not finished, need further verification
    const void *asanFiberAddr = nullptr;
    size_t asanFiberSize = 0;
#endif
    CoCtx ctx;
    uint64_t allocatedSize; // CoRoutine allocated size
    bool isTaskDone = false;
    /* do not add item after stkMem */
    StackMem stkMem;
};

struct CoStackAttr {
public:
    explicit CoStackAttr(uint64_t coSize = STACK_SIZE, CoStackProtectType coType =
        CoStackProtectType::CO_STACK_WEAK_PROTECT)
    {
        size = coSize;
        type = coType;
    }
    ~CoStackAttr() {}
    uint64_t size;
    CoStackProtectType type;

    static inline CoStackAttr* Instance(uint64_t coSize = STACK_SIZE,
        CoStackProtectType coType = CoStackProtectType::CO_STACK_WEAK_PROTECT)
    {
        static CoStackAttr inst(coSize, coType);
        return &inst;
    }
};

class CoRoutineFactory {
public:
    using CowakeCB = std::function<void (ffrt::CoTask*, CoWakeType)>;

    static CoRoutineFactory &Instance();

    static void CoWakeFunc(ffrt::CoTask* task, CoWakeType type)
    {
        return Instance().cowake_(task, type);
    }

    static void RegistCb(const CowakeCB &cowake)
    {
        Instance().cowake_ = cowake;
    }
private:
    CowakeCB cowake_;
};

void CoStackFree(void);
void CoWorkerExit(void);

int CoStart(ffrt::CoTask* task, CoRoutineEnv* coRoutineEnv);
void CoYield(void);

void CoWait(const std::function<bool(ffrt::CoTask*)>& pred);
void CoWake(ffrt::CoTask* task, CoWakeType type);

CoRoutineEnv* GetCoEnv(void);

inline void* GetCoStackAddr(CoRoutine* co)
{
    return static_cast<void*>(reinterpret_cast<char*>(co) + sizeof(CoRoutine) - STACK_MEM_SIZE);
}

#ifdef FFRT_TASK_LOCAL_ENABLE
void TaskTsdDeconstruct(ffrt::CPUEUTask* task);
#endif

#endif
