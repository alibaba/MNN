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

#ifndef FFRT_TYPES_HPP
#define FFRT_TYPES_HPP

#include <cstdint>

namespace ffrt {
#ifdef CLOSE_COROUTINE_MODE
constexpr bool USE_COROUTINE = false;
#else
constexpr bool USE_COROUTINE = true;
#endif
enum DT {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    FP16,
    FP32,
    FP64,
};

enum class DevType {
    CPU,
    DEVMAX,
};

enum class TaskType {
    ROOT,
    DEFAULT,
};

enum class DataStatus {
    IDLE, // 默认状态
    READY, // 当前版本被生产出来，标志着这个版本的所有消费者可以执行
    CONSUMED, // 同时也是RELEASE，当前版本的所有消费者已经执行完成，标志着下一个版本的生产者可以执行
    MERGED, // 嵌套场景下，标志一个子任务的version已经被父任务的version合并
};

enum class NestType {
    DEFAULT, // 不存在嵌套关系
    PARENTOUT, // 同parent的输出嵌套
    PARENTIN, // 同parent的输入嵌套
};

/* Note: do not change the order of the enum values.
 * If a new value is added, or the order changed
 * make sure to update `StatusToString` as well.
 */
enum class TaskStatus : uint8_t {
    PENDING,         // 任务创建后的初始状态
    ENQUEUED,         // 队列任务插入队列中 (串行/并发队列任务)
    DEQUEUED,         // 队列任务从队列中取出 (串行/并发队列任务)
    SUBMITTED,       // 任务存在数据依赖
    READY,           // 任务没有依赖/依赖解除
    POPPED,           // 任务从ReadyQueue中取出，等待执行
    EXECUTING,       // 任务执行在worker线程
    THREAD_BLOCK,    // 任务由于FFRT同步原语进入线程阻塞状态
    COROUTINE_BLOCK, // 任务由于FFRT同步原语进入协程阻塞状态
    FINISH,          // 任务执行完成，可解除依赖
    WAIT_RELEASING,  // 任务资源等待回收 (父子嵌套依赖时，父任务完成但子任务还未完成，可以进入此状态)
    CANCELED,        // 任务未执行前被取消 (cancel/skip语义)
};

enum class AliveStatus : uint8_t {
    UNITINITED,
    INITED,
    RELEASED,
};

enum class BlockType : uint8_t {
    BLOCK_COROUTINE,
    BLOCK_THREAD
};

enum class Dependence : uint8_t {
    DEPENDENCE_INIT,
    DATA_DEPENDENCE,
    CALL_DEPENDENCE,
    CONDITION_DEPENDENCE,
};

enum class SpecTaskType {
    EXIT_TASK,
    SLEEP_TASK,
    SPEC_TASK_MAX,
};

enum SkipStatus : uint8_t {
    SUBMITTED,
    EXECUTED,
    SKIPPED,
};
#ifndef _MSC_VER
#define FFRT_LIKELY(x) (__builtin_expect(!!(x), 1))
#define FFRT_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#define FFRT_LIKELY(x) (x)
#define FFRT_UNLIKELY(x) (x)
#endif

#define FORCE_INLINE
} // namespace ffrt
#endif
