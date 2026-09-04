//
//  ThreadPool.cpp
//  MNN
//
//  Created by MNN on 2019/06/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_USE_THREAD_POOL
#include "backend/cpu/ThreadPool.hpp"
#include <string.h>
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <MNN/MNNDefine.h>
#include "ThreadPool.hpp"

#define MNN_THREAD_POOL_MAX_TASKS 2
namespace MNN {
static std::unordered_map<long int, ThreadPool*> gInstances;
static std::mutex gInitMutex;

// Number of cheap in-core backoff iterations before falling back to a real
// scheduler yield. std::this_thread::yield() is a syscall (swtch_pri on
// Darwin); calling it on every spin iteration dominates per-op barrier cost
// for LLM decode, where each token issues hundreds of tiny parallel tasks.
static constexpr uint32_t kThreadPoolSpinBudget = 512;

// How long a worker may idle while the pool is active before blocking on the
// condition variable. Hot-spinning workers that hold no work share (e.g.
// threads excluded from compute-bound prefill work on heterogeneous 4P+6E
// SoCs) poison the OS scheduler's core placement and steal performance-core
// time from working threads; blocking removes them until enqueueInternal
// flags them again. Timed rather than counted because yield cost varies by
// an order of magnitude with system load; sized above decode's
// sub-millisecond inter-token gaps so decode workers never block.
static constexpr auto kWorkerIdleTimeout = std::chrono::milliseconds(8);

// mSleepMask tracks one worker per bit, so only workers below this index can be
// woken by enqueueInternal. Workers at or above it get no bit and stay on the
// spin/yield path instead of blocking, which keeps the mask arithmetic free of
// out-of-range shifts on hosts with more cores than the mask has bits.
static constexpr int kSleepMaskWidth = 64;

// Tiered backoff shared by the workers and by the enqueue-side completion wait.
// `spin` is advanced on every architecture, including those with no in-core
// hint instruction, so that the caller's tier checks and the worker's idle
// timeout behave the same everywhere.
static inline void MNNThreadPoolRelax(uint32_t& spin) {
    if (spin < kThreadPoolSpinBudget) {
        ++spin;
#if defined(__riscv)
        // Zihintpause encoding. Use the raw instruction so older assemblers do not need to recognize the mnemonic.
        asm volatile(".word 0x0100000f" ::: "memory");
        return;
#elif defined(__aarch64__) || defined(__arm64__)
        // `isb` is the usual ARM64 spin-wait backoff: a few tens of cycles of
        // delay without leaving the core, unlike the `yield` hint which is a
        // no-op on cores without SMT.
        asm volatile("isb sy" ::: "memory");
        return;
#endif
    }
    // Budget stays exhausted for the rest of this wait: short barriers are
    // served entirely by the spin above, while long waits (big-KV attention,
    // prefill tiles) fall back to yielding instead of spinning at full rate.
    std::this_thread::yield();
}

int ThreadPool::init(int numberThread, unsigned long cpuMask, ThreadPool*& threadPool) {
    if (1 >= numberThread) {
        numberThread = 1;
    }
    std::lock_guard<std::mutex> _l(gInitMutex);

    if (gInstances.find(cpuMask) == gInstances.end()){
        gInstances[cpuMask] = new ThreadPool(numberThread);
    }
    threadPool = gInstances[cpuMask];
    if (gInstances[cpuMask]->numberThread() < numberThread){
        return gInstances[cpuMask]->numberThread();
    }
    return numberThread;
}

void ThreadPool::destroy() {
    std::lock_guard<std::mutex> _l(gInitMutex);
    for (auto i= gInstances.begin(); i != gInstances.end(); i++){
        if (i->second){
            delete i->second;
        }
    }
    gInstances.clear();
}

ThreadPool::ThreadPool(int numberThread) {
    mNumberThread = numberThread;
    mActiveCount  = 0;
    mTaskAvailable.resize(MNN_THREAD_POOL_MAX_TASKS);
    mTasks.resize(MNN_THREAD_POOL_MAX_TASKS);
    for (int t = 0; t < mTasks.size(); ++t) {
        mTaskAvailable[t] = true;
        for (int i = 0; i < mNumberThread; ++i) {
            mTasks[t].second.emplace_back(new std::atomic_bool{false});
        }
    }
    for (int i = 1; i < mNumberThread; ++i) {
        int threadIndex = i;
        mWorkers.emplace_back([this, threadIndex]() {
            const uint64_t mySleepBit = threadIndex < kSleepMaskWidth ? (uint64_t(1) << threadIndex) : 0;
            while (!mStop) {
                uint32_t spin = 0;
                uint32_t idleYields = 0;
                auto idleStart = std::chrono::steady_clock::now();
                while (mActiveCount > 0) {
                    bool worked = false;
                    for (int i = 0; i < MNN_THREAD_POOL_MAX_TASKS; ++i) {
                        if (*mTasks[i].second[threadIndex]) {
                            mTasks[i].first->first(threadIndex);
                            { *mTasks[i].second[threadIndex] = false; }
                            worked = true;
                        }
                    }
                    if (worked) {
                        spin = 0;
                        idleYields = 0;
                        idleStart = std::chrono::steady_clock::now();
                        continue;
                    }
                    if (spin < kThreadPoolSpinBudget) {
                        MNNThreadPoolRelax(spin);
                        continue;
                    }
                    ++idleYields;
                    if (mySleepBit != 0 && (idleYields & 63) == 0 &&
                        std::chrono::steady_clock::now() - idleStart >= kWorkerIdleTimeout) {
                        // Prolonged idleness mid-run: block instead of
                        // spinning. Set the sleep bit before re-checking the
                        // flags so an enqueue racing us either sees the bit
                        // (and notifies) or has its flag observed by the
                        // predicate below.
                        mSleepMask.fetch_or(mySleepBit);
                        {
                            std::unique_lock<std::mutex> _l(mQueueMutex);
                            mCondition.wait(_l, [this, threadIndex] {
                                if (mStop || mActiveCount <= 0) {
                                    return true;
                                }
                                for (int i = 0; i < MNN_THREAD_POOL_MAX_TASKS; ++i) {
                                    if (*mTasks[i].second[threadIndex]) {
                                        return true;
                                    }
                                }
                                return false;
                            });
                        }
                        mSleepMask.fetch_and(~mySleepBit);
                        spin = 0;
                        idleYields = 0;
                        // idleStart is deliberately left at its pre-block
                        // value: a worker woken by a notify_all meant for
                        // someone else finds no flag, and the stale deadline
                        // sends it back to sleep after one short spin instead
                        // of letting it hot-spin for another full timeout. The
                        // worked branch above is what refreshes the deadline.
                        continue;
                    }
                    std::this_thread::yield();
                }
                std::unique_lock<std::mutex> _l(mQueueMutex);
                mCondition.wait(_l, [this] { return mStop || mActiveCount > 0; });
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> _l(mQueueMutex);
        mStop = true;
    }
    mCondition.notify_all();
    for (auto& worker : mWorkers) {
        worker.join();
    }
    for (auto& task : mTasks) {
        for (auto c : task.second) {
            delete c;
        }
    }
}

int ThreadPool::acquireWorkIndex() {
    std::lock_guard<std::mutex> _l(mQueueMutex);
    for (int i = 0; i < MNN_THREAD_POOL_MAX_TASKS; ++i) {
        if (mTaskAvailable[i]) {
            mTaskAvailable[i] = false;
            return i;
        }
    }
    return -1;
}
void ThreadPool::releaseWorkIndex(int index) {
    if (index < 0 || index >= MNN_THREAD_POOL_MAX_TASKS) {
        return;
    }
    std::lock_guard<std::mutex> _l(mQueueMutex);
    mTaskAvailable[index] = true;
}

void ThreadPool::active() {
    {
        std::lock_guard<std::mutex> _l(mQueueMutex);
        mActiveCount++;
    }
    mCondition.notify_all();
}
void ThreadPool::deactive() {
    mActiveCount--;
}

void ThreadPool::enqueue(TASK* taskp, int index) {
    auto& task = *taskp;
    if (1 >= task.second || 0 > index) {
        for (int i = 0; i < task.second; ++i) {
            task.first(i);
        }
        return;
    }
    enqueueInternal(taskp, index);
}
void ThreadPool::enqueueInternal(TASK* taskp, int index) {
    auto& task = *taskp;
    if (mActiveCount == 0) {
        for (int i = 0; i < task.second; ++i) {
            task.first(i);
        }
        return;
    }
    int workSize = task.second;
    TASK* tmpTask = nullptr;
    if (workSize > mNumberThread) {
        tmpTask = new TASK;
        *tmpTask = std::make_pair([workSize, &task, this](int tId) {
            for (int v = tId; v < workSize; v += mNumberThread) {
                task.first(v);
            }
        }, mNumberThread);
        mTasks[index].first = tmpTask;
        workSize = mNumberThread;
    } else {
        mTasks[index].first = taskp;
    }
    {
        for (int i = 1; i < workSize; ++i) {
            *mTasks[index].second[i] = true;
        }
    }
    // Wake sleepers only when a flagged worker is among them. The mutex hold
    // is required: without it the notify could land between a worker's
    // predicate check and its atomic block, leaving the flag unseen.
    const uint64_t flaggedMask = workSize >= kSleepMaskWidth
                                     ? ~uint64_t(1)
                                     : (((uint64_t(1) << workSize) - 1) & ~uint64_t(1));
    if ((mSleepMask.load() & flaggedMask) != 0) {
        {
            std::lock_guard<std::mutex> _l(mQueueMutex);
        }
        mCondition.notify_all();
    }
    mTasks[index].first->first(0);
    bool complete = true;
    uint32_t spin = 0;
    do {
        complete = true;
        for (int i = 1; i < workSize; ++i) {
            if (*mTasks[index].second[i]) {
                complete = false;
                break;
            }
        }
        MNNThreadPoolRelax(spin);
        // FUNC_PRINT(notComplete);
    } while (!complete);
    if (nullptr != tmpTask) {
        delete tmpTask;
    }
}
} // namespace MNN
#endif
