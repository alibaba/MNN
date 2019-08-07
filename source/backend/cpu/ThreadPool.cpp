//
//  ThreadPool.cpp
//  MNN
//
//  Created by MNN on 2019/06/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_USE_THREAD_POOL
#include "ThreadPool.hpp"
#include <string.h>
#include "MNNDefine.h"
#ifdef __ANDROID__
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#endif
//#define MNN_THREAD_LOCK_CPU

#define MNN_THREAD_POOL_MAX_TASKS 2
namespace MNN {
ThreadPool* ThreadPool::gInstance = nullptr;
static std::mutex gInitMutex;
int ThreadPool::init(int number) {
    if (1 >= number) {
        return 1;
    }
    std::lock_guard<std::mutex> _l(gInitMutex);
    if (nullptr != gInstance) {
        if (gInstance->number() < number) {
            return gInstance->number();
        }
    }
    if (nullptr == gInstance) {
        gInstance = new ThreadPool(number);
    }
    return number;
}
void ThreadPool::destroy() {
    std::lock_guard<std::mutex> _l(gInitMutex);
    if (nullptr != gInstance) {
        delete gInstance;
        gInstance = nullptr;
    }
}
#ifdef MNN_THREAD_LOCK_CPU
static int getNumberOfCPU() {
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp) {
        return 1;
    }
    int number = 0;
    char buffer[1024];
    while (!feof(fp)) {
        char* str = fgets(buffer, 1024, fp);
        if (!str) {
            break;
        }
        if (memcmp(buffer, "processor", 9) == 0) {
            number++;
        }
    }
    fclose(fp);
    if (number < 1) {
        number = 1;
    }
    return number;
}

static int getCPUMaxFreqKHz(int cpuID) {
    char path[256];
    sprintf(path, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuID);
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state", cpuID);
        fp = fopen(path, "rb");
        if (!fp) {
            sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuID);
            fp = fopen(path, "rb");
            if (!fp) {
                return -1;
            }
            int maxfrequency = -1;
            fscanf(fp, "%d", &maxfrequency);
            fclose(fp);
            return maxfrequency;
        }
    }
    int maxfrequency = 0;
    while (!feof(fp)) {
        int frequency = 0;
        int history   = fscanf(fp, "%d %*d", &frequency);
        if (history != 1) {
            break;
        }
        if (frequency > maxfrequency) {
            maxfrequency = frequency;
        }
    }
    fclose(fp);
    return maxfrequency;
}

static std::vector<int> sortCPUIDByMaxFrequency(int maxNumbers) {
    const int cpuNumbers = getNumberOfCPU();
    if (cpuNumbers == 0) {
        return {};
    }
    std::vector<int> cpuIDs;
    std::vector<std::pair<int, int>> cpusFrequency;
    cpusFrequency.resize(cpuNumbers);
    for (int i = 0; i < cpuNumbers; ++i) {
        int frequency           = getCPUMaxFreqKHz(i);
        cpusFrequency[i].first  = frequency;
        cpusFrequency[i].second = i;
    }
    maxNumbers = std::min(maxNumbers, cpuNumbers);
    std::sort(cpusFrequency.rbegin(), cpusFrequency.rend());
    cpuIDs.resize(maxNumbers);
    for (int i = 0; i < maxNumbers; ++i) {
        cpuIDs[i] = cpusFrequency[i].second;
    }
    // FUNC_PRINT(cpusFrequency[0].first);
    return cpuIDs;
}

static int setSchedAffinity(const std::vector<int>& cpuIDs) {
#define __NCPUBITS (8 * sizeof(unsigned long))
    typedef struct {
        unsigned long __bits[CPU_SETSIZE / __NCPUBITS];
    } cpu_set_t;

    // set affinity for thread

    pid_t pid = gettid();
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int i = 0; i < (int)cpuIDs.size(); i++) {
        CPU_SET(cpuIDs[i], &mask);
    }

    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
    if (syscallret) {
        MNN_PRINT("syscall error %d\n", syscallret);
        return -1;
    }

    return 0;
}

#endif // arch
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
#ifdef MNN_THREAD_LOCK_CPU
    std::vector<int> sortedCPUIDs = sortCPUIDByMaxFrequency(numberThread);
#endif
    for (int i = 1; i < mNumberThread; ++i) {
        int threadIndex = i;
#ifdef MNN_THREAD_LOCK_CPU
        mWorkers.emplace_back([this, sortedCPUIDs, threadIndex]() {
#else
        mWorkers.emplace_back([this, threadIndex]() {
#endif
#ifdef MNN_THREAD_LOCK_CPU
            int res = setSchedAffinity(sortedCPUIDs);
#endif
            while (!mStop) {
                while (mActiveCount > 0) {
                    for (int i = 0; i < MNN_THREAD_POOL_MAX_TASKS; ++i) {
                        if (*mTasks[i].second[threadIndex]) {
                            mTasks[i].first.first(threadIndex);
                            { *mTasks[i].second[threadIndex] = false; }
                        }
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
    mStop = true;
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
    if (nullptr == gInstance) {
        return -1;
    }
    std::lock_guard<std::mutex> _l(gInstance->mQueueMutex);
    for (int i = 0; i < MNN_THREAD_POOL_MAX_TASKS; ++i) {
        if (gInstance->mTaskAvailable[i]) {
            gInstance->mTaskAvailable[i] = false;
            return i;
        }
    }
    return -1;
}
void ThreadPool::releaseWorkIndex(int index) {
    if (nullptr == gInstance) {
        return;
    }
    if (index < 0 || index >= MNN_THREAD_POOL_MAX_TASKS) {
        return;
    }
    std::lock_guard<std::mutex> _l(gInstance->mQueueMutex);
    gInstance->mTaskAvailable[index] = true;
}

void ThreadPool::active() {
    if (nullptr == gInstance) {
        return;
    }
    gInstance->mActiveCount++;
    std::lock_guard<std::mutex> _l(gInstance->mQueueMutex);
    gInstance->mCondition.notify_all();
}
void ThreadPool::deactive() {
    if (nullptr == gInstance) {
        return;
    }
    gInstance->mActiveCount--;
}

void ThreadPool::enqueue(TASK&& task, int index) {
    if (1 >= task.second || 0 > index) {
        for (int i = 0; i < task.second; ++i) {
            task.first(i);
        }
        return;
    }
    MNN_ASSERT(nullptr != gInstance);
    gInstance->enqueueInternal(std::move(task), index);
}
void ThreadPool::enqueueInternal(TASK&& task, int index) {
    int workSize = task.second;
    if (workSize > mNumberThread) {
        mTasks[index].first = std::make_pair(
            [workSize, &task, this](int tId) {
                for (int v = tId; v < workSize; v += mNumberThread) {
                    task.first(v);
                }
            },
            mNumberThread);
        workSize = mNumberThread;
    } else {
        mTasks[index].first = std::move(task);
    }
    {
        for (int i = 1; i < workSize; ++i) {
            *mTasks[index].second[i] = true;
        }
    }
    mTasks[index].first.first(0);
    bool complete = true;
    do {
        std::this_thread::yield();
        complete = true;
        for (int i = 1; i < workSize; ++i) {
            if (*mTasks[index].second[i]) {
                complete = false;
                break;
            }
        }
        // FUNC_PRINT(notComplete);
    } while (!complete);
}
} // namespace MNN
#endif
