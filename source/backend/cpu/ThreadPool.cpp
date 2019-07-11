//
//  ThreadPool.cpp
//  MNN
//
//  Created by MNN on 2019/06/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_USE_THREAD_POOL
#include "ThreadPool.hpp"
#include "MNNDefine.h"
#include <string.h>
#ifdef __ANDROID__
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#endif
namespace MNN {
ThreadPool* ThreadPool::gInstance = nullptr;
static std::mutex gInitMutex;
void ThreadPool::init(int number) {
    std::unique_lock<std::mutex> _l(gInitMutex);
    if (nullptr != gInstance) {
        if (gInstance->number() < number) {
            delete gInstance;
        }
    }
    if (nullptr == gInstance) {
        gInstance = new ThreadPool(number);
    }
}
void ThreadPool::destroy() {
    std::unique_lock<std::mutex> _l(gInitMutex);
    if (nullptr != gInstance) {
        delete gInstance;
        gInstance = nullptr;
    }
}
#ifdef __ANDROID__
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
    //FUNC_PRINT(cpusFrequency[0].first);
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
    if (0 == mNumberThread) {
        mNumberThread = std::thread::hardware_concurrency();
    }
    mTasks.first.second = 0;
    mTasks.second       = 0;
#ifdef __ANDROID__
    std::vector<int> sortedCPUIDs = sortCPUIDByMaxFrequency(numberThread);
#endif
    for (int i = 1; i < mNumberThread; ++i) {
#ifdef __ANDROID__
        mWorkers.emplace_back([this, sortedCPUIDs]() {
#else
        mWorkers.emplace_back([this]() {
#endif
#ifdef __ANDROID__
            int res = setSchedAffinity(sortedCPUIDs);
#endif
            while (!mStop) {
                int index = 0;
                {
                    std::unique_lock<std::mutex> lock(mQueueMutex);
                    mCondition.wait(lock, [this] { return mStop || mTasks.second < mTasks.first.second; });
                    index = mTasks.second;
                    this->mTasks.second++;
                }
                if (mStop) {
                    return;
                }
                mTasks.first.first(index);
                {
                    std::unique_lock<std::mutex> lock(mTaskCompleteMutex);
                    mTaskCompleteCount += 1;
                }
                mTaskCompleteCondition.notify_one();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(mQueueMutex);
        mStop = true;
    }
    mCondition.notify_all();
    for (auto& worker : mWorkers) {
        worker.join();
    }
}

void ThreadPool::enqueue(TASK&& task) const {
    if (1 == task.second || 1 == mNumberThread) {
        for (int i=0; i<task.second; ++i) {
            task.first(i);
        }
        return;
    }
    std::unique_lock<std::mutex> __l(gInitMutex);
    {
        std::unique_lock<std::mutex> lock(mQueueMutex);
        mTasks.first       = std::move(task);
        mTasks.second      = 1;
        mTaskCompleteCount = 1;
        mCondition.notify_all();
    }
    mTasks.first.first(0);
    {
        std::unique_lock<std::mutex> lock(mTaskCompleteMutex);
        mTaskCompleteCondition.wait(lock, [this] { return mTaskCompleteCount >= mTasks.first.second; });
    }
}
} // namespace MNN
#endif
