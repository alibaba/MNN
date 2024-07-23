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
#include <MNN/MNNDefine.h>

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

ThreadPool::ThreadPool(int numberThread) {
    mNumberThread = numberThread;
    mActiveCount.resize(numberThread);
    for (int i=0; i<numberThread; ++i) {
        mActiveCount[i] = new std::atomic_int(0);
    }
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
            while (!mStop) {
                while (*mActiveCount[threadIndex] > 0) {
                    for (int i = 0; i < MNN_THREAD_POOL_MAX_TASKS; ++i) {
                        if (*mTasks[i].second[threadIndex]) {
                            mTasks[i].first.first(threadIndex);
                            { *mTasks[i].second[threadIndex] = false; }
                        }
                    }
                    std::this_thread::yield();
                }
                std::unique_lock<std::mutex> _l(mQueueMutex);
                mCondition.wait(_l, [this, threadIndex] { return mStop || *mActiveCount[threadIndex] > 0; });
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
    for (int i=0; i<mActiveCount.size(); ++i) {
        delete mActiveCount[i];
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

void ThreadPool::active(int threadNumber) {
    if (nullptr == gInstance) {
        return;
    }
    {
        std::lock_guard<std::mutex> _l(gInstance->mQueueMutex);
        for (int i=0; i<threadNumber; ++i) {
            (*gInstance->mActiveCount[i])++;
        }
    }
    gInstance->mCondition.notify_all();
}
void ThreadPool::deactive(int threadNumber) {
    if (nullptr == gInstance) {
        return;
    }
    for (int i=0; i<threadNumber; ++i) {
        (*gInstance->mActiveCount[i])--;
    }
}

void ThreadPool::enqueue(TASK&& task, int index, int threadNumber) {
    if (1 >= task.second || 0 > index) {
        for (int i = 0; i < task.second; ++i) {
            task.first(i);
        }
        return;
    }
    MNN_ASSERT(nullptr != gInstance);
    gInstance->enqueueInternal(std::move(task), index, threadNumber);
}
void ThreadPool::enqueueInternal(TASK&& task, int index, int threadNumber) {
    if (threadNumber <= 1) {
        for (int i = 0; i < task.second; ++i) {
            task.first(i);
        }
        return;
    }
    int workSize = task.second;
    if (workSize > threadNumber) {
        mTasks[index].first = std::make_pair(
            [workSize, &task, threadNumber, this](int tId) {
                for (int v = tId; v < workSize; v += threadNumber) {
                    task.first(v);
                }
            },threadNumber);
        workSize = threadNumber;
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
        complete = true;
        for (int i = 1; i < workSize; ++i) {
            if (*mTasks[index].second[i]) {
                complete = false;
                break;
            }
        }
        std::this_thread::yield();
        // FUNC_PRINT(notComplete);
    } while (!complete);
}
} // namespace MNN
#endif
