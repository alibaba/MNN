//
//  ThreadPool.hpp
//  MNN
//
//  Created by MNN on 2019/06/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPU_INTHREADPOOL_H
#define CPU_INTHREADPOOL_H
#ifdef MNN_USE_THREAD_POOL
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>
#include <MNN/MNNDefine.h>
namespace MNN {

class MNN_PUBLIC ThreadPool {
public:
    typedef std::pair<std::function<void(int)>, int> TASK;

    int number() const {
        return mNumberThread;
    }
    static void enqueue(TASK&& task, int index, int threadNumber);

    static void active(int threadNumber);
    static void deactive(int threadNumber);

    static int acquireWorkIndex();
    static void releaseWorkIndex(int index);

    static int init(int number);
    static void destroy();

private:
    void enqueueInternal(TASK&& task, int index, int threadNumber);

    static ThreadPool* gInstance;
    ThreadPool(int number = 0);
    ~ThreadPool();

    std::vector<std::thread> mWorkers;
    std::vector<bool> mTaskAvailable;
    std::atomic<bool> mStop = {false};

    std::vector<std::pair<TASK, std::vector<std::atomic_bool*>>> mTasks;
    std::condition_variable mCondition;
    std::mutex mQueueMutex;

    int mNumberThread            = 0;
    std::vector<std::atomic_int*> mActiveCount;
};
} // namespace MNN
#endif
#endif
