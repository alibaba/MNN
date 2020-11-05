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
    static void enqueue(TASK&& task, int index);

    static void active();
    static void deactive();

    static int acquireWorkIndex();
    static void releaseWorkIndex(int index);

    static int init(int number);
    static void destroy();

private:
    void enqueueInternal(TASK&& task, int index);

    static ThreadPool* gInstance;
    ThreadPool(int number = 0);
    ~ThreadPool();
    struct Worker {
        std::thread* workThread;
        std::condition_variable* condition;
        std::mutex* condMutex;
    };

    std::vector<Worker> mWorkers;
    std::atomic<bool> mStop = {false};
    std::mutex mQueueMutex;
    std::vector<bool> mTaskAvailable;
    std::vector<std::pair<TASK, std::vector<std::atomic_bool*>>> mTasks;
    int mNumberThread            = 0;
    std::atomic_int mActiveCount = {0};
};
} // namespace MNN
#endif
#endif
