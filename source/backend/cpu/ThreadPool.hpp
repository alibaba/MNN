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
namespace MNN {

class ThreadPool {
public:

    typedef std::pair<std::function<void(int)>, int> TASK;

    int number() const {
        return mNumberThread;
    }
    void enqueue(TASK&& task) const;

    static void init(int number);
    static void destroy();
    static ThreadPool* get() {return gInstance;}
private:
    static ThreadPool* gInstance;
    ThreadPool(int number = 0);
    ~ThreadPool();

    std::vector<std::thread> mWorkers;
    bool mStop = false;

    mutable std::pair<TASK, int> mTasks;
    mutable std::condition_variable mTaskCompleteCondition;
    mutable std::mutex mTaskCompleteMutex;
    mutable int mTaskCompleteCount;

    mutable std::mutex mQueueMutex;
    mutable std::condition_variable mCondition;

    int mNumberThread = 0;
};
} // namespace MNN
#endif
#endif
