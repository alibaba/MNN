//
//  ThreadPoolTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/06/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_USE_THREAD_POOL
#include <MNN/MNNDefine.h>
#include "MNNTestSuite.h"
#include "backend/cpu/ThreadPool.hpp"

using namespace MNN;

class ThreadPoolTest : public MNNTestCase {
public:
    virtual ~ThreadPoolTest() = default;
    virtual bool run(int precision) {
        std::vector<std::thread> threads;
        for (int i = 0; i < 10; ++i) {
            threads.emplace_back([i]() {
                MNN::ThreadPool* threadPool = nullptr;
                MNN::ThreadPool::init(10 - i, 0, threadPool);
                // initializer
                auto workIndex = threadPool->acquireWorkIndex();
                FUNC_PRINT(workIndex);
                threadPool->active();
                ThreadPool::TASK task = std::make_pair([](int index) {
                    FUNC_PRINT(index);
                    std::this_thread::yield();
                }, 10);
                threadPool->enqueue(&task, workIndex);
                threadPool->deactive();
                threadPool->releaseWorkIndex(workIndex);
            });
        }
        for (auto& t : threads) {
            t.join();
        }
        MNN::ThreadPool::destroy();
        return true;
    }
};

MNNTestSuiteRegister(ThreadPoolTest, "core/threadpool");
#endif
