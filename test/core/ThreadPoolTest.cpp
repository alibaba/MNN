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
                int number = MNN::ThreadPool::init(10 - i);
                // initializer
                auto workIndex = ThreadPool::acquireWorkIndex();
                FUNC_PRINT(workIndex);
                ThreadPool::active(number);
                auto func = [](int index) {
                    FUNC_PRINT(index);
                    std::this_thread::yield();
                };
                ThreadPool::enqueue(std::make_pair(std::move(func), 10), workIndex, number);
                ThreadPool::deactive(number);
                ThreadPool::releaseWorkIndex(workIndex);
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
