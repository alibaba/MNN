//
//  ThreadPoolTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/06/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_USE_THREAD_POOL
#include "MNNDefine.h"
#include "MNNTestSuite.h"
#include "ThreadPool.hpp"

using namespace MNN;

class ThreadPoolTest : public MNNTestCase {
public:
    virtual ~ThreadPoolTest() = default;
    virtual bool run() {
        // initializer
        MNN::ThreadPool::init(4);
        MNN::ThreadPool* pool = MNN::ThreadPool::get();
        auto func = [](int index) { FUNC_PRINT(index); };
        pool->enqueue(std::make_pair(func, 10));
        MNN::ThreadPool::destroy();
        return true;
    }
};

MNNTestSuiteRegister(ThreadPoolTest, "core/threadpool");
#endif
