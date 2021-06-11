//
//  ExecutorScopeMultiThreadTest.cpp
//  MNNTests
//
//  Created by MNN on 2020/2/2.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExecutorScope.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include <thread>

using namespace MNN::Express;

class ExecutorScopeMultiThreadTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        std::vector<std::thread> threads;
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&]() {
                ExecutorScope scope(ExecutorScope::Current());
            });
        }
        for (auto& t : threads) {
            t.join();
        }
        return true;
    }
};

MNNTestSuiteRegister(ExecutorScopeMultiThreadTest, "expr/ExecutorScopeMultiThread");
