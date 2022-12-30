//
//  AllAnyTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include <MNN/expr/ExecutorScope.hpp>

using namespace MNN::Express;

class AllAnyTest : public MNNTestCase {
public:
    bool _run(int precision, bool lazy) {
        auto y                = _Input({4}, NHWC, halide_type_of<int32_t>());
        std::vector<int> seq0 = {1, 0, 0, 1};
        std::vector<int> seq1 = {1, 1, 1, 1};
        std::vector<int> seq2 = {0, 0, 0, 0};
        auto yPtr             = y->writeMap<int32_t>();
        ::memcpy(yPtr, seq0.data(), seq0.size() * sizeof(int32_t));
        auto zAny     = _ReduceAny(y, {0});
        auto zAll     = _ReduceAll(y, {0});
        auto zAnyinfo = zAny->getInfo();
        if (zAny->readMap<int32_t>()[0] != 1) {
            FUNC_PRINT(1);
            return false;
        }
        if (zAll->readMap<int32_t>()[0] != 0) {
            FUNC_PRINT(1);
            return false;
        }
        if (!lazy) {
            return true;
        }
        // Call WriteMap to Refresh Compute
        yPtr = y->writeMap<int32_t>();
        ::memcpy(yPtr, seq1.data(), seq1.size() * sizeof(int32_t));
        if (zAny->readMap<int32_t>()[0] != 1) {
            FUNC_PRINT(1);
            return false;
        }
        if (zAll->readMap<int32_t>()[0] != 1) {
            FUNC_PRINT(1);
            return false;
        }
        // Call WriteMap to Refresh Compute
        yPtr = y->writeMap<int32_t>();
        ::memcpy(yPtr, seq2.data(), seq2.size() * sizeof(int32_t));
        if (zAny->readMap<int32_t>()[0] != 0) {
            FUNC_PRINT(1);
            return false;
        }
        if (zAll->readMap<int32_t>()[0] != 0) {
            FUNC_PRINT(1);
            return false;
        }
        return true;
    }
    virtual bool run(int precision) {
        ExecutorScope::Current()->lazyEval = false;
        auto res = _run(precision, false);
        if (!res) {
            FUNC_PRINT(1);
            return false;
        }
        ExecutorScope::Current()->lazyEval = true;
        res = _run(precision, true);
        return res;
    }
};
MNNTestSuiteRegister(AllAnyTest, "expr/AllAny");
