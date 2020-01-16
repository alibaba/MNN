//
//  ReverseSequenceTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/08/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"

using namespace MNN::Express;

class ReverseSequenceTest : public MNNTestCase {
public:
    virtual bool run() {
        auto y               = _Input({4}, NHWC, halide_type_of<int32_t>());
        std::vector<int> seq = {7, 2, 3, 5};
        auto yPtr            = y->writeMap<int32_t>();
        ::memcpy(yPtr, seq.data(), seq.size() * sizeof(int32_t));
        auto x    = _Input({10, 4, 8}, NHWC, halide_type_of<int32_t>());
        auto xPtr = x->writeMap<int32_t>();
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 8; ++k) {
                    xPtr[32 * i + 8 * j + k] = 100 * i + 10 * j + k;
                }
            }
        }
        auto ry    = _ReverseSequence(x, y, 1, 0);
        auto ryPtr = ry->readMap<int32_t>();
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 4; ++j) {
                auto req = seq[j];
                for (int k = 0; k < 8; ++k) {
                    auto compute = ryPtr[32 * i + 8 * j + k];
                    auto need    = 100 * i + 10 * j + k;
                    if (i < req) {
                        need = 100 * (req - i - 1) + 10 * j + k;
                    }
                    if (need != compute) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ReverseSequenceTest, "expr/ReverseSequence");
