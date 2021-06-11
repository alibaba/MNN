//
//  CosineSimilarityTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <string>
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"
using namespace MNN::Express;
class CosineSimilarityTest : public MNNTestCase {
public:
    virtual ~CosineSimilarityTest() = default;
    virtual bool run(int precision) {
        auto input_a              = _Input({1, 4, 2, 3}, NCHW, halide_type_of<float>());
        auto input_b              = _Input({1, 4, 2, 3}, NCHW, halide_type_of<float>());
        auto input_dim            = _Input({1}, NCHW, halide_type_of<int32_t>());
        auto output               = _CosineSimilarity(input_a, input_b, input_dim);
        std::vector<float> data_a = {// channel 0
                                     1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                     // channel 1
                                     1.1, 2.1, 3.1, 4.1, 5.1, 6.1,
                                     // channel 2
                                     1.2, 2.2, 3.2, 4.2, 5.2, 6.2,
                                     // channel 3
                                     1.3, 2.3, 3.3, 4.3, 5.3, 6.3};

        std::vector<float> data_b     = {// channel 0
                                     4.3, 1.3, 6.3, 0.0, -2.3, 3.3,
                                     // channel 1
                                     -4.0, 5.0, 6.0, 1.0, 2.0, 3.0,
                                     // channel 2
                                     4.1, 5.1, 6.1, 1.1, 2.1, 3.1,
                                     // channel 3
                                     4.2, 5.2, -6.2, 1.2, 2.2, 0.2};
        std::vector<int32_t> data_dim = {1};
        std::vector<float> data_c     = {
            0.53578, 0.94357, 0.471428, 0.874999, 0.479708, 0.876127,
        };

        ::memcpy(input_a->writeMap<float>(), data_a.data(), data_a.size() * sizeof(float));
        ::memcpy(input_b->writeMap<float>(), data_b.data(), data_b.size() * sizeof(float));
        ::memcpy(input_dim->writeMap<float>(), data_dim.data(), data_dim.size() * sizeof(int32_t));

        if (!checkVectorByRelativeError<float>(output->readMap<float>(), data_c.data(), data_c.size(), 0.005)) {
            MNN_ERROR("CosineSimilarityTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(CosineSimilarityTest, "op/CosineSimilarity");
