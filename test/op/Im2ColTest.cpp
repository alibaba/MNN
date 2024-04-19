//
//  Im2Col.cpp
//  MNNTests
//
//  Created by MNN on 2021/04/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class Im2ColTest : public MNNTestCase {
public:
    virtual ~Im2ColTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({2, 3, 4, 4}, NCHW, halide_type_of<float>());
        auto inputPtr         = input->writeMap<float>();
        for (int i = 0; i < 96; i++) {
            inputPtr[i] = 0.1*i;
        }
        auto output = _Im2Col(input, {2,2}, {1,1}, {0,0}, {2,2});
        std::vector<float> expectedOutput {
            0.000000, 0.200000, 0.800000, 1.000000,
            4.800000, 5.000000, 5.600000, 5.800000,
            0.100000, 0.300000, 0.900000, 1.100000,
            4.900000, 5.100000, 5.700000, 5.900000,
            0.400000, 0.600000, 1.200000, 1.400000,
            5.200000, 5.400000, 6.000000, 6.200000,
            0.500000, 0.700000, 1.300000, 1.500000,
            5.300000, 5.500000, 6.100000, 6.300000,
            1.600000, 1.800000, 2.400000, 2.600000,
            6.400000, 6.600000, 7.200000, 7.400000,
            1.700000, 1.900000, 2.500000, 2.700000,
            6.500000, 6.700000, 7.300000, 7.500000,
            2.000000, 2.200000, 2.800000, 3.000000,
            6.800000, 7.000000, 7.600000, 7.800000,
            2.100000, 2.300000, 2.900000, 3.100000,
            6.900000, 7.100000, 7.700000, 7.900000,
            3.200000, 3.400000, 4.000000, 4.200000,
            8.000000, 8.200000, 8.800000, 9.000000,
            3.300000, 3.500000, 4.100000, 4.300000,
            8.100000, 8.300000, 8.900000, 9.100000,
            3.600000, 3.800000, 4.400000, 4.600000,
            8.400000, 8.600000, 9.200000, 9.400000,
            3.700000, 3.900000, 4.500000, 4.700000,
            8.500000, 8.700000, 9.300000, 9.500000
        };
        auto gotOutput = output->readMap<float>();
        float thredhold = 0.0001f;
        if (precision >= 2) {
            thredhold = 0.05f;
        }
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 96, thredhold)) {
            MNN_ERROR("Im2ColTest test failed!\n");
            return false;
        }
        return true;
    }
};

class Col2ImTest : public MNNTestCase {
public:
    virtual ~Col2ImTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({2, 12, 4}, NCHW, halide_type_of<float>());
        auto inputPtr         = input->writeMap<float>();
        for (int i = 0; i < 2*48; i++) {
            inputPtr[i] = 0.1*i;
        }
        std::vector<int> shape {4, 4};
        auto outputShape                      = _Const(shape.data(), {2}, NCHW, halide_type_of<int>());
        auto output                           = _Col2Im(input, outputShape, {2,2}, {1,1}, {0,0}, {2,2});
        std::vector<float> expectedOutput {
            0.000000, 0.400000, 0.100000, 0.500000,
            0.800000, 1.200000, 0.900000, 1.300000,
            0.200000, 0.600000, 0.300000, 0.700000,
            1.000000, 1.400000, 1.100000, 1.500000,
            1.600000, 2.000000, 1.700000, 2.100000,
            2.400000, 2.800000, 2.500000, 2.900000,
            1.800000, 2.200000, 1.900000, 2.300000,
            2.600000, 3.000000, 2.700000, 3.100000,
            3.200000, 3.600000, 3.300000, 3.700000,
            4.000000, 4.400000, 4.100000, 4.500000,
            3.400000, 3.800000, 3.500000, 3.900000,
            4.200000, 4.600000, 4.300000, 4.700000,

            4.800000, 5.200000, 4.900000, 5.300000,
            5.600000, 6.000000, 5.700000, 6.100000,
            5.000000, 5.400000, 5.100000, 5.500000,
            5.800000, 6.200000, 5.900000, 6.300000,
            6.400000, 6.800000, 6.500000, 6.900000,
            7.200000, 7.600000, 7.300000, 7.700000,
            6.600000, 7.000000, 6.700000, 7.100000,
            7.400000, 7.800000, 7.500000, 7.900000,
            8.000000, 8.400000, 8.100000, 8.500000,
            8.800000, 9.200000, 8.900000, 9.300000,
            8.200000, 8.600000, 8.300000, 8.700000,
            9.000000, 9.400000, 9.100000, 9.500000
        };
        auto gotOutput = output->readMap<float>();
        float thredhold = 0.0001f;
        if (precision >= 2) {
            thredhold = 0.05f;
        }
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 96, thredhold)) {
            MNN_ERROR("Col2ImTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(Im2ColTest, "op/im2col");
MNNTestSuiteRegister(Col2ImTest, "op/col2im");
