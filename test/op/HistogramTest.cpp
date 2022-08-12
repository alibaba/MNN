//
//  HistogramTest.cpp
//  MNNTests
//
//  Created by MNN on 2022/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include <MNN/AutoTime.hpp>
using namespace MNN::Express;
class HistogramTest : public MNNTestCase {
public:
    virtual ~HistogramTest() = default;
    virtual bool run(int precision) {
        // float
        {
            {
                const float inpudata[] = {1, 2, 1, 4};
                auto input = _Const(inpudata, {4}, NHWC, halide_type_of<float>());
                const float expected[] = {0, 2, 1, 0};
                auto output = _Histogram(input, 4, 0, 3);
                if (!checkVector<float>(output->readMap<float>(), expected, 4, 1e-4)) {
                    MNN_ERROR("HistogramTest <float> is wrong!\n");
                    return false;
                }
            }
        }
        // int
        {
            {
                const int inpudata[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
                auto input = _Const(inpudata, {4, 2}, NHWC, halide_type_of<int>());
                const float expected[] = {1, 1, 1, 1, 1, 1, 1, 1, 0};
                auto output = _Histogram(input, 9, 0, 9);
                if (!checkVector<float>(output->readMap<float>(), expected, 9, 1e-4)) {
                    MNN_ERROR("HistogramTest <int> is wrong!\n");
                    return false;
                }
            }
        }
        // uint8
        {
            {
                const unsigned char inpudata[] = {1, 2, 2, 3, 3, 3, 4, 4, 5};
                auto input = _Const(inpudata, {3, 3}, NHWC, halide_type_of<uint8_t>());
                const float expected[] = {2, 0, 0, 3, 0, 0, 2};
                auto output = _Histogram(input, 7, 2, 4);
                if (!checkVector<float>(output->readMap<float>(), expected, 7, 1e-4)) {
                    MNN_ERROR("HistogramTest <uint8_t> is wrong!\n");
                    return false;
                }
            }
        }
        //  uint8 with channel
        {
            {
                const unsigned char inpudata[] = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
                auto input = _Const(inpudata, {2, 2, 3}, NHWC, halide_type_of<uint8_t>());
                const float expected[] = {0, 4, 0, 0};
                auto output = _Histogram(input, 4, 0, 3, 0);
                if (!checkVector<float>(output->readMap<float>(), expected, 4, 1e-4)) {
                    MNN_ERROR("HistogramTest <channel=0> is wrong!\n");
                    return false;
                }
                const float expected1[] = {0, 0, 4, 0};
                auto output1 = _Histogram(input, 4, 0, 3, 1);
                if (!checkVector<float>(output1->readMap<float>(), expected1, 4, 1e-4)) {
                    MNN_ERROR("HistogramTest <channel=1> is wrong!\n");
                    return false;
                }
                const float expected2[] = {0, 0, 0, 4};
                auto output2 = _Histogram(input, 4, 0, 3, 2);
                if (!checkVector<float>(output2->readMap<float>(), expected2, 4, 1e-4)) {
                    MNN_ERROR("HistogramTest <channel=2> is wrong!\n");
                    return false;
                }
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(HistogramTest, "op/histogram");
