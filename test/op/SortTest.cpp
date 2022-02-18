//
//  SortTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/12/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
using namespace MNN::Express;
class SortTest : public MNNTestCase {
public:
    virtual ~SortTest() = default;
    virtual bool run(int precision) {
        auto input_nhwc = _Input({4, 4}, NHWC);
        input_nhwc->setName("input_tensor_nhwc");
        // set input data
        const float inpudata[] = {-1.0, 2.0,   -3.0, 4.0,
                                  5.0,  -6.0, 7.0,   -8.0,
                                  -9.0, -10.0, 11.0, 12.0,
                                  13.0, 14.0, -15.0, -16.0};
        auto inputPtr          = input_nhwc->writeMap<float>();
        memcpy(inputPtr, inpudata, 16 * sizeof(float));
        const std::vector<float> expectedOutput_0 = {-9.0, -10.0, -15.0, -16.0,
                                                     -1.0,  -6.0,  -3.0,  -8.0,
                                                      5.0,   2.0,   7.0,   4.0,
                                                     13.0,  14.0,  11.0,  12.0};
        auto output_0                           = _Sort(input_nhwc, 0);
        auto gotOutput_0                        = output_0->readMap<float>();
        if (!checkVector<float>(gotOutput_0, expectedOutput_0.data(), 16, 0)) {
            MNN_ERROR("SortTest test axis_0 failed!\n");
            return false;
        }
        const std::vector<float> expectedOutput_1 = {-3.0, -1.0,  2.0, 4.0,
                                                     -8.0, -6.0,  5.0, 7.0,
                                                    -10.0, -9.0,  11.0, 12.0,
                                                    -16.0, -15.0, 13.0, 14.0};
        auto output_1                           = _Sort(input_nhwc, 1);
        auto gotOutput_1                        = output_1->readMap<float>();
        if (!checkVector<float>(gotOutput_1, expectedOutput_1.data(), 16, 0)) {
            MNN_ERROR("SortTest test axis_1 failed!\n");
            return false;
        }
        const std::vector<int> expectedOutput_2 = { 2, 2, 3, 3,
                                                    0, 1, 0, 1,
                                                    1, 0, 1, 0,
                                                    3, 3, 2, 2 };
        auto output_2                           = _Sort(_Clone(input_nhwc, true), 0, true);
        auto gotOutput_2                        = output_2->readMap<int>();
        if (!checkVector<int>(gotOutput_2, expectedOutput_2.data(), 16, 0)) {
            MNN_ERROR("ArgSortTest test axis_0 failed!\n");
            return false;
        }
        const std::vector<int> expectedOutput_3 = { 2, 0, 1, 3,
                                                    3, 1, 0, 2,
                                                    1, 0, 2, 3,
                                                    3, 2, 0, 1 };
        auto output_3                           = _Sort(_Clone(input_nhwc, true), 1, true);
        auto gotOutput_3                        = output_3->readMap<int>();
        if (!checkVector<int>(gotOutput_3, expectedOutput_3.data(), 16, 0)) {
            MNN_ERROR("ArgSortTest test axis_1 failed!\n");
            return false;
        }
        const std::vector<int> expectedOutput_4 = { 3, 3, 2, 2,
                                                    1, 0, 1, 0,
                                                    0, 1, 0, 1,
                                                    2, 2, 3, 3 };
        auto output_4                           = _Sort(_Clone(input_nhwc, true), 0, true, true);
        auto gotOutput_4                        = output_4->readMap<int>();
        if (!checkVector<int>(gotOutput_4, expectedOutput_4.data(), 16, 0)) {
            MNN_ERROR("ArgSortTest test axis_0, descend failed!\n");
            return false;
        }
        auto input_nchw = _Input({5}, NC4HW4);
        inputPtr          = input_nchw->writeMap<float>();
        const float inpudatax[] = { 0.4, 0.2, 0.5, 0.1, 0.3 };
        memcpy(inputPtr, inpudatax, 5 * sizeof(float));
        auto output_5 = _Sort(input_nchw, 0, true);
        auto gotOutput_5 = output_5->readMap<int>();
        const std::vector<int> expectedOutput_5 = { 3, 1, 4, 0, 2 };
        if (!checkVector<int>(gotOutput_5, expectedOutput_5.data(), 5, 0)) {
            MNN_ERROR("ArgSortTest test axis_0 failed!\n");
            return false;
        }
        return true;
    }
};

MNNTestSuiteRegister(SortTest, "op/sort");
