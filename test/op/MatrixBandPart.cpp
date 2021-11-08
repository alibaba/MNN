//
//  MatrixBandPart.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
using namespace MNN::Express;
class MatrixBandPartTest : public MNNTestCase {
public:
    virtual ~MatrixBandPartTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({4, 4}, NHWC);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {0.0, 1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0, -3.0, -2.0, -1.0, 0.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 16 * sizeof(float));
        input->unMap();
        int lower_data                          = 1;
        int higher_data                         = -1;
        auto lower                              = _Const(&lower_data, {}, NCHW, halide_type_of<int>());
        auto higher                             = _Const(&higher_data, {}, NCHW, halide_type_of<int>());
        auto output                             = _MatrixBandPart(input, lower, higher);
        const std::vector<float> expectedOutput = {0.0, 1.0,  2.0, 3.0, -1.0, 0.0, 1.0,  2.0,
                                                   0.0, -1.0, 0.0, 1.0, 0.0,  0.0, -1.0, 0.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 16, 0.01)) {
            MNN_ERROR("MatrixBandPartTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(MatrixBandPartTest, "op/matrixbandpart");
