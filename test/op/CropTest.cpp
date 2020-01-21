//
//  CropTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
using namespace MNN::Express;
class CropTest : public MNNTestCase {
public:
    virtual ~CropTest() = default;
    virtual bool run() {
        auto input = _Input({1, 1, 4, 4}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 16 * sizeof(float));
        input->unMap();
        const float size_data[] = {0.0, 0.0, 0.0, 0.0};
        auto size = _Const(size_data, {1, 1, 2, 2}, NCHW);
        input = _Convert(input, NC4HW4);
        auto output = _Crop(input, size, 2, {1, 1});
        output = _Convert(output, NCHW);
        const std::vector<float> expectedOutput = {6.0, 7.0, 10.0, 11.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("CropTest test failed!\n");
            return false;
        }
        const std::vector<int> expectedDim = {1, 1, 2, 2};
        auto gotDim = output->getInfo()->dim;
        if (!checkVector<int>(gotDim.data(), expectedDim.data(), 4, 0)) {
            MNN_ERROR("CropTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(CropTest, "op/crop");
