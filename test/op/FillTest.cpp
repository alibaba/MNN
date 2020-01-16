//
//  FillTest.cpp
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
class FillTest : public MNNTestCase {
public:
    virtual ~FillTest() = default;
    virtual bool run() {
        auto input = _Input({4}, NCHW, halide_type_of<int>());
        input->setName("input_tensor");
        //set input data
        const int inputdata[] = {1, 1, 1, 4};
        auto inputPtr          = input->writeMap<int>();
        memcpy(inputPtr, inputdata, 4 * sizeof(int));
        input->unMap();
        const int fill_data = 1;
        auto fill = _Const(&fill_data,{},NCHW, halide_type_of<int>());
        auto output = _Fill(input, fill);
        const std::vector<int> expectedOutput = {1, 1, 1, 1};
        auto gotOutput = output->readMap<int>();
        if (!checkVector<int>(gotOutput, expectedOutput.data(), 4, 0)) {
            MNN_ERROR("FillTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(FillTest, "op/fill");
