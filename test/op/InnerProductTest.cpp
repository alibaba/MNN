//
//  InnerProductTest.cpp
//  MNNTests
//
//  Created by MNN on 2020/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <string>
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"
using namespace MNN::Express;
class InnerProductTest : public MNNTestCase {
public:
    virtual ~InnerProductTest() = default;
    virtual bool run(int precision) {
        int batch         = 1;
        int outputChannel = 2;
        auto input        = _Input({batch, 4 * 2 * 3}, NCHW, halide_type_of<float>());

        std::vector<float> inputData = {// inputChannel 0
                                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                        // inputChannel 1
                                        1.1, 2.1, 3.1, 4.1, 5.1, 6.1,
                                        // inputChannel 2
                                        1.2, 2.2, 3.2, 4.2, 5.2, 6.2,
                                        // inputChannel 3
                                        1.3, 2.3, 3.3, 4.3, 5.3, 6.3};

        std::vector<float> weightData = {/* outputChannel 0*/
                                         // inputChannel 0
                                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                         // inputChannel 1
                                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                         // inputChannel 2
                                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                         // inputChannel 3
                                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0,

                                         /* outputChannel 1*/
                                         // inputChannel 0
                                         4.3, 1.3, 6.3, 0.0, -2.3, 3.3,
                                         // inputChannel 1
                                         -4.0, 5.0, 6.0, 1.0, 2.0, 3.0,
                                         // inputChannel 2
                                         4.1, 5.1, 6.1, 1.1, 2.1, 3.1,
                                         // inputChannel 3
                                         4.2, 5.2, -6.2, 1.2, 2.2, 0.2};
        std::vector<float> biasData = {1.0, 0.0};
        std::vector<float> outputData = {88.6, 176.86};

        INTS outputShape = {batch, outputChannel};
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        auto output = _InnerProduct(std::move(weightData), std::move(biasData), input, outputShape);
        auto outPtr = output->readMap<float>();

        if (!checkVectorByRelativeError<float>(outPtr, outputData.data(), outputData.size(), 0.005)) {
            MNN_ERROR("InnerProductTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(InnerProductTest, "op/InnerProduct");
