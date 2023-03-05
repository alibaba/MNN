//
//  TopKV2GradTest.cpp
//  MNNTests
//
//  Created by MNN on 2022/08/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "../tools/train/source/grad/OpGrad.hpp"

using namespace MNN;
using namespace MNN::Express;

class TopKV2GradTest : public MNNTestCase {
public:
    char name[20] = "TopKV2";
    virtual ~TopKV2GradTest() = default;

    virtual bool run(int precision) {
        std::vector<int> shape = {2, 3, 2, 3};
        const int len = shape[0] * shape[1] * shape[2] * shape[3];
        auto input = _Input(shape, NCHW);
        const float inpudata[] = {  0.5500, 0.6721, 0.4343, 0.8518, 0.9456, 0.6444, 0.5927, 0.4439, 0.9329,
                                    0.1434, 0.6933, 0.0180, 0.3173, 0.2903, 0.4159, 0.8706, 0.1812, 0.5890,
                                    0.3834, 0.0335, 0.9997, 0.7504, 0.5379, 0.9836, 0.3202, 0.4824, 0.9982,
                                    0.8029, 0.2889, 0.8386, 0.2282, 0.6912, 0.2678, 0.9031, 0.7055, 0.9389};
        auto inputPtr = input->writeMap<float>();
        memcpy(inputPtr, inpudata, len * sizeof(float));

        int kInt = 2;
        auto k = _Scalar<int>(kInt);
        auto output = _TopKV2(input, k);

        auto values = output[0];
        auto indices = output[1];

        auto vptr = values->readMap<float>();
        auto iptr = indices->readMap<int>();

        auto opExpr = values->expr().first;
        auto grad = OpGrad::get(opExpr->get()->type());
        const int len2 = shape[0] * shape[1] * shape[2] * kInt;
        const float outputDiff[] = {  0.6534, 0.3231, 0.9053, 0.3514, 0.0295, 0.6043, 0.4028, 0.0500, 0.0187,
                                    0.5509, 0.0573, 0.6394, 0.8483, 0.2786, 0.5789, 0.4515, 0.7059, 0.3444,
                                    0.2242, 0.1954, 0.2002, 0.2493, 0.1952, 0.1997};
        auto inputGrad = grad->onGrad(opExpr, {_Const(outputDiff, {shape[0], shape[1], shape[2], kInt})});

        const std::vector<float> expectedOutput = { 0.3231, 0.6534, 0.0000, 0.3514, 0.9053, 0.0000, 0.6043, 0.0000, 0.0295,
                                                    0.0500, 0.4028, 0.0000, 0.5509, 0.0000, 0.0187, 0.0573, 0.0000, 0.6394,
                                                    0.2786, 0.0000, 0.8483, 0.4515, 0.0000, 0.5789, 0.0000, 0.3444, 0.7059,
                                                    0.1954, 0.0000, 0.2242, 0.0000, 0.2002, 0.2493, 0.1997, 0.0000, 0.1952};
        auto gotOutput = inputGrad[0]->readMap<float>();

        for (int i = 0; i < len; ++i) {
            auto diff = ::fabsf(gotOutput[i] - expectedOutput[i]);
            if (diff > 0.0001) {
                MNN_ERROR("%d: %s grad test failed, expected: %f, but got: %f!\n", i, name, expectedOutput[i], gotOutput[i]);
                return false;
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(TopKV2GradTest, "grad/topkv2");
