//
//  ShapeTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNN_generated.h"
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class ShapeTest : public MNNTestCase {
public:
    virtual ~ShapeTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({1, 1, 1, 4}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output                           = _Shape(input);
        const std::vector<int> expectedOutput = {1, 1, 1, 4};
        auto gotOutput                        = output->readMap<int>();
        if (!checkVector<int>(gotOutput, expectedOutput.data(), 4, 0)) {
            MNN_ERROR("ShapeTest test failed!\n");
            return false;
        }
        return true;
    }
};
class ShapeSliceTest : public MNNTestCase {
public:
    virtual ~ShapeSliceTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({2, 3, 5, 7}, NCHW);
        input->setName("input_tensor");

        std::unique_ptr<MNN::OpT> op(new MNN::OpT);
        op->type                      = MNN::OpType_Shape;
        op->main.type                 = MNN::OpParameter_ShapeParam;
        op->main.value                = new MNN::ShapeParamT;
        op->main.AsShapeParam()->hasStart = true;
        op->main.AsShapeParam()->start    = 1;
        op->main.AsShapeParam()->hasEnd   = true;
        op->main.AsShapeParam()->end      = 3;
        auto output = Variable::create(Expr::create(std::move(op), {input}));

        const std::vector<int> expectedOutput = {3, 5};
        auto gotOutput                        = output->readMap<int>();
        if (!checkVector<int>(gotOutput, expectedOutput.data(), (int)expectedOutput.size(), 0)) {
            MNN_ERROR("ShapeSliceTest positive slice failed!\n");
            return false;
        }

        std::unique_ptr<MNN::OpT> negativeOp(new MNN::OpT);
        negativeOp->type                      = MNN::OpType_Shape;
        negativeOp->main.type                 = MNN::OpParameter_ShapeParam;
        negativeOp->main.value                = new MNN::ShapeParamT;
        negativeOp->main.AsShapeParam()->hasStart = true;
        negativeOp->main.AsShapeParam()->start    = -2;
        auto negativeOutput = Variable::create(Expr::create(std::move(negativeOp), {input}));

        const std::vector<int> negativeExpected = {5, 7};
        auto negativeGot                        = negativeOutput->readMap<int>();
        if (!checkVector<int>(negativeGot, negativeExpected.data(), (int)negativeExpected.size(), 0)) {
            MNN_ERROR("ShapeSliceTest negative start failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(ShapeTest, "op/shape");
MNNTestSuiteRegister(ShapeSliceTest, "op/shape/slice");
