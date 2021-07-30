//
//  UniqueTest.cpp
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
class UniqueTest : public MNNTestCase {
public:
    virtual ~UniqueTest() = default;
    virtual bool run(int precision) {
        auto input_x = _Input({9}, NHWC, halide_type_of<int>());
        input_x->setName("input_x");
        // set input data
        const int x_data[] = {1, 1, 2, 4, 4, 4, 7, 8, 8};
        auto xPtr          = input_x->writeMap<int>();
        memcpy(xPtr, x_data, 9 * sizeof(int));
        input_x->unMap();
        std::unique_ptr<MNN::OpT> op(new MNN::OpT);
        op->type       = MNN::OpType_Unique;
        op->main.type = MNN::OpParameter_NONE;
        op->main.value = nullptr;
        auto expr = Expr::create(std::move(op), {input_x}, 2);
        auto output0 = Variable::create(expr, 0);
        auto output1 = Variable::create(expr, 1);
        const std::vector<int> expectedOutput0 = {1, 2, 4, 7, 8};
        const std::vector<int> expectedOutput1 = {0, 0, 1, 2, 2, 2, 3, 4, 4};
        auto gotOutput0 = output0->readMap<int>();
        auto gotOutput1 = output1->readMap<int>();
        if (!checkVector<int>(gotOutput0, expectedOutput0.data(), 5, 0) && !checkVector<int>(gotOutput1, expectedOutput1.data(), 9, 0)) {
            MNN_ERROR("UniqueTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(UniqueTest, "op/unique");
