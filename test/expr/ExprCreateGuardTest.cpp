//
//  ExprCreateGuardTest.cpp
//  MNNTests
//
//  Created by MNN on 2026/09/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"

using namespace MNN;
using namespace MNN::Express;

class ExprCreateGuardTest : public MNNTestCase {
public:
    virtual ~ExprCreateGuardTest() = default;
    virtual bool run(int precision) {
        {
            std::unique_ptr<OpT> op(new OpT);
            op->name = "bad_const";
            op->type = OpType_Const;
            auto expr = Expr::create(op.get(), {}, 1);
            MNNTEST_ASSERT(nullptr == expr);
        }
        {
            std::unique_ptr<OpT> op(new OpT);
            op->name = "bad_input";
            op->type = OpType_Input;
            auto expr = Expr::create(op.get(), {}, 1);
            MNNTEST_ASSERT(nullptr == expr);
        }
        return true;
    }
};
MNNTestSuiteRegister(ExprCreateGuardTest, "expr/create_guard");
