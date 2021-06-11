//
//  ExtraTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"

using namespace MNN::Express;

class ExtraTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        auto x = _Input({4}, NHWC, halide_type_of<int32_t>());
        std::shared_ptr<MNN::OpT> extraOp(new MNN::OpT);
        extraOp->type = MNN::OpType_Extra;
        auto y        = Variable::create(Expr::create(extraOp.get(), {x}));
        if (nullptr != y->getInfo() || nullptr != y->readMap<int>()) {
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(ExtraTest, "expr/Extra");
