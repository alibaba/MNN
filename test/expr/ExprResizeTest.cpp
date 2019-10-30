//
//  ExprResizeTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ExprCreator.hpp"
#include "MNNTestSuite.h"

using namespace MNN::Express;

class ExprResizeTest : public MNNTestCase {
public:
    virtual bool run() {
        auto x               = _Input({4}, NHWC, halide_type_of<int32_t>());
        auto newX = _Input({4}, NHWC, halide_type_of<int32_t>());
        Variable::replace(x, newX);
        std::vector<int> x0 = {0, 1, 2, 3, 4, 5, 6};
        ::memcpy(x->writeMap<int>(), x0.data(), x->getInfo()->size*sizeof(int32_t));
        auto y = _Sum(_Mul(x, x), {});
        if (14 != y->readMap<int>()[0]) {
            return false;
        }
        
        x->resize({5});
        ::memcpy(x->writeMap<int>(), x0.data(), x->getInfo()->size*sizeof(int32_t));
        if (30 != y->readMap<int>()[0]) {
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(ExprResizeTest, "expr/ExprResize");
