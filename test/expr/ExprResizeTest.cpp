//
//  ExprResizeTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"

using namespace MNN::Express;

class ExprResizeTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        auto x    = _Input({4}, NHWC, halide_type_of<int32_t>());
        auto newX = _Input({4}, NHWC, halide_type_of<int32_t>());
        Variable::replace(x, newX);
        std::vector<int> x0 = {0, 1, 2, 3, 4, 5, 6};
        auto y              = _ReduceSum(_Multiply(x, x), {});
        ::memcpy(x->writeMap<int>(), x0.data(), x->getInfo()->size * sizeof(int32_t));
        if (14 != y->readMap<int>()[0]) {
            return false;
        }

        x->resize({5});
        ::memcpy(x->writeMap<int>(), x0.data(), x->getInfo()->size * sizeof(int32_t));
        if (30 != y->readMap<int>()[0]) {
            MNN_PRINT("%d  - Error: %d\n", 30, y->readMap<int>()[0]);
            return false;
        }
        auto z = _Cast<int>(_ReduceMean(_Cast<float>(x + x)));
        z.fix(VARP::CONSTANT);
        if (4 != z->readMap<int>()[0]) {
            MNN_PRINT("%d - Error = %d\n", 4, z->readMap<int>()[0]);
            return false;
        }
        x->resize({6});
        ::memcpy(x->writeMap<int>(), x0.data(), x->getInfo()->size * sizeof(int32_t));
        std::vector<int> shape{2, 3};
        auto tempShape = _Input({2, 3}, NCHW);
        auto xR        = _Reshape(x, _Shape(tempShape));
        auto xRPtr     = xR->readMap<int>();
        if (nullptr == xRPtr) {
            FUNC_PRINT(1);
            return false;
        }
        for (int i = 0; i < 6; ++i) {
            if (xRPtr[i] != x0[i]) {
                FUNC_PRINT(1);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ExprResizeTest, "expr/ExprResize");
