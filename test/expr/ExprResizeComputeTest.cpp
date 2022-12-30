//
//  ExprResizeComputeTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"

using namespace MNN::Express;

class ExprResizeComputeTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        {
            auto x      = _Input({2, 16, 36, 39}, NC4HW4, halide_type_of<float>());
            auto sx     = _Shape(x, true);
            auto wh     = _StridedSlice(sx, _Unsqueeze(_Scalar<int32_t>(2), {0}), _Unsqueeze(_Scalar<int32_t>(4), {0}),
                                    _Unsqueeze(_Scalar<int32_t>(1), {0}), 0, 0, 0, 0, 0);
            wh          = wh * _Scalar<int32_t>(2);
            auto y      = _Interp({x, wh}, 0.0f, 0.0f, 0, 0, 1, true);
            auto yShape = y->getInfo();
            if (yShape->dim[2] != 72 || yShape->dim[3] != 78) {
                return false;
            }
            {
                auto ptr = y->readMap<float>();
                if (nullptr != ptr) {
                    FUNC_PRINT(1);
                    return false;
                }
            }
            {
                x->writeMap<float>();
                auto ptr = y->readMap<float>();
                if (nullptr == ptr) {
                    return false;
                }
            }
            x->resize({1, 16, 44, 45});
            yShape = y->getInfo();
            if (yShape->dim[2] != 88 || yShape->dim[3] != 90) {
                return false;
            }
            {
                x->writeMap<float>();
                auto ptr = y->readMap<float>();
                if (nullptr == ptr) {
                    return false;
                }
            }
        }
        {
            auto x      = _Input({2, 16, 36, 39}, NC4HW4, halide_type_of<float>());
            auto whi    = _Input({2}, NHWC, halide_type_of<int32_t>());
            auto whPtr  = whi->writeMap<int32_t>();
            whPtr[0]    = 66;
            whPtr[1]    = 77;
            auto wh     = whi * _Scalar<int32_t>(2);
            auto y      = _Interp({x, wh}, 0.0f, 0.0f, 0, 0, 1, true);
            auto yShape = y->getInfo();
            if (yShape->dim[2] != 66 * 2 || yShape->dim[3] != 77 * 2) {
                return false;
            }
            {
                x->writeMap<float>();
                auto ptr = y->readMap<float>();
                if (nullptr == ptr) {
                    return false;
                }
            }
            whPtr    = whi->writeMap<int32_t>();
            whPtr[0] = 11;
            whPtr[1] = 22;
            yShape   = y->getInfo();
            if (yShape->dim[2] != 11 * 2 || yShape->dim[3] != 22 * 2) {
                return false;
            }

            {
                x->writeMap<float>();
                auto ptr = y->readMap<float>();
                if (nullptr == ptr) {
                    return false;
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ExprResizeComputeTest, "expr/ExprResizeCompute");
