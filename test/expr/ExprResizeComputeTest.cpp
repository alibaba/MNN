//
//  ExprResizeComputeTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNN_generated.h"
#include "MNNTestSuite.h"
#include "TestUtils.h"

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
        {
            auto x = _Input({1, 2, 3, 4, 5}, NCHW, halide_type_of<float>());
            auto inputPtr = x->writeMap<float>();
            for (int i = 0; i < x->getInfo()->size; ++i) {
                inputPtr[i] = (float)i;
            }
            x->unMap();

            std::unique_ptr<MNN::InterpT> interp(new MNN::InterpT);
            interp->resizeType = 1;
            interp->widthScale = 2.0f;
            interp->heightScale = 2.0f;
            interp->depthScale = 2.0f;

            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type = MNN::OpType_Interp3D;
            op->main.type = MNN::OpParameter_Interp;
            op->main.value = interp.release();

            auto y = Variable::create(Expr::create(op.get(), {x}));
            auto yShape = y->getInfo();
            if (yShape == nullptr) {
                return false;
            }
            const std::vector<int> expectedDim = {1, 2, 6, 8, 10};
            if (!checkVector<int>(yShape->dim.data(), expectedDim.data(), 5, 0)) {
                return false;
            }
            if (nullptr == y->readMap<float>()) {
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ExprResizeComputeTest, "expr/ExprResizeCompute");
