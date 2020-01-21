//
//  InterpGrad.cpp
//  MNN
//
//  Created by MNN on 2019/12/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class InterpGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto op = expr->get();
        // FIXME, the grad may be compute a little error
        auto shapeInfo = expr->inputs()[0]->getInfo();
        MNN_ASSERT(nullptr != shapeInfo && shapeInfo->dim.size() == 4);
        std::vector<VARP> res{nullptr};
        std::vector<int> shapeSize(shapeInfo->dim[2], shapeInfo->dim[3]);
        VARP interpShape = _Const(shapeSize.data(), {2}, NHWC);
        std::unique_ptr<OpT> interpOp(new OpT);
        interpOp->type       = OpType_Interp;
        interpOp->main.type  = OpParameter_Interp;
        interpOp->main.value = new InterpT;
        if (OpType_Resize == op->type()) {
            interpOp->main.AsInterp()->alignCorners = false;
            interpOp->main.AsInterp()->resizeType   = 2; // Bilinear
        } else {
            MNN_ASSERT(OpType_Interp == op->type());
            auto originInterpParam                  = op->main_as_Interp();
            interpOp->main.AsInterp()->resizeType   = originInterpParam->resizeType();
            interpOp->main.AsInterp()->alignCorners = originInterpParam->alignCorners();
        }
        res[0] = Variable::create(Expr::create(interpOp.get(), {backwardOutput[0], interpShape}));
        return res;
    }
};

static const auto gRegister = []() {
    static InterpGrad _c;
    OpGrad::insert((int)OpType_Interp, &_c);
    OpGrad::insert((int)OpType_Resize, &_c);
    return true;
}();
