//
//  PermuteGrad.cpp
//  MNN
//
//  Created by MNN on 2019/12/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class TransposeGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto shapeInfo = expr->inputs()[1]->getInfo();
        auto shape     = expr->inputs()[1]->readMap<int>();
        std::vector<VARP> res{nullptr};
        if (nullptr == shape || nullptr == shapeInfo) {
            MNN_ERROR("Can't get shape info\n");
            return res;
        }
        MNN_ASSERT(nullptr != shape);
        auto dimSize = shapeInfo->size;
        std::vector<int> dims(dimSize);
        for (int i = 0; i < dimSize; ++i) {
            for (int j = 0; j < dimSize; ++j) {
                if (shape[j] == i) {
                    dims[i] = j;
                    break;
                }
            }
        }
        res[0] = _Transpose(backwardOutput[0], dims);
        return res;
    }
};

class PermuteGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        MNN_ASSERT(expr->inputs().size() == 1);
        auto op    = expr->get();
        auto shape = op->main_as_Permute()->dims();
        std::vector<VARP> res{nullptr};
        MNN_ASSERT(nullptr != shape);
        std::unique_ptr<OpT> permuteOp(new OpT);
        permuteOp->type       = OpType_Permute;
        permuteOp->main.type  = OpParameter_Permute;
        permuteOp->main.value = new PermuteT;
        auto dimSize          = shape->size();
        std::vector<int> dims(dimSize);
        for (int i = 0; i < dimSize; ++i) {
            for (int j = 0; j < dimSize; ++j) {
                if (shape->data()[j] == i) {
                    dims[i] = j;
                    break;
                }
            }
        }
        permuteOp->main.AsPermute()->dims = dims;
        res[0]                            = Variable::create(Expr::create(permuteOp.get(), {backwardOutput[0]}));
        return res;
    }
};

static const auto gRegister = []() {
    static PermuteGrad _c;
    OpGrad::insert((int)OpType_Permute, &_c);
    static TransposeGrad _d;
    OpGrad::insert((int)OpType_Transpose, &_d);
    return true;
}();
