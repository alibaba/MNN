//
//  TensorConvertGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TensorConvertGrad.hpp"
#include "Macro.h"
using namespace std;
using namespace MNN;

class TensorConvertGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr, const std::vector<Express::VARP>& output, const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result{nullptr};
        std::unique_ptr<OpT> forwardOp(expr->get()->UnPack());

        unique_ptr<OpT> newOp(new OpT);
//        newOp->name          = forwardOp->name + "_Grad";
//        newOp->inputIndexes  = {outputDiff};
//        newOp->outputIndexes = {gradTensors[0]};
        newOp->type          = OpType_ConvertTensor;
        newOp->main.type     = OpParameter_TensorConvertInfo;
        auto cInfo           = new TensorConvertInfoT;
        cInfo->dest          = forwardOp->main.AsTensorConvertInfo()->source;
        cInfo->source        = forwardOp->main.AsTensorConvertInfo()->dest;
        newOp->main.value    = cInfo;
        
        result[0] = Express::Variable::create(Express::Expr::create(std::move(newOp), {backwardOutput[0]}));
        return result;
    }
};
static const auto gRegister = []() {
    static TensorConvertGrad _c;
    OpGrad::insert(OpType_ConvertTensor, &_c);
    return true;
}();
