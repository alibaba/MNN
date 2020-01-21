//
//  ReduceGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class ReduceGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result;
        auto inputs = expr->inputs();
        result.resize(inputs.size());
        std::unique_ptr<OpT> forwardOp(expr->get()->UnPack());
        std::vector<int> reductionDims = forwardOp->main.AsReductionParam()->dim;
        auto keepDim                   = forwardOp->main.AsReductionParam()->keepDims;
        if (inputs.size() > 1) {
            reductionDims.clear();
            auto ptr   = inputs[1]->readMap<int32_t>();
            auto shape = inputs[1]->getInfo();
            for (int i = 0; i < shape->size; ++i) {
                reductionDims.emplace_back(ptr[i]);
            }
            inputs[1]->unMap();
        }
        if (reductionDims.empty()) {
            auto shape = inputs[0]->getInfo();
            for (int i = 0; i < shape->dim.size(); ++i) {
                reductionDims.emplace_back(i);
            }
        }
        VARP init;
        {
            unique_ptr<OpT> newOp(new OpT);
            newOp->name = forwardOp->name + "__Zero";
            newOp->type = OpType_ZerosLike;
            init        = Variable::create(Expr::create(std::move(newOp), {inputs[0]}));
        }
        auto outputDiff = backwardOutput[0];

        // implement other reduction op's grad below
        if (forwardOp->main.AsReductionParam()->operation == ReductionType_SUM) {
            // do not need to modify grads, just copy them, so, pass
        }
        if (forwardOp->main.AsReductionParam()->operation == ReductionType_MEAN) {
            float gradCount  = outputDiff->getInfo()->size;
            float inputCount = inputs[0]->getInfo()->size;
            outputDiff       = _Multiply(outputDiff, _Const(gradCount / inputCount));
        }

        // this should be common operations, to expand grads to inputs shape
        if (!keepDim) {
            // Create Unsqueeze Op
            unique_ptr<OpT> newOp(new OpT);
            newOp->name                               = forwardOp->name + "__Unsqueeze";
            newOp->type                               = OpType_Unsqueeze;
            newOp->main.type                          = OpParameter_SqueezeParam;
            newOp->main.value                         = new SqueezeParamT;
            newOp->main.AsSqueezeParam()->squeezeDims = reductionDims;
            outputDiff                                = Variable::create(Expr::create(std::move(newOp), {outputDiff}));
        }
        result[0] = _Add(init, outputDiff);

        return result;
    }
};
class FillGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        return {backwardOutput[0].sum({})};
    }
};

static const auto gRegister = []() {
    static ReduceGrad _c;
    OpGrad::insert(OpType_Reduction, &_c);
    static FillGrad _d;
    OpGrad::insert(OpType_Fill, &_d);
    return true;
}();
