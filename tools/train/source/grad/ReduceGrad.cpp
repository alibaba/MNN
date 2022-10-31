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
        }
        if (reductionDims.empty()) {
            auto shape = inputs[0]->getInfo();
            for (int i = 0; i < shape->dim.size(); ++i) {
                reductionDims.emplace_back(i);
            }
        }

        VARP mask = _ZerosLike(inputs[0]) + _Scalar<float>(1.0f);
        auto outputDiff = backwardOutput[0];

        // implement other reduction op's grad below
        if (forwardOp->main.AsReductionParam()->operation == ReductionType_SUM) {
            // do not need to modify grads, just copy them, so, pass
        }

        if (forwardOp->main.AsReductionParam()->operation == ReductionType_MEAN) {
            auto gradCount = _Size(outputDiff);
            auto inputCount = _Size(inputs[0]);
            outputDiff = _Multiply(outputDiff, _Cast<float>(gradCount) / _Cast<float>(inputCount));
        }

        if (forwardOp->main.AsReductionParam()->operation == ReductionType_MAXIMUM) {
            auto output = Variable::create(expr);
            if (!keepDim) {
                output = _Unsqueeze(output, reductionDims);
            }
            mask = _Sign(inputs[0] - output) + _Scalar<float>(1.0f);
            mask = mask / _ReduceSum(mask);
        }

        if (forwardOp->main.AsReductionParam()->operation == ReductionType_MINIMUM) {
            auto output = Variable::create(expr);
            if (!keepDim) {
                output = _Unsqueeze(output, reductionDims);
            }
            mask = _Sign(output - inputs[0]) + _Scalar<float>(1.0f);
            mask = mask / _ReduceSum(mask);
        }

        if (forwardOp->main.AsReductionParam()->operation == ReductionType_PROD) {
            auto output = Variable::create(expr);
            if (!keepDim) {
                output = _Unsqueeze(output, reductionDims);
            }
            mask = output / inputs[0];
        }

        // this should be common operations, to expand grads to inputs shape
        if (!keepDim) {
            outputDiff = _Unsqueeze(outputDiff, reductionDims);
        }

        result[0] = mask * outputDiff;
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
