//
//  BinaryGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "BinaryGrad.hpp"
#include "Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class BinaryGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr, const std::vector<Express::VARP>& output, const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<VARP> res;
        auto inputs = expr->inputs();
        res.resize(inputs.size());
        auto op = expr->get();
        auto outputDiff = backwardOutput[0];
        switch (op->main_as_BinaryOp()->opType()) {
            case BinaryOpOperation_ADD: {
                res[0] = outputDiff;
                res[1] = outputDiff;
                break;
            }
            case BinaryOpOperation_SUB: {
                res[0] = outputDiff;
                res[1] = _Neg(outputDiff);
                break;
            }
            case BinaryOpOperation_MUL: {
                res[0] = _Mul(outputDiff, inputs[1]);
                res[1] = _Mul(outputDiff, inputs[0]);
                break;
            }
            case BinaryOpOperation_REALDIV: {
                res[0] = _Div(outputDiff, inputs[1]);
                // d (u / v) = dx / v , -dx*u(1/v)*(1/v)
                res[1] = _Neg(_Mul(outputDiff, _Div(output[0], inputs[1])));
                break;
            }
            default:
                break;
        }
        for (int i=0; i<inputs.size(); ++i) {
            auto inputShape = inputs[i]->getInfo();
            auto backShape = res[i]->getInfo();
            std::vector<int> reduceDims;
            bool keepDim = true;
            MNN_ASSERT(inputShape->dim.size() <= backShape->dim.size());
            if (inputShape->dim.size() < backShape->dim.size()) {
                auto diff = (int)backShape->dim.size() - (int)inputShape->dim.size();
                for (int i=0; i<diff; ++i) {
                    reduceDims.emplace_back(i);
                }
                keepDim = false;
            } else {
                for (int i=0; i<backShape->dim.size(); ++i) {
                    if (backShape->dim[i] > 1 && inputShape->dim[i] == 1) {
                        reduceDims.emplace_back(i);
                    }
                }
                keepDim = true;
            }
            if (!reduceDims.empty()) {
                res[i] = _Sum(res[i], reduceDims, keepDim);
            }
        }
        return res;
    }
};

static const auto gRegister = []() {
    static BinaryGrad _c;
    OpGrad::insert((int)OpType_BinaryOp, &_c);
    return true;
}();
