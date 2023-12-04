//
//  BinaryGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "BinaryGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;
class EltwiseGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<VARP> res;
        auto inputs = expr->inputs();
        res.resize(inputs.size());
        auto op         = expr->get();
        auto outputDiff = backwardOutput[0];
        switch (op->main_as_Eltwise()->type()) {
            case MNN::EltwiseType_SUM: {
                for (int i = 0; i < res.size(); ++i) {
                    res[i] = outputDiff;
                }
                break;
            }
            case MNN::EltwiseType_SUB: {
                res[0]       = outputDiff;
                auto negDiff = _Negative(outputDiff);
                for (int i = 1; i < res.size(); ++i) {
                    res[i] = negDiff;
                }
                break;
            }
            case MNN::EltwiseType_PROD: {
                for (int i = 0; i < res.size(); ++i) {
                    std::vector<VARP> prods{outputDiff};
                    for (int j = 0; j < inputs.size(); ++j) {
                        if (j == i) {
                            continue;
                        }
                        prods.emplace_back(inputs[j]);
                    }
                    std::unique_ptr<OpT> eltOp(new OpT);
                    eltOp->type                   = OpType_Eltwise;
                    eltOp->main.type              = OpParameter_Eltwise;
                    eltOp->main.value             = new EltwiseT;
                    eltOp->main.AsEltwise()->type = EltwiseType_PROD;
                    res[i]                        = Variable::create(Expr::create(eltOp.get(), prods));
                }
                break;
            }
            case MNN::EltwiseType_MAXIMUM: {
                for (int i = 0; i < inputs.size(); ++i) {
                    auto mask = _Sign(inputs[i] - Variable::create(expr, 0)) + _Const(1.0f, {}, NCHW);
                    res[i]    = mask * outputDiff;
                }
                break;
            }
            default:
                return res;
        }
        return res;
    }
};
class BinaryGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<VARP> res;
        auto inputs = expr->inputs();
        res.resize(inputs.size());
        auto op         = expr->get();
        auto outputDiff = backwardOutput[0];
        std::vector<VARP> output(expr->outputSize());
        for (int i = 0; i < expr->outputSize(); ++i) {
            output[i] = Variable::create(expr, i);
        }
        int activateType = op->main_as_BinaryOp()->activationType();
        if (activateType == 1) { // relu
            auto mask = _Cast<float>(_Greater(output[0], _Scalar(0.0f)));
            outputDiff = mask * backwardOutput[0];
        }
        switch (op->main_as_BinaryOp()->opType()) {
            case BinaryOpOperation_ADD: {
                res[0] = outputDiff;
                res[1] = outputDiff;
                break;
            }
            case BinaryOpOperation_SUB: {
                res[0] = outputDiff;
                res[1] = _Negative(outputDiff);
                break;
            }
            case BinaryOpOperation_MUL: {
                res[0] = outputDiff * inputs[1];
                res[1] = outputDiff * inputs[0];
                break;
            }
            case BinaryOpOperation_MAXIMUM: {
                auto mask0 = _Sign(inputs[0] - output[0]) + _Const(1.0f, {}, NCHW);
                auto mask1 = _Sign(inputs[1] - output[0]) + _Const(1.0f, {}, NCHW);
                auto maskSum = mask0 + mask1;
                res[0]     = outputDiff * mask0 / maskSum;
                res[1]     = outputDiff * mask1 / maskSum;
                break;
            }
            case BinaryOpOperation_MINIMUM: {
                auto mask0 = _Sign(output[0] - inputs[0]) + _Const(1.0f, {}, NCHW);
                auto mask1 = _Sign(output[0] - inputs[1]) + _Const(1.0f, {}, NCHW);
                auto maskSum = mask0 + mask1;
                res[0]     = outputDiff * mask0 / maskSum;
                res[1]     = outputDiff * mask1 / maskSum;
                break;
            }
            case BinaryOpOperation_REALDIV: {
                res[0] = _Divide(outputDiff, inputs[1]);
                // d (u / v) = dx / v , -dx*u(1/v)*(1/v)
                res[1] = _Negative(_Multiply(outputDiff, _Divide(output[0], inputs[1])));
                break;
            }
            case BinaryOpOperation_POW: {
                // d (pow(x, y)) = dv * pow(x, y) / x * y , dv * pow(x, y) * ln(x)
                res[0] = outputDiff * output[0] * OpGrad::divideAvoidZero(inputs[1], inputs[0]);
                res[1] = outputDiff * output[0] * _Log(inputs[0]);
                break;
            }
            case BinaryOpOperation_ATAN2: {
                // d atan(x/y) = (y/(x^2 + y^2), -x/(x^2 + y^2)) * outputDiff
                auto x2y2 = _Square(inputs[0]) + _Square(inputs[1]);
                res[0] = inputs[1] / x2y2 * outputDiff;
                res[1] = _Negative(inputs[0]) / x2y2 * outputDiff;
                break;
            }
            case BinaryOpOperation_SquaredDifference: {
                // d (x - y)^2 = (2 * (x - y), -2 * (x - y)) * outputDiff
                auto two = _Scalar(2.0f);
                auto xmy = inputs[0] - inputs[1];
                res[0] = two * xmy * outputDiff;
                res[1] = _Negative(res[0]);
                break;
            }
            default:
                MNN_ERROR("Can't grad for binary: %d\n", op->main_as_BinaryOp()->opType());
                return res;
        }
        for (int i = 0; i < inputs.size(); ++i) {
            auto inputShape = inputs[i]->getInfo();
            auto backShape  = res[i]->getInfo();
            std::vector<int> reduceDims;
            bool keepDim = true;
            MNN_ASSERT(inputShape->dim.size() <= backShape->dim.size());
            if (inputShape->dim.size() < backShape->dim.size()) {
                // case like: shape(7, 2, 3, 3) + shape(2, 3, 1)
                // will only be handled a part here
                // because we need keepDim = false for dim[0] = 7
                // and keepDim = true for dim[-1] = 3
                auto diff = (int)backShape->dim.size() - (int)inputShape->dim.size();
                for (int i = 0; i < diff; ++i) {
                    reduceDims.emplace_back(i);
                }
                keepDim = false;
            } else {
                for (int i = 0; i < backShape->dim.size(); ++i) {
                    if (backShape->dim[i] > 1 && inputShape->dim[i] == 1) {
                        reduceDims.emplace_back(i);
                    }
                }
                keepDim = true;
            }
            if (!reduceDims.empty()) {
                res[i] = _ReduceSum(res[i], reduceDims, keepDim);
                // for case like: shape(7, 2, 3, 3) + shape(2, 3, 1)
                if (keepDim == false) {
                    reduceDims.clear();
                    auto diff = (int)backShape->dim.size() - (int)inputShape->dim.size();
                    for (int j = 0; j < inputShape->dim.size(); j++) {
                        if (backShape->dim[j + diff] > 1 && inputShape->dim[j] == 1) {
                            reduceDims.emplace_back(j);
                        }
                    }
                    keepDim = true;
                    if (!reduceDims.empty()) {
                        res[i] = _ReduceSum(res[i], reduceDims, keepDim);
                    }
                }
            }
        }
        return res;
    }
};

static const auto gRegister = []() {
    static BinaryGrad _c;
    OpGrad::insert((int)OpType_BinaryOp, &_c);
    static EltwiseGrad _d;
    OpGrad::insert((int)OpType_Eltwise, &_d);
    return true;
}();
