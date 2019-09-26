//
//  UnaryGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Macro.h"
#include "OpGrad.hpp"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class UnaryGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr, const std::vector<Express::VARP>& output, const std::vector<Express::VARP>& backwardOutput) override {
        std::unique_ptr<OpT> forwardOp(expr->get()->UnPack());
        auto outputDiff = backwardOutput[0];
        auto input = expr->inputs()[0];

        switch (forwardOp->main.AsUnaryOp()->opType) {
            case MNN::UnaryOpOperation_LOG1P: {
                // d log(1+x) = 1/(1+x) * dx = dx / (1+x)
                auto oneConst = _Const(1.0f, {}, NHWC);
                auto addOne = _Add(input, oneConst);
                auto div = _Div(outputDiff, addOne);
                return {div};
            }
            case MNN::UnaryOpOperation_EXP: {
                // d Exp(x) = Exp(x) * dx
                return {_Mul(outputDiff, output[0])};
            }
            case MNN::UnaryOpOperation_LOG: {
                // d Log(x) =  dx / x
                return {_Div(outputDiff, input)};
            }
            case MNN::UnaryOpOperation_NEG: {
                // d (-x) = - dx
                return {_Neg(outputDiff)};
            }
            case MNN::UnaryOpOperation_SQRT: {
                // d (-sqrt(x)) = 0.5 / sqrt(x) * dx
                auto oneConst = _Const(0.5f, {}, NHWC);
                auto mul = _Mul(outputDiff, oneConst);
                auto div = _Div(mul, output[0]);
                return {div};
            }
            case MNN::UnaryOpOperation_SQUARE: {
                // d (x^2) = (x*dx + x*dx)
                auto mul = _Mul(input, outputDiff);
                return {_Add(mul, mul)};
            }
            default:
                MNN_ASSERT(false);
                break;
        }

        return {};
    }
};
class SigmoidGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr, const std::vector<Express::VARP>& output, const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result{nullptr};
        auto outputDiff = backwardOutput[0];

        // y = (1/(1+e(-x))) , dy = y(1-y) * dx = (y*y - y)*dx
        auto mul = _Mul(output[0], output[0]);
        auto sub = _Sub(mul, output[0]);
        auto grad = _Mul(sub, outputDiff);
        result[0] = grad;
        return result;
    }
};


static const auto gRegister = []() {
    static UnaryGrad _c;
    static SigmoidGrad _s;
    OpGrad::insert(OpType_UnaryOp, &_c);
    OpGrad::insert(OpType_Sigmoid, &_s);
    return true;
}();
