//
//  UnaryGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class UnaryGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::unique_ptr<OpT> forwardOp(expr->get()->UnPack());
        auto outputDiff = backwardOutput[0];
        auto input      = expr->inputs()[0];
        std::vector<Express::VARP> res(1, nullptr);
        std::vector<Express::VARP> output{Variable::create(expr, 0)};

        switch (forwardOp->main.AsUnaryOp()->opType) {
            case MNN::UnaryOpOperation_LOG1P: {
                // d log(1+x) = 1/(1+x) * dx = dx / (1+x)
                auto oneConst = _Const(1.0f, {}, NHWC);
                auto addOne   = _Add(input, oneConst);
                res[0]        = _Divide(outputDiff, addOne);
                break;
            }
            case MNN::UnaryOpOperation_EXP: {
                // d Exp(x) = Exp(x) * dx
                res[0] = _Multiply(outputDiff, output[0]);
                break;
            }
            case MNN::UnaryOpOperation_LOG: {
                // d Log(x) =  dx / x
                res[0] = _Divide(outputDiff, input);
                break;
            }
            case MNN::UnaryOpOperation_NEG: {
                // d (-x) = - dx
                res[0] = _Negative(outputDiff);
                break;
            }
            case MNN::UnaryOpOperation_SQRT: {
                // d (-sqrt(x)) = 0.5 / sqrt(x) * dx
                auto oneConst = _Const(0.5f, {}, NHWC);
                auto mul      = _Multiply(outputDiff, oneConst);
                res[0]        = _Divide(mul, output[0]);
                break;
            }
            case MNN::UnaryOpOperation_SQUARE: {
                // d (x^2) = (x*dx + x*dx)
                auto mul = _Multiply(input, outputDiff);
                res[0]   = _Add(mul, mul);
                break;
            }
            default:
                return res;
        }

        res[0]->setName(expr->name() + "_Grad");
        return res;
    }
};
class SigmoidGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result(1, nullptr);
        auto outputDiff = backwardOutput[0];
        std::vector<Express::VARP> output{Variable::create(expr, 0)};

        // y = (1/(1+e(-x))) , dy = y(1-y) * dx = (y*y - y)*dx
        auto mul  = _Multiply(output[0], output[0]);
        auto sub  = _Subtract(mul, output[0]);
        auto grad = _Multiply(sub, outputDiff);
        result[0] = grad;
        result[0]->setName(expr->name() + "_Grad");
        return result;
    }
};

class TanhGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result{nullptr};
        std::vector<Express::VARP> output{Variable::create(expr, 0)};

        auto outputDiff = backwardOutput[0];
        // d tanh(x) = (1-tanh(x)^2)dx
        result[0] = (_Const(1.0f, {}, NCHW) - _Square(output[0])) * outputDiff;
        return result;
    }
};

static const auto gRegister = []() {
    static UnaryGrad _c;
    static SigmoidGrad _s;
    static TanhGrad _t;
    OpGrad::insert(OpType_UnaryOp, &_c);
    OpGrad::insert(OpType_Sigmoid, &_s);
    OpGrad::insert(OpType_TanH, &_t);
    return true;
}();
