//
//  UnaryGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include "core/Macro.h"

#define MNN_PI 3.14159265358979323846

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
            case MNN::UnaryOpOperation_COS: {
                // d Sin(x) =  -dx * Sin(x)
                res[0] = _Negative(outputDiff) * _Sin(input);
                break;
            }
            case MNN::UnaryOpOperation_SIN: {
                // d Sin(x) =  dx * Cos(x)
                res[0] = outputDiff * _Cos(input);
                break;
            }
            case MNN::UnaryOpOperation_ABS: {
                // d Abs(x) =  dx * (x > 0 ? 1 : -1)
                res[0] = outputDiff * _Sign(input);
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
                res[0]        = OpGrad::divideAvoidZero(mul, output[0]);
                break;
            }
            case MNN::UnaryOpOperation_SQUARE: {
                // d (x^2) = (x*dx + x*dx)
                auto mul = _Multiply(input, outputDiff);
                res[0]   = _Add(mul, mul);
                break;
            }
            case MNN::UnaryOpOperation_SIGMOID: {
                auto grad = OpGrad::get(OpType_Sigmoid);
                res[0] = grad->onGrad(expr, backwardOutput)[0];
                break;
            }
            case MNN::UnaryOpOperation_TANH: {
                auto grad = OpGrad::get(OpType_TanH);
                res[0] = grad->onGrad(expr, backwardOutput)[0];
                break;
            }
            case MNN::UnaryOpOperation_RSQRT: {
                // d (x^(-1/2)) = -1/2 * x^(-3/2) * dx
                res[0]   = _Scalar(-0.5f) * _Pow(input, _Scalar(-1.5f)) * outputDiff;
                break;
            }
            case MNN::UnaryOpOperation_TAN: {
                // d tan(x) = 1 / (cos(x))^2 * dx
                res[0]   = _Scalar(1.0f) / _Square(_Cos(input)) * outputDiff;
                break;
            }
            case MNN::UnaryOpOperation_ASIN: {
                // d asin(x) = 1 / sqrt(1 - x^2) * dx
                res[0]   = _Scalar(1.0f) / _Sqrt(_Scalar(1.0f) - _Square(input)) * outputDiff;
                break;
            }
            case MNN::UnaryOpOperation_ACOS: {
                // d acos(x) = -1 / sqrt(1 - x^2) * dx
                res[0]   = _Scalar(-1.0f) / _Sqrt(_Scalar(1.0f) - _Square(input)) * outputDiff;
                break;
            }
            case MNN::UnaryOpOperation_ATAN: {
                // d atan(x) = 1 / (1 + x^2) * dx
                res[0]   = _Scalar(1.0f) / (_Scalar(1.0f) + _Square(input)) * outputDiff;
                break;
            }
            case MNN::UnaryOpOperation_RECIPROCAL: {
                // d x^-1 = - x^-2 * dx
                res[0]   = _Negative(_Pow(input, _Scalar(-2.0f))) * outputDiff;
                break;
            }
            case MNN::UnaryOpOperation_ACOSH: {
                // d acosh(x) = 1 / sqrt(x^2 - 1) * dx
                res[0]   = _Scalar(1.0f) / _Sqrt(_Square(input) - _Scalar(1.0f)) * outputDiff;
                break;
            }
            case MNN::UnaryOpOperation_SINH: {
                // d sinh(x) = cosh(x) * dx
                res[0]   = _Cosh(input) * outputDiff;
                break;
            }
            case MNN::UnaryOpOperation_COSH: {
                // d cosh(x) = sinh(x) * dx
                res[0]   = _Sinh(input) * outputDiff;
                break;
            }
            case MNN::UnaryOpOperation_ASINH: {
                // d asinh(x) = 1 / sqrt(x^2 + 1) * dx
                res[0]   = _Scalar(1.0f) / _Sqrt(_Square(input) + _Scalar(1.0f)) * outputDiff;
                break;
            }
            case MNN::UnaryOpOperation_ATANH: {
                // d atanh(x) = 1 / (1 - x^2) * dx
                res[0]   = _Scalar(1.0f) / (_Scalar(1.0f) - _Square(input)) * outputDiff;
                break;
            }
            case MNN::UnaryOpOperation_ERF: {
                // d erf(x) = 2 / sqrt(pi) * exp(- x^2) * dx
                res[0]   = _Scalar(2.0f) / _Sqrt(_Scalar(float(MNN_PI))) * _Exp(_Negative(_Square(input))) * outputDiff;
                break;
            }
            case MNN::UnaryOpOperation_ERFC: {
                // d erfc(x) = -2 / sqrt(pi) * exp(- x^2) * dx
                res[0]   = _Scalar(-2.0f) / _Sqrt(_Scalar(float(MNN_PI))) * _Exp(_Negative(_Square(input))) * outputDiff;
                break;
            }
            case MNN::UnaryOpOperation_ERFINV: {
                // d erfinv(x) = sqrt(pi) / 2 * exp(erfinv(x)^2) * dx
                res[0]   = _Sqrt(_Scalar(float(MNN_PI))) / _Scalar(2.0f) * _Exp(_Square(_Erfinv(input))) * outputDiff;
                break;
            }
            case MNN::UnaryOpOperation_EXPM1: {
                // d expm1(x) = exp(x) * dx
                res[0]   = _Exp(input) * outputDiff;
                break;
            }
            case MNN::UnaryOpOperation_HARDSWISH: {
                // d hardswish(x) = (relu6(x+3) + x * relu6'(x+3)) / 6 * dx
                auto inputp3 = input + _Scalar(3.0f);
                auto mask0 = _Cast<float>(_Greater(inputp3, _Scalar(0.0f)));
                auto mask1 = _Cast<float>(_Less(inputp3, _Scalar(6.0f)));
                auto relu6Xp3Grad = mask0 * mask1;
                res[0]   = (_Relu6(inputp3) + input * relu6Xp3Grad) / _Scalar(6.0f) * outputDiff;
                break;
            }
            case MNN::UnaryOpOperation_GELU:
            case MNN::UnaryOpOperation_GELU_STANDARD: {
                // d gelu(x) = 0.5 * ( (1 + erf(x / sqrt(2))) + x * (erf'(x / sqrt(2)) / sqrt(2)) ) * dx
                auto const05 = _Scalar(0.5f);
                auto inputx = input / _Sqrt(_Scalar(2.0f));
                auto part1 = _Scalar(1.0f) + _Erf(inputx);
                auto erfGrad = _Scalar(2.0f) / _Sqrt(_Scalar(float(MNN_PI))) * _Exp(_Negative(_Square(inputx)));
                auto part2 = input * erfGrad / _Sqrt(_Scalar(2.0f));
                res[0]   = const05 * (part1 + part2) * outputDiff;
                break;
            }
            default:
                MNN_ERROR("Can't grad for unary: %d\n", forwardOp->main.AsUnaryOp()->opType);

                return res;
        }

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

        // y = (1/(1+e(-x))) , dy = y(1-y) * dx = (y - y*y)*dx
        auto mul  = _Multiply(output[0], output[0]);
        auto sub  = _Subtract(output[0], mul);
        auto grad = _Multiply(sub, outputDiff);
        result[0] = grad;
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
