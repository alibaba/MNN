//
//  FuseTemplateOp.cpp
//  MNNConverter
//
//  Created by MNN on 2021/04/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN/expr/MathOp.hpp"
#include "MNN/expr/NeuralNetWorkOp.hpp"
#include "MNN_generated.h"
#include "config.hpp"
namespace MNN {
namespace Express {
// left is pattern, right is dest
static bool isSameOp(const MNN::Op* op0, const MNN::Op* op1) {
    if (op0->type() != op1->type()) {
        return false;
    }
    if (op0->main_type() != op1->main_type()) {
        return false;
    }
    if (op0->main_type() == OpParameter_NONE) {
        return true;
    }
    if (op0->type() == OpType_ReLU) {
        return op0->main_as_Relu()->slope() == op1->main_as_Relu()->slope();
    }
    if (op0->type() == OpType_ReLU6) {
        return op0->main_as_Relu6()->maxValue() == op1->main_as_Relu6()->maxValue() && op0->main_as_Relu6()->minValue() == op1->main_as_Relu6()->minValue();
    }
    if (op0->main_type() == OpParameter_UnaryOp) {
        return op0->main_as_UnaryOp()->opType() == op1->main_as_UnaryOp()->opType();
    }
    if (op0->main_type() == OpParameter_BinaryOp) {
        return op0->main_as_BinaryOp()->opType() == op1->main_as_BinaryOp()->opType();
    }
    return false;
}
static bool isTheSameRec(EXPRP left, EXPRP right, std::map<EXPRP, VARP>& inputConst) {
    auto lop = left->get();
    auto rop = right->get();
    if (nullptr == lop) {
        if (nullptr != rop) {
            return false;
        }
    }
    if (left->inputs().size() != right->inputs().size()) {
        return false;
    }
    if (left->outputSize() != right->outputSize()) {
        return false;
    }
    if (nullptr != lop && nullptr == rop) {
        return false;
    }
    if (nullptr == lop && nullptr == rop) {
        // Constant
        if (left->inputType() != right->inputType()) {
            return false;
        }
        auto leftV = Variable::create(left);
        auto rightV = Variable::create(right);
        if (leftV->getInfo() == nullptr || rightV->getInfo() == nullptr) {
            return false;
        }
        auto lInfo = leftV->getInfo();
        auto rInfo = rightV->getInfo();
        if (lInfo->type != rInfo->type) {
            return false;
        }
        if (lInfo->dim != rInfo->dim) {
            return false;
        }
        if (lInfo->size != rInfo->size) {
            return false;
        }
        auto lPtr = leftV->readMap<void>();
        auto rPtr = rightV->readMap<void>();
        if (nullptr == lPtr || nullptr == rPtr) {
            return false;
        }
        if (0 != ::memcmp(lPtr, rPtr, lInfo->size * lInfo->type.bytes())) {
            return false;
        }
        return true;
    }
    // Check Op
    if (!isSameOp(lop, rop)) {
        return false;
    }
    for (int i=0; i<left->inputs().size(); ++i) {
        auto leftExpr = left->inputs()[i]->expr();
        auto rightExpr = right->inputs()[i]->expr();
        auto subLop = leftExpr.first->get();
        if (nullptr == subLop) {
            if (leftExpr.first->inputType() == VARP::INPUT) {
                auto iter = inputConst.find(leftExpr.first);
                if (iter == inputConst.end()) {
                    inputConst.insert(std::make_pair(leftExpr.first, right->inputs()[i]));
                    continue;
                }
                auto iterExpr = iter->second->expr();
                if (iterExpr.first.get() != rightExpr.first.get() || iterExpr.second != rightExpr.second) {
                    return false;
                }
                continue;
            }
        }
        if (!isTheSameRec(left->inputs()[i]->expr().first, right->inputs()[i]->expr().first, inputConst)) {
            return false;
        }
    }
    return true;
}

static auto gRegister = []() {
    {
        // Turn DIV Const to Multi
        auto match = [](EXPRP expr) {
            if (expr->get() == nullptr) {
                return false;
            }
            if (OpType_BinaryOp != expr->get()->type()) {
                return false;
            }
            if (BinaryOpOperation_REALDIV != expr->get()->main_as_BinaryOp()->opType()) {
                return false;
            }
            auto i1 = expr->inputs()[1];
            auto i1Info = i1->getInfo();
            if (nullptr == i1Info || i1Info->type.code != halide_type_float) {
                return false;
            }
            auto i1Ptr = i1->readMap<void>();
            if (nullptr == i1Ptr) {
                return false;
            }
            return true;
        };

        auto transform = [](EXPRP expr) {
            auto i1 = expr->inputs()[1];
            i1 = _Reciprocal(i1);
            i1.fix(VARP::CONSTANT);
            auto newVar = _Multiply(expr->inputs()[0], i1);
            newVar->setName(expr->name());
            Expr::replace(expr, newVar->expr().first);
            return true;
        };
        TemplateMerge::getInstance("Merge").insertTemplate("ConstDivToMul", match, transform);
    }
    {
        auto input = _Input({}, NCHW);
        auto left = _Relu6(_Add(input, _Scalar<float>(3)));
        auto res = _Multiply(_Multiply(input, left), _Scalar<float>(1.0f/6.0f));
        auto res2 = _Multiply(input, _Multiply(left, _Scalar<float>(1.0f/6.0f)));
        std::vector<EXPRP> templatesExprs = {
            res->expr().first,
            res2->expr().first
        };
        auto transform2 = [templatesExprs, input](EXPRP expr) {
            auto config = Global<modelConfig>::Get();
            auto version = config->targetVersion;
            if (version < 1.2f) {
                // For target version < 1.2 , don't support hardswish
                return false;
            }
            for (auto templateExpr : templatesExprs) {
                std::map<EXPRP, VARP> inputConst;
                if (isTheSameRec(templateExpr, expr, inputConst)) {
                    auto inputVarIter = inputConst.find(input->expr().first);
                    if (inputVarIter == inputConst.end()) {
                        MNN_ERROR("Invalid Match, may be something is wrong for Fuse\n");
                        continue;
                    }
                    auto inputVar = inputVarIter->second;
                    std::unique_ptr<MNN::OpT> newOp(new OpT);
                    newOp->type = OpType_UnaryOp;
                    newOp->main.value = new UnaryOpT;
                    newOp->main.type = OpParameter_UnaryOp;
                    newOp->main.AsUnaryOp()->opType = UnaryOpOperation_HARDSWISH;
                    auto newVar = Variable::create(Expr::create(newOp.get(), {inputVar}, 1));
                    newVar->setName(expr->outputName(0));
                    Expr::replace(expr, newVar->expr().first);
                    return true;
                }
            }
            return false;
        };
        TemplateMerge::getInstance("Merge").insertTemplateV2("FuseHardSwish", transform2);
    }
    {
        auto zero0 = _Scalar<int>(0);
        auto zero1 = _Scalar<float>(0);
        auto one0 = _Scalar<int>(1);
        auto one1 = _Scalar<float>(1);
        auto input0 = _Input({}, NHWC, halide_type_of<int>());
        auto input1 = _Input({}, NHWC, halide_type_of<float>());
        std::vector<MNN::Express::VARP> binaryAddZero({
            zero0 + input1,
            input1 + zero0,
            zero1 + input1,
            input1 + zero1,
            input0 - zero0,
            input1 - zero1,
            input0 * one0,
            one0 * input0,
            input1 * one1,
            one1 * input1,
        });
        auto transform2 = [binaryAddZero, input0, input1](EXPRP expr) {
            std::map<EXPRP, VARP> inputConst;
            for (int index=0; index<binaryAddZero.size(); ++index) {
                auto res = binaryAddZero[index];
                if (isTheSameRec(res->expr().first, expr, inputConst)) {
                    auto inputVarIter0 = inputConst.find(input0->expr().first);
                    auto inputVarIter1 = inputConst.find(input1->expr().first);
                    MNN::Express::VARP inputVar;
                    if (inputVarIter0 == inputConst.end() && inputVarIter1 == inputConst.end()) {
                        MNN_ERROR("Invalid Match, may be something is wrong for Fuse\n");
                        return false;
                    }
                    if (inputVarIter0 != inputConst.end()) {
                        inputVar = inputVarIter0->second;
                    } else {
                        inputVar = inputVarIter1->second;
                    }
                    std::unique_ptr<MNN::OpT> newOp(new OpT);
                    newOp->type = OpType_Identity;
                    newOp->main.type = OpParameter_NONE;
                    auto newVar = Variable::create(Expr::create(newOp.get(), {inputVar}, 1));
                    newVar->setName(expr->outputName(0));
                    Expr::replace(expr, newVar->expr().first);
                    return true;
                }
            }
            return false;
        };
        TemplateMerge::getInstance("Merge").insertTemplateV2("RemoveUselessBinary", transform2);
    }
    {
        auto input0 = _Input({}, NCHW);
        auto input1 = _Input({}, NCHW);
        auto diff = input0 - input1;
        auto res0 = diff * diff;
        auto transform2 = [res0, input0, input1](EXPRP expr) {
            std::map<EXPRP, VARP> inputConst;
            if (isTheSameRec(res0->expr().first, expr, inputConst)) {
                auto inputVarIter0 = inputConst.find(input0->expr().first);
                auto inputVarIter1 = inputConst.find(input1->expr().first);
                if (inputVarIter0 == inputConst.end() || inputVarIter1 == inputConst.end()) {
                    MNN_ERROR("Invalid Match, may be something is wrong for Fuse\n");
                    return false;
                }
                auto inputVar = inputVarIter0->second;
                std::unique_ptr<MNN::OpT> newOp(new OpT);
                newOp->type = OpType_BinaryOp;
                newOp->main.value = new BinaryOpT;
                newOp->main.type = OpParameter_BinaryOp;
                newOp->main.AsBinaryOp()->opType = BinaryOpOperation_SquaredDifference;
                auto newVar = Variable::create(Expr::create(newOp.get(), {inputVarIter0->second, inputVarIter1->second}, 1));
                newVar->setName(expr->outputName(0));
                Expr::replace(expr, newVar->expr().first);
                return true;
            }
            return false;
        };
        TemplateMerge::getInstance("Merge").insertTemplateV2("FuseSquaredDifference", transform2);
    }
    {
        auto input = _Input({}, NCHW);
        auto sqr = _Sqrt(input);
        auto sqrdiv = _Reciprocal(sqr);
        auto sqrdiv2 = _Scalar<float>(1.0f) / sqr;
        std::vector<EXPRP> templatesExprs = {
            sqrdiv->expr().first,
            sqrdiv2->expr().first
        };

        auto transform = [templatesExprs, input](EXPRP expr) {
            for (auto templateExpr : templatesExprs) {
                std::map<EXPRP, VARP> inputConst;
                if (isTheSameRec(templateExpr, expr, inputConst)) {
                    auto inputVarIter = inputConst.find(input->expr().first);
                    if (inputVarIter == inputConst.end()) {
                        MNN_ERROR("Invalid Match, may be something is wrong for Fuse\n");
                        continue;
                    }
                    auto inputVar = inputVarIter->second;
                    std::unique_ptr<MNN::OpT> newOp(new OpT);
                    newOp->type = OpType_UnaryOp;
                    newOp->main.value = new UnaryOpT;
                    newOp->main.type = OpParameter_UnaryOp;
                    newOp->main.AsUnaryOp()->opType = UnaryOpOperation_RSQRT;
                    auto newVar = Variable::create(Expr::create(newOp.get(), {inputVar}, 1));
                    newVar->setName(expr->outputName(0));
                    Expr::replace(expr, newVar->expr().first);
                    return true;
                }
            }
            return false;
        };
        TemplateMerge::getInstance("Merge").insertTemplateV2("FuseRsqrt", transform);
    }
    {
        auto input = _Input({}, NCHW);
        auto inputSquare = _Pow(input, _Scalar<float>(2.0f));
        std::vector<EXPRP> templatesExprs = {
            inputSquare->expr().first,
        };

        auto transform = [templatesExprs, input](EXPRP expr) {
            for (auto templateExpr : templatesExprs) {
                std::map<EXPRP, VARP> inputConst;
                if (isTheSameRec(templateExpr, expr, inputConst)) {
                    auto inputVarIter = inputConst.find(input->expr().first);
                    if (inputVarIter == inputConst.end()) {
                        MNN_ERROR("Invalid Match, may be something is wrong for Fuse\n");
                        continue;
                    }
                    auto inputVar = inputVarIter->second;
                    std::unique_ptr<MNN::OpT> newOp(new OpT);
                    newOp->type = OpType_UnaryOp;
                    newOp->main.value = new UnaryOpT;
                    newOp->main.type = OpParameter_UnaryOp;
                    newOp->main.AsUnaryOp()->opType = UnaryOpOperation_SQUARE;
                    auto newVar = Variable::create(Expr::create(newOp.get(), {inputVar}, 1));
                    newVar->setName(expr->outputName(0));
                    Expr::replace(expr, newVar->expr().first);
                    return true;
                }
            }
            return false;
        };
        TemplateMerge::getInstance("Merge").insertTemplateV2("FusePow2ToSquare", transform);
    }
    {
        auto input = _Input({}, NCHW);
        auto input1 = _Input({}, NCHW);
        auto res = _Divide(input, _Sqrt(input1));
        std::vector<EXPRP> templatesExprs = {
            res->expr().first,
        };

        auto transform = [templatesExprs, input, input1](EXPRP expr) {
            for (auto templateExpr : templatesExprs) {
                std::map<EXPRP, VARP> inputConst;
                if (isTheSameRec(templateExpr, expr, inputConst)) {
                    auto inputVarIter = inputConst.find(input->expr().first);
                    if (inputVarIter == inputConst.end()) {
                        MNN_ERROR("Invalid Match, may be something is wrong for Fuse\n");
                        continue;
                    }
                    auto inputVar = inputVarIter->second;
                    auto input1VarIter = inputConst.find(input1->expr().first);
                    if (input1VarIter == inputConst.end()) {
                        MNN_ERROR("Invalid Match, may be something is wrong for Fuse\n");
                        continue;
                    }
                    auto input1Var = input1VarIter->second;
                    auto newVar = _Multiply(inputVar, _Rsqrt(input1Var));
                    newVar->setName(expr->outputName(0));
                    Expr::replace(expr, newVar->expr().first);
                    return true;
                }
            }
            return false;
        };
        TemplateMerge::getInstance("Merge").insertTemplateV2("TurnDivSqrtToMulRSqrt", transform);
    }
    {
        auto input = _Input({}, NHWC);
        auto const707 = _Scalar<float>(0.707106769);
        auto constOne = _Scalar<float>(1.0f);
        auto constHalf = _Scalar<float>(0.5);
        auto res = (MNN::Express::_Erf(input * const707) + constOne) * input * constHalf;
        auto res2 = input * (MNN::Express::_Erf(input * const707) + constOne) * constHalf;
        std::vector<EXPRP> templatesExprs = {
            res->expr().first, res2->expr().first
        };

        auto transform = [templatesExprs, input](EXPRP expr) {
            auto config = Global<modelConfig>::Get();
            auto unaryType = UnaryOpOperation_GELU_STANDARD;
            if (config->optimizeLevel == 2) {
                unaryType = UnaryOpOperation_GELU;
            }
            for (auto templateExpr : templatesExprs) {
                std::map<EXPRP, VARP> inputConst;
                if (isTheSameRec(templateExpr, expr, inputConst)) {
                    auto inputVarIter = inputConst.find(input->expr().first);
                    if (inputVarIter == inputConst.end()) {
                        MNN_ERROR("Invalid Match, may be something is wrong for Fuse\n");
                        continue;
                    }
                    auto inputVar = inputVarIter->second;
                    std::unique_ptr<MNN::OpT> newUnary(new MNN::OpT);
                    newUnary->type = OpType_UnaryOp;
                    newUnary->main.type = OpParameter_UnaryOp;
                    newUnary->main.value = new UnaryOpT;
                    newUnary->main.AsUnaryOp()->opType = unaryType;
                    auto newVar = MNN::Express::Variable::create(MNN::Express::Expr::create(newUnary.get(), {inputVar}));
                    newVar->setName(expr->outputName(0));
                    Expr::replace(expr, newVar->expr().first);
                    return true;
                }
            }
            return false;
        };
        TemplateMerge::getInstance("Merge").insertTemplateV2("FuseGELU", transform);
    }
    return true;
}();

}
} // namespace MNN
