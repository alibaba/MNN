//
//  TensorConverterMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN_generated.h"
#include <MNN/expr/ExprCreator.hpp>

namespace MNN {
namespace Express {

#define CONVERT(src, dst, f)\
if (f == src) return dst;

static int __convertFormat(Dimensionformat format) {
    CONVERT(NCHW, MNN_DATA_FORMAT_NCHW, format);
    CONVERT(NHWC, MNN_DATA_FORMAT_NHWC, format);
    CONVERT(NC4HW4, MNN_DATA_FORMAT_NC4HW4, format);
    return MNN_DATA_FORMAT_UNKNOWN;
}

static Express::Dimensionformat __revertFormat(int format) {
    CONVERT(MNN_DATA_FORMAT_NCHW, Express::NCHW, format);
    CONVERT(MNN_DATA_FORMAT_NHWC, Express::NHWC, format);
    CONVERT(MNN_DATA_FORMAT_NC4HW4, Express::NC4HW4, format);
    return NCHW;
}

static auto gRegister = []() {
    {
        auto compare = [](EXPRP expr) {
            if (nullptr == expr->get()) {
                return false;
            }
            if (expr->get()->type() != OpType_BinaryOp) {
                return false;
            }
            auto opType = expr->get()->main_as_BinaryOp()->opType();
            int code = -1;
#define CONVERTBINARY_ELT(src, dst) if (opType == src) code = dst;
            CONVERTBINARY_ELT(BinaryOpOperation_ADD, EltwiseType_SUM);
            CONVERTBINARY_ELT(BinaryOpOperation_SUB, EltwiseType_SUB);
            CONVERTBINARY_ELT(BinaryOpOperation_MAXIMUM, EltwiseType_MAXIMUM);
            CONVERTBINARY_ELT(BinaryOpOperation_MUL, EltwiseType_PROD);
            
            if (-1 == code) {
                return false;
            }
            auto inputs = expr->inputs();
            MNN_ASSERT(inputs.size() == 2);
            auto input0Info = inputs[0]->getInfo();
            auto input1Info = inputs[1]->getInfo();
            if (nullptr == input0Info || nullptr == input1Info) {
                return false;
            }
            if (input0Info->size <= 0 || input1Info->size <= 0) {
                return false;
            }
            if (input0Info->order != input1Info->order) {
                return false;
            }
            if (input0Info->type.code != halide_type_float || input1Info->type.code != halide_type_float) {
                return false;
            }
            if (input0Info->dim.size() != input1Info->dim.size()) {
                return false;
            }
            for (int i=0; i<input1Info->dim.size(); ++i) {
                if (input1Info->dim[i] != input0Info->dim[i]) {
                    return false;
                }
            }
            return true;
        };
        auto modify = [](EXPRP expr) {
            auto inputs = expr->inputs();
            auto opType = expr->get()->main_as_BinaryOp()->opType();
            int code = -1;
            CONVERTBINARY_ELT(BinaryOpOperation_ADD, EltwiseType_SUM);
            CONVERTBINARY_ELT(BinaryOpOperation_SUB, EltwiseType_SUB);
            CONVERTBINARY_ELT(BinaryOpOperation_MAXIMUM, EltwiseType_MAXIMUM);
            CONVERTBINARY_ELT(BinaryOpOperation_MUL, EltwiseType_PROD);
            std::unique_ptr<OpT> newOp(new OpT);
            newOp->type = OpType_Eltwise;
            newOp->main.type = OpParameter_Eltwise;
            newOp->main.value = new EltwiseT;
            newOp->main.AsEltwise()->type = (EltwiseType)code;
            auto newExpr = Expr::create(newOp.get(), inputs);
#undef CONVERTBINARY_ELT

            newExpr->setName(expr->name());
            Expr::replace(expr, newExpr);
            return true;
        };
        TemplateMerge::getInstance("Merge").insertTemplate("TurnBinaryToElementwise", compare, modify);
    }
    {
        auto compare = [](EXPRP expr) {
            if (nullptr == expr->get()) {
                return false;
            }
            if (expr->get()->type() != OpType_ConvertTensor) {
                return false;
            }
            auto inputs = expr->inputs();
            auto inputExpr = inputs[0]->expr().first;
            if (nullptr == inputExpr->get()) {
                return false;
            }
            auto inputOp = inputExpr->get();
            if (inputOp->type() != OpType_ConvertTensor) {
                return false;
            }
            return true;
        };
        auto modify = [](EXPRP expr) {
            auto inputs = expr->inputs();
            auto inputExpr = inputs[0]->expr().first;
            auto subInputs = inputExpr->inputs();
            auto newExpr = Expr::create(expr->extra(), std::move(subInputs));
            newExpr->setName(expr->name());
            Expr::replace(expr, newExpr);
            return true;
        };
        TemplateMerge::getInstance("Merge").insertTemplate("TensorConverterMerge", compare, modify);
    }
    {
        auto compare = [](EXPRP expr) {
            if (nullptr == expr->get()) {
                return false;
            }
            if (expr->get()->type() == OpType_ConvertTensor) {
                return false;
            }
            auto inputs = expr->inputs();
            for (auto input : inputs) {
                if (input->expr().first->get() == nullptr) {
                    continue;
                }
                auto subOp = input->expr().first->get();
                if (subOp->type() != OpType_ConvertTensor) {
                    continue;
                }
                auto inputInfo = input->expr().first->inputs()[0]->getInfo();
                if (nullptr == inputInfo) {
                    continue;
                }
                if (subOp->main_as_TensorConvertInfo()->dest() == __convertFormat(inputInfo->order)) {
                    return true;
                }
            }
            return false;
        };
        auto modify = [](EXPRP expr) {
            auto inputs = expr->inputs();
            std::vector<VARP> newInput = inputs;;
            for (int i=0; i<inputs.size(); ++i) {
                auto input = inputs[i];
                if (input->expr().first->get() == nullptr) {
                    continue;
                }
                auto subOp = input->expr().first->get();
                if (subOp->type() != OpType_ConvertTensor) {
                    continue;
                }
                auto inputInfo = input->expr().first->inputs()[0]->getInfo();
                if (nullptr == inputInfo) {
                    continue;
                }
                if (subOp->main_as_TensorConvertInfo()->dest() == __convertFormat(inputInfo->order)) {
                    newInput[i] = input->expr().first->inputs()[0];
                }
            }
            auto newExpr = Expr::create(expr->extra(), std::move(newInput), expr->outputSize());
            newExpr->setName(expr->name());
            Expr::replace(expr, newExpr);
            return true;
        };
        TemplateMerge::getInstance("Merge").insertTemplate("TensorConverterSameMerge", compare, modify);
    }
    {
        // Turn Input connected with convert to not convert
        auto compare = [](EXPRP expr) {
            if (nullptr == expr->get()) {
                return false;
            }
            if (expr->get()->type() != OpType_ConvertTensor) {
                return false;
            }
            auto inputs = expr->inputs();
            auto inputOp =  inputs[0]->expr().first->get();
            if (nullptr != inputOp) {
                return false;
            }
            return true;
        };
        auto modify = [](EXPRP expr) {
            auto input = expr->inputs()[0];
            auto inputExpr = input->expr().first;
            auto tempInputs = {input};
            EXPRP newInputExpr;
            if (VARP::INPUT == inputExpr->inputType()) {
                auto outputInfo = inputExpr->outputInfo(0);
                Variable::Info newInfo = *outputInfo;
                auto newOrder = __revertFormat(expr->get()->main_as_TensorConvertInfo()->dest());
                if (newOrder == NC4HW4 && newInfo.order == NHWC && newInfo.dim.size() > 3) {
                    int h = newInfo.dim[1];
                    int w = newInfo.dim[2];
                    int c = newInfo.dim[3];
                    newInfo.dim[1] = c;
                    newInfo.dim[2] = h;
                    newInfo.dim[3] = w;
                }
                newInfo.order = __revertFormat(expr->get()->main_as_TensorConvertInfo()->dest());
                newInfo.syncSize();
                newInfo.ptr = nullptr;
                newInputExpr = Expr::create(std::move(newInfo));
            } else {
                auto newInput = Variable::create(Expr::create(expr->extra(), std::move(tempInputs)));
                newInput.fix(inputExpr->inputType());
                newInputExpr = newInput->expr().first;
            }
            newInputExpr->setName(inputExpr->name());
            Expr::replace(expr, newInputExpr);
            return true;
        };
        TemplateMerge::getInstance("Merge").insertTemplate("TensorConverterInputMerge", compare, modify);
    }
    {
        auto compare = [](EXPRP expr) {
            if (nullptr == expr->get()) {
                return false;
            }
            if (OpType_ConvertTensor == expr->get()->type()) {
                return false;
            }
            if (expr->outputSize() > 1) {
                return false;
            }
            auto inputs = expr->inputs();
            if (inputs.empty()) {
                return false;
            }
            for (int i=0; i<inputs.size(); ++i) {
                auto inputOp =  inputs[i]->expr().first->get();
                if (nullptr == inputOp) {
                    return false;
                }
                if (inputOp->type() != OpType_ConvertTensor) {
                    return false;
                }
                if (inputs[i]->getInfo() == nullptr) {
                    return false;
                }
                if (inputs[i]->getInfo()->order == NC4HW4) {
                    return false;
                }
            }
            auto type = expr->get()->type();
#define SUPPORT(t) if (type == t) return true;
            SUPPORT(OpType_UnaryOp);
            SUPPORT(OpType_ReLU);
            SUPPORT(OpType_ReLU6);
            SUPPORT(OpType_Cast);
            SUPPORT(OpType_ELU);
            SUPPORT(OpType_Sigmoid);
            SUPPORT(OpType_Selu);
            SUPPORT(OpType_Permute);
            SUPPORT(OpType_Concat);
            SUPPORT(OpType_Slice);
            SUPPORT(OpType_Eltwise);

#undef SUPPORT
            return false;
        };
        auto modify = [](EXPRP expr) {
            std::vector<VARP> tempInputs;
            for (int i=0; i<expr->inputs().size(); ++i) {
                tempInputs.emplace_back(expr->inputs()[i]->expr().first->inputs()[0]);
            }
            EXPRP newInputExpr;
            auto order = expr->inputs()[0]->getInfo()->order;
            auto newInput = Variable::create(Expr::create(expr->extra(), std::move(tempInputs), expr->outputSize()));
            newInput->setName(expr->name());
            newInput = _Convert(newInput, order);
            newInputExpr = newInput->expr().first;
            Expr::replace(expr, newInputExpr);
            return true;
        };
        TemplateMerge::getInstance("Merge").insertTemplate("TurnCompabilityOpAsNC4HW4", compare, modify);
    }
    return true;
}();
}
} // namespace MNN
