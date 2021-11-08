//
//  Conv1dQuantToConv2dQuant.cpp
//  MNNConverter
//
//  Created by MNN on 2020/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace Express {

static auto gRegister = []() {
    auto match = [](EXPRP expr) {
        if (nullptr == expr->get()) {
            return false;
        }
        if (expr->get()->type() != OpType::OpType_ConvertTensor) {
            return false;
        }

        auto input1 = expr->inputs()[0];
        auto expr1  = input1->expr().first;
        if (expr1->get() == nullptr) {
            return false;
        }
        if (expr1->get()->type() != OpType::OpType_Int8ToFloat) {
            return false;
        }

        auto input2 = expr1->inputs()[0];
        auto expr2  = input2->expr().first;
        if (expr2->get() == nullptr) {
            return false;
        }
        if (expr2->get()->type() != OpType::OpType_FloatToInt8) {
            return false;
        }

        auto input3 = expr2->inputs()[0];
        auto expr3  = input3->expr().first;
        if (expr3->get() == nullptr) {
            return false;
        }
        if (expr3->get()->type() != OpType::OpType_ConvertTensor) {
            return false;
        }

        auto input4 = expr3->inputs()[0];
        auto expr4  = input4->expr().first;
        if (expr4->get() == nullptr) {
            return false;
        }

        auto input5 = expr4->inputs()[0];
        auto expr5  = input5->expr().first;
        if (expr5->get() == nullptr) {
            return false;
        }

        auto input6 = expr5->inputs()[0];
        auto expr6  = input6->expr().first;
        if (expr6->get() == nullptr) {
            return false;
        }

        int squeezeIndex = 0;
        if (expr4->get()->type() == OpType::OpType_Squeeze) {
            squeezeIndex = 0;
        } else if (expr5->get()->type() == OpType::OpType_Squeeze) {
            squeezeIndex = 1;
        } else if (expr6->get()->type() == OpType::OpType_Squeeze) {
            squeezeIndex = 2;
        } else {
            return false; // should find squeeze in 3 steps after OpType_ConvertTensor
        }

        EXPRP squeezeExpr = nullptr;
        if (squeezeIndex == 0) {
            squeezeExpr = expr4;
        }
        if (squeezeIndex == 1) {
            squeezeExpr = expr5;
            if (expr4->get()->type() == OpType_BinaryOp) {
                auto binaryOp     = expr4->get();
                auto binaryParams = binaryOp->main_as_BinaryOp();
                if (binaryParams->opType() != BinaryOpOperation_ADD) {
                    return false;
                }
            } else if ((expr4->get()->type() != OpType_ReLU) && (expr4->get()->type() != OpType_ReLU6)) {
                return false;
            } else {
                // pass
            }
        }
        if (squeezeIndex == 2) {
            squeezeExpr = expr6;
            if ((expr4->get()->type() != OpType_ReLU) && (expr4->get()->type() != OpType_ReLU6)) {
                return false;
            }

            if (expr5->get()->type() != OpType_BinaryOp) {
                return false;
            }
            auto binaryOp     = expr5->get();
            auto binaryParams = binaryOp->main_as_BinaryOp();
            if (binaryParams->opType() != BinaryOpOperation_ADD) {
                return false;
            }
        }

        auto input7 = squeezeExpr->inputs()[0];
        auto expr7  = input7->expr().first;
        if (expr7->get() == nullptr) {
            return false;
        }
        if (expr7->get()->type() != OpType::OpType_Convolution) {
            return false;
        }

        auto input8 = expr7->inputs()[0];
        auto expr8  = input8->expr().first;
        if (expr8->get() == nullptr) {
            return false;
        }
        if (expr8->get()->type() != OpType_ConvertTensor) {
            return false;
        }

        auto input9 = expr8->inputs()[0];
        auto expr9  = input9->expr().first;
        if (expr9->get() == nullptr) {
            return false;
        }
        if (expr9->get()->type() != OpType_Int8ToFloat) {
            return false;
        }

        auto input10 = expr9->inputs()[0];
        auto expr10  = input10->expr().first;
        if (expr10->get() == nullptr) {
            return false;
        }
        if (expr10->get()->type() != OpType_FloatToInt8) {
            return false;
        }

        auto input11 = expr10->inputs()[0];
        auto expr11  = input11->expr().first;
        if (expr11->get() == nullptr) {
            return false;
        }
        if (expr11->get()->type() != OpType_ConvertTensor) {
            return false;
        }

        auto input12 = expr11->inputs()[0];
        auto expr12  = input12->expr().first;
        if (expr12->get() == nullptr) {
            return false;
        }
        if (expr12->get()->type() != OpType_ExpandDims) {
            return false;
        }

        return true;
    };

    auto transform = [](EXPRP expr) {
        // OpType_Int8ToFloat
        auto input1 = expr->inputs()[0];
        auto expr1  = input1->expr().first;
        // OpType_FloatToInt8
        auto input2 = expr1->inputs()[0];
        auto expr2  = input2->expr().first;
        // OpType_ConvertTensor
        auto input3 = expr2->inputs()[0];
        auto expr3  = input3->expr().first;

        auto input4 = expr3->inputs()[0];
        auto expr4  = input4->expr().first;

        auto input5 = expr4->inputs()[0];
        auto expr5  = input5->expr().first;

        auto input6 = expr5->inputs()[0];
        auto expr6  = input6->expr().first;

        int squeezeIndex = 0;
        if (expr4->get()->type() == OpType::OpType_Squeeze) {
            squeezeIndex = 0;
        } else if (expr5->get()->type() == OpType::OpType_Squeeze) {
            squeezeIndex = 1;
        } else if (expr6->get()->type() == OpType::OpType_Squeeze) {
            squeezeIndex = 2;
        } else {
            // should find squeeze in 3 steps after OpType_ConvertTensor
        }

        EXPRP squeezeExpr = nullptr;
        if (squeezeIndex == 0) {
            squeezeExpr = expr4;
        }
        if (squeezeIndex == 1) {
            squeezeExpr = expr5;
            // then, expr4 should be relu or relu6 or bias_add
        }
        if (squeezeIndex == 2) {
            squeezeExpr = expr6;
            // then, expr4 should be relu or relu6
            // expr5 should be bias_add
        }

        // OpType_Convolution
        auto input7 = squeezeExpr->inputs()[0];
        auto expr7  = input7->expr().first;
        // OpType_ConvertTensor
        auto input8 = expr7->inputs()[0];
        auto expr8  = input8->expr().first;
        // OpType_Int8ToFloat
        auto input9 = expr8->inputs()[0];
        auto expr9  = input9->expr().first;
        // OpType_FloatToInt8
        auto input10 = expr9->inputs()[0];
        auto expr10  = input10->expr().first;
        // OpType_ConvertTensor
        auto input11 = expr10->inputs()[0];
        auto expr11  = input11->expr().first;
        // OpType_ExpandDims
        auto input12 = expr11->inputs()[0];
        auto expr12  = input12->expr().first;

        // now, begin reorder
        auto convInput = expr7->inputs()[0];

        std::unique_ptr<OpT> op7(expr7->get()->UnPack());
        auto newExpr7 = Expr::create(op7.get(), {convInput});
        newExpr7->setName(expr7->name());
        auto output = Variable::create(newExpr7);
        output->setName(expr7->outputName(0));

        if (squeezeIndex == 0) {
            // pass
        }
        if (squeezeIndex == 1) {
            std::vector<VARP> inputs{output};
            if (expr4->get()->type() == OpType_BinaryOp) {
                inputs.push_back(expr4->inputs().at(1));
            }
            std::unique_ptr<OpT> op4(expr4->get()->UnPack());
            auto newExpr4 = Expr::create(op4.get(), inputs);
            newExpr4->setName(expr4->name());
            output = Variable::create(newExpr4);
            output->setName(expr4->outputName(0));
        }
        if (squeezeIndex == 2) {
            std::unique_ptr<OpT> op5(expr5->get()->UnPack());
            auto newExpr5 = Expr::create(op5.get(), {output, expr5->inputs().at(1)});
            newExpr5->setName(expr5->name());
            output = Variable::create(newExpr5);
            output->setName(expr5->outputName(0));

            std::unique_ptr<OpT> op4(expr4->get()->UnPack());
            auto newExpr4 = Expr::create(op4.get(), {output});
            newExpr4->setName(expr4->name());
            output = Variable::create(newExpr4);
            output->setName(expr4->outputName(0));
        }

        std::unique_ptr<OpT> op3(expr3->get()->UnPack());
        auto newExpr3 = Expr::create(op3.get(), {output});
        newExpr3->setName(expr3->name());
        output = Variable::create(newExpr3);
        output->setName(expr3->outputName(0));

        std::unique_ptr<OpT> op2(expr2->get()->UnPack());
        auto newExpr2 = Expr::create(op2.get(), {output});
        newExpr2->setName(expr2->name());
        output = Variable::create(newExpr2);
        output->setName(expr2->outputName(0));

        std::unique_ptr<OpT> op1(expr1->get()->UnPack());
        auto newExpr1 = Expr::create(op1.get(), {output});
        newExpr1->setName(expr1->name());
        output = Variable::create(newExpr1);
        output->setName(expr1->outputName(0));

        std::unique_ptr<OpT> op(expr->get()->UnPack());
        auto newExpr = Expr::create(op.get(), {output});
        newExpr->setName(expr->name());
        output = Variable::create(newExpr);
        output->setName(expr->outputName(0));

        std::unique_ptr<OpT> squeezeOp(squeezeExpr->get()->UnPack());
        auto newSqueezeExpr = Expr::create(squeezeOp.get(), {output});
        newSqueezeExpr->setName(squeezeExpr->name());
        output = Variable::create(newSqueezeExpr);
        output->setName(squeezeExpr->outputName(0));

        Expr::replace(expr, output->expr().first);
        return true;
    };

    TemplateMerge::getInstance("Merge").insertTemplate("Conv1dQuantToConv2dQuant", match, transform,
                                                       PASS_PRIORITY_HIGH);
    return true;
}();

}
} // namespace MNN
