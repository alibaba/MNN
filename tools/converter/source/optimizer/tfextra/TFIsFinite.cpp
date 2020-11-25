//
//  TFIsFinite.cpp
//  MNNConverter
//
//  Created by MNN on 2020/05/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TFExtraManager.hpp"

namespace MNN {
namespace Express {

class IsFiniteTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        MNN_THROW_CHECK(expr->inputs().size() == 1, "Tf IsFinite needs one inputs.");
        auto attrs = expr->get()->main_as_Extra()->attr();
        auto it =
            std::find_if(attrs->begin(), attrs->end(), [](const Attribute* attr) { return attr->key()->str() == "T"; });
        MNN_ASSERT(it != attrs->end());
        // IsFinit only support float data type, such as float16, bfloat16,
        // float32 and float64.
        MNN_ASSERT(it->type() == DataType_DT_FLOAT);

        auto input = expr->inputs()[0];
        auto op    = expr->get();

        std::unique_ptr<MNN::OpT> lessOp(new OpT);
        lessOp->type       = OpType_BinaryOp;
        lessOp->name       = op->name()->str();
        lessOp->main.type  = OpParameter_BinaryOp;
        lessOp->main.value = new BinaryOpT;

        auto* param   = lessOp->main.AsBinaryOp();
        param->opType = MNN::BinaryOpOperation_LESS;
        param->T      = DataType_DT_FLOAT;

        auto finite = _Scalar(std::numeric_limits<float>::max() - 0.1f);
        finite->setName(lessOp->name + "_const_finite");
        return Expr::create(lessOp.get(), {input, finite}, 1);
    }
};

static auto gRegister = []() {
    TFExtraManager::get()->insert("IsFinite", std::shared_ptr<TFExtraManager::Transform>(new IsFiniteTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
