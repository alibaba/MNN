//
//  TFlitePrelu.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TFliteExtraManager.hpp"

namespace MNN {
namespace Express {

/*See ConvolutionTflite.cpp for detail attribute*/
class TFlitePrelu : public TFliteExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        MNN_ASSERT(inputs.size() == 2);
        auto input     = inputs[0];
        auto slope    = inputs[1];
        auto slopePtr = slope->readMap<float>();
        if (nullptr != slopePtr) {
            std::unique_ptr<OpT> op(new OpT);
            op->type = OpType_PReLU;
            op->main.type = OpParameter_PRelu;
            op->main.value = new PReluT;
            op->main.AsPRelu()->slope.resize(slope->getInfo()->size);
            ::memcpy(op->main.AsPRelu()->slope.data(), slopePtr, slope->getInfo()->size * sizeof(float));
            op->main.AsPRelu()->slopeCount = slope->getInfo()->size;
            auto outputExpr = Expr::create(op.get(), {input});
            outputExpr->setName(expr->name());
            return outputExpr;
        }
        MNN_ERROR("Don't support not const prelu of tflite currently\n");
        return nullptr;
    }
};
static auto gRegister = []() {
    TFliteExtraManager::get()->insert("PRELU", std::shared_ptr<TFliteExtraManager::Transform>(new TFlitePrelu));
    return true;
}();
} // namespace Express
} // namespace MNN
