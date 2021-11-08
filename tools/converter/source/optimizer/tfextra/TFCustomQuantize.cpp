//
//  TFCustomQuantize.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../merge/MergeHelpers.hpp"
#include "MNN_generated.h"
#include "TFExtraManager.hpp"
#include <string>

namespace MNN {
namespace Express {

class CustomQuantizeTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        MNN_ASSERT(nullptr != op->main_as_Extra());
        auto attr = op->main_as_Extra()->attr();

        int nbit         = 8;
        float zero_point = 0.f;
        float clamp_min = -128.0f;
        float clamp_max = 127.0f;
        int method;
        std::vector<float> scale_val;
        for (int i = 0; i < attr->size(); ++i) {
            auto attr_value = attr->GetAs<Attribute>(i);
            if (attr_value->key()->str() == "nbit") {
                nbit = attr_value->i();
            }
            if (attr_value->key()->str() == "zero_point") {
                zero_point = attr_value->f();
            }
            if (attr_value->key()->str() == "scale") {
                auto* list_value = attr_value->list()->f();
                scale_val.resize(list_value->size());
                for (int i = 0; i < list_value->size(); ++i) {
                    scale_val[i] = list_value->Get(i);
                }
            }
            if (attr_value->key()->str() == "clamp_min") {
                clamp_min = attr_value->f();
            }
            if (attr_value->key()->str() == "clamp_max") {
                clamp_max = attr_value->f();
            }
            if (attr_value->key()->str() == "method") {
                method = attr_value->i();
            }
        }

        VARP input = helpers::ConvertLayout(inputs[0], NC4HW4, NHWC);
        input->setName(expr->name() + "_convert_to_nc4hw4");
        input->expr().first->setName(input->name());

        std::unique_ptr<OpT> quant_op(new OpT);
        quant_op->type       = OpType_FloatToInt8;
        quant_op->main.type  = OpParameter_QuantizedFloatParam;
        quant_op->main.value = new QuantizedFloatParamT;

        auto* quant_param  = quant_op->main.AsQuantizedFloatParam();
        quant_param->nbits = nbit;
        quant_param->zeroPoint = zero_point;
        quant_param->clampMin = clamp_min;
        quant_param->clampMax = clamp_max;
        quant_param->method = MNN::QuantizeAlgo(method);
        quant_param->tensorScale = {scale_val[0]};
        EXPRP quant_expr = Expr::create(quant_op.get(), {input});
        quant_expr->setName(expr->name());
        return quant_expr;
    }
};

static auto gRegister = []() {
    TFExtraManager::get()->insert("CustomQuantize",
                                  std::shared_ptr<TFExtraManager::Transform>(new CustomQuantizeTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
