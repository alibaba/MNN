//
//  TFCustomDequantize.cpp
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

class CustomDequantizeTransform : public TFExtraManager::Transform {
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
            if (attr_value->key()->str() == "clamp_min") {
                clamp_min = attr_value->f();
            }
            if (attr_value->key()->str() == "clamp_max") {
                clamp_max = attr_value->f();
            }
            if (attr_value->key()->str() == "method") {
                method = attr_value->i();
            }
            if (attr_value->key()->str() == "scale") {
                auto* list_value = attr_value->list()->f();
                scale_val.resize(list_value->size());
                for (int i = 0; i < list_value->size(); ++i) {
                    scale_val[i] = list_value->Get(i);
                }
            }
        }

        VARP input = inputs[0];
        std::unique_ptr<OpT> dequant_op(new OpT);
        dequant_op->type       = OpType_Int8ToFloat;
        dequant_op->main.type  = OpParameter_QuantizedFloatParam;
        dequant_op->main.value = new QuantizedFloatParamT;

        auto* dequant_param  = dequant_op->main.AsQuantizedFloatParam();
        dequant_param->nbits = nbit;
        dequant_param->clampMin = clamp_min;
        dequant_param->clampMax = clamp_max;
        dequant_param->method = MNN::QuantizeAlgo(method);
        dequant_param->tensorScale = {scale_val[0]};
        EXPRP dequant_expr = Expr::create(dequant_op.get(), {input});
        dequant_expr->setName(expr->name());

        VARP dequant = Variable::create(dequant_expr);
        dequant->setName(expr->name());
        VARP output = helpers::ConvertLayout(dequant, NHWC, NC4HW4);
        output->setName(expr->name() + "_convert_to_nhwc");
        output->expr().first->setName(output->name());
        return output->expr().first;
    }
};

static auto gRegister = []() {
    TFExtraManager::get()->insert("CustomDequantize",
                                  std::shared_ptr<TFExtraManager::Transform>(new CustomDequantizeTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
