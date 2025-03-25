//
//  OnnxQuantizeLinear.cpp
//  MNNConverter
//
//  Created by MNN on 2023/03/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

static VARP _Float2Int8(VARP x, VARP scale, VARP zero) {
    int size = 1;
    if (scale->getInfo()->size > 1) {
        size = scale->getInfo()->size;
    }
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_FloatToInt8;
    op->main.type = OpParameter_QuantizedFloatParam;
    op->main.value = new QuantizedFloatParamT;
    op->main.AsQuantizedFloatParam()->tensorScale.resize(size);
    op->main.AsQuantizedFloatParam()->floatzeros.resize(size);
    ::memcpy(op->main.AsQuantizedFloatParam()->tensorScale.data(), scale->readMap<float>(), size * sizeof(float));
    ::memcpy(op->main.AsQuantizedFloatParam()->floatzeros.data(), zero->readMap<float>(), size * sizeof(float));
    return Variable::create(Expr::create(op.get(), {x}));
}

/* Given a float input value x, it quantizes x to corresponding int8 value quant_x using scales and zeroPoint. */
class OnnxQuantizeLinearTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);
        auto inputs = expr->inputs();
        if (inputs.size() < 2) {
            MNN_ERROR("Onnx QuantizeLinear input error: inputs size<2\n");
            return nullptr;
        }
        auto input = inputs[0];
        auto scale = inputs[1];
        auto dataType = halide_type_int;
        VARP zeropoint = _Const(0.f);
        auto offset = _Const(0.f);
        if (inputs.size() > 2) {
            zeropoint = _Cast<float>(inputs[2]);
            if (inputs[2]->getInfo()) {
                dataType = static_cast<halide_type_code_t>(inputs[2]->getInfo()->type.code);
            }
        }
        if (dataType == halide_type_uint) {
            offset = _Const(128.f);
        }
        MNN_ASSERT(scale->readMap<float>() != nullptr);
        auto newvar = _Float2Int8(input, _Reciprocal(scale), zeropoint - offset);
        newvar->expr().first->setName(expr->name());
        return newvar->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("QuantizeLinear",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxQuantizeLinearTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
