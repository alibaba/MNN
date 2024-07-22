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
        
        if (nullptr == scale || nullptr == input) {
            MNN_ERROR("QuantizeLinear should provide scale and input\n");
            return nullptr;
        }
        uint8_t dataType = halide_type_int;
        VARP zeropoint = _Const(0.f);
        auto offset = _Const(0.f);
        if (inputs.size() > 2) {
            zeropoint = _Cast<float>(inputs[2]);
            dataType = inputs[2]->getInfo()->type.code;
        }
        if (dataType == halide_type_uint) {
            offset = _Const(128.f);
        }
        auto scaleReq = _Reciprocal(scale);
        // auto output = _Cast<int8_t>(_Round(_Relu6(_Round(input * scaleReq) + zeropoint, -128.0f, 127.0f)));
        auto output = _FloatToInt8(input, scaleReq, -128, 127, static_cast<int8_t>(zeropoint->readMap<float>()[0] - offset->readMap<float>()[0]));
        std::unique_ptr<MNN::OpT> iden(new MNN::OpT);
        iden->type = OpType_FloatToInt8;
        std::vector<int32_t> inputDim = {};
        
        if (input->getInfo()) {
            inputDim = input->getInfo()->dim;
        }
        auto _shape  = _Const(inputDim.data(), {static_cast<int32_t>(inputDim.size())}, NHWC, halide_type_of<int>());
        auto newExpr = MNN::Express::Expr::create(iden.get(), {input, output, scale, zeropoint - offset, _shape}, 5);
        newExpr->setName(expr->name());
        return newExpr;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("QuantizeLinear",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxQuantizeLinearTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
