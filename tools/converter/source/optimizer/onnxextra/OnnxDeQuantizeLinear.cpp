//
//  OnnxDequantizeLinear.cpp
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

class OnnxDequantizeLinearTransform : public OnnxExtraManager::Transform {
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
        VARP zeropoint = nullptr;
        if (inputs.size() > 2) {
            zeropoint = inputs[2];
        }
        
        std::vector<int32_t> inputDim = {};
        if (input->getInfo()) {
            inputDim = input->getInfo()->dim;
        }
        if (!scale->getInfo()->dim.empty()) {
            zeropoint = _Unsqueeze(zeropoint, {1,2,3});
            scale = _Unsqueeze(scale, {1, 2, 3});
        } else {
            scale = _Reshape(scale, {1});
            zeropoint = _Reshape(zeropoint, {1});
        }
        auto _shape  = _Const(inputDim.data(), {static_cast<int32_t>(inputDim.size())}, NHWC, halide_type_of<int>());
        auto output = (_Cast<float>(input) - _Cast<float>(zeropoint)) * scale;
        std::unique_ptr<MNN::OpT> iden(new MNN::OpT);
        iden->type = OpType_Int8ToFloat;

        auto newExpr = MNN::Express::Expr::create(iden.get(), {input, output, scale, _Cast<float>(zeropoint), _shape}, 5);
        newExpr->setName(expr->name());
        return newExpr;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("DequantizeLinear",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxDequantizeLinearTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
