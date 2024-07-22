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

        uint8_t dataType = halide_type_int;
        VARP zeropoint = _Const(0.f);
        if (inputs.size() > 2) {
            if (inputs[2]->getInfo() == nullptr) {
                MNN_ERROR("DequantizeLinear layer inputs.size>2, but zeroPoint is not const\n");
            }
            MNN_ASSERT(inputs[2]->getInfo() != nullptr);
            auto zeroDim = inputs[2]->getInfo()->dim;
            dataType = inputs[2]->getInfo()->type.code;
            std::vector<float> fp32Zero(inputs[2]->getInfo()->size);
            if (dataType == halide_type_int) {
                const int8_t* zeroPtr = inputs[2]->readMap<int8_t>();
                for (int j = 0; j < fp32Zero.size(); ++j) {
                    fp32Zero[j] = static_cast<float>(zeroPtr[j]);
                }
                zeropoint = _Const(fp32Zero.data(), zeroDim, inputs[2]->getInfo()->order, halide_type_of<float>());
            } else {
                const uint8_t* zeroPtr = inputs[2]->readMap<uint8_t>();
                for (int j = 0; j < fp32Zero.size(); ++j) {
                    fp32Zero[j] = static_cast<float>(zeroPtr[j]) - 128.f;
                }
                zeropoint = _Const(fp32Zero.data(), zeroDim, inputs[2]->getInfo()->order, halide_type_of<float>());
            }
            zeropoint = _Cast<float>(inputs[2]);
        }
        
        std::vector<int32_t> inputDim = {};
        if (input->getInfo()) {
            inputDim = input->getInfo()->dim;
            dataType = input->getInfo()->type.code;
        }
        auto offset = _Const(0.f);
        if (dataType == halide_type_uint) {
            offset = _Const(128.f);
        }
        // if (!scale->getInfo()->dim.empty()) {
        //     zeropoint = _Unsqueeze(zeropoint, {1,2,3});
        //     scale = _Unsqueeze(scale, {1, 2, 3});
        // } else {
        //     scale = _Reshape(scale, {1});
        //     zeropoint = _Reshape(zeropoint, {1});
        // }
        auto _shape  = _Const(inputDim.data(), {static_cast<int32_t>(inputDim.size())}, NHWC, halide_type_of<int>());
        auto output = (_Cast<float>(input) - zeropoint) * scale;
        std::unique_ptr<MNN::OpT> iden(new MNN::OpT);
        iden->type = OpType_Int8ToFloat;

        auto newExpr = MNN::Express::Expr::create(iden.get(), {input, output, scale, zeropoint - offset, _shape}, 5);
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
