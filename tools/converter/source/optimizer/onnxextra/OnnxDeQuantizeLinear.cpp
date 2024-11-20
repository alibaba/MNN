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

static VARP _Int8ToFloat(VARP x, VARP scale, VARP zero) {
    MNN_ASSERT(scale->getInfo() && zero->getInfo());
    MNN_ASSERT(scale->getInfo()->size == zero->getInfo()->size || zero->getInfo()->size <= 1);
    auto size = 1;
    if (scale->getInfo()->size > 1) {
        size = scale->getInfo()->size;
    }
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_Int8ToFloat;
    op->main.type = OpParameter_QuantizedFloatParam;
    op->main.value = new QuantizedFloatParamT;
    op->main.AsQuantizedFloatParam()->tensorScale.resize(size);
    if (scale->readMap<float>()) {
        ::memcpy(op->main.AsQuantizedFloatParam()->tensorScale.data(), scale->readMap<float>(), size * sizeof(float));
    }
    op->main.AsQuantizedFloatParam()->floatzeros.resize(size);
    if (zero->readMap<float>()) {
        auto zerosize = 1;
        if (zero->getInfo()->size > 1) {
            zerosize = zero->getInfo()->size;
        }
        ::memcpy(op->main.AsQuantizedFloatParam()->floatzeros.data(), zero->readMap<float>(), zerosize * sizeof(float));
    }
    return Variable::create(Expr::create(op.get(), {x}));
}

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
        bool int32Dequant = false;
        auto input = inputs[0];
        auto scale = inputs[1];

        auto dataType = halide_type_int;
        VARP zeropoint = _Const(0.f);
        if (inputs.size() > 2) {
            if (inputs[2]->getInfo()) {
                dataType = static_cast<halide_type_code_t>(inputs[2]->getInfo()->type.code);
            }
            zeropoint = _Cast<float>(inputs[2]);

        }
        
        std::vector<int32_t> inputDim = {};
        if (input->getInfo()) {
            inputDim = input->getInfo()->dim;
            dataType = static_cast<halide_type_code_t>(input->getInfo()->type.code);
            if (input->getInfo()->type.bits == 32) {
                // from onnx document.
                auto floatinput = _Cast<float>(input);
                auto output = floatinput * scale;
                output->expr().first->setName(expr->name());
                return output->expr().first;
            }
            if (dataType == halide_type_uint && input->readMap<uint8_t>()) {
                auto floatinput = _Cast<float>(input);
                auto output = (floatinput - zeropoint) * scale;
                output->expr().first->setName(expr->name());
                return output->expr().first;
            }
        }
        auto offset = _Const(0.f);
        if (dataType == halide_type_uint) {
            offset = _Const(128.f);
        }
        std::unique_ptr<MNN::OpT> iden(new MNN::OpT);
        iden->type = OpType_Int8ToFloat;

        if (input->getInfo() && input->getInfo()->dim.size() == 4) { // convolution weight
            auto shape_ = input->getInfo()->dim;
            int size = scale->getInfo()->dim[0];
            // [oc,ic,kx,ky] -> [ic,oc,kx,ky]
            auto x = _Permute(input, {1, 0, 2, 3});
            auto y = _Int8ToFloat(x, scale, zeropoint - offset);
            y->expr().first->setName(expr->name());
            return y->expr().first;
        }
        if (scale->readMap<float>() && input->getInfo() && input->getInfo()->type.bits == 8) { // matmul B const
            auto newvar = _Int8ToFloat(input, scale, (zeropoint- offset));
            newvar->expr().first->setName(expr->name());
            return newvar->expr().first;
        }
        
        if (scale->readMap<float>() == nullptr) { // dynamic layer's input
            auto int8ToFloatvar = _Int8ToFloat(input, _Const(1.0f), _Const(0.f));
            auto output = (int8ToFloatvar - zeropoint) * scale;
            output->expr().first->setName(expr->name());
            return output->expr().first;
        }
        auto newvar = _Int8ToFloat(input, scale, (zeropoint- offset));
        newvar->expr().first->setName(expr->name());
        return newvar->expr().first;
        
        
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("DequantizeLinear",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxDequantizeLinearTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
