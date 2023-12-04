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

static VARP _DequantizeLinear(VARP x, VARP scales, VARP zeroPoints) {
     auto scaleInfo = scales->getInfo();
     auto zeroInfo = zeroPoints->getInfo();
     INTS zeroShape;
     auto& scaleShape = scaleInfo->dim; // e.g [1,1,D,1], D>1
     if (nullptr != zeroInfo) {
         zeroShape = zeroInfo->dim;   // e.g [1,1,D,1], D>1
     }
     if (zeroInfo && scaleShape.size() != zeroShape.size()) {
         MNN_ERROR("QuantizeLinear scale and zeroPoints should be the same shape!\n");
         return nullptr;
     }
     int scaleSize = 1, zeroSize = 1;
     int scaleAxis = 0, zeroAxis = 0;
     for (int d = 0; d < scaleShape.size(); ++d) {
         if (scaleShape[d] > scaleSize) {
             scaleSize = scaleShape[d];
             scaleAxis = d;
         }
         if (zeroInfo && zeroShape[d] > zeroSize) {
             zeroSize = zeroShape[d];
             zeroAxis = d;
         }
     }
     if (zeroInfo && (scaleSize != zeroSize || scaleAxis != zeroAxis)) {
         MNN_ERROR("QuantizeLinear scale and zeroPoints should be the same size and same axis!\n");
         return nullptr;
     }

     std::unique_ptr<OpT> dequantizeLinear(new OpT);
     dequantizeLinear->type = OpType_DequantizeLinear;
     dequantizeLinear->main.type = OpParameter_DequantizeLinear;
     dequantizeLinear->main.value = new DequantizeLinearT;
     dequantizeLinear->main.AsDequantizeLinear()->scaleSize = scaleSize;
     dequantizeLinear->main.AsDequantizeLinear()->scaleAxis = scaleAxis;
     return (Variable::create(Expr::create(dequantizeLinear.get(), {x, scales, zeroPoints})));
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
        auto output = _DequantizeLinear(input, scale, zeropoint);
        output->setName(expr->name());
        return output->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("DequantizeLinear",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxDequantizeLinearTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
