//
//  OnnxSTFT.cpp
//  MNNConverter
//
//  Created by MNN on 2025/01/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/**
 Ref from onnxruntime stft_decomposition.cc
 */

#include <numeric>

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {
/*
 Subgraph pattern 1: STFT with optional Window parameter set
           [root]--(signal)--------------------+
           [root]--(frame_step)---------------+|
           [root]--(window)------------------+||
           [root]--(frame_length) ----------+|||
                                            ||||
                                            vvvv
                                           [STFT]--(output)-->
 After Fusion:
           [root]--(signal)-------------------------+
           [root]                                   |
           [root]--(window)--+                      |
           [root]            |                      |
                             v                      v
      (only for non-fp32) [Cast]             +--[Reshape]
                             |               |      |
                             v               |      v
                         [Reshape]-->[Mul]---|-->[Conv]-------+
                             |               |                |
                             |               +-----|          |
                             |                     v          v
                             +------>[Mul]------>[Conv]-->[Concat]-->[Reshape]-->[Transpose]--(output)-->

 Subgraph pattern 2: STFT without optional Window parameter set
          [root]--(signal)-------------------+
          [root]--(frame_step)--------------+|
          [root]                             |
          [root]--(frame_length) ----------+||
                                           |||
                                           vvv
                                          [STFT]--(output)-->
After Fusion:
          [root]--(signal)-->[Reshape]-->[Conv]
          [root]                 |         |
          [root]                 |         v
          [root]                 +------>[Conv]-->[Concat]-->[Reshape]-->[Transpose]--(output)-->
*/
class OnnxSTFTTransformer : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        int outside = 1;
        auto attrs = expr->get()->main_as_Extra()->attr();
        if (nullptr != attrs) {
            for (int i=0; i<attrs->size(); ++i) {
                auto attr = attrs->GetAs<Attribute>(i);
                if (attr->key()->str() == "onesided") {
                    outside = attr->i();
                    break;
                }
            }
        }
        int dft_size = 0;
        std::vector<VARP> validInputs;
        if (inputs[2] == nullptr) {
            auto frameLengthVar = inputs[3];
            // Make Virtual window
            auto shapeExpand = _Unsqueeze(frameLengthVar, {0});
            auto windows = _Fill(shapeExpand, _Scalar<float>(1.0f));
            validInputs = {inputs[0], inputs[1], windows};
        } else {
            validInputs = {inputs[0], inputs[1], inputs[2]};
        }
        std::unique_ptr<MNN::OpT> op(new MNN::OpT);
        op->type = OpType_Stft;
        op->main.value = new StftParamT;
        op->main.type = OpParameter_StftParam;
        op->main.AsStftParam()->hop_length = dft_size;
        op->main.AsStftParam()->abs = outside;
        auto newExpr = Expr::create(op.get(), validInputs, expr->outputSize());
        newExpr->setName(expr->name());
        
        return newExpr;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("STFT", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxSTFTTransformer));
    return true;
}();

} // namespace Express
} // namespace MNN
