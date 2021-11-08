//
//  EmbedTransform.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <vector>
#include "CaffeExtraManager.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace Express {

class EmbedTransform : public CaffeExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op         = expr->get();
        auto inputs     = expr->inputs();
        auto numOutput  = op->main_as_Extra()->attr()->GetAs<Attribute>(0)->i();
        auto inputDim   = op->main_as_Extra()->attr()->GetAs<Attribute>(1)->i();
        auto biasTerm   = op->main_as_Extra()->attr()->GetAs<Attribute>(2)->b();
        auto weightsPtr = op->main_as_Extra()->attr()->GetAs<Attribute>(3)->tensor()->float32s()->data();
        auto shape      = op->main_as_Extra()->attr()->GetAs<Attribute>(3)->tensor()->dims();
        MNN_ASSERT(shape->size() == 2);

        auto oneHot = _OneHot(_Cast(inputs[0], halide_type_of<int>()), _Cast(_Const(inputDim), halide_type_of<int>()),
                              _Const(1.0f), _Const(0.0f));
        auto weight = _Const(weightsPtr, {shape->data()[0], shape->data()[1]}, NCHW);
        auto xW     = _MatMul(oneHot, weight);

        if (!biasTerm) {
            return xW->expr().first;
        } else {
            auto biasPtr   = op->main_as_Extra()->attr()->GetAs<Attribute>(4)->tensor()->float32s()->data();
            auto biasShape = op->main_as_Extra()->attr()->GetAs<Attribute>(4)->tensor()->dims();
            MNN_ASSERT(biasShape->size() == 2); // in caffe source code embed_layer.cpp, bias_shape is 1 * num_output
            auto bias   = _Const(biasPtr, {1, biasShape->data()[1]}, NCHW);
            auto output = _Add(xW, bias);

            return output->expr().first;
        }
    }
};

static auto gRegister = []() {
    CaffeExtraManager::get()->insert("Embed", std::shared_ptr<CaffeExtraManager::Transform>(new EmbedTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
