//
//  FullConnect.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "../../tflite/liteOpConverter.hpp"
#include "TFliteExtraManager.hpp"

namespace MNN {
namespace Express {

/*See ConvolutionTflite.cpp for detail attribute*/
class FCTransform : public TFliteExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto extra = expr->get()->main_as_Extra();
        tflite::ActivationFunctionType activation = tflite::ActivationFunctionType_NONE;
        if (nullptr != extra) {
            if (nullptr != extra->attr()) {
                for (int i=0; i<extra->attr()->size(); ++i) {
                    auto attr = extra->attr()->GetAs<Attribute>(i);
                    if (attr->key()->str() == "fused_activation_function") {
                        activation = (tflite::ActivationFunctionType)attr->i();
                    }
                }
            }
        }
        MNN_ASSERT(inputs.size() == 3);
        auto input     = inputs[0];
        auto weight    = inputs[1];
        auto bias      = inputs[2];
        input = _Reshape(input, {0, -1}, NHWC);
        auto newOutput = _MatMul(input, weight, false, true) + bias;
        if (activation == tflite::ActivationFunctionType_RELU) {
            newOutput = _Relu(newOutput);
        } else if (activation == tflite::ActivationFunctionType_RELU6) {
            newOutput = _Relu6(newOutput);
        }
        newOutput->setName(expr->name());
        return newOutput->expr().first;
    }
};
static auto gRegister = []() {
    TFliteExtraManager::get()->insert("FULL_CONNECT", std::shared_ptr<TFliteExtraManager::Transform>(new FCTransform));
    return true;
}();
} // namespace Express
} // namespace MNN
