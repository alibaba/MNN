//
//  TFPrelu.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TFliteExtraManager.hpp"
#include "MNN_generated.h"


namespace MNN {
namespace Express {

/*See ConvolutionTflite.cpp for detail attribute*/
class FCTransform : public TFliteExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        MNN_ASSERT(inputs.size() == 3);
        auto input = inputs[0];
        auto weight = inputs[1];
        auto bias = inputs[2];
        auto newOutput = _MatMul(input, weight, false, true) + bias;
        newOutput->setName(expr->name());
        return newOutput->expr().first;
    }
};
static auto gRegister = []() {
    TFliteExtraManager::get()->insert("FULL_CONNECT", std::shared_ptr<TFliteExtraManager::Transform>(new FCTransform));
    return true;
}();
}
} // namespace MNN
