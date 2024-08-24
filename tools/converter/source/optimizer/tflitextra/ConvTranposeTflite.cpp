//
//  ConvTranposeTflite.cpp
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

/*See CustomTflite.cpp for detail attribute*/
class ConvTranposeTflite : public TFliteExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto weight = inputs[1];
        auto bias = inputs[2];
        weight = _Transpose(weight, {3, 0, 1, 2});
        auto weightInfo = weight->getInfo();
        auto biasInfo = bias->getInfo();
        
        auto extra = expr->get()->main_as_Extra();
        std::unique_ptr<MNN::OpT> deconvOp(flatbuffers::GetRoot<MNN::Op>(extra->info()->data())->UnPack());
        auto weightPtr = weight->readMap<float>();
        auto biasPtr = bias->readMap<float>();
        EXPRP newExpr;
        if (nullptr == weightPtr || nullptr == biasPtr) {
            newExpr = Expr::create(deconvOp.get(), {inputs[0], weight, bias});
        } else {
            auto conv = deconvOp->main.AsConvolution2D();
            conv->weight.resize(weightInfo->size);
            ::memcpy(conv->weight.data(), weightPtr, weightInfo->size * sizeof(float));
            conv->bias.resize(biasInfo->size);
            ::memcpy(conv->bias.data(), biasPtr, biasInfo->size * sizeof(float));
            newExpr = Expr::create(deconvOp.get(), {inputs[0]});
        }
        auto newOutput = Variable::create(newExpr);
        newOutput->setName(expr->name());
        return newOutput->expr().first;
    }
};
static auto gRegister = []() {
    TFliteExtraManager::get()->insert("Convolution2DTransposeBias", std::shared_ptr<TFliteExtraManager::Transform>(new ConvTranposeTflite));
    return true;
}();
} // namespace Express
} // namespace MNN
