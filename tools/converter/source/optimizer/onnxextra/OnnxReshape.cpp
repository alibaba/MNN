//
//  OnnxReshape.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxReshapeTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();

        MNN_CHECK(inputs.size() == 2, "Onnx Reshape should have 2 inputs!");

        auto shape        = inputs[1];
        auto shapeDataPtr = shape->readMap<int32_t>();

        std::unique_ptr<OpT> mergedReshape(new OpT);
        mergedReshape->name = expr->name();
        mergedReshape->type      = OpType_Reshape;
        mergedReshape->main.type = OpParameter_Reshape;

        std::unique_ptr<ReshapeT> reshapeParam(new ReshapeT);
        reshapeParam->dimType = MNN::MNN_DATA_FORMAT_NCHW;

        if (true) {
            mergedReshape->main.value = reshapeParam.release();
            return Expr::create(mergedReshape.get(), {inputs[0], inputs[1]});
        }

        // shape is Constant
        auto shapeInfo = shape->getInfo();
        MNN_CHECK(shapeInfo != nullptr, "The Second input of Reshape shoud be Constant!");

        const int dimSize = shapeInfo->size;
        reshapeParam->dims.resize(dimSize);
        memcpy(reshapeParam->dims.data(), shapeDataPtr, dimSize * sizeof(int32_t));

        mergedReshape->main.value = reshapeParam.release();

        auto newExpr = Expr::create(mergedReshape.get(), {inputs[0]});
        newExpr->setName(expr->name());
        return newExpr;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Reshape", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxReshapeTransform));
    return true;
}();

} // namespace Express

} // namespace MNN
