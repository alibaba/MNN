//
//  OnnxPrelu.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxPreluTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        MNN_CHECK(inputs.size() == 2, "Onnx Prelu Should have 2 inputs!");

        auto slope     = inputs[1];
        auto slopeInfo = slope->getInfo();
        MNN_CHECK(slopeInfo != nullptr, "Slope should be Constant node!");

        const int slopeSize = slopeInfo->size;

        std::unique_ptr<PReluT> preluParam(new PReluT);

        preluParam->slopeCount = slopeSize;

        auto slopeData = slope->readMap<float>();
        MNN_CHECK(slopeData != nullptr, "Slope should be Constant node!");

        preluParam->slope.resize(slopeSize);
        memcpy(preluParam->slope.data(), slopeData, slopeSize * sizeof(float));

        // prelu(input, slope) => mergedPrelu(input)
        std::unique_ptr<OpT> mergedOp(new OpT);
        mergedOp->name = expr->name();
        mergedOp->type       = OpType_PReLU;
        mergedOp->main.type  = OpParameter_PRelu;
        mergedOp->main.value = preluParam.release();
        auto newExpr = Expr::create(mergedOp.get(), {_Convert(inputs[0], NC4HW4)});
        newExpr->setName(expr->name());
        auto res = _Convert(Variable::create(newExpr), NCHW);
        return res->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("PRelu", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxPreluTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
