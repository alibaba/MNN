//
//  OnnxConstantOfShape.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
namespace MNN {
namespace Express {

class OnnxConstantOfShapeTransform : public OnnxExtraManager::Transform{
public:
    virtual EXPRP onExecute(EXPRP expr) const override{
        auto inputs = expr->inputs();
        MNN_CHECK(1 == inputs.size(), "Onnx ConstantOfShape should have one input!");
        
        std::unique_ptr<OpT> mnnFill(new OpT);
        mnnFill->name = expr->name();
        mnnFill->defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
        mnnFill->type = OpType_Fill;
        mnnFill->main.type = OpParameter_Fill;
        mnnFill->main.value = nullptr;
        
        // get value from attribute
        float value = 0.0f;
        auto op = expr->get();
        auto extraParam = op->main_as_Extra();
        const int size  = extraParam->attr()->size();
        for (int i = 0; i < size; ++i) {
            auto attr       = extraParam->attr()->GetAs<Attribute>(i);
            const auto& key = attr->key()->str();
            if (key == "value") {
                auto blob = attr->tensor();
                MNN_CHECK(blob->float32s()->size() == 1, "Onnx ConstantOfShape value is tensor, defalut have one float value!");
                value = blob->float32s()->data()[0];
            }
        }
        auto theSecondInputOfFill = _Const(value, {1});
        
        return Expr::create(mnnFill.get(), {inputs[0], theSecondInputOfFill});
    }
};


static auto gRegister = [](){
    OnnxExtraManager::get()->insert("ConstantOfShape", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxConstantOfShapeTransform));
    return true;
}();

}
}
