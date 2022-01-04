//
//  OnnxLSTMMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/11/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxLSTMTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        MNN_ASSERT(inputs.size() >= 4);
        std::unique_ptr<OpT> lstm(new OpT);
        lstm->name       = expr->name();
        if (expr->get()->main_as_Extra()->type()->str() == "RNN") {
            lstm->type = OpType_RNN;
        } else {
            lstm->type = OpType_LSTM;
        }
        lstm->main.type  = OpParameter_LSTM;
        lstm->main.value = new LSTMT;
        {
            auto extra = expr->get()->main_as_Extra();
            auto attr  = extra->attr();
            if (nullptr != attr) {
                for (int i = 0; i < attr->size(); ++i) {
                    auto attUnit = attr->GetAs<Attribute>(i);
                    if (attUnit->key()->str() == "hidden_size") {
                        lstm->main.AsLSTM()->outputCount = attUnit->i();
                    }
                }
            }
        }
        // onnx docs guarantee bias shape is [num_direction, 8 * hidden_size], we split it to 2x [num_dicection, 4 * hidden_size] (W/R), then add together
        auto biasWR = _Split(inputs[3], {2}, 1);
        inputs[3] = _Add(biasWR[0], biasWR[1]);
        // Y, Y_h, Y_c
        auto originLSTM = Expr::create(lstm.get(), inputs, (lstm->type == OpType_RNN ? 2 : 3));
        originLSTM->setName(expr->name());
        for (int i = 0; i < expr->outputSize(); ++i) {
            Variable::create(originLSTM, i)->setName(expr->outputName(i));
        }
        return originLSTM;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("LSTM", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxLSTMTransform));
    OnnxExtraManager::get()->insert("RNN", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxLSTMTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
