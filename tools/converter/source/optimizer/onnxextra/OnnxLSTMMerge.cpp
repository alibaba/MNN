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
        lstm->type       = OpType_LSTM;
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
        auto hiddenSize = lstm->main.AsLSTM()->outputCount;
        {
            // Merge Bias
            auto bias = inputs[3];
            auto info = bias->getInfo();
            auto ptr  = bias->readMap<float>();
            if (nullptr == info || nullptr == ptr) {
                MNN_ERROR("Can't solve LSTM because bias is not const\n");
                return nullptr;
            }
            if (8 * hiddenSize == info->dim[1]) {
                std::vector<float> biasVector(hiddenSize * 4 * info->dim[0]);
                for (int z = 0; z < info->dim[0]; ++z) {
                    auto src = ptr + z * info->dim[1];
                    auto dst = biasVector.data() + z * 4 * hiddenSize;
                    for (int i = 0; i < hiddenSize * 4; ++i) {
                        dst[i] = src[i] + src[i + hiddenSize * 4];
                    }
                }
                auto newBias = _Const(biasVector.data(), {info->dim[0], hiddenSize * 4}, NCHW);
                inputs[3]    = newBias;
            }
        }
        // Y, Y_h, Y_c
        auto originLSTM = Expr::create(lstm.get(), inputs, 3);
        originLSTM->setName(expr->name());
        return originLSTM;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("LSTM", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxLSTMTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
