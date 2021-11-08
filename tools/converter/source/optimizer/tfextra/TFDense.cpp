//
//  TFDense.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TFExtraManager.hpp"

namespace MNN {
namespace Express {
class DenseTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        MNN_ASSERT(nullptr != op->main_as_Extra());
        auto extraAttr  = op->main_as_Extra()->attr();
        bool transposeA = false;
        bool transposeB = false;
        bool bias       = false;
        if (nullptr != extraAttr) {
            for (int i = 0; i < extraAttr->size(); ++i) {
                auto attr = extraAttr->GetAs<Attribute>(i);
                if ("use_bias" == attr->key()->str()) {
                    bias = attr->b();
                    continue;
                }
                if ("transpose_b" == attr->key()->str()) {
                    transposeB = attr->b();
                    continue;
                }
                if ("transpose_a" == attr->key()->str()) {
                    transposeA = attr->b();
                    continue;
                }
            }
        }
        auto output = _MatMul(inputs[0], inputs[1], transposeA, transposeB);
        if (bias) {
            output = output + inputs[2];
        }
        output->setName(expr->name());
        return output->expr().first;
    }
};
static auto gRegister = []() {
    TFExtraManager::get()->insert("Dense", std::shared_ptr<TFExtraManager::Transform>(new DenseTransform));
    return true;
}();
} // namespace Express
} // namespace MNN
