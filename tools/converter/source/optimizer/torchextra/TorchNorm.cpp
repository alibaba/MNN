//
//  TorchNorm.cpp
//  MNNConverter
//
//  Created by MNN on 2022/04/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TorchExtraManager.hpp"
#include "logkit.h"

namespace MNN {
namespace Express {

class TorchNormTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        auto info = op->main_as_Extra();
        int ord = 2;
        std::vector<int> dims = {1};
        bool keepDim = true;
        if (nullptr != info->attr()) {
            for (int i = 0; i < info->attr()->size(); ++i) {
                const auto attr          = info->attr()->GetAs<Attribute>(i);
                const auto attributeName = attr->key()->str();
                if (attributeName == "ord") {
                    ord = attr->i();
                } else if (attributeName == "dim") {
                    dims[0] = attr->i();
                } else if (attributeName == "keepDim") {
                    keepDim = attr->i();
                }
            }
        }
        auto x = inputs[0];
        if (ord == 2) {
            auto res = _Sqrt(_ReduceSum(x*x, dims, keepDim));
            res->setName(expr->name());
            return res->expr().first;
        }
        auto res = _Pow(_ReduceSum(_Pow(x, _Scalar(ord)), dims, keepDim), _Scalar(-ord));
        res->setName(expr->name());
        return res->expr().first;
    }
};

static auto gRegister = []() {
    TorchExtraManager::get()->insert("norm", std::shared_ptr<TorchExtraManager::Transform>(new TorchNormTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
