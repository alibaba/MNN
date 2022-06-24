//
//  TorchFlip.cpp
//  MNNConverter
//
//  Created by MNN on 2022/03/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TorchExtraManager.hpp"
#include "logkit.h"

namespace MNN {
namespace Express {

class TorchFlipTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op     = expr->get();
        auto extra  = op->main_as_Extra();
        MNN_ASSERT(nullptr != extra);
        auto attr = extra->attr();
        int dim = 1;
        if (nullptr != attr) {
            for (int i = 0; i < attr->size(); ++i) {
                auto att = attr->GetAs<Attribute>(i);
                auto key = att->key()->str();
                if ("dims" == key) {
                    dim = att->i();
                }
            }
        }
        auto input   = expr->inputs()[0];
        auto end = _Shape(input);
        auto begin = _ZerosLike(end);
        auto dims = _Scalar(dim);
        auto index = _Add(dims, _Multiply(_Rank(input), _Less(dims, _Scalar(0))));
        auto strided = _ScatterNd(_Unsqueeze(index, {0}), _Unsqueeze(_Scalar(-1), {0}), _Shape(end), begin + _Scalar(1));
        auto newExpr = _StridedSlice(input, begin, end, strided, 31, 31, 0, 0, 0)->expr().first;
        newExpr->setName(expr->name());
        return newExpr;
    }
};

static auto gRegister = []() {
    TorchExtraManager::get()->insert("flip", std::shared_ptr<TorchExtraManager::Transform>(new TorchFlipTransform));
    return true;
}();
} // namespace Express
} // namespace MNN
