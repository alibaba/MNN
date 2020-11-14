//
//  ReductionTransform.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string>
#include "CaffeExtraManager.hpp"
#include "MNN_generated.h"
#include "logkit.h"

namespace MNN {
namespace Express {

class ReductionTransform : public CaffeExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op     = expr->get();
        auto inputs = expr->inputs();

        std::vector<int> reductionDim;
        auto beginAxis = op->main_as_Extra()->attr()->GetAs<Attribute>(0)->i();
        for (int i = beginAxis; i < 4; i++) {
            reductionDim.emplace_back(i);
        }

        auto opType = op->main_as_Extra()->attr()->GetAs<Attribute>(0)->key()->str();
        if (opType == "SUM") {
            auto newVar = _ReduceSum(inputs[0], reductionDim, false);
            return newVar->expr().first;
        }
        if (opType == "MEAN") {
            auto newVar = _ReduceMean(inputs[0], reductionDim, false);
            return newVar->expr().first;
        }
        if (opType == "ASUM") {
            auto absVar = _Abs(inputs[0]);
            auto newVar = _ReduceSum(absVar, reductionDim, false);
            return newVar->expr().first;
        }
        if (opType == "SUMSQ") {
            auto sqVar  = _Square(inputs[0]);
            auto newVar = _ReduceSum(sqVar, reductionDim, false);
            return newVar->expr().first;
        }
        DLOG(FATAL) << "not supported caffe reduction type";
        return nullptr;
    }
};

static auto gRegister = []() {
    CaffeExtraManager::get()->insert("Reduction",
                                     std::shared_ptr<CaffeExtraManager::Transform>(new ReductionTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
