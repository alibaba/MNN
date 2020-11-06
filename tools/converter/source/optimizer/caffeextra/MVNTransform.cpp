//
//  MVNTransform.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <vector>
#include "CaffeExtraManager.hpp"
#include "MNN_generated.h"
namespace MNN {
namespace Express {

class MVNTransform : public CaffeExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op                = expr->get();
        auto inputs            = expr->inputs();
        auto acrossChannels    = op->main_as_Extra()->attr()->GetAs<Attribute>(0)->b();
        auto eps               = op->main_as_Extra()->attr()->GetAs<Attribute>(1)->f();
        auto normalizeVariance = op->main_as_Extra()->attr()->GetAs<Attribute>(2)->b();

        std::vector<int> reduceDims;
        if (acrossChannels) {
            reduceDims = {1, 2, 3};
        } else {
            reduceDims = {2, 3};
        }

        auto mean    = _ReduceMean(inputs[0], reduceDims, true);
        auto subMean = _Subtract(inputs[0], mean); // of input shape

        if (!normalizeVariance) {
            return subMean->expr().first;
        } else {
            auto s2         = _Square(subMean); // element wise of input shape
            auto variance   = _ReduceMean(s2, reduceDims, true);
            auto stdv       = _Add(_Sqrt(variance), _Const(eps));
            auto normedData = _Divide(inputs[0], stdv);
            return normedData->expr().first;
        }
    }
};

static auto gRegister = []() {
    CaffeExtraManager::get()->insert("MVN", std::shared_ptr<CaffeExtraManager::Transform>(new MVNTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
