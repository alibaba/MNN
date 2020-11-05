//
//  OnnxReduceL2.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <map>
#include <string>
#include <vector>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
#include "logkit.h"

namespace MNN {
namespace Express {

class OnnxReduceL2Transform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        std::vector<int> axis;
        auto info     = op->main_as_Extra();
        bool keepDims = false;
        for (int i = 0; i < info->attr()->size(); ++i) {
            const auto attr          = info->attr()->GetAs<Attribute>(i);
            const auto attributeName = attr->key()->str();
            if (attributeName == "axes") {
                if (nullptr != attr->list() && nullptr != attr->list()->i()) {
                    axis.resize(attr->list()->i()->size());
                    ::memcpy(axis.data(), attr->list()->i()->data(), axis.size() * sizeof(int));
                }
            } else if (attributeName == "keepdims") {
                if (attr->i() > 0) {
                    keepDims = true;
                }
            }
        }
        // ReduceL2(x) = sqrt(Sum(x*x))
        auto x = _Multiply(inputs[0], inputs[0]);
        x      = _ReduceSum(x, axis, keepDims);
        x      = _Sqrt(x);
        x->setName(opName);
        return x->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("ReduceL2",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxReduceL2Transform));
    return true;
}();

} // namespace Express
} // namespace MNN
