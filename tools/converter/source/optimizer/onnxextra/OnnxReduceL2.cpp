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

class OnnxReduceTransform : public OnnxExtraManager::Transform {
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
        auto type = op->main_as_Extra()->type()->str();
        VARP x = inputs[0], y;
        if (type == "ReduceL1") {
            // ReduceL1(x) = Sum(Abs(x))
            y = _ReduceSum(_Abs(x), axis, keepDims);
        } else if (type == "ReduceL2") {
            // ReduceL2(x) = sqrt(Sum(x*x))
            y = _Sqrt(_ReduceSum(_Multiply(x, x), axis, keepDims));
        } else if (type == "ReduceLogSum") {
            // ReduceLogSum(x) = Log(Sum(x))
            y = _Log(_ReduceSum(x, axis, keepDims));
        } else if (type == "ReduceLogSumExp") {
            // ReduceLogSumExp(x) = Log(Sum(Exp(x)))
            y = _Log(_ReduceSum(_Exp(x), axis, keepDims));
        } else if (type == "ReduceSumSquare") {
            y = _ReduceSum(_Multiply(x, x), axis, keepDims);
        }
        y->setName(opName);
        return y->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("ReduceL2",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxReduceTransform));
    OnnxExtraManager::get()->insert("ReduceL1",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxReduceTransform));
    OnnxExtraManager::get()->insert("ReduceLogSum",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxReduceTransform));
    OnnxExtraManager::get()->insert("ReduceLogSumExp",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxReduceTransform));
    OnnxExtraManager::get()->insert("ReduceSumSquare",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxReduceTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
