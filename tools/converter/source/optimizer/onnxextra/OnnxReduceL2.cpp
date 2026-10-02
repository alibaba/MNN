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
        auto info = op->main_as_Extra();
        bool keepDims = true;
        bool noopWithEmptyAxes = false;
        for (int i = 0; i < info->attr()->size(); ++i) {
            const auto attr = info->attr()->GetAs<Attribute>(i);
            const auto attributeName = attr->key()->str();
            if (attributeName == "axes") {
                if (nullptr != attr->list() && nullptr != attr->list()->i()) {
                    axis.resize(attr->list()->i()->size());
                    ::memcpy(axis.data(), attr->list()->i()->data(), axis.size() * sizeof(int));
                }
            } else if (attributeName == "keepdims") {
                keepDims = attr->i() != 0;
            } else if (attributeName == "noop_with_empty_axes") {
                noopWithEmptyAxes = attr->i() != 0;
            }
        }
        if (inputs.size() > 1 && nullptr != inputs[1]) {
            const auto axesExpr = inputs[1]->expr().first;
            const auto axesOp = axesExpr->get();
            const bool constantAxes = axesOp ? axesOp->type() == OpType_Const : axesExpr->inputType() == VARP::CONSTANT;
            const auto axesInfo = inputs[1]->getInfo();
            if (!constantAxes || axesInfo == nullptr) {
                MNN_ERROR("ONNX composite reduction requires constant axes\n");
                return nullptr;
            }
            axis.resize(axesInfo->size);
            if (!axis.empty()) {
                const auto axesData = inputs[1]->readMap<int>();
                if (axesData == nullptr) {
                    MNN_ERROR("Cannot read ONNX composite reduction axes\n");
                    return nullptr;
                }
                ::memcpy(axis.data(), axesData, axis.size() * sizeof(int));
            }
        }
        auto reduceSum = [&](VARP value) {
            // Empty axes disable only the reduction, not the surrounding math.
            if (noopWithEmptyAxes && axis.empty()) {
                return value;
            }
            return _ReduceSum(value, axis, keepDims);
        };
        auto type = op->main_as_Extra()->type()->str();
        VARP x = inputs[0], y;
        if (type == "ReduceL1") {
            // ReduceL1(x) = Sum(Abs(x))
            y = reduceSum(_Abs(x));
        } else if (type == "ReduceL2") {
            // ReduceL2(x) = sqrt(Sum(x*x))
            y = _Sqrt(reduceSum(_Multiply(x, x)));
        } else if (type == "ReduceLogSum") {
            // ReduceLogSum(x) = Log(Sum(x))
            y = _Log(reduceSum(x));
        } else if (type == "ReduceLogSumExp") {
            // ReduceLogSumExp(x) = Log(Sum(Exp(x)))
            y = _Log(reduceSum(_Exp(x)));
        } else if (type == "ReduceSumSquare") {
            y = reduceSum(_Multiply(x, x));
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
