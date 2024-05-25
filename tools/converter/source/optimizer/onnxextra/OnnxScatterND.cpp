//
//  OnnxScatterND.cpp
//  MNNConverter
//
//  Created by MNN on 2020/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <numeric>

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxScatterNdTransformer : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);
        auto inputs = expr->inputs();
        if (3 != inputs.size()) {
            MNN_ERROR("Onnx ScatterND error for inputs: %d\n", (int)inputs.size());
            return nullptr;
        }
        // Onnx Scatter = data + MNN::Scatter(indice, update, shape)
        auto data   = inputs[0];
        auto info   = data->getInfo();
        auto type   = halide_type_of<float>();
        if (nullptr != info) {
            type = info->type;
        }
        auto indice = inputs[1];
        auto update = inputs[2];
        auto config = Global<modelConfig>::Get();
        auto version = config->targetVersion;
        auto shape  = _Shape(data, true);
        if (version < 2.0f) {
            // For target version < 2.0 , don't support 4 input scatternd
            auto tfRes  = _ScatterNd(indice, update, shape);
            VARP tfMask;
            if (type.code == halide_type_float) {
                auto updateOne = _Fill(_Shape(update, NCHW), _Scalar<float>(1.0f));
                auto mask = _ScatterNd(indice, updateOne, shape);
                tfMask = _Cast<float>(_Less(mask, _Scalar<float>(0.5f)));
            } else {
                auto updateOne = _Fill(_Shape(update, NCHW), _Scalar<int>(1));
                auto mask = _ScatterNd(indice, updateOne, shape);
                tfMask = _Less(mask, _Scalar<int>(1));
            }
            auto dst    = data * tfMask + tfRes;
            dst->setName(expr->name());
            return dst->expr().first;
        }
        auto tfRes  = _ScatterNd(indice, update, shape, data);
        tfRes->setName(expr->name());
        return tfRes->expr().first;
    }
};

class OnnxScatterElementsTransformer : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs   = expr->inputs();
        int axis      = 0;
        auto op       = expr->get();
        int reduction = -1;
        if (nullptr != op->main_as_Extra()->attr()) {
            for (int i = 0; i < op->main_as_Extra()->attr()->size(); ++i) {
                auto attr = op->main_as_Extra()->attr()->GetAs<Attribute>(i);
                auto key  = attr->key()->str();
                if (key == "axis") {
                    axis = attr->i();
                    break;
                }
                if (key == "reduction") {
                    auto reductionStr = attr->s()->str();
                    if (reductionStr == "add") {
                        reduction = BinaryOpOperation_ADD;
                    } else if (reductionStr == "mul") {
                        reduction = BinaryOpOperation_MUL;
                    }
                    break;
                }
            }
        }
        auto input = inputs[0], indice = inputs[1], update = inputs[2];
        auto dst   = Express::_ScatterElements(input, indice, update, _Scalar(axis), reduction);
        dst->setName(expr->name());
        return dst->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("ScatterND",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxScatterNdTransformer));
    OnnxExtraManager::get()->insert("Scatter",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxScatterElementsTransformer));
    OnnxExtraManager::get()->insert("ScatterElements",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxScatterElementsTransformer));
    return true;
}();

} // namespace Express
} // namespace MNN
