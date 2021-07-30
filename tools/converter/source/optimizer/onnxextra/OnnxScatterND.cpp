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
        auto shape  = _Shape(data, true);
        auto tfRes  = _ScatterNd(indice, update, shape);
        VARP tfMask;
        if (type.code == halide_type_float) {
            tfMask = _Scalar<float>(1.0f) - _Abs(_Sign(tfRes));
        } else {
            tfMask = _Scalar<int>(1.0f) - _Abs(_Sign(tfRes));
        }
        auto dst    = data * tfMask + tfRes;
        dst->setName(expr->name());
        return dst->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("ScatterND",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxScatterNdTransformer));
    return true;
}();

} // namespace Express
} // namespace MNN
