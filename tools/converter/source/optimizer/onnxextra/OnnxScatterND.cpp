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
            auto updateOne = _ZerosLike(update) + _Scalar<float>(1.0f);
            auto mask = _ScatterNd(indice, updateOne, shape);
            tfMask = _Cast<float>(_Less(mask, _Scalar<float>(0.5f)));
        } else {
            auto updateOne = _ZerosLike(update) + _Scalar<int>(1);
            auto mask = _ScatterNd(indice, updateOne, shape);
            tfMask = _Less(mask, _Scalar<int>(1));
        }
        auto dst    = data * tfMask + tfRes;
        dst->setName(expr->name());
        return dst->expr().first;
    }
};

class OnnxScatterElementsTransformer : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        int axis    = 0;
        auto op     = expr->get();
        if (nullptr != op->main_as_Extra()->attr()) {
            for (int i = 0; i < op->main_as_Extra()->attr()->size(); ++i) {
                auto attr = op->main_as_Extra()->attr()->GetAs<Attribute>(i);
                auto key  = attr->key()->str();
                if (key == "axis") {
                    axis = attr->i();
                    break;
                }
            }
        }
        // Reshape the input as outside, axis, inside
        auto input = inputs[0], indice = inputs[1], update = inputs[2];
        auto info = input->getInfo();
        if (nullptr == info) {
            MNN_ERROR("Currently don't suport ScatterElements with no rank input\n");
            return nullptr;
        }
        auto type = info->type;
        auto shape = _Shape(indice, NCHW);
        auto zeroV = _Unsqueeze(_Scalar<int>(0), {0}), oneV = _Unsqueeze(_Scalar<int>(1), {0});
        VARPS subIndices;
        int dimSize = info->dim.size();
        for (int i = 0; i < dimSize; ++i) {
            if (i == axis) {
                subIndices.push_back(_Unsqueeze(indice, {-1}));
                continue;
            }
            INTS broadCastDims(dimSize, 1), mask(dimSize, 1);
            broadCastDims[i] = -1;
            mask[i] = 0;
            auto subIndice = _Range(zeroV, _Slice(shape, _Unsqueeze(_Scalar<int>(i), {0}), oneV), oneV);
            auto tileNum = _Select(_Const(mask.data(), {dimSize}, NCHW, halide_type_of<int>()), shape, oneV);
            subIndice = _Tile(_Reshape(subIndice, broadCastDims, NCHW), tileNum);
            subIndices.push_back(_Unsqueeze(subIndice, {-1}));
        }
        indice = _Concat(subIndices, -1);
        shape = _Shape(input, NCHW);
        auto tfRes = _ScatterNd(indice, update, shape);
        VARP tfMask;
        if (type.code == halide_type_float) {
            auto updateOne = _ZerosLike(update) + _Scalar<float>(1.0f);
            auto mask = _ScatterNd(indice, updateOne, shape);
            tfMask = _Cast<float>(_Less(mask, _Scalar<float>(0.5f)));
        } else {
            auto updateOne = _ZerosLike(update) + _Scalar<int>(1);
            auto mask = _ScatterNd(indice, updateOne, shape);
            tfMask = _Less(mask, _Scalar<int>(1));
        }
        auto dst    = input * tfMask + tfRes;
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
