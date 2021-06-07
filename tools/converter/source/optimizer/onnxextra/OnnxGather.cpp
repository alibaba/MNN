//
//  OnnxGather.cpp
//  MNNConverter
//
//  Created by MNN on 2020/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxGatherTransform : public OnnxExtraManager::Transform {
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
        auto output = _GatherV2(inputs[0], inputs[1], _Scalar<int>(axis));
        output->setName(expr->name());
        return output->expr().first;
    }
};

class OnnxGatherNDTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_GatherND;
        auto outputExpr = Expr::create(op.get(), inputs, 1);
        outputExpr->setName(expr->name());
        return outputExpr;
    }
};

class OnnxGatherElementTransform : public OnnxExtraManager::Transform {
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
        if (inputs.size() < 2) {
            MNN_ERROR("GatherElements should has two inputs\n");
            return nullptr;
        }
        // Reshape the input as outside, axis, inside
        auto index = inputs[1];
        auto input = inputs[0];
        auto info = input->getInfo();
        if (nullptr == info) {
            MNN_ERROR("Currently don't suport GatherElements with no rank input\n");
            return nullptr;
        }
        auto shape = _Shape(input, NCHW);
        if (axis < 0) {
            axis = axis + info->dim.size();
        }
        VARP outsideDims;
        auto zero = _Scalar<int>(0);
        auto oneV = _Scalar<int>(1);
        if (axis > 0) {
            auto outsideDimsAll = _Slice(shape, _Unsqueeze(zero, {0}), _Unsqueeze(_Scalar<int>(axis), {0}));
            outsideDims = _ReduceProd(outsideDimsAll);
        } else {
            outsideDims = oneV;
        }
        auto axisLength = _Slice(shape, _Unsqueeze(_Scalar<int>(axis), {0}), _Unsqueeze(oneV, {0}));
        VARP insideDims;
        if (axis + 1 < info->dim.size()) {
            auto length = info->dim.size() - (axis + 1);
            auto dimsAll = _Slice(shape, _Unsqueeze(_Scalar<int>(axis+1), {0}), _Unsqueeze(_Scalar<int>(length), {0}));
            insideDims = _ReduceProd(dimsAll);
        } else {
            insideDims = oneV;
        }
        // Compute auto index
        auto outputRange = _Range(zero, outsideDims, oneV) * axisLength * insideDims;
        auto insideRange = _Range(zero, insideDims, oneV);
        // Fuse outputRange and insideRange
        std::vector<int> outputReshapeDims = {-1, 1, 1};
        std::vector<int> insideReshapeDims = {1, 1, -1};
        if (axis == 0) {
            // No outside
            outputReshapeDims.erase(outputReshapeDims.begin());
            insideReshapeDims.erase(insideReshapeDims.begin());
        }
        if (axis + 1 == info->dim.size()) {
            outputReshapeDims.erase(outputReshapeDims.end() - 1);
            insideReshapeDims.erase(insideReshapeDims.end() - 1);
        }
        auto autoIndex = _Reshape(outputRange, outputReshapeDims) + _Reshape(insideRange, insideReshapeDims);
        auto shapeIndex = _Shape(index);

        // index -> index * insideDims + autoindex
        index = index * _Fill(shapeIndex, insideDims);
        index = index + autoIndex;

        auto output = _GatherV2(_Reshape(input, {-1}), _Reshape(index, {-1}), _Scalar<int>(0));
        output = _Reshape(output, shapeIndex);
        output->setName(expr->name());
        return output->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Gather", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxGatherTransform));
    OnnxExtraManager::get()->insert("GatherND", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxGatherNDTransform));
    OnnxExtraManager::get()->insert("GatherElements", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxGatherElementTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
