//
//  OnnxGather.cpp
//  MNNConverter
//
//  Created by MNN on 2020/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
#include "config.hpp"

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
        auto axisVar = _Scalar<int>(axis);
        auto config = Global<modelConfig>::Get();
        if (config->optimizeLevel < 2) {
            // Add negative protect, may decrease performance
            auto rankVar = _Rank(inputs[0]);
            axisVar = _Mod(axisVar + rankVar, rankVar);
            auto shapeVar = _Shape(inputs[0], true);
            auto axisLengthVar = _Squeeze(_StridedSlice(shapeVar, _Unsqueeze(axisVar, {0}), _Unsqueeze(axisVar + _Scalar<int>(1), {0}),  _Unsqueeze(_Scalar<int32_t>(1), {0}), 0, 0, 0, 0, 0));
            inputs[1] = _Mod(inputs[1] + axisLengthVar, axisLengthVar);
        }
        auto output = _GatherV2(inputs[0], inputs[1], axisVar);
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
        auto shape = _Shape(input, NCHW), shapeIndex = _Shape(index, NCHW);
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
        auto sliceShapeLeft = [=](VARP shape) {
            std::vector<int32_t> broadDimData(info->dim.size() - axis, 1);
            auto broadDim = _Const(broadDimData.data(), {(int32_t)broadDimData.size()}, NCHW, halide_type_of<int32_t>());
            if (axis == 0) {
                return broadDim;
            }
            return _Concat({_Slice(shape, _Unsqueeze(_Scalar<int>(0), {0}), _Unsqueeze(_Scalar<int>(axis), {0})), broadDim}, 0);
        };
        auto sliceShapeRight = [=](VARP shape) {
            std::vector<int32_t> broadDimData(axis + 1, 1);
            auto broadDim = _Const(broadDimData.data(), {(int32_t)broadDimData.size()}, NCHW, halide_type_of<int32_t>());
            if (axis == info->dim.size() - 1) {
                return broadDim;
            }
            return _Concat({broadDim, _Slice(shape, _Unsqueeze(_Unsqueeze(_Scalar<int>(axis + 1), {0})), _Unsqueeze(_Scalar<int>(info->dim.size() - axis - 1), {0}))}, 0);
        };
        auto reshapeSliceRange = [=](VARP range, std::function<VARP(VARP)> sliceFunc) {
            std::vector<int32_t> beginData(info->dim.size(), 0);
            auto begin = _Const(beginData.data(), {(int32_t)beginData.size()}, NCHW, halide_type_of<int32_t>());
            return _Slice(_Reshape(range, sliceFunc(shape)), begin, sliceFunc(shapeIndex));
        };
        auto autoIndex = reshapeSliceRange(outputRange, sliceShapeLeft) + reshapeSliceRange(insideRange, sliceShapeRight);
        // index -> index * insideDims + autoindex
        index = index * insideDims + autoIndex;

        auto output = _GatherV2(_Reshape(input, {-1}), _Reshape(index, {-1}), _Scalar<int>(0));
        output = _Reshape(output, shapeIndex);
        output->setName(expr->name());
        return output->expr().first;
    }
};

class OnnxCompressTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        int axis = 0, axisExist = 0;
        auto op = expr->get();
        for (int i = 0; i < op->main_as_Extra()->attr()->size(); ++i) {
            auto attr = op->main_as_Extra()->attr()->GetAs<Attribute>(i);
            auto key  = attr->key()->str();
            if (key == "axis") {
                axis = attr->i();
                axisExist = 1;
                break;
            }
        }
        VARP input = inputs[0];
        if (axisExist == 0) {
            input = _Reshape(input, {-1});
        }
        std::unique_ptr<OpT> whereOp(new OpT);
        whereOp->type = OpType_Where;
        whereOp->main.type = OpParameter_Extra;
        whereOp->main.value = new ExtraT;
        auto cond = Variable::create(Expr::create(std::move(whereOp), {inputs[1]}));
        
        auto res = _GatherV2(input, _Reshape(cond, {-1}), _Scalar<int32_t>(axis));
        res->setName(expr->name());
        return res->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Gather", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxGatherTransform));
    OnnxExtraManager::get()->insert("GatherND", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxGatherNDTransform));
    OnnxExtraManager::get()->insert("GatherElements", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxGatherElementTransform));
    OnnxExtraManager::get()->insert("Compress", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxCompressTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
