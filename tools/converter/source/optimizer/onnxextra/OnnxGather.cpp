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
static VARP _ReshapeF(VARP x, VARP shape, MNN::MNN_DATA_FORMAT format) {
    MNN_ASSERT(nullptr != x);
    std::unique_ptr<OpT> reshape(new OpT);
    reshape->type                      = OpType_Reshape;
    reshape->main.type                 = OpParameter_Reshape;
    reshape->main.value                = new ReshapeT;
    reshape->main.AsReshape()->dimType = format;
    return (Variable::create(Expr::create(reshape.get(), {x, shape})));
}
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
        auto indice = inputs[1];
        auto param = inputs[0];
        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_GatherND;
        // Default is 0, Ref https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
        int batch_dims    = 0;
        auto oriop     = expr->get();
        if (nullptr != oriop->main_as_Extra()->attr()) {
            for (int i = 0; i < oriop->main_as_Extra()->attr()->size(); ++i) {
                auto attr = oriop->main_as_Extra()->attr()->GetAs<Attribute>(i);
                auto key  = attr->key()->str();
                if (key == "batch_dims") {
                    batch_dims = attr->i();
                    break;
                }
            }
        }
        if (batch_dims != 0) {
            op->main.value = new AxisT;
            op->main.type = OpParameter_Axis;
            op->main.AsAxis()->axis = batch_dims;
            // Add Extra offset for indice
            auto indiceShape = _Shape(indice, true);
            auto startV = _Unsqueeze(_Scalar<int>(0), {0});
            auto batchV = _Unsqueeze(_Scalar<int>(batch_dims), {0});
            batchV.fix(VARP::CONSTANT);
            startV.fix(VARP::CONSTANT);
            auto rankIndice = _Unsqueeze(_Rank(indice), {0});
            auto totalSize = _Slice(indiceShape, startV, batchV);
            auto start = _Scalar<int>(0);
            auto delta = _Scalar<int>(1);
            // Make offset from range: [B, 1]
            auto offset = _Range(start, _ReduceProd(totalSize), delta);
            VARP dims = rankIndice - batchV;
            auto oneSize = _Fill(dims, _Scalar<int>(1));
            offset = _ReshapeF(offset, _Concat({totalSize, oneSize}, 0), MNN::MNN_DATA_FORMAT_NCHW);

            // Compute Stride
            auto lastDim = _Slice(indiceShape, rankIndice - delta, _Unsqueeze(delta, {0}));
            auto paramShape = _Shape(inputs[0], true);
            paramShape->setName(inputs[0]->name() + "_shape");
            auto rankInput = _Unsqueeze(_Rank(inputs[0]), {0});
            auto inputDimOffset = rankInput - batchV;
            auto inputSDim = inputDimOffset - lastDim;
            auto lastInputShapes = _Slice(paramShape, batchV, rankInput - batchV - inputSDim);
            auto strideParameter = _ReduceProd(lastInputShapes);
            strideParameter->setName(param->name() + "_stride");
            offset = offset * strideParameter;
            indice = indice + offset;
        }
        auto outputExpr = Expr::create(op.get(), {param, indice}, 1);
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
        auto dst = Express::_GatherElements(input, index, _Scalar(axis));
        dst->setName(expr->name());
        return dst->expr().first;
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
