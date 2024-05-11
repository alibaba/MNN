//
//  OnnxBatchNormMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
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

class OnnxBatchNormTransform : public OnnxExtraManager::Transform {
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();

        MNN_THROW_CHECK(inputs.size() == 5, "BatchNorm should have 5 inputs");

        int channels  = 1;
        float epsilon = 1e-10;

        auto bnOp       = expr->get();
        auto extraParam = bnOp->main_as_Extra();
        int size        = 0;
        if (nullptr != extraParam->attr()) {
            size = extraParam->attr()->size();
            for (int i = 0; i < size; ++i) {
                auto attr       = extraParam->attr()->GetAs<Attribute>(i);
                const auto& key = attr->key()->str();
                if (key == "epsilon") {
                    epsilon = attr->f();
                }
            }
        }

        auto gamma    = inputs[1];
        auto beta     = inputs[2];
        auto mean     = inputs[3];
        auto variance = inputs[4];

        MNN_THROW_CHECK(gamma->getInfo() != nullptr, "BatchNorm second input should be Constant!");
        MNN_THROW_CHECK(beta->getInfo() != nullptr, "BatchNorm second input should be Constant!");
        MNN_THROW_CHECK(mean->getInfo() != nullptr, "BatchNorm second input should be Constant!");
        MNN_THROW_CHECK(variance->getInfo() != nullptr, "BatchNorm second input should be Constant!");
        auto gammaSize    = gamma->getInfo()->size;
        auto betaSize     = beta->getInfo()->size;
        auto meanSize     = mean->getInfo()->size;
        auto varianceSize = variance->getInfo()->size;

        // find the max value(incase broadcast mode)
        channels = gammaSize > betaSize ? gammaSize : betaSize;
        channels = channels > meanSize ? channels : meanSize;
        channels = channels > varianceSize ? channels : varianceSize;

        std::unique_ptr<MNN::BatchNormT> batchnorm(new MNN::BatchNormT);
        batchnorm->slopeData.resize(channels);
        batchnorm->biasData.resize(channels);
        batchnorm->meanData.resize(channels);
        batchnorm->varData.resize(channels);
        batchnorm->channels = channels;

        // TODO check data length, then support broadcast mode
        auto gammaDataPtr = gamma->readMap<float>();
        MNN_THROW_CHECK(gammaDataPtr != nullptr, "BatchNorm's gamma not valid!");
        memcpy(batchnorm->slopeData.data(), gammaDataPtr, gamma->getInfo()->size * sizeof(float));

        auto betaDataPtr = beta->readMap<float>();
        MNN_THROW_CHECK(betaDataPtr != nullptr, "BatchNorm's beta not valid!");
        memcpy(batchnorm->biasData.data(), betaDataPtr, beta->getInfo()->size * sizeof(float));
        auto meanDataPtr = mean->readMap<float>();
        MNN_THROW_CHECK(meanDataPtr != nullptr, "BatchNorm's mean not valid!");
        memcpy(batchnorm->meanData.data(), meanDataPtr, mean->getInfo()->size * sizeof(float));
        auto varPtr = variance->readMap<float>();
        MNN_THROW_CHECK(varPtr != nullptr, "BatchNorm's var not valid!");
        for (int i = 0; i < channels; ++i) {
            batchnorm->varData[i] = varPtr[i];
        }

        std::unique_ptr<OpT> mnnBnOp(new OpT);
        mnnBnOp->name      = expr->name();
        mnnBnOp->type      = OpType_BatchNorm;
        mnnBnOp->main.type = OpParameter_BatchNorm;
        {
            auto bnParam        = new MNN::BatchNormT;
            mnnBnOp->main.value = bnParam;
            bnParam->channels   = batchnorm->channels;
            bnParam->slopeData.resize(batchnorm->channels);
            bnParam->biasData.resize(batchnorm->channels);
            bnParam->meanData.resize(batchnorm->channels);
            bnParam->varData.resize(batchnorm->channels);
            const float* slopeDataPtr = batchnorm->slopeData.data();
            const float* biasDataPtr  = batchnorm->biasData.data();
            const float* meanDataPtr  = batchnorm->meanData.data();
            const float* varDataPtr   = batchnorm->varData.data();

            for (int i = 0; i < batchnorm->channels; i++) {
                bnParam->slopeData[i] = slopeDataPtr[i];
                bnParam->biasData[i]  = biasDataPtr[i];
                bnParam->meanData[i]  = meanDataPtr[i];
                bnParam->varData[i]   = varDataPtr[i];
            }
            bnParam->epsilon = epsilon;
        }
        // create merged op
        auto newExpr = Expr::create(mnnBnOp.get(), {inputs[0]});
        newExpr->setName(expr->name());
        auto res = Variable::create(newExpr);
        return res->expr().first;
    }
};

static VARP _OnnxReshape(VARP x, VARP shape) {
    std::unique_ptr<OpT> reshape(new OpT);
    reshape->type = OpType_Reshape;
    reshape->main.type = OpParameter_Reshape;
    reshape->main.value = new ReshapeT;
    reshape->main.AsReshape()->dimType = MNN_DATA_FORMAT_NCHW;
    return (Variable::create(Expr::create(reshape.get(), {x, shape})));
}

class OnnxInstanceNormalTransform : public OnnxExtraManager::Transform {
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();

        MNN_THROW_CHECK(inputs.size() == 3, "InstanceNormal should have 3 inputs");
        auto input = inputs[0];

        int channels  = 1;
        float epsilon = 1e-10;

        auto bnOp       = expr->get();
        auto extraParam = bnOp->main_as_Extra();
        int size        = 0;
        if (nullptr != extraParam->attr()) {
            size = extraParam->attr()->size();
            for (int i = 0; i < size; ++i) {
                auto attr       = extraParam->attr()->GetAs<Attribute>(i);
                const auto& key = attr->key()->str();
                if (key == "epsilon") {
                    epsilon = attr->f();
                }
            }
        }
        bool needScale = true;
        bool scaleConst = false;
        do {
            auto biasPtr = inputs[2]->readMap<float>();
            auto scalePtr = inputs[1]->readMap<float>();
            if (nullptr == biasPtr || nullptr == scalePtr) {
                break;
            }
            scaleConst = true;
            auto oneVar = _Scalar<float>(1.0f);
            auto scaleOff = inputs[1] - oneVar;
            auto scaleSum = _ReduceSum(scaleOff * scaleOff);
            if (scaleSum->readMap<float>()[0] > 0.000001f) {
                break;
            }
            auto biasSum = _ReduceSum(inputs[2] * inputs[2]);
            if (biasSum->readMap<float>()[0] > 0.000001f) {
                break;
            }
            needScale = false;
        } while (false);
        auto originShape = _Shape(inputs[0], NCHW);
        auto inputDim3 = _Reshape(inputs[0], {0, 0, -1}, NCHW);
        
        // Turn to layernorm
        std::unique_ptr<MNN::OpT> layerNormOp(new MNN::OpT);
        layerNormOp->type = OpType_LayerNorm;
        layerNormOp->main.value = new LayerNormT;
        layerNormOp->main.type = OpParameter_LayerNorm;
        {
            auto param = layerNormOp->main.AsLayerNorm();
            param->axis = {-1}; // Layernorm only need axis's size as 1
            param->epsilon = epsilon;
            param->group = 1;
        }
        auto res = Variable::create(Expr::create(layerNormOp.get(), {inputDim3}));
        res = _ReshapeF(res, originShape, MNN_DATA_FORMAT_NCHW);
        if (needScale) {
            if (scaleConst) {
                auto biasPtr = inputs[2]->readMap<float>();
                auto scalePtr = inputs[1]->readMap<float>();
                int channels = inputs[1]->getInfo()->size;
                std::vector<float> scales(channels);
                std::vector<float> bias(channels);
                ::memcpy(bias.data(), biasPtr, channels * sizeof(float));
                ::memcpy(scales.data(), scalePtr, channels * sizeof(float));
                res = _Scale(res, channels, std::move(scales), std::move(bias));
            } else {
                auto compatShape = _Concat({_Shape(inputs[1], true), _Fill(_Unsqueeze(_Size(_Shape(input, true)) - _Scalar<int>(2), {0}), _Scalar<int>(1))}, 0);
                auto scale      = _OnnxReshape(inputs[1], compatShape);
                auto bias       = _OnnxReshape(inputs[2], compatShape);
                res = res * scale + bias;
            }
        }
        res->setName(expr->name());
        return res->expr().first;
    }
};

class OnnxMeanVarianceNormTransform : public OnnxExtraManager::Transform {
    virtual EXPRP onExecute(EXPRP expr) const override {
        std::vector<int> axes {0, 2, 3};
        auto attrs = expr->get()->main_as_Extra()->attr();
        if (attrs != nullptr) {
            for (const auto& attr : *attrs) {
                if (attr->key()->str() == "axes") {
                    axes.clear();
                    for (int i = 0; i < attr->list()->i()->size(); ++i) {
                        axes.push_back(attr->list()->i()->Get(i));
                    }
                }
            }
        }
        auto input = expr->inputs()[0];
        auto mean = _ReduceMean(input, axes, true);
        auto temp = input - mean;
        auto var = _ReduceMean(temp * temp, axes, true);
        auto res = temp * _Rsqrt(var);
        res->setName(expr->name());
        return res->expr().first;
    }
};

class OnnxLpNormTransform : public OnnxExtraManager::Transform {
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto input = expr->inputs()[0];
        int p = 2, axis = -1;
        auto attrs = expr->get()->main_as_Extra()->attr();
        if (attrs != nullptr) {
            for (const auto& attr : *attrs) {
                auto attrName = attr->key()->str();
                if (attrName == "axis") {
                    axis = attr->i();
                } else if (attrName == "p") {
                    p = attr->i();
                }
            }
        }
        if (p != 1 && p != 2) {
            MNN_ERROR("Onnx's LpNormalization only support attr p is 1 or 2");
            return nullptr;
        }
        VARP res;
        if (p == 1) {
            res = input / _ReduceSumMutable(_Abs(input), _Scalar<int>(axis), true);
        } else {
            res = input * _Rsqrt(_ReduceSumMutable(input * input, _Scalar<int>(axis), true));
        }
        res->setName(expr->name());
        return res->expr().first;
    }
};

class OnnxLayerNormTransform : public OnnxExtraManager::Transform {
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto input = expr->inputs()[0];
        int axis = -1;
        float eps = 1e-05;
        auto attrs = expr->get()->main_as_Extra()->attr();
        if (attrs != nullptr) {
            for (const auto& attr : *attrs) {
                auto attrName = attr->key()->str();
                if (attrName == "axis") {
                    axis = attr->i();
                }
                if (attrName == "epsilon") {
                    eps = attr->f();
                }
            }
        }
        auto axisVar = _Scalar<int>(axis);
        // Add negative protect, may decrease performance
        auto rankVar = _Rank(inputs[0]);
        axisVar = _Mod(axisVar + rankVar, rankVar);
        auto reduceAxis = _Range(axisVar, rankVar, _Scalar<int>(1));
        auto mean = _ReduceMeanMutable(input, reduceAxis, true);
        auto sub = input - mean;
        auto normal = _Rsqrt(_ReduceMeanMutable(_Square(sub), reduceAxis, true) + _Scalar<float>(eps));
        auto y = sub * normal * inputs[1];
        if (inputs.size() > 2) {
            y = y + inputs[2];
        }
        std::vector<VARP> identityOutputs = {y};
        if (expr->outputSize() > 1) {
            identityOutputs.emplace_back(mean);
        }
        if (expr->outputSize() > 2) {
            identityOutputs.emplace_back(normal);
        }
        std::unique_ptr<OpT> copyOp(new OpT);
        copyOp->type = OpType_Identity;
        auto resultExpr = Expr::create(copyOp.get(), identityOutputs, identityOutputs.size());
        resultExpr->setName(expr->name());
        for (int i=0; i<expr->outputSize(); ++i) {
            auto var = MNN::Express::Variable::create(resultExpr, i);
            var->setName(expr->outputName(i));
        }
        return resultExpr;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("BatchNormalization",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxBatchNormTransform));
    OnnxExtraManager::get()->insert("InstanceNormalization",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxInstanceNormalTransform));
    OnnxExtraManager::get()->insert("MeanVarianceNormalization",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxMeanVarianceNormTransform));
    OnnxExtraManager::get()->insert("LpNormalization",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxLpNormTransform));
    OnnxExtraManager::get()->insert("LayerNormalization",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxLayerNormTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
