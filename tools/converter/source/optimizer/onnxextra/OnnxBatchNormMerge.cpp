//
//  OnnxBatchNormMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
#include <math.h>
namespace MNN {
namespace Express {

class OnnxBatchNormTransform : public OnnxExtraManager::Transform {
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();

        MNN_CHECK(inputs.size() == 5, "BatchNorm should have 5 inputs");

        int channels  = 1;
        float epsilon = 1e-10;

        auto bnOp       = expr->get();
        auto extraParam = bnOp->main_as_Extra();
        int size = 0;
        if (nullptr != extraParam->attr()) {
            size  = extraParam->attr()->size();
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

        MNN_CHECK(gamma->getInfo() != nullptr, "BatchNorm second input should be Constant!");
        MNN_CHECK(beta->getInfo() != nullptr, "BatchNorm second input should be Constant!");
        MNN_CHECK(mean->getInfo() != nullptr, "BatchNorm second input should be Constant!");
        MNN_CHECK(variance->getInfo() != nullptr, "BatchNorm second input should be Constant!");
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
        memcpy(batchnorm->slopeData.data(), gammaDataPtr, gamma->getInfo()->size * sizeof(float));

        auto betaDataPtr = beta->readMap<float>();
        memcpy(batchnorm->biasData.data(), betaDataPtr, beta->getInfo()->size * sizeof(float));
        auto meanDataPtr = mean->readMap<float>();
        memcpy(batchnorm->meanData.data(), meanDataPtr, mean->getInfo()->size * sizeof(float));
        auto varPtr = variance->readMap<float>();
        for (int i = 0; i < channels; ++i) {
            batchnorm->varData[i] = varPtr[i];
        }

        std::unique_ptr<OpT> mnnBnOp(new OpT);
        mnnBnOp->name = expr->name();
        mnnBnOp->type       = OpType_BatchNorm;
        mnnBnOp->main.type  = OpParameter_BatchNorm;
        {
            auto bnParam      = new MNN::BatchNormT;
            mnnBnOp->main.value = bnParam;
            bnParam->channels = batchnorm->channels;
            bnParam->slopeData.resize(batchnorm->channels);
            bnParam->biasData.resize(batchnorm->channels);
            bnParam->meanData.resize(batchnorm->channels);
            bnParam->varData.resize(batchnorm->channels);
            const float* slopeDataPtr = batchnorm->slopeData.data();
            const float* biasDataPtr = batchnorm->biasData.data();
            const float* meanDataPtr = batchnorm->meanData.data();
            const float* varDataPtr  = batchnorm->varData.data();

            for (int i = 0; i < batchnorm->channels; i++) {
                bnParam->slopeData[i] = slopeDataPtr[i];
                bnParam->biasData[i]  = biasDataPtr[i];
                bnParam->meanData[i] = meanDataPtr[i];
                bnParam->varData[i] = varDataPtr[i];
            }
            bnParam->epsilon = epsilon;
        }
        // create bn op
        auto newExpr = Expr::create(mnnBnOp.get(), {_Convert(inputs[0], NC4HW4)});
        newExpr->setName(expr->name());
        auto res = _Convert(Variable::create(newExpr), NCHW);
        return res->expr().first;
    }
};

class OnnxInstanceNormalTransform : public OnnxExtraManager::Transform {
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();

        MNN_CHECK(inputs.size() == 3, "InstanceNormal should have 3 inputs");
        auto changedInput = _Convert(inputs[0], NC4HW4);

        int channels  = 1;
        float epsilon = 1e-10;

        auto bnOp       = expr->get();
        auto extraParam = bnOp->main_as_Extra();
        int size = 0;
        if (nullptr != extraParam->attr()) {
            size  = extraParam->attr()->size();
            for (int i = 0; i < size; ++i) {
                auto attr       = extraParam->attr()->GetAs<Attribute>(i);
                const auto& key = attr->key()->str();
                if (key == "epsilon") {
                    epsilon = attr->f();
                }
            }
        }
        auto scale    = inputs[1];
        auto bias     = inputs[2];
        channels = scale->getInfo()->size;
        if (bias->getInfo()->size != channels) {
            MNN_ASSERT(false);
            return nullptr;
        }
        std::unique_ptr<MNN::OpT> instanse(new MNN::OpT);
        instanse->type = OpType_InstanceNorm;
        instanse->main.type = OpParameter_BatchNorm;
        std::unique_ptr<MNN::BatchNormT> batchnorm(new MNN::BatchNormT);
        batchnorm->channels = channels;
        batchnorm->epsilon = epsilon;
        batchnorm->slopeData.resize(channels);
        auto scalePtr = scale->readMap<float>();
        ::memcpy(batchnorm->slopeData.data(), scalePtr, channels * sizeof(float));
        batchnorm->biasData.resize(channels);
        auto biasPtr = bias->readMap<float>();
        ::memcpy(batchnorm->biasData.data(), biasPtr, channels * sizeof(float));
        instanse->main.value = batchnorm.release();
        auto meanVar = _Moments(changedInput, {1}, nullptr, true);

        EXPRP newExpr = Expr::create(std::move(instanse), {changedInput, meanVar[0], meanVar[1]});
        newExpr->setName(expr->name());
        auto res = _Convert(Variable::create(newExpr), NCHW);
        return res->expr().first;
    }
};



static auto gRegister = []() {
    OnnxExtraManager::get()->insert("BatchNormalization",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxBatchNormTransform));
    OnnxExtraManager::get()->insert("InstanceNormalization",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxInstanceNormalTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
