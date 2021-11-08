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

        std::vector<int> dims {0};
        int len = 4;
        auto info = input->getInfo();
        if (info) {
            len = info->dim.size();
        }
        for (int i = 2; i < len; i++) {
            dims.push_back(i);
        }
        auto scale      = _Unsqueeze(inputs[1], dims);
        auto bias       = _Unsqueeze(inputs[2], dims);
        auto epsilonVar = _Scalar<float>(epsilon);
        auto mean       = _ReduceMean(input, {2, 3}, true);
        auto temp       = input - mean;
        temp            = temp * temp;
        auto var        = _ReduceMean(temp, {2, 3}, true);
        auto varRev     = _Rsqrt(var + epsilonVar);
        auto alpha      = scale * varRev;
        auto beta       = bias - alpha * mean;
        auto res        = input * alpha + beta;
        res->setName(expr->name());
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
