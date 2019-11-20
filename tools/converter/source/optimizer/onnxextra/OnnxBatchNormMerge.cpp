//
//  OnnxBatchNormMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
namespace MNN {
namespace Express {

class OnnxBatchNormTransform : public OnnxExtraManager::Transform {
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();

        MNN_CHECK(inputs.size() == 5, "BatchNorm should have 5 inputs");

        int channels  = 1;
        float epsilon = 0.001;

        auto bnOp       = expr->get();
        auto extraParam = bnOp->main_as_Extra();
        const int size  = extraParam->attr()->size();
        for (int i = 0; i < size; ++i) {
            auto attr       = extraParam->attr()->GetAs<Attribute>(i);
            const auto& key = attr->key()->str();
            if (key == "epsilon") {
                epsilon = attr->f();
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
            batchnorm->varData[i] = varPtr[i] + epsilon;
        }

        // create merged op
        std::unique_ptr<OpT> mergedOp(new OpT);
        mergedOp->name = expr->name();
        mergedOp->type       = OpType_BatchNorm;
        mergedOp->main.type  = OpParameter_BatchNorm;
        mergedOp->main.value = batchnorm.release();

        return Expr::create(mergedOp.get(), {inputs[0]});
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("BatchNormalization",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxBatchNormTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
