//
//  TFBatchNormalMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TFExtraManager.hpp"

namespace MNN {
namespace Express {
class BatchNormalTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs                 = expr->inputs();
        auto op                     = expr->get();
        std::vector<VARP> subInputs = {inputs[0]};
        std::unique_ptr<MNN::OpT> BatchNormalOp(new OpT);
        BatchNormalOp->type       = OpType_BatchNorm;
        BatchNormalOp->name       = op->name()->str();
        BatchNormalOp->main.type  = OpParameter_BatchNorm;
        BatchNormalOp->main.value = new BatchNormT;
        auto batchnorm            = BatchNormalOp->main.AsBatchNorm();
        batchnorm->epsilon = 0.001f;
        bool train = false;
        auto extra         = op->main_as_Extra();
        bool nhwc = true;
        if (nullptr != extra->attr()) {
            for (int i = 0; i < extra->attr()->size(); ++i) {
                auto attr = extra->attr()->GetAs<Attribute>(i);
                if (attr->key()->str() == "epsilon") {
                    batchnorm->epsilon = attr->f();
                }
                if (attr->key()->str() == "is_training") {
                    train = attr->b();
                }
                if (attr->key()->str() == "data_format") {
                    if (nullptr != attr->s()) {
                        nhwc = attr->s()->str() == "NHWC";
                    }
                }
            }
        }
        auto scaleNode = inputs[1];
        auto biasNode  = inputs[2];
        if (train) {
            std::vector<int> reduceDims = {0, 1, 2};
            std::vector<int> reshapeDims;
            if (!nhwc) {
                // NCHW
                reduceDims = {0, 2, 3};
                reshapeDims = {1, -1, 1, 1};
                scaleNode = _Reshape(scaleNode, reshapeDims);
                biasNode = _Reshape(biasNode, reshapeDims);
            }
            // NHWC, mean for NHW
            auto mean = _ReduceMean(inputs[0], reduceDims, true);
            auto xSub = inputs[0] - mean;
            auto sampleVar      = _ReduceMean(_Square(xSub), reduceDims,
                                         true); // variance for each channel in the batch
            auto rSampleStd     = _Reciprocal(_Sqrt(sampleVar + _Const(batchnorm->epsilon)));
            auto normalizedData = xSub * rSampleStd;
            auto outputData          = normalizedData * scaleNode + biasNode;
            outputData->setName(expr->name());
            return outputData->expr().first;
        }
        auto meanNode  = inputs[3];
        auto varNode   = inputs[4];
        batchnorm->channels = 0;
        {
            auto info = scaleNode->getInfo();
            auto ptr  = scaleNode->readMap<float>();
            if (nullptr == info || nullptr == ptr) {
                MNN_ERROR("Don't support not const scale node \n");
                return nullptr;
            }
            batchnorm->channels = info->size;
            batchnorm->slopeData.resize(batchnorm->channels);
            batchnorm->biasData.resize(batchnorm->channels);
            batchnorm->meanData.resize(batchnorm->channels);
            batchnorm->varData.resize(batchnorm->channels);
            ::memcpy(batchnorm->slopeData.data(), ptr, info->size * sizeof(float));
        }
        {
            auto info = biasNode->getInfo();
            auto ptr  = biasNode->readMap<float>();
            if (nullptr == info || nullptr == ptr) {
                MNN_ERROR("Don't support not const bias node \n");
                return nullptr;
            }
            if (info->size != batchnorm->channels) {
                MNN_ERROR("Don't match channels: %d -> %d\n", batchnorm->channels, info->size);
                return nullptr;
            }
            ::memcpy(batchnorm->biasData.data(), ptr, info->size * sizeof(float));
        }
        {
            auto info = meanNode->getInfo();
            auto ptr  = meanNode->readMap<float>();
            if (nullptr == info || nullptr == ptr) {
                MNN_ERROR("Don't support not const meanNode node \n");
                return nullptr;
            }
            if (info->size != batchnorm->channels) {
                MNN_ERROR("Don't match channels: %d -> %d\n", batchnorm->channels, info->size);
                return nullptr;
            }
            ::memcpy(batchnorm->meanData.data(), ptr, info->size * sizeof(float));
        }
        {
            auto info = varNode->getInfo();
            auto ptr  = varNode->readMap<float>();
            if (nullptr == info || nullptr == ptr) {
                MNN_ERROR("Don't support not const varNode node \n");
                return nullptr;
            }
            if (info->size != batchnorm->channels) {
                MNN_ERROR("Don't match channels: %d -> %d\n", batchnorm->channels, info->size);
                return nullptr;
            }
            for (int i = 0; i < batchnorm->varData.size(); ++i) {
                batchnorm->varData[i] = ptr[i];
            }
        }
        auto newExpr = Expr::create(BatchNormalOp.get(), subInputs, expr->outputSize());
        return newExpr;
    }
};
static auto gRegister = []() {
    TFExtraManager::get()->insert("FusedBatchNorm",
                                  std::shared_ptr<TFExtraManager::Transform>(new BatchNormalTransform));
    TFExtraManager::get()->insert("FusedBatchNormV3",
                                  std::shared_ptr<TFExtraManager::Transform>(new BatchNormalTransform));
    return true;
}();
} // namespace Express
} // namespace MNN
