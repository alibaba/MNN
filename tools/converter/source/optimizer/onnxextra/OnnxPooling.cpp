//
//  OnnxPooling.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxPoolingTransform : public OnnxExtraManager::Transform {
public:
    static bool setUp3DPooling(OpT* dstOp, const Extra* info) {
        const auto& type = info->type()->str();
        std::unique_ptr<MNN::Pool3DT> pool(new MNN::Pool3DT);
        if (type == "MaxPool") {
            pool->type = MNN::PoolType_MAXPOOL;
        } else if (type == "AveragePool") {
            pool->type = MNN::PoolType_AVEPOOL;
        } else {
            return false;
        }
        pool->padType = MNN::PoolPadType_CAFFE;
        for (int i = 0; i < info->attr()->size(); ++i) {
            const auto attr = info->attr()->GetAs<Attribute>(i);
            const auto attributeName = attr->key()->str();
            auto list = attr->list();
            if (nullptr == list || nullptr == list->i()) {
                continue;
            }
            auto vec = std::vector<int>({
                static_cast<int>(list->i()->data()[0]),
                static_cast<int>(list->i()->data()[1]),
                static_cast<int>(list->i()->data()[2]),
            });
            if (attributeName == "kernel_shape") {
                pool->kernels = vec;
            } else if (attributeName == "strides") {
                pool->strides = vec;
            } else if (attributeName == "pads") {
                pool->pads = vec;
            }
        }
        dstOp->type = MNN::OpType_Pooling3D;
        dstOp->main.type = MNN::OpParameter_Pool3D;
        dstOp->main.value = pool.release();
        return true;
    }
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op = expr->get();
        std::unique_ptr<OpT> poolOp(new OpT);
        auto extraParam    = op->main_as_Extra();
        bool is3DPooling = false;
        int attrSize = 0;
        if (extraParam->attr() != nullptr) {
            attrSize = extraParam->attr()->size();
            for (int i = 0; i < attrSize; ++i) {
                auto attr       = extraParam->attr()->GetAs<Attribute>(i);
                const auto& key = attr->key()->str();
                if (key == "kernel_shape") {
                    auto kernelSize= attr->list()->i()->size();
                    if (kernelSize == 3) {
                        is3DPooling = true;
                    }
                    break;
                }
            }
        }
        auto type = extraParam->type()->str();
        if (is3DPooling) {
            bool res = setUp3DPooling(poolOp.get(), extraParam);
            if (!res) {
                return nullptr;
            }
        } else {
            poolOp->type = OpType_Pooling;
            auto poolParam  = new MNN::PoolT;
            poolOp->main.type = OpParameter_Pool;
            poolOp->main.value = poolParam;
            poolParam->ceilModel = false;
            do {
                if (type == "GlobalAveragePool") {
                    poolParam->type     = MNN::PoolType_AVEPOOL;
                    poolParam->isGlobal = true;
                    break;
                }
                if (type == "GlobalMaxPool") {
                    poolParam->type     = MNN::PoolType_MAXPOOL;
                    poolParam->isGlobal = true;
                    break;
                }
                if (type == "MaxPool") {
                    poolParam->type = MNN::PoolType_MAXPOOL;
                } else {
                    poolParam->type = MNN::PoolType_AVEPOOL;
                }
                poolParam->isGlobal = false;
                for (int i = 0; i < attrSize; ++i) {
                    auto attr       = extraParam->attr()->GetAs<Attribute>(i);
                    auto attributeName = attr->key()->str();
                    if (attributeName == "auto_pad") {
                        auto type = attr->s()->str();
                        if (type == "NOTSET") {
                            poolParam->padType = PoolPadType_CAFFE;
                        } else if (type == "SAME_UPPER" || type == "SAME_LOWER") {
                            poolParam->padType = PoolPadType_SAME;
                        } else if (type == "VALID") {
                            poolParam->padType = PoolPadType_VALID;
                        }
                    }
                    if (attributeName == "ceil_mode") {
                        poolParam->ceilModel = static_cast<bool>(attr->i());
                        continue;
                    }
                    auto list = attr->list();
                    if (nullptr == list || nullptr == list->i()) {
                        continue;
                    }
                    auto vec = list->i()->data();
                    auto intDataSize = list->i()->size();
                    if (attributeName == "pads") {
                        // TODO Support Asymmetrical pads
                        poolParam->padY = vec[0];
                        poolParam->padX = 0;
                        if (intDataSize > 2) {
                            poolParam->padX = vec[1];
                        }
                        poolParam->pads.resize(intDataSize);
                        for (int u=0; u<intDataSize; ++u) {
                            poolParam->pads[u] = vec[u];
                        }
                        continue;
                    }
                    if (attributeName == "kernel_shape") {
                        poolParam->kernelY = vec[0];
                        poolParam->kernelX = 1;
                        if (intDataSize > 1) {
                            poolParam->kernelX = vec[1];
                        }
                        continue;
                    }
                    if (attributeName == "strides") {
                        poolParam->strideY = vec[0];
                        poolParam->strideX = 1;
                        if (intDataSize > 1) {
                            poolParam->strideX = vec[1];
                        }
                        continue;
                    }
                }
            } while (false);
        }
        auto poolExpr = Expr::create(poolOp.get(), {_Convert(inputs[0], NC4HW4)});
        auto res = _Convert(Variable::create(poolExpr), NCHW);
        poolExpr->setName(expr->name());
        return res->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("MaxPool", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxPoolingTransform));
    OnnxExtraManager::get()->insert("GlobalMaxPool", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxPoolingTransform));
    OnnxExtraManager::get()->insert("GlobalAveragePool", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxPoolingTransform));
    OnnxExtraManager::get()->insert("AveragePool", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxPoolingTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
