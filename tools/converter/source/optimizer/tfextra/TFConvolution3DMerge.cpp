//
//  TFConvolution3DMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include "MNN_generated.h"
#include "TFExtraManager.hpp"

namespace MNN {
namespace Express {

class Convolution3DTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op               = expr->get();
        auto inputs           = expr->inputs();
        auto weight           = inputs[1];
        auto weightInfo       = weight->getInfo();
        auto weightTensorData = weight->readMap<float>();
        if (nullptr == weightInfo || nullptr == weightTensorData) {
            MNN_ERROR("For %s Convolution3D weight is not const\n", expr->name().c_str());
            return nullptr;
        }

        std::unique_ptr<Convolution3DT> conv3d(new MNN::Convolution3DT);
        int depth        = weightInfo->dim[0];
        int kh           = weightInfo->dim[1];
        int kw           = weightInfo->dim[2];
        int num_input    = weightInfo->dim[3];
        int num_output   = weightInfo->dim[4];
        weight           = _Transpose(weight, {4, 3, 0, 1, 2});
        weightInfo       = weight->getInfo();
        weightTensorData = weight->readMap<float>();
        conv3d->bias.resize(num_output);
        std::fill(conv3d->bias.begin(), conv3d->bias.end(), 0.0f);

        conv3d->weight.resize(weightInfo->size);
        ::memcpy(conv3d->weight.data(), weightTensorData, weightInfo->size * sizeof(float));
        conv3d->common.reset(new MNN::Convolution3DCommonT);
        auto common = conv3d->common.get();

        common->relu = common->relu6 = false;
        common->outputCount          = num_output;
        common->inputCount           = num_input;
        common->kernels              = std::vector<int>({depth, kh, kw});

        auto extra = op->main_as_Extra();
        if (extra == nullptr || extra->attr() == nullptr) {
            return nullptr;
        }
        for (int i = 0; i < extra->attr()->size(); ++i) {
            auto attr      = extra->attr()->GetAs<Attribute>(i);
            const auto key = attr->key()->str();
            if (key == "dilations" || key == "rates") {
                auto values     = attr->list()->i()->data();
                common->dilates = std::vector<int>({values[1], values[2], values[3]});
            } else if (key == "strides") {
                auto values     = attr->list()->i()->data();
                common->strides = std::vector<int>({values[1], values[2], values[3]});
            } else if (key == "padding") {
                common->padMode  = MNN::PadMode_SAME;
                auto paddingType = attr->s()->str();
                if (paddingType == "VALID") {
                    common->padMode = MNN::PadMode_VALID;
                    common->pads    = std::vector<int>({0, 0, 0});
                }
            }
        }

        std::unique_ptr<OpT> newOp(new OpT);
        newOp->name       = expr->name();
        newOp->type       = OpType_Convolution3D;
        newOp->main.type  = OpParameter_Convolution3D;
        newOp->main.value = conv3d.release();

        auto newExpr = Expr::create(newOp.get(), {inputs[0]}, 1);
        return newExpr;
    }
};

static auto gRegister = []() {
    TFExtraManager::get()->insert("Conv3D", std::shared_ptr<TFExtraManager::Transform>(new Convolution3DTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
