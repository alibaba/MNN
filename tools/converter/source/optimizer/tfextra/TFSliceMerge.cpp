//
//  TFSliceMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TFExtraManager.hpp"

namespace MNN {
namespace Express {
class SplitTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);
        auto type   = op->main_as_Extra()->type()->str();
        auto inputs = expr->inputs();
        MNN_ASSERT(inputs.size() == 2);
        std::vector<VARP> subInputs = {inputs[1]};
        std::unique_ptr<MNN::OpT> sliceOp(new OpT);
        sliceOp->type       = OpType_Slice;
        sliceOp->name       = op->name()->str();
        sliceOp->main.type  = OpParameter_Slice;
        sliceOp->main.value = new SliceT;
        auto sliceParameter = sliceOp->main.AsSlice();
        auto slicePointAttr = op->main_as_Extra()->attr();
        for (int i = 0; i < slicePointAttr->size(); ++i) {
            auto attr = slicePointAttr->GetAs<Attribute>(i);
            if (attr->key()->str() == "num_split") {
                if (nullptr == attr->tensor()) {
                    sliceParameter->slicePoints.resize(1);
                    sliceParameter->slicePoints[0] = attr->i();
                    break;
                }
                MNN_ASSERT(nullptr != attr->tensor()->int32s());
                auto intArray = attr->tensor()->int32s();
                sliceParameter->slicePoints.resize(intArray->size());
                ::memcpy(sliceParameter->slicePoints.data(), intArray->data(),
                         sliceParameter->slicePoints.size() * sizeof(int32_t));
                break;
            }
        }
        {
            auto axisNode        = inputs[0];
            sliceParameter->axis = 0;
            auto slicePointInfo  = axisNode->getInfo();
            auto slicePointPtr   = axisNode->readMap<int32_t>();
            if (nullptr == slicePointInfo || nullptr == slicePointPtr) {
                MNN_ERROR("Don't support not const axis point\n");
                return nullptr;
            }
            sliceParameter->axis = slicePointPtr[0];
        }
        sliceParameter->sourceType = MNN::NetSource_TENSORFLOW;
        auto newExpr               = Expr::create(sliceOp.get(), subInputs, expr->outputSize());
        return newExpr;
    }
};
class SliceTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);
        auto type   = op->main_as_Extra()->type()->str();
        auto inputs = expr->inputs();
        MNN_ASSERT(inputs.size() == 3);
        std::vector<VARP> subInputs = {inputs[0]};
        std::unique_ptr<MNN::OpT> sliceOp(new OpT);
        sliceOp->type       = OpType_Slice;
        sliceOp->name       = op->name()->str();
        sliceOp->main.type  = OpParameter_Slice;
        sliceOp->main.value = new SliceT;
        auto sliceParameter = sliceOp->main.AsSlice();
        {
            auto slicePoint     = inputs[1];
            auto slicePointInfo = slicePoint->getInfo();
            auto slicePointPtr  = slicePoint->readMap<int32_t>();
            if (nullptr == slicePointInfo || nullptr == slicePointPtr) {
                MNN_ERROR("Don't support not const slice point\n");
                return nullptr;
            }
            MNN_ASSERT(1 == slicePointInfo->dim.size());
            sliceParameter->slicePoints.resize(slicePointInfo->size);
            for (int i = 0; i < slicePointInfo->size; ++i) {
                sliceParameter->slicePoints[i] = slicePointPtr[i];
            }
        }
        {
            auto axisNode        = inputs[2];
            sliceParameter->axis = 0;
            auto slicePointInfo  = axisNode->getInfo();
            auto slicePointPtr   = axisNode->readMap<int32_t>();
            if (nullptr == slicePointInfo || nullptr == slicePointPtr) {
                MNN_ERROR("Don't support not const axis point\n");
                return nullptr;
            }
            sliceParameter->axis = slicePointPtr[0];
        }
        sliceParameter->sourceType = MNN::NetSource_TENSORFLOW;
        auto newExpr               = Expr::create(sliceOp.get(), subInputs, expr->outputSize());
        return newExpr;
    }
};
static auto gRegister = []() {
    TFExtraManager::get()->insert("SplitV", std::shared_ptr<TFExtraManager::Transform>(new SliceTransform));
    TFExtraManager::get()->insert("Split", std::shared_ptr<TFExtraManager::Transform>(new SplitTransform));
    return true;
}();
} // namespace Express
} // namespace MNN
