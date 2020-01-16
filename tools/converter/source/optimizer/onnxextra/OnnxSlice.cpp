//
//  OnnxSlice.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OnnxExtraManager.hpp"
#include "MNN_generated.h"


namespace MNN {
namespace Express {
class OnnxSliceTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);
        auto type   = op->main_as_Extra()->type()->str();
        auto inputs = expr->inputs();
        auto input = inputs[0];
        auto inputInfo = input->getInfo();
        if (nullptr == inputInfo) {
            MNN_ERROR("Onnx slice must use the same dimensition\n");
            return nullptr;
        }
        std::unique_ptr<MNN::OpT> sliceOp(new OpT);
        sliceOp->type = OpType_SliceTf;
        sliceOp->name = op->name()->str();
        auto attrs = op->main_as_Extra()->attr();
        if (inputs.size() == 1 && nullptr == attrs) {
            MNN_PRINT("Attrs of Slice in ONNX must not be null when inputs.size == 1\n");
            return nullptr;
        }
        std::vector<int> starts;
        std::vector<int> ends;
        std::vector<int> axes;
        if (inputs.size() == 1) {
            auto copyFunction = [](std::vector<int>& dst, const MNN::Attribute* attr) {
                MNN_ASSERT(nullptr != attr->list());
                MNN_ASSERT(nullptr != attr->list()->i());
                dst.resize(attr->list()->i()->size());
                ::memcpy(dst.data(), attr->list()->i()->data(), dst.size() * sizeof(int));
            };
            for (int i=0; i<attrs->size(); ++i) {
                auto attr = attrs->GetAs<Attribute>(i);
                if (nullptr == attr->list()) {
                    return nullptr;
                }
                if (attr->key()->str() == "axes") {
                    copyFunction(axes, attr);
                }
                else if (attr->key()->str() == "ends") {
                    copyFunction(ends, attr);
                }
                else if (attr->key()->str() == "starts") {
                    copyFunction(starts, attr);
                }
            }
        } else if (inputs.size() == 4) {
            auto copyFunction = [](std::vector<int>& dst, const VARP& var) {
                MNN_ASSERT(nullptr != var);
                auto varInfo = var->getInfo();
                auto varData = var->readMap<int>();
                MNN_ASSERT(nullptr != varInfo && nullptr != varData);
                dst.resize(varInfo->size);
                ::memcpy(dst.data(), varData, dst.size() * sizeof(int));
            };
            copyFunction(starts, inputs[1]);
            copyFunction(ends, inputs[2]);
            copyFunction(axes, inputs[3]);
        } else {
            MNN_ERROR("Slice onnx must have 1 or 4 input\n");
            return nullptr;
        }
        if (starts.size() != ends.size() || ends.size() != axes.size() || starts.size() != axes.size()) {
            return nullptr;
        }
        std::vector<int> tfBegin(inputInfo->dim.size());
        std::vector<int> tfSize(inputInfo->dim.size());
        for (int i=0; i<inputInfo->dim.size(); ++i) {
            tfBegin[i] = 0;
            tfSize[i] = inputInfo->dim[i];
        }
        int axisSize = starts.size();
        for (int i=0; i<axisSize; ++i) {
            auto axis = axes[i];
            auto sta = starts[i];
            auto fin = ends[i];
            if (fin > inputInfo->dim[axis]) {
                fin = inputInfo->dim[axis];
            }
            tfBegin[axis] = sta;
            tfSize[axis] = fin - sta;
        }
        auto beginVar = _Const(tfBegin.data(), {(int)inputInfo->dim.size()}, NCHW, halide_type_of<int32_t>());
        auto sizeVar = _Const(tfSize.data(), {(int)inputInfo->dim.size()}, NCHW, halide_type_of<int32_t>());

        auto newExpr = Expr::create(sliceOp.get(), {input, beginVar, sizeVar}, expr->outputSize());
        return newExpr;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Slice", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxSliceTransform));
    return true;
}();
}
} // namespace MNN
