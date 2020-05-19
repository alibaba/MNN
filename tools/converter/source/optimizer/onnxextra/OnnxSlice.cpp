//
//  OnnxSlice.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <climits>
#include <numeric>

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

// Returns true if all elements in the array are 1, otherwise returns false.
static bool IsAllOne(const std::vector<int> &array) {
    for (const int &i : array) {
        if (i != 1)
            return false;
    }
    return array.size() ? true : false;
}

class OnnxSliceTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);

        auto type      = op->main_as_Extra()->type()->str();
        auto inputs    = expr->inputs();
        auto input     = inputs[0];
        auto inputInfo = input->getInfo();
        if (nullptr == inputInfo) {
            MNN_ERROR("Onnx slice must use the same dimensition\n");
            return nullptr;
        }
        auto attrs = op->main_as_Extra()->attr();
        if (inputs.size() == 1 && nullptr == attrs) {
            MNN_PRINT("Attrs of Slice in ONNX must not be null when inputs.size == 1\n");
            return nullptr;
        }
        std::vector<int> starts, ends, axes, strides;
        if (inputs.size() == 1) {
            auto copyFunction = [](std::vector<int> &dst, const MNN::Attribute *attr) {
                MNN_ASSERT(nullptr != attr->list());
                MNN_ASSERT(nullptr != attr->list()->i());
                dst.resize(attr->list()->i()->size());
                ::memcpy(dst.data(), attr->list()->i()->data(), dst.size() * sizeof(int));
            };
            for (int i = 0; i < attrs->size(); ++i) {
                auto attr = attrs->GetAs<Attribute>(i);
                MNN_ASSERT(nullptr != attr->list());

                if (attr->key()->str() == "axes") {
                    copyFunction(axes, attr);
                } else if (attr->key()->str() == "ends") {
                    copyFunction(ends, attr);
                } else if (attr->key()->str() == "starts") {
                    copyFunction(starts, attr);
                } else if (attr->key()->str() == "steps") {
                    copyFunction(strides, attr);
                }
            }
        } else if (inputs.size() >= 4) {
            auto copyFunction = [](std::vector<int> &dst, const VARP &var) {
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
            if (inputs.size() > 4) {
                copyFunction(strides, inputs[4]);
            }
        } else {
            MNN_ERROR(
                "Onnx slice must have 1 or 4 inputs for Slice1, \
                       or 5 inputs for Slice11.\n");
            return nullptr;
        }

        MNN_ASSERT(starts.size() == ends.size());
        if (axes.size() == 0) {
            axes.resize(ends.size());
            std::iota(axes.begin(), axes.end(), 0);
        } else {
            MNN_ASSERT(axes.size() == ends.size());
        }

        auto MakeConstVecVar = [](const std::vector<int> &array) {
            auto data_type = halide_type_of<int32_t>();
            return _Const(array.data(), {static_cast<int>(array.size())}, NCHW, data_type);
        };

        std::unique_ptr<MNN::OpT> sliceOp(new OpT);
        sliceOp->name = op->name()->str();

        int ndim = inputInfo->dim.size();
        // If strides is not empty and all elements are not 1, then we should
        // convert the op to tensorflow stride_slice op, otherwise convert to
        // slice op.
        if (strides.size() && !IsAllOne(strides)) {
            MNN_ASSERT(strides.size() == ends.size());
            std::vector<int> tfBegin(ndim, 0);
            std::vector<int> tfEnd = inputInfo->dim;
            std::vector<int> tfStrides(ndim, 1);
            // int begin_mask = 0, end_mask = 0;

            for (int i = 0; i < axes.size(); ++i) {
                int axis      = axes[i];
                tfBegin[axis] = starts[i];
                // MNN only support int32 instead of int64, and int64 will be limit
                // to (1 << 30) for saturation.
                if (ends[i] == (1 << 30)) {
                    tfEnd[axis] = inputInfo->dim[axis];
                } else if (ends[i] == -(1 << 30)) {
                    tfEnd[axis] = 0;
                } else {
                    tfEnd[axis] = ends[i];
                }
                tfStrides[axis] = strides[i];
            }
            auto beginVar   = MakeConstVecVar(tfBegin);
            auto EndVar     = MakeConstVecVar(tfEnd);
            auto StridesVar = MakeConstVecVar(tfStrides);
            sliceOp->type   = OpType_StridedSlice;
            return Expr::create(sliceOp.get(), {input, beginVar, EndVar, StridesVar}, expr->outputSize());
        } else {
            std::vector<int> tfBegin(ndim, 0);
            std::vector<int> tfSize = inputInfo->dim;
            for (int i = 0; i < axes.size(); ++i) {
                int axis = axes[i];
                auto fin = ends[i];
                if (fin > inputInfo->dim[axis]) {
                    fin = inputInfo->dim[axis];
                }
                if (starts[i] < 0) {
                    starts[i] = inputInfo->dim[axis] + starts[i];
                }
                tfBegin[axis] = starts[i];
                tfSize[axis]  = fin - starts[i];
            }
            auto beginVar = MakeConstVecVar(tfBegin);
            auto sizeVar  = MakeConstVecVar(tfSize);
            sliceOp->type = OpType_SliceTf;
            return Expr::create(sliceOp.get(), {input, beginVar, sizeVar}, expr->outputSize());
        }
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Slice", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxSliceTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
