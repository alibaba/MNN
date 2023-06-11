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

        auto type   = op->main_as_Extra()->type()->str();
        auto inputs = expr->inputs();
        auto input  = inputs[0];
        auto attrs  = op->main_as_Extra()->attr();
        if (inputs.size() == 1 && nullptr == attrs) {
            MNN_PRINT("Attrs of Slice in ONNX must not be null when inputs.size == 1\n");
            return nullptr;
        }
        VARP startVar;
        VARP endVar;
        VARP axisVar;
        VARP strideVar;
        if (nullptr != attrs) {
            // Copy from attribute
            std::vector<int> starts, ends, axes, strides;
            auto copyFunction = [](std::vector<int> &dst, const MNN::Attribute *attr) {
                MNN_ASSERT(nullptr != attr->list());
                MNN_ASSERT(nullptr != attr->list()->i());
                dst.resize(attr->list()->i()->size());
                ::memcpy(dst.data(), attr->list()->i()->data(), dst.size() * sizeof(int));
            };
            for (int i = 0; i < attrs->size(); ++i) {
                auto attr = attrs->GetAs<Attribute>(i);
                if (attr->key()->str() == "axes") {
                    copyFunction(axes, attr);
                    axisVar = _Const(axes.data(), {(int)axes.size()}, NCHW, halide_type_of<int>());
                } else if (attr->key()->str() == "ends") {
                    copyFunction(ends, attr);
                    endVar = _Const(ends.data(), {(int)ends.size()}, NCHW, halide_type_of<int>());
                } else if (attr->key()->str() == "starts") {
                    copyFunction(starts, attr);
                    startVar = _Const(starts.data(), {(int)starts.size()}, NCHW, halide_type_of<int>());
                } else if (attr->key()->str() == "steps") {
                    copyFunction(strides, attr);
                    strideVar = _Const(strides.data(), {(int)strides.size()}, NCHW, halide_type_of<int>());
                }
            }
        }
        {
            // If has input, use input instead of attribute
            if (inputs.size() > 1) {
                startVar = inputs[1];
            }
            if (inputs.size() > 2) {
                endVar = inputs[2];
            }
            if (inputs.size() > 3) {
                axisVar = inputs[3];
            }
            if (inputs.size() > 4) {
                strideVar = inputs[4];
            }
        }
        
        std::unique_ptr<MNN::OpT> sliceOp(new OpT);
        sliceOp->name = op->name()->str();

        sliceOp->type       = OpType_StridedSlice;
        sliceOp->main.type  = OpParameter_StridedSliceParam;
        auto param          = new StridedSliceParamT;
        param->fromType     = 1;
        sliceOp->main.value = param;
        if(nullptr != axisVar && nullptr != strideVar) {
            return Expr::create(sliceOp.get(), {input, startVar, endVar, axisVar, strideVar}, expr->outputSize());
        }
        if(nullptr != axisVar) {
            return Expr::create(sliceOp.get(), {input, startVar, endVar, axisVar}, expr->outputSize());
        }
        return Expr::create(sliceOp.get(), {input, startVar, endVar}, expr->outputSize());    
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Slice", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxSliceTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
