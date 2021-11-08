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
        // Use TF's stridedslice, turn onnx slice attribute to tf format
        auto rank = _Unsqueeze(_Rank(input), {0});
        if (nullptr != axisVar) {
            auto axisPtr = axisVar->readMap<int>();
            if (nullptr != axisPtr) {
                if (0 > axisPtr[0]) {
                    axisVar = axisVar + _Rank(input);
                }
            }
            auto axisVarScatter = _Unsqueeze(axisVar, {1});
            if (nullptr != axisPtr) {
                axisVarScatter.fix(VARP::CONSTANT);
            }
            auto shape      = _Shape(input, true);
            auto defaultVar = _Fill(_Shape(axisVar, true), _Scalar<int>(1));
            auto mask       = _Scalar<int>(1) - _ScatterNd(axisVarScatter, defaultVar, rank);
            startVar        = _ScatterNd(axisVarScatter, startVar, rank);
            endVar          = _ScatterNd(axisVarScatter, endVar, rank) + mask * shape;
            if (nullptr != strideVar) {
                strideVar = _ScatterNd(axisVarScatter, strideVar - _Scalar<int>(1), rank) + _Fill(rank, _Scalar<int32_t>(1));
            }
        }
        if (nullptr == strideVar) {
            strideVar = _Fill(rank, _Scalar<int32_t>(1));
        }

        std::unique_ptr<MNN::OpT> sliceOp(new OpT);
        sliceOp->name = op->name()->str();

        sliceOp->type       = OpType_StridedSlice;
        sliceOp->main.type  = OpParameter_StridedSliceParam;
        auto param          = new StridedSliceParamT;
        sliceOp->main.value = param;
        return Expr::create(sliceOp.get(), {input, startVar, endVar, strideVar}, expr->outputSize());
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Slice", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxSliceTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
