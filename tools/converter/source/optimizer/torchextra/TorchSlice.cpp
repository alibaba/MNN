//
//  TorchSlice.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TorchExtraManager.hpp"

namespace MNN {
namespace Express {

class TorchSliceTransform : public TorchExtraManager::Transform {
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
            for (int i = 0; i < attrs->size(); ++i) {
                auto attr = attrs->GetAs<Attribute>(i);
                if (attr->key()->str() == "dim") {
                    int dim = attr->i();
                    axisVar = _Const(&dim, {1}, NCHW, halide_type_of<int>());
                } else if (attr->key()->str() == "end") {
                    int end = attr->i();
                    endVar = _Const(&end, {1}, NCHW, halide_type_of<int>());
                } else if (attr->key()->str() == "start") {
                    int start = attr->i();
                    startVar = _Const(&start, {1}, NCHW, halide_type_of<int>());
                } else if (attr->key()->str() == "stride") {
                    int stride = attr->i();
                    strideVar = _Const(&stride, {1}, NCHW, halide_type_of<int>());
                }
            }
        }
        {
            // If has input, use input instead of attribute
            if (inputs.size() > 1) {
                axisVar = inputs[1];
                if (axisVar->getInfo() && axisVar->getInfo()->dim.empty()) {
                    axisVar = _Unsqueeze(axisVar, {0});
                }
            }
            if (inputs.size() > 2) {
                startVar = inputs[2];
                if (startVar->getInfo() && startVar->getInfo()->dim.empty()) {
                    startVar = _Unsqueeze(startVar, {0});
                }
            }
            if (inputs.size() > 3) {
                endVar = inputs[3];
                if (endVar->getInfo() && endVar->getInfo()->dim.empty()) {
                    endVar = _Unsqueeze(endVar, {0});
                }
            }
            if (inputs.size() > 4) {
                strideVar = inputs[4];
                if (strideVar->getInfo() && strideVar->getInfo()->dim.empty()) {
                    strideVar = _Unsqueeze(strideVar, {0});
                }
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
            auto shape      = _Shape(input, true);
            auto defaultVar = _Fill(_Shape(axisVar, true), _Scalar<int>(1));
            auto mask       = _Scalar<int>(1) - _ScatterNd(axisVar, defaultVar, rank);
            startVar        = _ScatterNd(axisVar, startVar, rank);
            endVar          = _ScatterNd(axisVar, endVar, rank) + mask * shape;
            if (nullptr != strideVar) {
                strideVar = _ScatterNd(axisVar, strideVar - _Scalar<int>(1), rank) + _Fill(rank, _Scalar<int32_t>(1));
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
        param->Index        = DataType_DT_INT32;
        param->T            = DataType_DT_FLOAT;
        sliceOp->main.value = param;
        return Expr::create(sliceOp.get(), {input, startVar, endVar, strideVar}, expr->outputSize());
    }
};

static auto gRegister = []() {
    TorchExtraManager::get()->insert("slice", std::shared_ptr<TorchExtraManager::Transform>(new TorchSliceTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
