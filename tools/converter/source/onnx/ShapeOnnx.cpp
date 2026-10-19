//
//  ShapeOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/06/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include "onnxOpConverter.hpp"

static int32_t saturateShapeBound(int64_t bound) {
    // ShapeParam uses int32 bounds; preserve out-of-range signs for the runtime rank clamp.
    bound = std::max<int64_t>(bound, std::numeric_limits<int32_t>::min());
    bound = std::min<int64_t>(bound, std::numeric_limits<int32_t>::max());
    return static_cast<int32_t>(bound);
}

DECLARE_OP_CONVERTER(ShapeOnnx);

MNN::OpType ShapeOnnx::opType() {
    return MNN::OpType_Shape;
}
MNN::OpParameter ShapeOnnx::type() {
    return MNN::OpParameter_NONE;
}

void ShapeOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode, OnnxScope* scope) {
    bool hasStart = false, hasEnd = false;
    int start = 0, end = 0;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName = attributeProto.name();
        if (attributeName == "start") {
            hasStart = true;
            start = saturateShapeBound(attributeProto.i());
        }
        if (attributeName == "end") {
            hasEnd = true;
            end = saturateShapeBound(attributeProto.i());
        }
    }
    // Only set ShapeParam when start/end are specified, to keep backward compatibility with old engines
    if (hasStart || hasEnd) {
        std::unique_ptr<MNN::ShapeParamT> shapeParam(new MNN::ShapeParamT);
        shapeParam->hasStart = hasStart;
        shapeParam->start = start;
        shapeParam->hasEnd = hasEnd;
        shapeParam->end = end;
        dstOp->main.type = MNN::OpParameter_ShapeParam;
        dstOp->main.value = shapeParam.release();
    }
    dstOp->defaultDimentionFormat = MNN::MNN_DATA_FORMAT_NCHW;
}

REGISTER_CONVERTER(ShapeOnnx, Shape);

DECLARE_OP_CONVERTER(SizeOnnx);

MNN::OpType SizeOnnx::opType() {
    return MNN::OpType_Size;
}
MNN::OpParameter SizeOnnx::type() {
    return MNN::OpParameter_NONE;
}

void SizeOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode, OnnxScope* scope) {
    dstOp->defaultDimentionFormat = MNN::MNN_DATA_FORMAT_NCHW;
}

REGISTER_CONVERTER(SizeOnnx, Size);
