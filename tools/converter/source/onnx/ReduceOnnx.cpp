//
//  ReduceOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ReduceOnnx);

MNN::OpType ReduceOnnx::opType() {
    return MNN::OpType_Reduction;
}
MNN::OpParameter ReduceOnnx::type() {
    return MNN::OpParameter_ReductionParam;
}

void ReduceOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                     OnnxScope* scope) {
    std::vector<int> axes;
    bool keepdims          = true;
    bool noopWithEmptyAxes = false;
    const auto attrSize = onnxNode->attribute_size();
    for (int i = 0; i < attrSize; ++i) {
        const auto &attributeProto = onnxNode->attribute(i);
        const auto &attributeName  = attributeProto.name();

        if (attributeName == "axes") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS) << "Node Attribute ERROR";
            const int size = attributeProto.ints_size();
            for (int k = 0; k < size; ++k) {
                axes.push_back(attributeProto.ints(k));
            }
        } else if (attributeName == "keepdims") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT) << "Node Attribute ERROR";
            keepdims = static_cast<bool>(attributeProto.i());
        } else if (attributeName == "noop_with_empty_axes") {
            DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT) << "Node Attribute ERROR";
            noopWithEmptyAxes = attributeProto.i() != 0;
        }
    }

    const auto& type = onnxNode->op_type();
    const bool hasAxesInput = scope->mOpsetVersion >= (type == "ReduceSum" ? 13 : 18);
    if (hasAxesInput) {
        bool emptyAxes = onnxNode->input_size() < 2 || onnxNode->input(1).empty();
        if (!emptyAxes) {
            const auto initializer = scope->mInitializers.find(onnxNode->input(1));
            if (initializer != scope->mInitializers.end()) {
                const auto* tensor = initializer->second;
                emptyAxes = tensor->dims_size() == 1 && tensor->dims(0) == 0;
            }
        }
        if (emptyAxes) {
            // A single-input Reduction without axes reduces over all dimensions.
            dstOp->inputIndexes.resize(1);
            if (noopWithEmptyAxes) {
                dstOp->type      = MNN::OpType_Identity;
                dstOp->main.type = MNN::OpParameter_NONE;
                return;
            }
        }
    }

    auto param = new MNN::ReductionParamT;

    if (type == "ReduceMean") {
        param->operation = MNN::ReductionType_MEAN;
    } else if (type == "ReduceMax") {
        param->operation = MNN::ReductionType_MAXIMUM;
    } else if (type == "ReduceMin") {
        param->operation = MNN::ReductionType_MINIMUM;
    } else if (type == "ReduceProd") {
        param->operation = MNN::ReductionType_PROD;
    } else if (type == "ReduceSum") {
        param->operation = MNN::ReductionType_SUM;
    } else {
        DLOG(ERROR) << "TODO ==> " << type;
    }

    param->dType      = MNN::DataType_DT_FLOAT;
    param->dim        = axes;
    param->keepDims   = keepdims;
    dstOp->main.value = param;
}

REGISTER_CONVERTER(ReduceOnnx, ReduceMean);
REGISTER_CONVERTER(ReduceOnnx, ReduceMax);
REGISTER_CONVERTER(ReduceOnnx, ReduceMin);
REGISTER_CONVERTER(ReduceOnnx, ReduceProd);
REGISTER_CONVERTER(ReduceOnnx, ReduceSum);
