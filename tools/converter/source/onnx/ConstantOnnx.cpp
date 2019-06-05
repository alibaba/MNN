//
//  ConstantOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/05/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ConstantOnnx);

MNN::OpType ConstantOnnx::opType() {
    return MNN::OpType_Const;
}
MNN::OpParameter ConstantOnnx::type() {
    return MNN::OpParameter_Blob;
}

void ConstantOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                       std::vector<const onnx::TensorProto *> initializers) {
    auto constantParam = new MNN::BlobT;

    const onnx::TensorProto *constantTp;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto &attributeProto = onnxNode->attribute(i);
        const auto &attributeName  = attributeProto.name();
        if (attributeName == "value") {
            constantTp = &attributeProto.t();
        }
    }
    if (!constantTp) {
        DLOG(FATAL) << "Constant No TensorProto Data!!!==> " << dstOp->name;
    }

    std::map<::onnx::TensorProto_DataType, MNN::DataType> dataTypeMap{
        {onnx::TensorProto_DataType_FLOAT, MNN::DataType_DT_FLOAT},
        {onnx::TensorProto_DataType_INT8, MNN::DataType_DT_INT8},
        {onnx::TensorProto_DataType_INT32, MNN::DataType_DT_INT32},
        {onnx::TensorProto_DataType_INT64, MNN::DataType_DT_INT32}, // For compability, use int32 instead of int64
        {onnx::TensorProto_DataType_UINT8, MNN::DataType_DT_UINT8},
    };
    auto iter = dataTypeMap.find(constantTp->data_type());
    if (iter == dataTypeMap.end()) {
        bool isSupport = false;
        DCHECK(isSupport) << "Constant Data Type Not Supported!!!==> " << constantTp->data_type();
    }
    auto dataType = iter->second;

    constantParam->dataType   = dataType;
    constantParam->dataFormat = MNN::MNN_DATA_FORMAT_NCHW;

    size_t dimSize = constantTp->dims().size();
    constantParam->dims.resize(dimSize);
    size_t dataSize = 1;
    for (int i = 0; i < dimSize; ++i) {
        constantParam->dims[i] = constantTp->dims(i);
        dataSize               = dataSize * constantTp->dims(i);
    }

    const void *tensor_content = nullptr;
    if (dataSize == 1 || dimSize == 0) {
        // scalar or one dim data(only one data)
        switch (dataType) {
            case MNN::DataType_DT_INT64:
                tensor_content = constantTp->int64_data().data();
                break;
            case MNN::DataType_DT_INT32:
                tensor_content = constantTp->int32_data().data();
                break;
            default:
                tensor_content = constantTp->float_data().data();
                break;
        }
        // some Const node is Scalar, but must
        // access to data from tensor_content
        if (!tensor_content) {
            tensor_content = constantTp->raw_data().data();
        }

    } else {
        tensor_content = constantTp->raw_data().data();
    }
    if (!tensor_content) {
        DLOG(FATAL) << "Convert no data, "
                       "Please make sure "
                    << dstOp->name;
    }

    switch (iter->first) {
        case onnx::TensorProto_DataType_INT64: {
            constantParam->int32s.resize(dataSize);
            auto source = (int64_t *)tensor_content;

            for (int i = 0; i < dataSize; ++i) {
                constantParam->int32s[i] = source[i];
            }
            break;
        }
        case onnx::TensorProto_DataType_INT32: {
            auto source = (int32_t *)tensor_content;
            constantParam->int32s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->int32s[i] = source[i];
            }
            break;
        }
        case onnx::TensorProto_DataType_UINT8: {
            auto source = (uint8_t *)tensor_content;
            constantParam->uint8s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->uint8s[i] = source[i];
            }
            break;
        }
        default: {
            float *tempFloatData = (float *)tensor_content;
            constantParam->float32s.resize(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                constantParam->float32s[i] = tempFloatData[i];
            }
            break;
        }
    }

    dstOp->main.value = constantParam;
    DCHECK(onnxNode->input_size() == 0) << "Constant Should Not Have Input!!! ===> " << dstOp->name;
}

REGISTER_CONVERTER(ConstantOnnx, Constant);
