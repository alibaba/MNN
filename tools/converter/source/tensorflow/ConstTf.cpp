//
//  ConstTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(ConstTf);

MNN::OpType ConstTf::opType() {
    return MNN::OpType_Const;
}
MNN::OpParameter ConstTf::type() {
    return MNN::OpParameter_Blob;
}

void ConstTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto parameter = new MNN::BlobT;

    tensorflow::AttrValue weightsValue;
    if (!find_attr_value(srcNode->tfNode, "value", weightsValue)) {
        LOG(ERROR) << "Const Node Have Not Data!!!==> " << srcNode->opName;
    }

    parameter->dataFormat = MNN::MNN_DATA_FORMAT_NHWC;

    MNN::DataType dataType = MNN::DataType_DT_INVALID;
    dataType               = (MNN::DataType)weightsValue.tensor().dtype();

    MNN::DataType supporting[] = {MNN::DataType_DT_FLOAT, MNN::DataType_DT_INT32, MNN::DataType_DT_INT64,
                                  MNN::DataType_DT_QUINT8};
    bool isSupport = false;
    for (int i = 0; i < sizeof(supporting) / sizeof(supporting[0]); i++) {
        if (dataType == supporting[i]) {
            isSupport = true;
            break;
        }
    }
    CHECK(isSupport) << "Const Data Type Not Supported!!!==> " << dataType;
    CHECK(dataType <= MNN::DataType_MAX) << "Const Data Type Not Supported!!!==> " << dataType;

    parameter->dataType = dataType;

    size_t dimSize = weightsValue.tensor().tensor_shape().dim_size();

    parameter->dims.resize(dimSize);
    size_t dataSize = 1;
    for (int i = 0; i < dimSize; i++) {
        dataSize           = dataSize * weightsValue.tensor().tensor_shape().dim(i).size();
        parameter->dims[i] = weightsValue.tensor().tensor_shape().dim(i).size();
    }

    const void *tensor_content = nullptr;
    if (dataSize == 1 || dimSize == 0) {
        // scalar or one dim data(only one data)
        switch (dataType) {
            case MNN::DataType_DT_INT64:
                tensor_content = weightsValue.tensor().int64_val().data();
                break;
            case MNN::DataType_DT_INT32:
                tensor_content = weightsValue.tensor().int_val().data();
                break;
            default:
                tensor_content = weightsValue.tensor().float_val().data();
                break;
        }
        // some Const node is Scalar, but must
        // access to data from tensor_content
        if (!tensor_content) {
            tensor_content = weightsValue.tensor().tensor_content().data();
        }

    } else {
        tensor_content = weightsValue.tensor().tensor_content().data();
    }
    if (!tensor_content) {
        DLOG(FATAL) << "Convert no data, "
                       "Please make sure "
                    << srcNode->opName;
    }

    switch (dataType) {
        case MNN::DataType_DT_INT64: {
            //Use Int32 instead of int64
            parameter->dataType = MNN::DataType_DT_INT32;
            int64_t *tempInt64Data = (int64_t *)tensor_content;
            parameter->int32s.resize(dataSize);
            for (int i = 0; i < dataSize; i++) {
                parameter->int32s[i] = tempInt64Data[i];
            }
            break;
        }
        case MNN::DataType_DT_QUINT8: {
            unsigned char *tempInt64Data = (unsigned char *)tensor_content;
            parameter->uint8s.resize(dataSize);
            for (int i = 0; i < dataSize; i++) {
                parameter->uint8s[i] = tempInt64Data[i];
            }
            break;
        }
        case MNN::DataType_DT_INT32: {
            int32_t *tempInt32Data = (int32_t *)tensor_content;
            parameter->int32s.resize(dataSize);
            for (int i = 0; i < dataSize; i++) {
                parameter->int32s[i] = tempInt32Data[i];
            }
            break;
        }
        default: {
            float *tempFloatData = (float *)tensor_content;
            parameter->float32s.resize(dataSize);
            for (int i = 0; i < dataSize; i++) {
                parameter->float32s[i] = tempFloatData[i];
            }
            break;
        }
    }

    dstOp->main.value = parameter;
    CHECK(srcNode->inTensors.size() == 0) << "Const Should Not Have Input!!! ===> " << srcNode->opName;
}

REGISTER_CONVERTER(ConstTf, Const);
