//
//  ReshapeOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ReshapeOnnx);

MNN::OpType ReshapeOnnx::opType() {
    return MNN::OpType_Reshape;
}
MNN::OpParameter ReshapeOnnx::type() {
    return MNN::OpParameter_Reshape;
}

void ReshapeOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                      std::vector<const onnx::TensorProto*> initializers) {
    auto reshape      = new MNN::ReshapeT;
    reshape->dimType  = MNN::MNN_DATA_FORMAT_NCHW;
    dstOp->main.value = reshape;
    if (initializers.size() == 0) {
        return;
    }
    DCHECK(initializers.size() == 1) << "Reshape Input ERROR! ==> " << dstOp->name << ":" << initializers.size();
    auto shape = initializers[0];
    DCHECK(shape->data_type() == ::onnx::TensorProto_DataType_INT64) << "Reshape Data Type ERROR!";
    const int dimSize = shape->dims(0);
    reshape->dims.resize(dimSize);

    if (shape->int64_data_size() != 0) {
        for (int i = 0; i < dimSize; ++i) {
            reshape->dims[i] = static_cast<int>(shape->int64_data(i));
        }
    } else if (shape->raw_data().data()) {
        const int64_t* shapeData = reinterpret_cast<const int64_t*>(shape->raw_data().data());
        for (int i = 0; i < dimSize; ++i) {
            reshape->dims[i] = static_cast<int>(shapeData[i]);
        }
    } else {
        DLOG(ERROR) << "Reshape Shape Data ERROR! ==> " << dstOp->name;
    }
}

REGISTER_CONVERTER(ReshapeOnnx, Reshape);
