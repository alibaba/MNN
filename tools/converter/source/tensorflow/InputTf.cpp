//
//  InputTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(InputTf);

MNN::OpType InputTf::opType() {
    return MNN::OpType_Input;
}
MNN::OpParameter InputTf::type() {
    return MNN::OpParameter_Input;
}

void InputTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto inputParam = new MNN::InputT;

    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "shape", value)) {
        const tensorflow::TensorShapeProto &shape = value.shape();
        int64_t dimSize                           = shape.dim_size();
        inputParam->dims.resize(dimSize);
        DCHECK(dimSize <= 5) << "Placeholder Dim must less than "
                               "or equal to 5, is "
                            << dimSize << " " << srcNode->opName << std::endl;

        for (int i = 0; i < dimSize; ++i) {
            auto dimValue       = shape.dim(i).size();
            inputParam->dims[i] = dimValue;
        }
    }

    find_attr_value(srcNode->tfNode, "dtype", value);
    inputParam->dtype = (MNN::DataType)value.type();

    inputParam->dformat = MNN::MNN_DATA_FORMAT_NHWC;

    dstOp->main.value = inputParam;
}

REGISTER_CONVERTER(InputTf, Placeholder);
REGISTER_CONVERTER(InputTf, PlaceholderWithDefault);
