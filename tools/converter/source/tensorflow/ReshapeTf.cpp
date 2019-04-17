//
//  ReshapeTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(ReshapeTf);
MNN::OpType ReshapeTf::opType() {
    return MNN::OpType_Reshape;
}
MNN::OpParameter ReshapeTf::type() {
    return MNN::OpParameter_Reshape;
}

void ReshapeTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto reshape = new MNN::ReshapeT;

    TmpNode *shapeNode = tempGraph->_getTmpNode(srcNode->inEdges[1]);
    if (shapeNode->opType != "Const") {
        dstOp->main.value = reshape;
        return;
    }

    // Const Shape
    tensorflow::AttrValue value;
    if (find_attr_value(shapeNode->tfNode, "value", value)) {
        MNN::DataType dataType = (MNN::DataType)value.tensor().dtype();
        CHECK(dataType == MNN::DataType_DT_INT32) << "Shape Dtype ERROR" << srcNode->opName;

        reshape->dimType = MNN::MNN_DATA_FORMAT_NHWC;
        if (!value.tensor().tensor_content().empty()) // int32
        {
            const int *data = reinterpret_cast<const int *>(value.tensor().tensor_content().c_str());
            int size        = value.tensor().tensor_content().size() / sizeof(int);
            CHECK(size > 1) << "Shape Data ERROR!!! ===> " << srcNode->opName;
            reshape->dims.resize(size);
            for (int i = 0; i < size; ++i) {
                reshape->dims[i] = data[i];
            }
        } else {
            // only one int value
            reshape->dims.resize(1);
            reshape->dims[0] = value.tensor().int_val(0);
        }
    }

    dstOp->main.value = reshape;
}

REGISTER_CONVERTER(ReshapeTf, Reshape);
