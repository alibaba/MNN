//
//  QuantizedRelu6Tf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(QuantizedRelu6);

MNN::OpType QuantizedRelu6::opType() {
    return MNN::OpType_QuantizedRelu6;
}
MNN::OpParameter QuantizedRelu6::type() {
    return MNN::OpParameter_QuantizedRelu6;
}

void QuantizedRelu6::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto QuantizedRelu6 = new MNN::QuantizedRelu6T;

    dstOp->main.value = QuantizedRelu6;

    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "Tinput", value)) {
        const auto dateType = static_cast<MNN::DataType>(value.type());
        switch (dateType) {
            case MNN::DataType_DT_QUINT8:
                QuantizedRelu6->type = MNN::DataType_DT_QUINT8;
                break;
            case MNN::DataType_DT_QINT32:
                QuantizedRelu6->type = MNN::DataType_DT_QINT32;
                break;
            default:
                QuantizedRelu6->type = MNN::DataType_DT_QUINT8;
                break;
        }
    }

    CHECK(srcNode->inEdges.size() == 1) << "QuantizedRelu Input ERROR";
}

REGISTER_CONVERTER(QuantizedRelu6, QuantizedRelu6);
