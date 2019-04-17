//
//  QuantizedReluTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(QuantizedRelu);

MNN::OpType QuantizedRelu::opType() {
    return MNN::OpType_QuantizedRelu;
}
MNN::OpParameter QuantizedRelu::type() {
    return MNN::OpParameter_QuantizedRelu;
}

void QuantizedRelu::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto QuantizedRelu = new MNN::QuantizedReluT;

    dstOp->main.value = QuantizedRelu;

    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "Tinput", value)) {
        const auto dateType = static_cast<MNN::DataType>(value.type());
        switch (dateType) {
            case MNN::DataType_DT_QUINT8:
                QuantizedRelu->type = MNN::DataType_DT_QUINT8;
                break;
            case MNN::DataType_DT_QINT32:
                QuantizedRelu->type = MNN::DataType_DT_QINT32;
                break;
            default:
                DLOG(FATAL) << "unsupported type";
        }
    }

    CHECK(srcNode->inEdges.size() == 1) << "QuantizedRelu Input ERROR";
}

REGISTER_CONVERTER(QuantizedRelu, QuantizedRelu);
