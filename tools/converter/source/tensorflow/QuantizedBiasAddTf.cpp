//
//  QuantizedBiasAddTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(QuantizedBiasAdd);

MNN::OpType QuantizedBiasAdd::opType() {
    return MNN::OpType_QuantizedBiasAdd;
}
MNN::OpParameter QuantizedBiasAdd::type() {
    return MNN::OpParameter_QuantizedBiasAdd;
}

void QuantizedBiasAdd::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto QuantizedBiasAdd = new MNN::QuantizedBiasAddT;

    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "T1", value)) {
        const auto dateType = static_cast<MNN::DataType>(value.type());
        switch (dateType) {
            case MNN::DataType_DT_QUINT8:
                QuantizedBiasAdd->inputType = MNN::DataType_DT_QUINT8;
                break;
            case MNN::DataType_DT_QINT8:
                QuantizedBiasAdd->inputType = MNN::DataType_DT_QINT8;
                break;
            default:
                DLOG(FATAL) << "unsupported type";
        }
    }

    if (find_attr_value(srcNode->tfNode, "out_type", value)) {
        const auto dateType = static_cast<MNN::DataType>(value.type());
        switch (dateType) {
            case MNN::DataType_DT_QINT32:
                QuantizedBiasAdd->outputType = MNN::DataType_DT_QINT32;
                break;
            default:
                DLOG(FATAL) << "unsupported type";
        }
    }

    dstOp->main.value = QuantizedBiasAdd;

    DCHECK(srcNode->inEdges.size() == 4) << "QuantizedBiasAdd Input ERROR";
}

REGISTER_CONVERTER(QuantizedBiasAdd, QuantizedBiasAdd);
