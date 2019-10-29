//
//  DequantizeTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(DequantizeTf);

MNN::OpType DequantizeTf::opType() {
    return MNN::OpType_Dequantize;
}
MNN::OpParameter DequantizeTf::type() {
    return MNN::OpParameter_Dequantize;
}

void DequantizeTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto Dequantize = new MNN::DequantizeT;
    tensorflow::AttrValue value;

    if (find_attr_value(srcNode->tfNode, "mode", value)) {
        if (value.s() == "MIN_COMBINED") {
            Dequantize->mode = MNN::QuantizeMode_MIN_COMBINED;
        } else if (value.s() == "MIN_FIRST") {
            Dequantize->mode = MNN::QuantizeMode_MIN_FIRST;
        } else if (value.s() == "SCALED") {
            Dequantize->mode = MNN::QuantizeMode_SCALED;
        }
    }

    if (find_attr_value(srcNode->tfNode, "T", value)) {
        const auto dateType = static_cast<MNN::DataType>(value.type());
        switch (dateType) {
            case MNN::DataType_DT_QUINT8:
                Dequantize->type = MNN::DataType_DT_QUINT8;
                break;
            case MNN::DataType_DT_QINT8:
                Dequantize->type = MNN::DataType_DT_QINT8;
                break;
            case MNN::DataType_DT_QUINT16:
                Dequantize->type = MNN::DataType_DT_QINT16;
                break;
            case MNN::DataType_DT_QINT16:
                Dequantize->type = MNN::DataType_DT_QUINT16;
                break;
            case MNN::DataType_DT_QINT32:
                Dequantize->type = MNN::DataType_DT_QINT32;
                break;
            default:
                DLOG(FATAL) << "unsupported type";
        }
    }

    dstOp->main.value = Dequantize;
}

REGISTER_CONVERTER(DequantizeTf, Dequantize);
