//
//  QuantizeV2Tf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(QuantizeV2Tf);

MNN::OpType QuantizeV2Tf::opType() {
    return MNN::OpType_QuantizeV2;
}
MNN::OpParameter QuantizeV2Tf::type() {
    return MNN::OpParameter_QuantizeV2;
}

void QuantizeV2Tf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto QuantizeV2 = new MNN::QuantizeV2T;

    tensorflow::AttrValue value;

    if (find_attr_value(srcNode->tfNode, "mode", value)) {
        if (value.s() == "MIN_COMBINED") {
            QuantizeV2->mode = MNN::QuantizeMode_MIN_COMBINED;
        } else if (value.s() == "MIN_FIRST") {
            QuantizeV2->mode = MNN::QuantizeMode_MIN_FIRST;
        } else if (value.s() == "SCALED") {
            QuantizeV2->mode = MNN::QuantizeMode_SCALED;
        }
    }

    if (find_attr_value(srcNode->tfNode, "T", value)) {
        const auto dateType = static_cast<MNN::DataType>(value.type());
        switch (dateType) {
            case MNN::DataType_DT_QUINT8:
                QuantizeV2->type = MNN::DataType_DT_QUINT8;
                break;
            case MNN::DataType_DT_QINT8:
                QuantizeV2->type = MNN::DataType_DT_QINT8;
                break;
            case MNN::DataType_DT_QUINT16:
                QuantizeV2->type = MNN::DataType_DT_QINT16;
                break;
            case MNN::DataType_DT_QINT16:
                QuantizeV2->type = MNN::DataType_DT_QUINT16;
                break;
            case MNN::DataType_DT_QINT32:
                QuantizeV2->type = MNN::DataType_DT_QINT32;
                break;
            default:
                DLOG(FATAL) << "unsupported type";
        }
    }

    if (find_attr_value(srcNode->tfNode, "round_mode", value)) {
        if (value.s() == "HALF_AWAY_FROM_ZERO") {
            QuantizeV2->roundMode = MNN::QuantizeRoundMode_HALF_AWAY_FROM_ZERO;
        } else if (value.s() == "HALF_TO_EVEN") {
            QuantizeV2->roundMode = MNN::QuantizeRoundMode_HALF_TO_EVEN;
        }
    }

    dstOp->main.value = QuantizeV2;

    // when in fuse the max/min case inEdges is 2
    CHECK(srcNode->inEdges.size() == 3 || srcNode->inEdges.size() == 2) << "QuantizeV2 Input ERROR";
}

REGISTER_CONVERTER(QuantizeV2Tf, QuantizeV2);
