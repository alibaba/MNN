//
//  QuantizeMaxMinTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(QuantizeMaxMinTf);

MNN::OpType QuantizeMaxMinTf::opType() {
    return MNN::OpType_QuantizeMaxMin;
}
MNN::OpParameter QuantizeMaxMinTf::type() {
    return MNN::OpParameter_QuantizeMaxMin;
}

void QuantizeMaxMinTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto QuantizeMaxMin = new MNN::QuantizeMaxMinT;
    if (srcNode->inTensors.size() == 2) {
        std::vector<std::string> tensor_tmp(srcNode->inTensors);
        srcNode->inTensors.clear();
        srcNode->inTensors.push_back(tensor_tmp.at(0));
    }
    TmpNode *SonNode = tempGraph->_getTmpNode(srcNode->outEdges[0]);
    if (SonNode->inTensors.size() == 6) {
        std::vector<std::string> tensor_tmp(SonNode->inTensors);
        SonNode->inTensors.clear();
        SonNode->inTensors.push_back(tensor_tmp.at(0));
        SonNode->inTensors.push_back(tensor_tmp.at(1));
        SonNode->inTensors.push_back(tensor_tmp.at(2));
    }
    dstOp->main.value = QuantizeMaxMin;
    DCHECK(srcNode->inTensors.size() == 1 && srcNode->outTensors.size() == 2 && SonNode->inTensors.size() == 3)
        << "QuantizeMaxMin Input or Output ERROR";
}

REGISTER_CONVERTER(QuantizeMaxMinTf, QuantizeMaxMin);
