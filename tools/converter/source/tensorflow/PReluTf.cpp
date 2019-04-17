//
//  PReluTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(PReluTf);

MNN::OpType PReluTf::opType() {
    return MNN::OpType_PReLU;
}
MNN::OpParameter PReluTf::type() {
    return MNN::OpParameter_PRelu;
}

void PReluTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto PRelu = new MNN::PReluT;
    if (srcNode->inTensors.size() == 4 && srcNode->inEdges.size() == 4) {
        TmpNode *preNode = tempGraph->_getTmpNode(srcNode->inEdges[0]);
        std::vector<std::string> tensor_tmp(srcNode->inTensors);
        std::vector<std::string> edge_tmp(srcNode->inEdges);
        preNode->outTensors.push_back(tensor_tmp.at(0));
        srcNode->inTensors.clear();
        srcNode->inEdges.clear();
        srcNode->inTensors.push_back(tensor_tmp.at(0));
        srcNode->inTensors.push_back(tensor_tmp.at(1));
        srcNode->inEdges.push_back(edge_tmp.at(2));
        srcNode->inEdges.push_back(edge_tmp.at(3));
    }

    TmpNode *slope = tempGraph->_getTmpNode(srcNode->inEdges[1]);
    tensorflow::AttrValue value;
    if (find_attr_value(slope->tfNode, "value", value)) {
        const tensorflow::TensorProto &slopeTensor = value.tensor();
        int slope_size                             = slopeTensor.tensor_shape().dim(0).size();
        std::vector<float> slopeData;
        slopeData.resize(slope_size);
        const float *slope_tensor_data = reinterpret_cast<const float *>(slopeTensor.tensor_content().data());
        for (int i = 0; i < slope_size; i++) {
            slopeData[i] = slope_tensor_data[i];
        }
        PRelu->slope      = slopeData;
        PRelu->slopeCount = slope_size;
    }

    dstOp->main.value = PRelu;
    DCHECK(srcNode->inTensors.size() == 2) << "PRelu Input ERROR";
}

REGISTER_CONVERTER(PReluTf, PRelu);
