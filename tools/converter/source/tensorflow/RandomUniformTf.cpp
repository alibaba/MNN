//
//  RandomUniformTf.cpp
//  MNNConverter
//
//  Created by MNN on 2020/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "tfOpConverter.hpp"
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"
#include "graph.pb.h"

using namespace MNN;
DECLARE_OP_CONVERTER(RandomUniformTf);

MNN::OpType RandomUniformTf::opType() {
    return MNN::OpType_RandomUniform;
}
MNN::OpParameter RandomUniformTf::type() {
    return MNN::OpParameter_RandomUniform;
}

void RandomUniformTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto parameter = new MNN::RandomUniformT;
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "seed", value)) {
        parameter->seed = value.i();
    }
    if (find_attr_value(srcNode->tfNode, "seed2", value)) {
        parameter->seed2 = value.i();
    }
    if (find_attr_value(srcNode->tfNode, "type", value)) {
        parameter->type = static_cast<MNN::DataType>(value.i());
    }
    dstOp->main.value = parameter;
}

REGISTER_CONVERTER(RandomUniformTf, RandomUniform);
