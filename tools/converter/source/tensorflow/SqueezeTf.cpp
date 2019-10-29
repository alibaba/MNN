//
//  SqueezeTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(SqueezeTf);

MNN::OpType SqueezeTf::opType() {
    return MNN::OpType_Squeeze;
}
MNN::OpParameter SqueezeTf::type() {
    return MNN::OpParameter_SqueezeParam;
}

void SqueezeTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto squeeze = new MNN::SqueezeParamT;

    tensorflow::AttrValue value_squeezeDims;
    if (find_attr_value(srcNode->tfNode, "squeeze_dims", value_squeezeDims)) {
        const int dimSize = value_squeezeDims.list().i_size();
        for (int i = 0; i < dimSize; i++) {
            squeeze->squeezeDims.push_back((int32_t)value_squeezeDims.list().i(i));
        }
    }

    dstOp->main.value = squeeze;
}

REGISTER_CONVERTER(SqueezeTf, Squeeze);
