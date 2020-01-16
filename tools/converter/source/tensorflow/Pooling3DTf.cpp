//
//  Pooling3DTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "graph.pb.h"
#include "tfOpConverter.hpp"

DECLARE_OP_CONVERTER(Pooling3DTf);

MNN::OpType Pooling3DTf::opType() {
    return MNN::OpType_Pooling3D;
}
MNN::OpParameter Pooling3DTf::type() {
    return MNN::OpParameter_Pool3D;
}

// input: tensor
void Pooling3DTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto pool3d = new MNN::Pool3DT;
    
    tensorflow::AttrValue value;
    
    int stride_h      = 1;
    int stride_w      = 1;
    
    if (srcNode->opType == "AvgPool3D") {
        pool3d->type = MNN::PoolType_AVEPOOL;
    } else if (srcNode->opType == "MaxPool3D") {
        pool3d->type = MNN::PoolType_MAXPOOL;
    } else {
        DLOG(ERROR) << "Not Support This Pooling Type: " << srcNode->opType;
    }
    
    if (find_attr_value(srcNode->tfNode, "ksize", value)) {
        std::vector<int32_t> kernels;
        for (int i = 1; i < 4; ++i) {
            kernels.push_back(value.list().i(i));
        }
        pool3d->kernels = kernels;
    }
    
    if (find_attr_value(srcNode->tfNode, "strides", value)) {
        std::vector<int32_t> strides;
        for (int i = 1; i < 4; ++i) {
            strides.push_back(value.list().i(i));
        }
        pool3d->strides = strides;
    }
    
    if (find_attr_value(srcNode->tfNode, "padding", value)) {
        if (value.s() == "VALID") {
            pool3d->padType = MNN::PoolPadType_VALID;
            pool3d->pads = std::vector<int32_t>(3, 0);
        } else if (value.s() == "SAME") {
            pool3d->padType = MNN::PoolPadType_SAME;
        } else {
            DLOG(ERROR) << "Not Support This Padding Mode";
        }
    }
        
    dstOp->main.value = pool3d;
}

REGISTER_CONVERTER(Pooling3DTf, MaxPool3D);
REGISTER_CONVERTER(Pooling3DTf, AvgPool3D);
