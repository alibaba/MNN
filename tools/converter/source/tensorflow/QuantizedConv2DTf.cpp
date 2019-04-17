//
//  QuantizedConv2DTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(QuantizedConv2DTf);

MNN::OpType QuantizedConv2DTf::opType() {
    return MNN::OpType_TfQuantizedConv2D;
}
MNN::OpParameter QuantizedConv2DTf::type() {
    return MNN::OpParameter_TfQuantizedConv2D;
}

void QuantizedConv2DTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto TfQuantizedConv2D = new MNN::TfQuantizedConv2DT;

    TmpNode *weightNode = tempGraph->_getTmpNode(srcNode->inEdges[1]);
    tensorflow::AttrValue value;

    int kh         = 1;
    int kw         = 1;
    int num_input  = 1;
    int num_output = 1;
    if (find_attr_value(weightNode->tfNode, "value", value)) {
        const tensorflow::TensorProto &weightTensor = value.tensor();
        const tensorflow::TensorShapeProto &shape   = weightTensor.tensor_shape();
        CHECK(shape.dim_size() == 4) << "Weight Shape Wrong!";
        kh         = shape.dim(0).size();
        kw         = shape.dim(1).size();
        num_input  = shape.dim(2).size();
        num_output = shape.dim(3).size();
    }

    TfQuantizedConv2D->common = std::unique_ptr<MNN::Convolution2DCommonT>(new MNN::Convolution2DCommonT);
    auto &common              = TfQuantizedConv2D->common;

    common->relu        = false;
    common->group       = 1;
    common->outputCount = num_output;
    common->kernelX     = kw;
    common->kernelY     = kh;

    int dilation_h = 1;
    int dilation_w = 1;
    if (find_attr_value(srcNode->tfNode, "rate", value)) {
        // height, width
        dilation_h = value.list().i(0);
        dilation_w = value.list().i(1);
    }
    common->dilateX = dilation_w;
    common->dilateY = dilation_h;

    int stride_h = 1;
    int stride_w = 1;
    if (find_attr_value(srcNode->tfNode, "strides", value)) {
        // batch, height, width, channels
        stride_h = value.list().i(1);
        stride_w = value.list().i(2);
    }
    common->strideX = stride_w;
    common->strideY = stride_h;

    common->padX = 0;
    common->padY = 0;

    common->padMode = MNN::PadMode_SAME;
    if (find_attr_value(srcNode->tfNode, "padding", value)) {
        if (value.s() == "VALID") {
            common->padMode = MNN::PadMode_VALID;
        }
    }

    dstOp->main.value = TfQuantizedConv2D;
}

REGISTER_CONVERTER(QuantizedConv2DTf, QuantizedConv2D);
