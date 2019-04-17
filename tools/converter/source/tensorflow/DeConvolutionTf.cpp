//
//  DeConvolutionTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(DeConvolutionTf);

MNN::OpType DeConvolutionTf::opType() {
    return MNN::OpType_Deconvolution;
}
MNN::OpParameter DeConvolutionTf::type() {
    return MNN::OpParameter_Convolution2D;
}

// input : stack, weight, inputTensor, (bias)
void DeConvolutionTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto deconvolution = new MNN::Convolution2DT;

    TmpNode *weightNode = tempGraph->_getTmpNode(srcNode->inEdges[1]);
    tensorflow::AttrValue value;

    int kh         = 1;
    int kw         = 1;
    int num_input  = 1;
    int num_output = 1;
    if (find_attr_value(weightNode->tfNode, "value", value)) {
        const tensorflow::TensorProto &weightTensor = value.tensor();
        const tensorflow::TensorShapeProto &shape   = weightTensor.tensor_shape();
        DCHECK(shape.dim_size() == 4) << "Weight Shape Wrong!";
        kh             = shape.dim(0).size();
        kw             = shape.dim(1).size();
        num_input      = shape.dim(2).size();
        num_output     = shape.dim(3).size();
        int weightSize = kh * kw * num_input * num_output;
        std::vector<float> weightData;
        weightData.resize(weightSize);
        const float *weightTensorData = reinterpret_cast<const float *>(weightTensor.tensor_content().data());
        float *weightDataTemp         = new float[weightSize];
        convertDataFormat(weightTensorData, weightDataTemp, kh, kw, num_input, num_output);
        for (int i = 0; i < weightSize; i++) {
            weightData[i] = weightDataTemp[i];
        }
        delete[] weightDataTemp;
        deconvolution->weight = weightData;
    }

    TmpNode *biasNode = srcNode->inEdges.size() == 4 ? tempGraph->_getTmpNode(srcNode->inEdges[3]) : nullptr;
    std::vector<float> biasData(num_input, 0.0f);
    if (biasNode != nullptr) {
        DCHECK(biasNode->opType == "Const") << "Deconvolution Op Bias ERROR!!!";
        if (find_attr_value(biasNode->tfNode, "value", value)) {
            const tensorflow::TensorProto &biasTensor = value.tensor();
            if (num_input == 1) {
                biasData[0] = biasTensor.float_val().data()[0];
            } else {
                const float *biasTensorData = reinterpret_cast<const float *>(biasTensor.tensor_content().data());
                for (int i = 0; i < num_input; i++) {
                    biasData[i] = biasTensorData[i];
                }
            }
        }
    }
    deconvolution->bias = biasData;

    deconvolution->common = std::unique_ptr<MNN::Convolution2DCommonT>(new MNN::Convolution2DCommonT);
    auto &common          = deconvolution->common;

    common->relu        = false;
    common->group       = 1;
    common->outputCount = num_input;
    common->inputCount = num_output;
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

    dstOp->main.value = deconvolution;
}

REGISTER_CONVERTER(DeConvolutionTf, Conv2DBackpropInput);
