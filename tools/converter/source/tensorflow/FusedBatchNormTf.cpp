//
//  FusedBatchNormTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(FusedBatchNormTf);

MNN::OpType FusedBatchNormTf::opType() {
    return MNN::OpType_BatchNorm;
}
MNN::OpParameter FusedBatchNormTf::type() {
    return MNN::OpParameter_BatchNorm;
}

// input: tensor, scale, bias, mean, var
void FusedBatchNormTf::run(MNN::OpT *dstOp, TmpNode *srcNode, TmpGraph *tempGraph) {
    auto batchnorm = new MNN::BatchNormT;

    const auto inputSize = srcNode->inEdges.size();
    tensorflow::AttrValue value;
    if (5 == inputSize) {
        // general BatchNorm
        TmpNode *scaleNode = tempGraph->_getTmpNode(srcNode->inEdges[1]);
        TmpNode *biasNode  = tempGraph->_getTmpNode(srcNode->inEdges[2]);
        TmpNode *meanNode  = tempGraph->_getTmpNode(srcNode->inEdges[3]);
        TmpNode *varNode   = tempGraph->_getTmpNode(srcNode->inEdges[4]);

        float epsilon = 0.001;
        if (find_attr_value(srcNode->tfNode, "epsilon", value)) {
            epsilon = value.f();
        }
        batchnorm->epsilon = epsilon;
        int channels       = 0;
        while (1) {
            // get channels value according to tensor.tensor_content(raw data) firstly,
            // if chnnels equal to 0, then set channels to be 1
            if (find_attr_value(scaleNode->tfNode, "value", value)) {
                const tensorflow::TensorProto &tmpTensor = value.tensor();
                channels                                 = tmpTensor.tensor_content().size() / sizeof(float);
            }
            if (channels > 0)
                break;
            if (find_attr_value(biasNode->tfNode, "value", value)) {
                const tensorflow::TensorProto &tmpTensor = value.tensor();
                channels                                 = tmpTensor.tensor_content().size() / sizeof(float);
            }
            if (channels > 0)
                break;
            if (find_attr_value(meanNode->tfNode, "value", value)) {
                const tensorflow::TensorProto &tmpTensor = value.tensor();
                channels                                 = tmpTensor.tensor_content().size() / sizeof(float);
            }
            if (channels > 0)
                break;
            if (find_attr_value(varNode->tfNode, "value", value)) {
                const tensorflow::TensorProto &tmpTensor = value.tensor();
                channels                                 = tmpTensor.tensor_content().size() / sizeof(float);
            }
            break;
        }
        if (channels == 0) {
            channels = 1;
        }

        batchnorm->channels = channels;
        batchnorm->varData.resize(channels);
        batchnorm->slopeData.resize(channels);
        batchnorm->biasData.resize(channels);
        batchnorm->meanData.resize(channels);

        auto processData = [&](TmpNode *node, std::vector<float> &dataContainer, bool addEps) {
            if (find_attr_value(node->tfNode, "value", value)) {
                const tensorflow::TensorProto &tmpTensor = value.tensor();
                if (tmpTensor.tensor_content().size() > 0) {
                    const float *tmpTensorData = reinterpret_cast<const float *>(tmpTensor.tensor_content().data());
                    for (int i = 0; i < channels; i++) {
                        if (addEps) {
                            dataContainer[i] = tmpTensorData[i] + epsilon;
                        } else {
                            dataContainer[i] = tmpTensorData[i];
                        }
                    }
                } else {
                    if (tmpTensor.float_val_size() > 0) {
                        auto data = tmpTensor.float_val().data();
                        for (int i = 0; i < channels; i++) {
                            if (addEps) {
                                dataContainer[i] = data[0] + epsilon;
                            } else {
                                dataContainer[i] = data[0];
                            }
                        }
                    }
                }
            }
        };

        processData(scaleNode, batchnorm->slopeData, false);
        processData(biasNode, batchnorm->biasData, false);
        processData(meanNode, batchnorm->meanData, false);
        processData(varNode, batchnorm->varData, true);

        DCHECK(srcNode->inTensors.size() == 1) << "FusedBatchNorm Input ERROR!!! ===> " << srcNode->opName;
    } else if (4 == inputSize) {
        // instance_norm
        TmpNode *scaleNode = tempGraph->_getTmpNode(srcNode->inEdges[1]);
        TmpNode *biasNode  = tempGraph->_getTmpNode(srcNode->inEdges[2]);

        float epsilon = 0.001;
        if (find_attr_value(srcNode->tfNode, "epsilon", value)) {
            epsilon = value.tensor().float_val(0);
        }
        batchnorm->epsilon = epsilon;
        int channels       = 0;
        if (find_attr_value(biasNode->tfNode, "value", value)) {
            const tensorflow::TensorProto &biasTensor = value.tensor();
            channels                                  = biasTensor.tensor_content().size() / sizeof(float);
            batchnorm->channels                       = channels;
            batchnorm->biasData.resize(channels);
            const float *biasTensorData = reinterpret_cast<const float *>(biasTensor.tensor_content().data());
            for (int i = 0; i < channels; i++) {
                batchnorm->biasData[i] = biasTensorData[i];
            }
        }
        batchnorm->slopeData.resize(channels);
        if (find_attr_value(scaleNode->tfNode, "value", value)) {
            const tensorflow::TensorProto &slopeTensor = value.tensor();
            if (slopeTensor.tensor_content().size() > 0) {
                const float *slopeTensorData = reinterpret_cast<const float *>(slopeTensor.tensor_content().data());
                for (int i = 0; i < channels; i++) {
                    batchnorm->slopeData[i] = slopeTensorData[i];
                }
            } else {
                if (slopeTensor.float_val_size() > 0) {
                    float slope = *slopeTensor.float_val().data();
                    for (int i = 0; i < channels; i++) {
                        batchnorm->slopeData[i] = slope;
                    }
                }
            }
        }

        DCHECK(srcNode->inTensors.size() == 3) << "FusedBatchNorm Input ERROR!!! ===> " << srcNode->opName;
    } else {
        DLOG(FATAL) << "FusedBatchNorm Input ERROR";
    }

    dstOp->main.value = batchnorm;
}

REGISTER_CONVERTER(FusedBatchNormTf, FusedBatchNorm);
REGISTER_CONVERTER(FusedBatchNormTf, InstanceNorm);
