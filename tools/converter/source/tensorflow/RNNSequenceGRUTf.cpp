//
//  RNNSequenceGRUTf.cpp
//  MNNConverter
//
//  Created by MNN on 2019/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

DECLARE_OP_CONVERTER(RNNSequenceGRUTf);
MNN::OpType RNNSequenceGRUTf::opType() {
    return MNN::OpType_RNNSequenceGRU;
}

MNN::OpParameter RNNSequenceGRUTf::type() {
    return MNN::OpParameter_RNNParam;
}

void RNNSequenceGRUTf::run(MNN::OpT* dstOp, TmpNode* srcNode, TmpGraph* tempGraph) {
    const int inputSize = srcNode->tfNode->input_size();
    DCHECK(inputSize == 5 || inputSize == 9) << "RNNSequenceGRU input error! ==> " << srcNode->opName;
    auto rnnGRUParam = new MNN::RNNParamT;

    tensorflow::AttrValue value;

    rnnGRUParam->isBidirectionalRNN = false;
    if (find_attr_value(srcNode->tfNode, "is_bidirectional_rnn", value)) {
        rnnGRUParam->isBidirectionalRNN = value.b();
    }
    rnnGRUParam->keepAllOutputs = false;
    if (find_attr_value(srcNode->tfNode, "keep_all_outputs", value)) {
        rnnGRUParam->keepAllOutputs = value.b();
    }

    std::function<void(tensorflow::AttrValue & value, MNN::BlobT * data)> weightProcess =
        [](tensorflow::AttrValue& value, MNN::BlobT* data) {
            const auto& weightTensor = value.tensor();
            DCHECK(2 == weightTensor.tensor_shape().dim_size()) << "Shape error";
            data->dataType   = MNN::DataType_DT_FLOAT;
            data->dataFormat = MNN::MNN_DATA_FORMAT_NHWC;
            data->dims.resize(2);
            data->dims[0]      = weightTensor.tensor_shape().dim(0).size();
            data->dims[1]      = weightTensor.tensor_shape().dim(1).size();
            const int dataSize = weightTensor.tensor_content().size() / sizeof(float);
            data->float32s.resize(dataSize);
            ::memcpy(data->float32s.data(), weightTensor.tensor_content().data(), dataSize * sizeof(float));
        };

    std::function<void(tensorflow::AttrValue & value, MNN::BlobT * data)> biasProcess = [](tensorflow::AttrValue& value,
                                                                                           MNN::BlobT* data) {
        const auto& biasTensor = value.tensor();
        DCHECK(1 == biasTensor.tensor_shape().dim_size()) << "Shape error";
        data->dataType   = MNN::DataType_DT_FLOAT;
        data->dataFormat = MNN::MNN_DATA_FORMAT_NHWC;
        data->dims.resize(1);
        data->dims[0]      = biasTensor.tensor_shape().dim(0).size();
        const int dataSize = biasTensor.tensor_content().size() / sizeof(float);
        data->float32s.resize(dataSize);
        ::memcpy(data->float32s.data(), biasTensor.tensor_content().data(), dataSize * sizeof(float));
    };

    // forward weight
    auto gateWeightNode      = tempGraph->_getTmpNode(srcNode->inEdges[1]);
    auto gateBiasNode        = tempGraph->_getTmpNode(srcNode->inEdges[2]);
    auto candidateWeightNode = tempGraph->_getTmpNode(srcNode->inEdges[3]);
    auto candidateBiasNode   = tempGraph->_getTmpNode(srcNode->inEdges[4]);
    if (find_attr_value(gateWeightNode->tfNode, "value", value)) {
        rnnGRUParam->fwGateWeight = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
        weightProcess(value, rnnGRUParam->fwGateWeight.get());
    } else {
        LOG(FATAL) << "ERROR!";
    }
    if (find_attr_value(gateBiasNode->tfNode, "value", value)) {
        rnnGRUParam->fwGateBias = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
        biasProcess(value, rnnGRUParam->fwGateBias.get());
    } else {
        LOG(FATAL) << "ERROR!";
    }

    if (find_attr_value(candidateWeightNode->tfNode, "value", value)) {
        rnnGRUParam->fwCandidateWeight = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
        weightProcess(value, rnnGRUParam->fwCandidateWeight.get());
    } else {
        LOG(FATAL) << "ERROR!";
    }

    if (find_attr_value(candidateBiasNode->tfNode, "value", value)) {
        rnnGRUParam->fwCandidateBias = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
        biasProcess(value, rnnGRUParam->fwCandidateBias.get());
    } else {
        LOG(FATAL) << "ERROR!";
    }

    // option: backward weight
    if (inputSize == 9) {
        // backward weight
        auto gateWeightNode      = tempGraph->_getTmpNode(srcNode->inEdges[5]);
        auto gateBiasNode        = tempGraph->_getTmpNode(srcNode->inEdges[6]);
        auto candidateWeightNode = tempGraph->_getTmpNode(srcNode->inEdges[7]);
        auto candidateBiasNode   = tempGraph->_getTmpNode(srcNode->inEdges[8]);
        if (find_attr_value(gateWeightNode->tfNode, "value", value)) {
            rnnGRUParam->bwGateWeight = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
            weightProcess(value, rnnGRUParam->bwGateWeight.get());
        } else {
            LOG(FATAL) << "ERROR!";
        }
        if (find_attr_value(gateBiasNode->tfNode, "value", value)) {
            rnnGRUParam->bwGateBias = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
            biasProcess(value, rnnGRUParam->bwGateBias.get());
        } else {
            LOG(FATAL) << "ERROR!";
        }

        if (find_attr_value(candidateWeightNode->tfNode, "value", value)) {
            rnnGRUParam->bwCandidateWeight = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
            weightProcess(value, rnnGRUParam->bwCandidateWeight.get());
        } else {
            LOG(FATAL) << "ERROR!";
        }

        if (find_attr_value(candidateBiasNode->tfNode, "value", value)) {
            rnnGRUParam->bwCandidateBias = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
            biasProcess(value, rnnGRUParam->bwCandidateBias.get());
        } else {
            LOG(FATAL) << "ERROR!";
        }
    }

    rnnGRUParam->numUnits = rnnGRUParam->fwCandidateBias->dims[0];

    dstOp->main.value = rnnGRUParam;
}

REGISTER_CONVERTER(RNNSequenceGRUTf, RNNSequenceGRU);
