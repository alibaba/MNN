//
//  onnxConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <iostream>

#include "MNN_generated.h"
#include "OnnxUtils.hpp"
#include "logkit.h"

#include "OnnxTmpGraph.hpp"
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"
#include "onnx.pb.h"
#include "onnxConverter.hpp"
#include "onnxOpConverter.hpp"

int onnx2MNNNet(const std::string inputModel, const std::string bizCode, std::unique_ptr<MNN::NetT>& netT) {
    onnx::ModelProto onnxModel;
    // read ONNX Model
    bool success = onnx_read_proto_from_binary(inputModel.c_str(), &onnxModel);
    DCHECK(success) << "read onnx model failed: " << inputModel;

    LOG(INFO) << "ONNX Model ir version: " << onnxModel.ir_version();

    const auto& onnxGraph = onnxModel.graph();
    const int nodeCount   = onnxGraph.node_size();

    std::shared_ptr<OnnxTmpGraph> onnxTempGraph(new OnnxTmpGraph(&onnxGraph));

    std::map<std::string, int> tensorsName;
    // find the inputs which do not have initializer
    const auto& initializers = onnxTempGraph->mInitializers;
    const auto& inputs       = onnxTempGraph->mInputs;
    const auto& outputs      = onnxTempGraph->mOutputs;
    for (const auto& iter : inputs) {
        bool notHaveInitializer = initializers.find(iter.first) == initializers.end();
        if (notHaveInitializer) {
            netT->tensorName.push_back(iter.first);
            tensorsName.insert(std::make_pair(iter.first, tensorsName.size()));
        }
    }

    // set input node to MNN net
    for (const auto& iter : tensorsName) {
        // here tensorsName are true Input node name
        MNN::OpT* MNNOp  = new MNN::OpT;
        MNNOp->name      = iter.first;
        MNNOp->type      = MNN::OpType_Input;
        MNNOp->main.type = MNN::OpParameter_Input;
        auto inputParam  = new MNN::InputT;
        const auto it    = inputs.find(iter.first);
        DCHECK(it != inputs.end()) << "Input Paramter ERROR ==> " << iter.first;
        const auto& tensorInfo = (it->second)->type().tensor_type();
        const int inputDimSize = tensorInfo.shape().dim_size();
        inputParam->dims.resize(inputDimSize);
        for (int i = 0; i < inputDimSize; ++i) {
            inputParam->dims[i] = tensorInfo.shape().dim(i).dim_value();
        }
        // TODO!!! onnx data type to MNN data type
        inputParam->dtype   = static_cast<MNN::DataType>(tensorInfo.elem_type());
        inputParam->dformat = MNN::MNN_DATA_FORMAT_NCHW;
        MNNOp->main.value   = inputParam;

        netT->oplists.emplace_back(MNNOp);
    }

    // onnx node ==> MNN node
    for (int i = 0; i < nodeCount; ++i) {
        const auto& onnxNode = onnxGraph.node(i);
        const auto& opType   = onnxNode.op_type();

        if (opType == "Dropout" || opType == "Identity")
            continue;

        MNN::OpT* MNNOp  = new MNN::OpT;
        auto opConverter = onnxOpConverterSuit::get()->search(opType);
        DCHECK(opConverter) << "MNN Converter NOT_SUPPORTED_OP: [ " << opType << " ]";
        // name maybe null, use the first output name as node-name
        const auto& name = onnxNode.output(0);
        MNNOp->name      = name;
        MNNOp->type      = opConverter->opType();
        MNNOp->main.type = opConverter->type();
        std::vector<const onnx::TensorProto*> opInitializers;
        for (int k = 0; k < onnxNode.input_size(); ++k) {
            const auto& inputName = onnxNode.input(k);
            const auto it         = initializers.find(inputName);
            if (it != initializers.end()) {
                opInitializers.push_back(it->second);
            }
        }

        if ((opType == "Upsample") && (onnxNode.input_size() == 2)) {
            const auto& parentNode = onnxTempGraph->_getTmpNode(onnxNode.input(1));
            for (int i = 0; i < parentNode->onnxNode->attribute_size(); ++i) {
                const auto& attributeProto = parentNode->onnxNode->attribute(i);
                if (attributeProto.name() == "value") {
                    opInitializers.push_back(&attributeProto.t());
                }
            }
        }

        opConverter->run(MNNOp, &onnxNode, opInitializers);

        netT->oplists.emplace_back(MNNOp);

        netT->tensorName.push_back(name);
        tensorsName.insert(std::make_pair(name, tensorsName.size()));
    }

    // set input-output tensor's index
    for (auto& op : netT->oplists) {
        const auto& name    = op->name;
        const auto& curNode = onnxTempGraph->_getTmpNode(name);
        if (curNode) {
            auto onnxnode = curNode->onnxNode;
            // output index
            for (int i = 0; i < onnxnode->output_size(); ++i) {
                const auto it = tensorsName.find(onnxnode->output(i));
                DCHECK(it != tensorsName.end()) << "Tensor Name Not Found!!! ==> " << onnxnode->output(i);
                op->outputIndexes.push_back(it->second);
            }
            // input index
            const int inEdgesNum = curNode->inEdges.size();
            if (inEdgesNum == 0) {
                // incase: this node's input is input not graph-node
                for (int k = 0; k < onnxnode->input_size(); ++k) {
                    const auto& inputname = onnxnode->input(k);
                    // check whether in initializer
                    bool haveInitializer = initializers.find(inputname) != initializers.end();
                    if (haveInitializer)
                        continue;
                    const auto it = tensorsName.find(inputname);
                    DCHECK(it != tensorsName.end()) << "Tensor Name Not Found!!! ==> " << inputname;
                    op->inputIndexes.push_back(it->second);
                }
            } else {
                // delete the const input edges for Upsample node!!! Must to do
                // Const node, others no delete
                if ((curNode->opType == "Upsample") && (inEdgesNum == 2)) {
                    const std::vector<std::string>::iterator it2delete = curNode->inEdges.begin() + 1;
                    curNode->inEdges.erase(it2delete);
                    DCHECK(curNode->inEdges.size() == 1) << "Upsample op Input ERROR!!! ===> " << name;
                }

                for (int j = 0; j < curNode->inEdges.size(); ++j) {
                    const auto it = tensorsName.find(curNode->inEdges[j]);
                    DCHECK(it != tensorsName.end()) << "Tensor Name Not Found!!! ==> " << curNode->inEdges[j];
                    op->inputIndexes.push_back(it->second);
                }
            }
        } else {
            // input node(output index)
            const auto it = tensorsName.find(name);
            DCHECK(it != tensorsName.end()) << "Tensor Name Not Found!!! ==> " << name;
            op->outputIndexes.push_back(it->second);
        }
    }

    netT->tensorNumber = tensorsName.size();
    // set MNN net output name
    for (const auto& iter : outputs) {
        netT->outputName.push_back(iter.first);
    }

    netT->sourceType = MNN::NetSource_CAFFE;
    netT->bizCode    = bizCode;

    return 0;
}
