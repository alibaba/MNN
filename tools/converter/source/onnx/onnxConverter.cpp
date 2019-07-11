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
        inputParam->dtype   = onnxOpConverter::convertDataType(tensorInfo.elem_type());
        inputParam->dformat = MNN::MNN_DATA_FORMAT_NC4HW4;
        MNNOp->main.value   = inputParam;

        netT->oplists.emplace_back(MNNOp);
    }
    static std::set<std::string> treatInitializerOp{"Conv", "Upsample", "Reshape", "Const", "Gemm", "BatchNormalization"};
    // onnx node ==> MNN node
    for (int i = 0; i < nodeCount; ++i) {
        const auto& onnxNode = onnxGraph.node(i);
        const auto& opType   = onnxNode.op_type();

        if (opType == "Dropout" || opType == "Identity") {
            LG << "Skip Dropout and Identity Node: " << onnxNode.name();
            continue;
        }

        auto opConverter = onnxOpConverterSuit::get()->search(opType);
        if (nullptr == opConverter) {
            LG << "MNN Converter NOT_SUPPORTED_OP: [ " << opType << " ]";
            break;
        }
        // DCHECK(opConverter) << "MNN Converter NOT_SUPPORTED_OP: [ " << opType << " ]";
        // name maybe null, use the first output name as node-name
        MNN::OpT* MNNOp  = new MNN::OpT;
        const auto& name = onnxNode.output(0);
        MNNOp->name      = name;
        MNNOp->type      = opConverter->opType();
        MNNOp->main.type = opConverter->type();

        if (treatInitializerOp.find(onnxNode.op_type()) == treatInitializerOp.end()) {
            // Init const value
            for (int k = 0; k < onnxNode.input_size(); ++k) {
                const auto& inputName = onnxNode.input(k);
                const auto it         = initializers.find(inputName);
                if (it != initializers.end() && tensorsName.find(it->first) == tensorsName.end()) {
                    // Create const Op
                    std::unique_ptr<MNN::OpT> constOp(new MNN::OpT);
                    constOp->type       = MNN::OpType_Const;
                    constOp->main.type  = MNN::OpParameter_Blob;
                    constOp->main.value = onnxOpConverter::convertTensorToBlob(it->second);
                    auto outputIndex    = (int)netT->tensorName.size();
                    constOp->name       = it->first;
                    tensorsName.insert(std::make_pair(it->first, outputIndex));
                    netT->tensorName.emplace_back(constOp->name);
                    netT->oplists.emplace_back(std::move(constOp));
                }
            }
        }
        std::vector<const onnx::TensorProto*> opInitializers;
        for (int k = 0; k < onnxNode.input_size(); ++k) {
            const auto& inputName = onnxNode.input(k);
            const auto it         = initializers.find(inputName);
            if (it != initializers.end()) {
                opInitializers.push_back(it->second);
            }
        }

        if ((opType == "Upsample") && (onnxNode.input_size() == 2)) {
            const auto& constantNode = onnxTempGraph->_getTmpNode(onnxNode.input(1));
            for (int i = 0; i < constantNode->onnxNode->attribute_size(); ++i) {
                const auto& attributeProto = constantNode->onnxNode->attribute(i);
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
            for (int i = 0; i < onnxnode->input_size(); ++i) {
                auto inputTensorName = onnxnode->input(i);
                if (i < curNode->inEdges.size()) {
                    inputTensorName = curNode->inEdges[i];
                }
                auto it = tensorsName.find(inputTensorName);
                if (it == tensorsName.end()) {
                    continue;
                }
                op->inputIndexes.emplace_back(it->second);
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
