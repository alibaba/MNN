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

    // op_name: name
    // get mnn op pointer conveniently, then manipulate the mnn op
    std::map<std::string, MNN::OpT*> mnnNodesMap;
    // all tensors container
    std::map<std::string, int> tensorsName;
    // find the inputs which do not have initializer
    const auto& initializers         = onnxTempGraph->mInitializers;
    const auto& inputs               = onnxTempGraph->mInputs;
    const auto& outputs              = onnxTempGraph->mOutputs;
    const auto& constantNodeToDelete = onnxTempGraph->mConstantNodeToDelete;
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
        inputParam->dformat = MNN::MNN_DATA_FORMAT_NCHW;
        MNNOp->outputIndexes.push_back(tensorsName[iter.first]);
        MNNOp->main.value = inputParam;
        mnnNodesMap.insert(std::make_pair(iter.first, MNNOp));
        netT->oplists.emplace_back(MNNOp);
    }

    // onnx node ==> MNN node
    for (int i = 0; i < nodeCount; ++i) {
        const auto& onnxNode = onnxGraph.node(i);
        const auto& opType   = onnxNode.op_type();

        // name maybe null, use the first output name as node-name
        const auto& name = onnxNode.output(0);

        // TODO not to use constantNodeToDelete anymore
        if (constantNodeToDelete.find(name) != constantNodeToDelete.end()) {
            continue;
        }

        auto opConverter = onnxOpConverterSuit::get()->search(opType);

        MNN::OpT* MNNOp  = new MNN::OpT;
        MNNOp->name      = name;
        MNNOp->type      = opConverter->opType();
        MNNOp->main.type = opConverter->type();
        mnnNodesMap.insert(std::make_pair(name, MNNOp));

        // convert initializer to be Constant node(op)
        for (int k = 0; k < onnxNode.input_size(); ++k) {
            const auto& inputName = onnxNode.input(k);
            const auto it         = initializers.find(inputName);
            if (it != initializers.end() && tensorsName.find(it->first) == tensorsName.end()) {
                // Create const Op
                MNN::OpT* constOp   = new MNN::OpT;
                constOp->type       = MNN::OpType_Const;
                constOp->main.type  = MNN::OpParameter_Blob;
                constOp->main.value = onnxOpConverter::convertTensorToBlob(it->second);
                mnnNodesMap.insert(std::make_pair(inputName, constOp));
                auto outputIndex = (int)netT->tensorName.size();
                constOp->name    = it->first;
                constOp->outputIndexes.push_back(outputIndex);
                tensorsName.insert(std::make_pair(it->first, outputIndex));
                netT->tensorName.emplace_back(constOp->name);
                netT->oplists.emplace_back(constOp);
            }
        }

        // TODO, delete the run() args opInitializers
        std::vector<const onnx::TensorProto*> opInitializers;
        for (int k = 0; k < onnxNode.input_size(); ++k) {
            const auto& inputName = onnxNode.input(k);
            const auto it         = initializers.find(inputName);
            if (it != initializers.end()) {
                opInitializers.push_back(it->second);
            }
        }
        opConverter->run(MNNOp, &onnxNode, opInitializers);

        netT->oplists.emplace_back(MNNOp);

        const int outputTensorSize = onnxNode.output_size();
        for (int ot = 0; ot < outputTensorSize; ++ot) {
            netT->tensorName.push_back(onnxNode.output(ot));
            tensorsName.insert(std::make_pair(onnxNode.output(ot), tensorsName.size()));
        }
    }

    // set input-output tensor's index
    for (int i = 0; i < nodeCount; ++i) {
        const auto& onnxNode = onnxGraph.node(i);

        auto iter = mnnNodesMap.find(onnxNode.output(0));
        DCHECK(iter != mnnNodesMap.end()) << "Can't find node: " << onnxNode.name();
        auto curOp = mnnNodesMap[onnxNode.output(0)];

        // set input index
        const int inputSize = onnxNode.input_size();
        for (int j = 0; j < inputSize; ++j) {
            const auto& inputName = onnxNode.input(j);
            // onnx have optional input, which may be a placeholder when pytorch export onnx model, so drop this input, but we should check it out sometimes.
            if(inputName == ""){
                LOG(INFO) << "Check it out ==> " << curOp->name << " has empty input, the index is " << j;
                continue;
            }
            auto iterTensor = tensorsName.find(inputName);
            DCHECK(iterTensor != tensorsName.end()) << "Can't find tensor: " << inputName;
            curOp->inputIndexes.push_back(iterTensor->second);
        }

        // set output index
        const int outputSize = onnxNode.output_size();
        for (int j = 0; j < outputSize; ++j) {
            const auto& outputName = onnxNode.output(j);
            auto iterTensor        = tensorsName.find(outputName);
            DCHECK(iterTensor != tensorsName.end()) << "Can't find tensor: " << outputName;
            curOp->outputIndexes.push_back(iterTensor->second);
        }
    }

    netT->tensorNumber = tensorsName.size();
    // set MNN net output name
    for (const auto& iter : outputs) {
        netT->outputName.push_back(iter.first);
    }

    netT->sourceType = MNN::NetSource_ONNX;
    netT->bizCode    = bizCode;

    return 0;
}
