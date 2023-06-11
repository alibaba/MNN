//
//  onnxConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <iostream>
#include <queue>

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

std::vector<int> topoSort(const ::onnx::GraphProto& onnxGraph) {
    std::vector<int> idxMap;
    const int nodeCount   = onnxGraph.node_size();
    std::map<std::string, int> outputMap;
    std::map<int, std::vector<int>> graph; // key --[in]--> values
    std::vector<int> inDegree(nodeCount);
    // build Graph and inDegree
    for (int i = 0; i < nodeCount; ++i) {
        const auto& onnxNode = onnxGraph.node(i);
        if (onnxNode.op_type() == "Loop" || onnxNode.op_type() == "If") {
            return idxMap;
        }
        for (int k = 0; k < onnxNode.output_size(); k++) {
            outputMap.insert(std::make_pair(onnxNode.output(k), i));
        }
    }
    for (int i = 0; i < nodeCount; ++i) {
        const auto& onnxNode = onnxGraph.node(i);
        for (int k = 0; k < onnxNode.input_size(); k++) {
            auto inputName = onnxNode.input(k);
            auto iter = outputMap.find(inputName);
            if (iter != outputMap.end()) {
                graph[iter->second].push_back(i);
            }
        }
    }
    for (auto node : graph) {
        for (auto output : node.second) {
            inDegree[output]++;
        }
    }
    // topo sort
    std::queue<int> validNode;
    for (int i = 0; i < nodeCount; i++) {
        if (!inDegree[i]) {
            validNode.push(i);
        }
    }
    while (!validNode.empty()) {
        int node = validNode.front();
        validNode.pop();
        idxMap.push_back(node);
        for (auto succ : graph[node]) {
            if (--inDegree[succ] == 0) {
                validNode.push(succ);
            }
        }
    }
    MNN_ASSERT(idxMap.size() == nodeCount);
    return idxMap;
}

int onnx2MNNNet(const std::string inputModel, const std::string bizCode,
                std::unique_ptr<MNN::NetT>& netT) {
    std::string modelDir;
    size_t pos = inputModel.find_last_of("\\/");
    if (pos != std::string::npos) {
        modelDir = inputModel.substr(0, pos + 1);
    }

    onnx::ModelProto onnxModel;
    // read ONNX Model
    bool success = onnx_read_proto_from_binary(inputModel.c_str(), &onnxModel);
    DCHECK(success) << "read onnx model failed: " << inputModel;
    if (!success) {
        MNN_ERROR("[ERROR] Model file is not onnx model.\n");
        return 1;
    }

    int opsetVersion = 13;
    auto opsetInfo = onnxModel.opset_import();
    if (!opsetInfo.empty()) {
        opsetVersion = static_cast<int>(opsetInfo.begin()->version());
    }
    LOG(INFO) << "ONNX Model ir version: " << onnxModel.ir_version();
    LOG(INFO) << "ONNX Model opset version: " << opsetVersion;

    const auto& onnxGraph = onnxModel.graph();
    const int nodeCount   = onnxGraph.node_size();

    std::unique_ptr<OnnxScope> scope(new OnnxScope(&onnxGraph, netT.get()));
    scope->mOpsetVersion = opsetVersion;
    // find the inputs which do not have initializer
    const auto& initializers         = scope->mInitializers;
    const auto& inputs               = scope->mInputs;
    const auto& outputs              = scope->mOutputs;
    // set input node to MNN net
    for (const auto& iter : inputs) {
        bool notHaveInitializer = initializers.find(iter.first) == initializers.end();
        if (notHaveInitializer) {
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
                const auto& dim = tensorInfo.shape().dim(i);
                if (dim.has_dim_value()) {
                    inputParam->dims[i] = static_cast<int32_t>(dim.dim_value());
                } else {
                    inputParam->dims[i] = -1;
                }
            }
            inputParam->dtype   = onnxOpConverter::convertDataType(tensorInfo.elem_type());
            inputParam->dformat = MNN::MNN_DATA_FORMAT_NCHW;
            MNNOp->outputIndexes.push_back(scope->declareTensor(iter.first));
            MNNOp->main.value = inputParam;
            netT->oplists.emplace_back(MNNOp);
        }
    }

    // onnx model not all topo sort graph, sort it
    std::vector<int> idxMap = topoSort(onnxGraph);

    // onnx node ==> MNN node
    for (int idx = 0; idx < nodeCount; ++idx) {
        int i = idxMap.size() == nodeCount ? idxMap[idx] : idx;
        const auto& onnxNode = onnxGraph.node(i);
        const auto& opType   = onnxNode.op_type();

        // name maybe null, use the first output name as node-name
        const auto& name = onnxNode.output(0);
        auto opConverter = onnxOpConverterSuit::get()->search(opType);

        MNN::OpT* MNNOp  = new MNN::OpT;
        MNNOp->name      = name;
        MNNOp->type      = opConverter->opType();
        MNNOp->main.type = opConverter->type();

        // convert initializer to be Constant node(op)
        for (int k = 0; k < onnxNode.input_size(); ++k) {
            const auto& inputName = onnxNode.input(k);
            const auto it         = initializers.find(inputName);
            if (it != initializers.end() && scope->lookupTensor(it->first) == -1) {
                // Create const Op
                MNN::OpT* constOp   = new MNN::OpT;
                constOp->type       = MNN::OpType_Const;
                constOp->main.type  = MNN::OpParameter_Blob;
                constOp->main.value = onnxOpConverter::convertTensorToBlob(it->second, modelDir);
                constOp->name    = it->first;
                constOp->outputIndexes.push_back(scope->declareTensor(it->first));
                netT->oplists.emplace_back(constOp);
            }
        }

        // build input and output
        for (int k = 0; k < onnxNode.input_size(); k++) {
            int inputIdx = scope->lookupTensor(onnxNode.input(k));
            if (inputIdx < 0) {
                LOG(INFO) << "Check it out ==> " << MNNOp->name << " has empty input, the index is " << k;
            }
            MNNOp->inputIndexes.push_back(inputIdx);
        }
        for (int k = onnxNode.input_size() - 1; k >= 0 && MNNOp->inputIndexes[k] < 0; --k) {
            MNNOp->inputIndexes.pop_back();
        }
        for (int k = 0; k < onnxNode.output_size(); k++) {
            MNNOp->outputIndexes.push_back(scope->declareTensor(onnxNode.output(k)));
        }
        // build op
        opConverter->run(MNNOp, &onnxNode, scope.get());
        netT->oplists.emplace_back(MNNOp);
    }
    netT->tensorNumber = netT->tensorName.size();
    // set MNN net output name
    for (const auto& iter : outputs) {
        netT->outputName.push_back(iter.first);
    }

    netT->sourceType = MNN::NetSource_ONNX;
    netT->bizCode    = bizCode;

    return 0;
}
