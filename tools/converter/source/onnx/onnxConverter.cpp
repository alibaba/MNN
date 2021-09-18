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

int onnx2MNNNet(const std::string inputModel, const std::string bizCode,
                std::unique_ptr<MNN::NetT>& netT) {
    onnx::ModelProto onnxModel;
    // read ONNX Model
    bool success = onnx_read_proto_from_binary(inputModel.c_str(), &onnxModel);
    DCHECK(success) << "read onnx model failed: " << inputModel;

    LOG(INFO) << "ONNX Model ir version: " << onnxModel.ir_version();

    const auto& onnxGraph = onnxModel.graph();
    const int nodeCount   = onnxGraph.node_size();

    std::unique_ptr<OnnxScope> scope(new OnnxScope(&onnxGraph, netT.get()));
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
                inputParam->dims[i] = tensorInfo.shape().dim(i).dim_value();
            }
            inputParam->dtype   = onnxOpConverter::convertDataType(tensorInfo.elem_type());
            inputParam->dformat = MNN::MNN_DATA_FORMAT_NCHW;
            MNNOp->outputIndexes.push_back(scope->declareTensor(iter.first));
            MNNOp->main.value = inputParam;
            netT->oplists.emplace_back(MNNOp);
        }
    }

    // onnx node ==> MNN node
    for (int i = 0; i < nodeCount; ++i) {
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
                constOp->main.value = onnxOpConverter::convertTensorToBlob(it->second);
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
                continue;
            }
            MNNOp->inputIndexes.push_back(inputIdx);
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
