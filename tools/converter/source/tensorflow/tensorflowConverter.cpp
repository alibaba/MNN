//
//  tensorflowConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <iostream>
#include <sstream>

#include "MNN_generated.h"
#include "TfUtils.hpp"
#include "TmpGraph.hpp"
#include "logkit.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"
#include "tfOpConverter.hpp"
#include "tensorflowConverter.hpp"
int tensorflow2MNNNet(const std::string inputModel, const std::string bizCode, std::unique_ptr<MNN::NetT> &netT) {
    tensorflow::GraphDef tfGraph;

    // load
    bool success = tf_read_proto_from_binary(inputModel.c_str(), &tfGraph);
    DCHECK(success) << "read_proto_from_binary failed";
    const int node_count = tfGraph.node_size();
    std::map<std::string, MNN::OpT *> nodes;
    std::map<std::string, int> tensorName;

    for (int i = 0; i < node_count; i++) {
        const tensorflow::NodeDef &tfNode = tfGraph.node(i);
        std::shared_ptr<TmpNode> tempNode(new TmpNode());
        tempNode->opName = tfNode.name();
        tempNode->opType = tfNode.op();
        tempNode->tfNode = &tfNode;
        MNN::OpT *op     = new MNN::OpT;
        auto creator     = tfOpConverterSuit::get()->search(tempNode->opType);
        DCHECK(creator) << "MNN Converter NOT_SUPPORTED_OP: [ " << tempNode->opType << " ]";
        op->name      = tempNode->opName;
        op->type      = creator->opType();
        op->main.type = creator->type();
        nodes.insert(std::make_pair(tfNode.name(), op));
        
        // resize the inputIndexes and outputIndexes
        auto inputSize = tfNode.input_size();
        op->inputIndexes.resize(inputSize);
        // -1 is placeholder value, and the number of -1 is the number of output tensors
        // defalut: every op output one tensor, if the number of the output tensors is bigger than 1, set the outputIndexes in the op converter(void run(MNN::OpT *dstOp, TmpNode *srcNode))
        op->outputIndexes = {-1};
        
        creator->run(op, tempNode.get());
        
        for (int j = 0; j < inputSize; j++) {
            std::string inputName = tfNode.input(j); // may be input or input:0 or input:1
            // delete the name that has "^"
            inputName       = inputName.substr(inputName.find("^") + 1, inputName.size());
            int tensorIndex = -1;
            if (tensorName.find(inputName) != tensorName.end()) {
                tensorIndex = tensorName[inputName];
            } else {
                tensorIndex = (int)tensorName.size();
                tensorName.insert(std::make_pair(inputName, tensorIndex));
                netT->tensorName.emplace_back(inputName);
            }
            op->inputIndexes[j] = tensorIndex;
        }
        netT->oplists.emplace_back(op);
    }
    for (int i = 0; i < node_count; i++) {
        const tensorflow::NodeDef &tfNode = tfGraph.node(i);
        auto inputSize                    = tfNode.input_size();
        for (int j = 0; j < inputSize; j++) {
            std::string inputName = tfNode.input(j); // may be input or input:0 or input:1
            // delete the name that has "^"
            inputName       = inputName.substr(inputName.find("^") + 1, inputName.size());
            int tensorIndex = tensorName[inputName];
            std::string prefix;
            std::string node_name;
            std::string suffix;
            TFModelOptimizer::NodeNamePartsFromInput(inputName, &prefix, &node_name, &suffix);
            auto inputNodeIter = nodes.find(node_name);
            // printf("%s\n", node_name.c_str());
            DCHECK(inputNodeIter != nodes.end()) << "Error for don't find " << tfNode.name() << ", " << j;
            auto inputOp    = inputNodeIter->second;
            int outputIndex = 0;
            if (!suffix.empty()) {
                suffix = suffix.substr(suffix.find(":") + 1, suffix.size());
                std::istringstream os(suffix);
                os >> outputIndex;
            }
            if (inputOp->outputIndexes.size() < outputIndex + 1) {
                auto originSize = inputOp->outputIndexes.size();
                inputOp->outputIndexes.resize(outputIndex + 1);
                for (int p = originSize; p <= outputIndex; ++p) {
                    inputOp->outputIndexes[p] = -1;
                }
            }
            inputOp->outputIndexes[outputIndex] = tensorIndex;
        }
    }
    // Add Extra Tensor Index for -1
    for (auto &op : netT->oplists) {
        auto outputSize = op->outputIndexes.size();
        for (int o = 0; o < outputSize; ++o) {
            if (op->outputIndexes[o] == -1) {
                auto newIndex        = (int)netT->tensorName.size();
                op->outputIndexes[o] = newIndex;
                if (o != 0) {
                    netT->tensorName.emplace_back(op->name + flatbuffers::NumToString(o));
                } else {
                    netT->tensorName.emplace_back(op->name);
                }
            }
        }
    }

    netT->sourceType = MNN::NetSource_TENSORFLOW;
    netT->bizCode    = bizCode;

    return 0;
}
