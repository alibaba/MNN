//
//  OnnxTmpGraph.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OnnxTmpGraph.hpp"
#include "logkit.h"

OnnxTmpNode::OnnxTmpNode() : opName(), opType(), onnxNode(nullptr) {
}

OnnxTmpNode::~OnnxTmpNode() {
}

OnnxTmpGraph::OnnxTmpGraph(const onnx::GraphProto* onnxGraph) : mOnnxGraph(onnxGraph) {
    _init();
    buildGraph();
    _genMinGraph();
}

OnnxTmpGraph::~OnnxTmpGraph() {
}

void OnnxTmpGraph::_init() {
    const int nodeCount = mOnnxGraph->node_size();
    for (int i = 0; i < nodeCount; ++i) {
        const auto& onnxNode = mOnnxGraph->node(i);
        std::shared_ptr<OnnxTmpNode> node(new OnnxTmpNode());
        node->opName   = onnxNode.output(0);
        node->opType   = onnxNode.op_type();
        node->onnxNode = &onnxNode;
        mTempNodes.insert(std::make_pair(onnxNode.output(0), node));
    }

    const int initializerCount = mOnnxGraph->initializer_size();
    for (int i = 0; i < initializerCount; ++i) {
        const auto& initializer = mOnnxGraph->initializer(i);
        mInitializers.insert(std::make_pair(initializer.name(), &initializer));
    }
    const int inputCount = mOnnxGraph->input_size();
    for (int i = 0; i < inputCount; ++i) {
        const auto& input = mOnnxGraph->input(i);
        mInputs.insert(std::make_pair(input.name(), &input));
    }
    const int outputCount = mOnnxGraph->output_size();
    for (int i = 0; i < outputCount; ++i) {
        const auto& output = mOnnxGraph->output(i);
        mOutputs.insert(std::make_pair(output.name(), &output));
    }
}

int OnnxTmpGraph::buildGraph() {
    const int nodeCount = mOnnxGraph->node_size();

    for (int i = 0; i < nodeCount; ++i) {
        const auto& onnxNode          = mOnnxGraph->node(i);
        const std::string curNodeName = onnxNode.output(0);
        const int inputSize           = onnxNode.input_size();
        const auto& curNode           = _getTmpNode(curNodeName);
        for (int j = 0; j < inputSize; ++j) {
            const std::string inputName = onnxNode.input(j);
            const auto& srcNode         = _getTmpNode(inputName);
            if (!srcNode)
                continue;
            _makeConnection(srcNode, curNode, inputName, curNodeName);
        }
    }

    return 0;
}

void OnnxTmpGraph::_genMinGraph() {
    for (auto& iter : mTempNodes) {
        auto& curNode           = iter.second;
        const auto& opType      = curNode->opType;
        const auto& curNodeName = curNode->opName;
        if (opType == "Dropout" || opType == "Identity") {
            DCHECK(curNode->inEdges.size() == 1) << "Dropout's input node num. != 1 [ " << curNode->opName << " ]";
            const auto& parentName = curNode->inEdges[0];
            const auto& parentNode = _getTmpNode(parentName);
            for (int i = 0; i < curNode->outEdges.size(); ++i) {
                const auto& sonName = curNode->outEdges[i];
                const auto& sonNode = _getTmpNode(sonName);
                _changInOutName(parentNode->outEdges, sonName, curNodeName);
                _changInOutName(sonNode->inEdges, parentName, curNodeName);
            }
        } else if (opType == "Upsample") {
            DCHECK(2 == curNode->inEdges.size()) << "Upsample Input ERROR!";
            // put [Upsample]'s second input(Constant) into mInitializers, and delete this Constant node
            const auto& constantTmpNode = _getTmpNode(curNode->inEdges[1]);
            for (int i = 0; i < constantTmpNode->onnxNode->attribute_size(); ++i) {
                const auto& attributeProto = constantTmpNode->onnxNode->attribute(i);
                if (attributeProto.name() == "value") {
                    mInitializers.insert(std::make_pair(constantTmpNode->opName, &attributeProto.t()));
                }
            }

            mConstantNodeToDelete.insert(curNode->inEdges[1]);
            auto it = curNode->inEdges.begin();
            curNode->inEdges.erase(it + 1);
        }
    }
}

int OnnxTmpGraph::_makeConnection(const std::shared_ptr<OnnxTmpNode>& srcNode,
                                  const std::shared_ptr<OnnxTmpNode>& dstNode, const std::string& srcName,
                                  const std::string& dstName) {
    // node1, node2
    this->_pushNoReaptedItem(srcNode->outEdges, dstName);
    this->_pushNoReaptedItem(dstNode->inEdges, srcName);

    return 0;
}

std::shared_ptr<OnnxTmpNode> OnnxTmpGraph::_getTmpNode(const std::string& nodeName) {
    const auto& it = mTempNodes.find(nodeName);
    if (it != mTempNodes.end()) {
        return it->second;
    } else {
        //        DLOG(INFO) << "Check The Node Name ===> [ " << nodeName << " ]";
        return 0;
    }
}

int OnnxTmpGraph::_pushNoReaptedItem(std::vector<std::string>& tensorNames, const std::string& item) {
    std::vector<std::string>::iterator it = tensorNames.begin();
    while (it != tensorNames.end()) {
        if (item == *it) {
            return -1; // item in tensor names
        }
        it++;
    }
    if (it == tensorNames.end()) {
        tensorNames.push_back(item);
    }

    return 0;
}

void OnnxTmpGraph::_changInOutName(std::vector<std::string>& inOutEdges, const std::string& name,
                                   const std::string& deleteName) {
    auto it = inOutEdges.begin();
    while (it != inOutEdges.end()) {
        if (deleteName == *it) {
            *it = name;
            return;
        }
        it++;
    }
    if (it == inOutEdges.end()) {
        inOutEdges.push_back(name);
    }
}
