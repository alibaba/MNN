//
//  OnnxTmpGraph.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OnnxTmpGraph_hpp
#define OnnxTmpGraph_hpp

#include <stdio.h>

#include "onnx.pb.h"

class OnnxTmpNode {
public:
    OnnxTmpNode();
    ~OnnxTmpNode();
    std::string opName;
    std::string opType;
    const onnx::NodeProto* onnxNode;

    std::vector<std::string> inEdges;
    std::vector<std::string> outEdges;

    //    std::vector<std::string> inTensors;
    //    std::vector<std::string> outTensors;
};

class OnnxTmpGraph {
public:
    OnnxTmpGraph(const onnx::GraphProto* onnxGraph);
    OnnxTmpGraph() = delete;
    ~OnnxTmpGraph();

    int buildGraph();
    std::shared_ptr<OnnxTmpNode> _getTmpNode(const std::string& nodeName);

    const onnx::GraphProto* mOnnxGraph;
    std::map<std::string, std::shared_ptr<OnnxTmpNode>> mTempNodes;
    std::map<std::string, const onnx::TensorProto*> mInitializers;
    std::map<std::string, const onnx::ValueInfoProto*> mInputs;
    std::map<std::string, const onnx::ValueInfoProto*> mOutputs;
    std::set<std::string> mConstantNodeToDelete;

private:
    void _init();
    void _genMinGraph();
    int _pushNoReaptedItem(std::vector<std::string>& tensorNames, const std::string& item);
    int _makeConnection(const std::shared_ptr<OnnxTmpNode>& srcNode, const std::shared_ptr<OnnxTmpNode>& dstNode,
                        const std::string& srcName, const std::string& dstName);
    void _changInOutName(std::vector<std::string>& inOutEdges, const std::string& name, const std::string& deleteName);
};

#endif /* OnnxTmpGraph_hpp */
