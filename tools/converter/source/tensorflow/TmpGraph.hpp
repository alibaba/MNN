//
//  TmpGraph.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TMPGRAPH_HPP
#define TMPGRAPH_HPP

#include <iostream>
#include <map>
#include <vector>

#include "graph.pb.h"

class TmpNode {
public:
    TmpNode();
    ~TmpNode();

public:
    std::string opName;
    std::string opType;

    const tensorflow::NodeDef *tfNode;

    std::vector<std::string> inEdges;  // node
    std::vector<std::string> outEdges; // node

    std::vector<std::string> inTensors;  // tensor names
    std::vector<std::string> outTensors; // tensor names

    std::string future;

    bool isCovered;
    bool isDelete;
    int leftInEdges;
    std::string DebugString() const;
};

class TmpGraph {
public:
    TmpGraph(const tensorflow::GraphDef &tfGraph);
    ~TmpGraph();

public:
    tensorflow::GraphDef _tfGraph;

    std::vector<TmpNode *> tmpNodes;
    std::map<std::string, TmpNode *> tmpNodeMap; // nodeName, TmpNode*

    // constant nodes which have no input
    std::vector<std::string> inputNodes;
    std::vector<std::string> outputNodes;

    std::vector<std::string> opsInOrder;

public:
    int buildGraph(); // build the min Graph
    TmpNode *_getTmpNode(const std::string &nodeName);

private:
    TmpGraph();
    bool _allOpSupported();
    int _setInOutTensorsName(TmpNode *parentNode, TmpNode *curNode, std::string inputName);

    int _setOuputTensorsName(std::vector<std::string> &tensorVector, std::string inputName, int index);

    int _makeConnection(TmpNode *srcNode, TmpNode *dstNode, const std::string srcName, const std::string dstName);

    void _genMinGraph();
    void _changInOutName(std::vector<std::string> &inOutEdges, std::string name, std::string deleteName);

    int _getOpsInorder(const std::vector<std::string> inputNodes);

    void _getInputNodes();

    int _pushNoReaptedItem(std::vector<std::string> &tensorNames, const std::string item);
    void _getTmpNodeMapAndConnection();
    int _optimizeTfModel();
    bool _hasContinuousConstantNode();
};

#endif // TMPGRAPH_HPP
