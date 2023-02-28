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


#endif // TMPGRAPH_HPP
