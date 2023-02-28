//
//  TmpGraph.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TmpGraph.hpp"
#include <set>
#include "TfUtils.hpp"
#include "logkit.h"

TmpNode::TmpNode() : opName(), opType(), tfNode(nullptr), isCovered(false), isDelete(false), leftInEdges(0) {
}

TmpNode::~TmpNode(){};

std::string TmpNode::DebugString() const {
    return tfNode->DebugString();
}

