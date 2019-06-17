//
//  OpGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include <sstream>
using namespace std;

static std::map<MNN::OpType, OpGrad::Creator*>& getConverter() {
    static std::map<MNN::OpType, OpGrad::Creator*> gConverterMap;
    return gConverterMap;
}

OpGrad::Creator* OpGrad::get(MNN::OpType type) {
    auto& converterMap = getConverter();
    auto iter          = converterMap.find(type);
    if (iter != converterMap.end()) {
        return iter->second;
    }
    return nullptr;
}

void OpGrad::insert(MNN::OpType type, OpGrad::Creator* converter) {
    auto& converterMap = getConverter();
    converterMap.insert(std::make_pair(type, converter));
}

bool OpGrad::onGradCommon(MNN::NetT* dest, const MNN::OpT* forwardOp,
                          std::map<int, std::vector<int>>& backwardTensors) {
    // Create New Diff Tensors
    std::vector<int> gradTensors;
    for (int i = 0; i < forwardOp->inputIndexes.size(); ++i) {
        int newTensorId = dest->tensorName.size();
        gradTensors.emplace_back(newTensorId);
        dest->tensorName.emplace_back(forwardOp->name + "_" + numberToString(i));
        auto inputIndex = forwardOp->inputIndexes[i];
        if (backwardTensors.find(inputIndex) == backwardTensors.end()) {
            backwardTensors.insert(make_pair(inputIndex, vector<int>{}));
        }
        backwardTensors[inputIndex].emplace_back(newTensorId);
    }
    auto result = this->onGrad(dest, forwardOp, backwardTensors, gradTensors);
    dest->tensorName.insert(dest->tensorName.end(), result.tensorNames.begin(), result.tensorNames.end());
    for (auto& op : result.opLists) {
        dest->oplists.emplace_back(std::move(op));
    }
    return true;
}
std::string numberToString(int index) {
    std::ostringstream os;
    os << index;
    return os.str();
}
