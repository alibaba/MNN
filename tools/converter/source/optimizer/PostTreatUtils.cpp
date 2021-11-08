//
//  PostTreatUtils.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "PostTreatUtils.hpp"
#include "OpCount.hpp"
#include <mutex>
#include <set>
using namespace MNN;

template <typename T>
bool inVector(const std::vector<T>& vec, const T& val) {
    return std::find(vec.begin(), vec.end(), val) != vec.end();
}
std::map<std::string, std::shared_ptr<PostConverter>>* PostConverter::getConvertMap() {
    static std::once_flag of;
    static std::map<std::string, std::shared_ptr<PostConverter>>* gConverter;
    std::call_once(of, [&]() {
        gConverter = new std::map<std::string, std::shared_ptr<PostConverter>>;
        auto count = MNN::OpCount::get();
        count->insertOp("TF", "Dropout");
        count->insertOp("TF", "NoOp");
        count->insertOp("TF", "Print");
        count->insertOp("CAFFE", "Dropout");
        count->insertOp("CAFFE", "Split");
        auto unuseExtraOpType = std::vector<std::string>({"Identity", "IdentityN", "NoOp", "Assign", "Print", "Assert", "StopGradient", "Enter", "NextIteration"});
        for (auto& s : unuseExtraOpType) {
            count->insertOp("TF", s);
        }
        std::set<std::string> controlOps{"Merge", "Switch", "LoopCond", "Enter", "Exit", "NextIteration"};
        for (auto& s : controlOps) {
            count->insertOp("TF", s);
        }
        count->insertOp("ONNX", "Identity");
    });
    return gConverter;
}
PostConverter* PostConverter::get(std::string key) {
    auto gConverter = getConvertMap();
    if (gConverter->find(key) != gConverter->end()) {
        return gConverter->at(key).get();
    }
    return nullptr;
}

void PostConverter::add(std::shared_ptr<PostConverter> converter, std::string key) {
    auto gConverter = getConvertMap();
    gConverter->insert(std::make_pair(key, converter));
}

bool PostTreatUtils::_isSingleInputOutput(const MNN::OpT* op) {
    if (op->inputIndexes.size() != 1 || op->outputIndexes.size() != 1) {
        return false;
    }
    return true;
}

MNN::OpT* PostTreatUtils::_findOpByOutputIndex(int outputIndex, const NetT* net) {
    for (auto& op : net->oplists) {
        if (inVector(op->outputIndexes, outputIndex)) {
            return op.get();
        }
    }
    return nullptr;
}

std::vector<MNN::OpT*> PostTreatUtils::_findOpByInputIndex(int inputIndex, const NetT* net) {
    std::vector<MNN::OpT*> ops;
    for (auto& op : net->oplists) {
        if (inVector(op->inputIndexes, inputIndex)) {
            ops.push_back(op.get());
        }
    }

    // check whether the next op is in_place op
    const int opsSize = ops.size();
    if (opsSize > 1) {
        auto realNextOp = ops[0];
        if (inVector(realNextOp->outputIndexes, inputIndex)) {
            ops.clear();
            ops.push_back(realNextOp);
        }
    }

    return ops;
}

int PostTreatUtils::_getOpDecestorCount(MNN::OpT* op, const MNN::NetT* mNet) {
    int decestorCount = 0;
    for (auto& otherOp : mNet->oplists) {
        if (otherOp.get() != op) {
            for (auto inputIndex : otherOp->inputIndexes) {
                if (inVector(op->outputIndexes, inputIndex)) {
                    decestorCount++;
                    break; // one decestor just count one.
                }
            }
        }
    }
    return decestorCount;
}

void PostTreatUtils::_removeOpInNet(MNN::OpT* op, MNN::NetT* net) {
    for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
        if (iter->get() == op) {
            // LOG(INFO) << "remove op: " << op->name;
            net->oplists.erase(iter);
            break;
        }
    }
}

bool PostTreatUtils::_replace(std::vector<int>& indexes, int freshIndex, int oldIndex) {
    auto iter = indexes.begin();
    while (iter != indexes.end()) {
        if (*iter == oldIndex) {
            *iter = freshIndex;
            return true;
        }
        ++iter;
    }
    return false;
}
