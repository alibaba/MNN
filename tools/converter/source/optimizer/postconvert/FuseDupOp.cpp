//
//  FuseDupOp.cpp
//  MNNConverter
//
//  Created by MNN on 2021/02/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNDefine.h>
#include "../PostTreatUtils.hpp"
#include <map>
#include <set>
using namespace MNN;
class FuseDupOp : public PostConverter {
public:
    static bool isSameIndexes(const MNN::OpT* op0, const MNN::OpT* op1) {
        if (op0->inputIndexes != op1->inputIndexes) {
            return false;
        }
        if (op0->outputIndexes.size() != op1->outputIndexes.size()) {
            return false;
        }
        return true;
    }
    static bool isSameOp(const MNN::OpT* op0, const MNN::OpT* op1) {
        if (op0->type != op1->type) {
            return false;
        }
        if (op0->main.type != op1->main.type) {
            return false;
        }
        if (op0->externalPath != op1->externalPath) {
            return false;
        }
        if (op0->main.type == OpParameter_NONE) {
            return true;
        }
        if (op0->type == OpType_ReLU) {
            return op0->main.AsRelu()->slope == op1->main.AsRelu()->slope;
        }
        if (op0->type == OpType_ReLU6) {
            return op0->main.AsRelu6()->maxValue == op1->main.AsRelu6()->maxValue && op0->main.AsRelu6()->minValue == op1->main.AsRelu6()->minValue;
        }
        if (op0->main.type == OpParameter_Blob) {
            auto v0 = op0->main.AsBlob();
            auto v1 = op1->main.AsBlob();
            if (v0->external != v1->external) {
                return false;
            }
            if (v0->dataFormat != v1->dataFormat) {
                return false;
            }
            if (v0->dataType != v1->dataType) {
                return false;
            }
            if (v0->dims != v1->dims) {
                return false;
            }
            if (v0->dataFormat != v1->dataFormat) {
                return false;
            }
            if (DataType_DT_INT32 == v0->dataType) {
                return v0->int32s == v1->int32s;
            }
            if (DataType_DT_FLOAT == v0->dataType) {
                return v0->float32s == v1->float32s;
            }
            if (DataType_DT_UINT8 == v0->dataType) {
                return v0->uint8s == v1->uint8s;
            }
            if (DataType_DT_INT8 == v0->dataType) {
                return v0->int8s == v1->int8s;
            }
        }
        if (op0->main.type == OpParameter_UnaryOp) {
            return op0->main.AsUnaryOp()->opType == op1->main.AsUnaryOp()->opType;
        }
        if (op0->main.type == OpParameter_BinaryOp) {
            return op0->main.AsBinaryOp()->opType == op1->main.AsBinaryOp()->opType;
        }
        if (op0->main.type == OpParameter_ReductionParam) {
            if (op0->main.AsReductionParam()->operation != op1->main.AsReductionParam()->operation) {
                return false;
            }
            if (op0->main.AsReductionParam()->keepDims != op1->main.AsReductionParam()->keepDims) {
                return false;
            }
            if (op0->main.AsReductionParam()->dim != op1->main.AsReductionParam()->dim) {
                return false;
            }
            return true;
        }
        return false;
    }
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        std::set<MNN::OpT*> unusefulOps;
        std::map<int, int> replaceIndexes;
        // outputNames can fuse, but need reserve outputName; updateNames can't fuse
        std::set<std::string> outputNames(net->outputName.begin(), net->outputName.end());
        std::set<std::string> updateNames;
        for (const auto& op : net->oplists) {
            if (op->type == OpType_While) {
                for (const auto& update : op->main.AsWhileParam()->aliases_updates) {
                    for (const auto& updateName : update->data) {
                        updateNames.insert(updateName);
                    }
                }
            }
        }
        std::map<int, std::vector<int>> sameOps;
        for (int i=0; i<net->oplists.size(); ++i) {
            auto originOp = net->oplists[i].get();
            if (nullptr == originOp || updateNames.find(originOp->name) != updateNames.end()) {
                continue;
            }
            std::vector<int> sameOpIndexes;
            for (int j=i+1; j < net->oplists.size(); ++j) {
                auto judgeOp = net->oplists[j].get();
                if (nullptr == judgeOp || updateNames.find(judgeOp->name) != updateNames.end()) {
                    continue;
                }
                if (isSameOp(originOp, judgeOp)) {
                    sameOpIndexes.emplace_back(j);
                }
            }
            sameOps.insert(std::make_pair(i, sameOpIndexes));
        }
        
        bool change = false;
        int step = 0;
        do {
            change = false;
            for (int i=0; i<net->oplists.size(); ++i) {
                auto originOp = net->oplists[i].get();
                if (nullptr == originOp) {
                    continue;
                }
                auto iter = sameOps.find(i);
                if (iter == sameOps.end()) {
                    continue;
                }
                for (auto j : iter->second) {
                    auto judgeOp = net->oplists[j].get();
                    if (nullptr == judgeOp || updateNames.find(judgeOp->name) != updateNames.end()) {
                        continue;
                    }
                    if (isSameIndexes(judgeOp, originOp)) {
                        auto keepOp = originOp, removeOp = judgeOp;
                        // outputs must keep
                        if (outputNames.find(removeOp->name) != outputNames.end()) {
                            keepOp = removeOp;
                            removeOp = originOp;
                        }
                        for (int v=0; v<judgeOp->outputIndexes.size(); ++v) {
                            auto originIndex = removeOp->outputIndexes[v];
                            auto newIndex = keepOp->outputIndexes[v];
                            if (originIndex != newIndex) {
                                auto replaceIter = replaceIndexes.find(newIndex);
                                if (replaceIter != replaceIndexes.end()) {
                                    newIndex = replaceIter->second;
                                }
                                replaceIndexes.insert(std::make_pair(originIndex, newIndex));
                            }
                        }
                        net->oplists[j].reset();
                        change = true;
                    }
                }
            }
            auto findFinalIndex = [&](int index) -> int {
                auto iter = replaceIndexes.find(index);
                if (iter == replaceIndexes.end()) {
                    return index;
                }
                return iter->second;
            };
            // Replace index
            for (auto& op : net->oplists) {
                if (nullptr == op.get()) {
                    continue;
                }
                for (int i=0; i<op->inputIndexes.size(); ++i) {
                    op->inputIndexes[i] = findFinalIndex(op->inputIndexes[i]);
                }
                for (int i=0; i<op->outputIndexes.size(); ++i) {
                    op->outputIndexes[i] = findFinalIndex(op->outputIndexes[i]);
                }
            }
            step++;
        } while (change);
#ifdef DEBUG
        MNN_PRINT("FuseDup run for %d step\n", step);
#endif
        // Remove nullptr op
        auto tempOpList = std::move(net->oplists);
        net->oplists.clear();
        for (int i=0; i<tempOpList.size(); ++i) {
            if (nullptr != tempOpList[i].get()) {
                net->oplists.emplace_back(std::move(tempOpList[i]));
            }
        }
        return true;
    }
};
static PostConverterRegister<FuseDupOp> __l("FuseDupOp");
