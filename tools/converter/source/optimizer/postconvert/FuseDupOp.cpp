//
//  FuseDupOp.cpp
//  MNNConverter
//
//  Created by MNN on 2021/02/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
#include <map>
#include <set>
using namespace MNN;
class FuseDupOp : public PostConverter {
public:
    static bool isSameOp(const MNN::OpT* op0, const MNN::OpT* op1) {
        if (op0->type != op1->type) {
            return false;
        }
        if (op0->main.type != op1->main.type) {
            return false;
        }
        if (op0->inputIndexes != op1->inputIndexes) {
            return false;
        }
        if (op0->outputIndexes.size() != op1->outputIndexes.size()) {
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
        }
        if (op0->main.type == OpParameter_UnaryOp) {
            return op0->main.AsUnaryOp()->opType == op1->main.AsUnaryOp()->opType;
        }
        if (op0->main.type == OpParameter_BinaryOp) {
            return op0->main.AsBinaryOp()->opType == op1->main.AsBinaryOp()->opType;
        }
        if (op0->main.type == OpParameter_ReductionParam) {
            return op0->main.AsReductionParam()->operation == op1->main.AsReductionParam()->operation;
        }
        return false;
    }
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        std::set<MNN::OpT*> unusefulOps;
        std::map<int, int> replaceIndexes;
        std::set<std::string> outputNames(net->outputName.begin(), net->outputName.end());
        for (int i=0; i<net->oplists.size(); ++i) {
            auto originOp = net->oplists[i].get();
            if (nullptr == originOp) {
                continue;
            }
            for (int j=i+1; j < net->oplists.size(); ++j) {
                auto judgeOp = net->oplists[j].get();
                if (nullptr == judgeOp) {
                    continue;
                }
                if (isSameOp(originOp, judgeOp)) {
                    auto keepOp = originOp, removeOp = judgeOp;
                    // outputs must keep
                    if (outputNames.find(removeOp->name) != outputNames.end()) {
                        keepOp = removeOp;
                        removeOp = originOp;
                    }
                    for (int v=0; v<judgeOp->outputIndexes.size(); ++v) {
                        replaceIndexes.insert(std::make_pair(removeOp->outputIndexes[v], keepOp->outputIndexes[v]));
                    }
                    net->oplists[j].reset();
                }
            }
        }
        // Remove nullptr op
        auto tempOpList = std::move(net->oplists);
        net->oplists.clear();
        for (int i=0; i<tempOpList.size(); ++i) {
            if (nullptr != tempOpList[i].get()) {
                net->oplists.emplace_back(std::move(tempOpList[i]));
            }
        }

        // Replace index
        for (auto& op : net->oplists) {
            for (int i=0; i<op->inputIndexes.size(); ++i) {
                auto iter = replaceIndexes.find(op->inputIndexes[i]);
                if (iter!=replaceIndexes.end()) {
                    op->inputIndexes[i] = iter->second;
                }
            }
            for (int i=0; i<op->outputIndexes.size(); ++i) {
                auto iter = replaceIndexes.find(op->outputIndexes[i]);
                if (iter!=replaceIndexes.end()) {
                    op->outputIndexes[i] = iter->second;
                }
            }
        }
        return true;
    }
};
static PostConverterRegister<FuseDupOp> __l("FuseDupOp");
