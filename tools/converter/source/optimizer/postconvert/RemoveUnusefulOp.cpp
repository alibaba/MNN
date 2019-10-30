//
//  RemoveUnusefulOp.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <vector>
#include <string>
#include <algorithm>
#include "../PostTreatUtils.hpp"
using namespace MNN;

class RemoveUnusefulOp : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {

        const MNN::NetT* const netPtr = net.get();
        
        /* The Op's output set as input */
        auto shouldDeleteJudge = [=](const MNN::OpT* op) {
            static auto unuseOpType = std::vector<OpType>({OpType_Seq2Out, OpType_Dropout});
            static auto unuseExtraOpType = std::vector<std::string>({"Identity", "NoOp", "Dropout", "Print", "Assert", "StopGradient"});
            if (std::find(unuseOpType.begin(), unuseOpType.end(), op->type) != unuseOpType.end()) {
                return true;
            }
            if (op->type == OpType_Extra) {
                if (std::find(unuseExtraOpType.begin(), unuseExtraOpType.end(), op->main.AsExtra()->type) != unuseExtraOpType.end()) {
                    return true;
                }
                if (netPtr->sourceType == MNN::NetSource_CAFFE && op->main.AsExtra()->type == "Split") {
                    return true;
                }
            }
            if (op->type == OpType_Cast) {
                if (op->main.AsCastParam()->dstT == op->main.AsCastParam()->srcT) {
                    return true;
                }
                if (op->main.AsCastParam()->dstT == MNN::DataType_DT_INT32 && op->main.AsCastParam()->srcT == MNN::DataType_DT_INT64) {
                    return true;
                }
                if (op->main.AsCastParam()->srcT == MNN::DataType_DT_INT32 && op->main.AsCastParam()->dstT == MNN::DataType_DT_INT64) {
                    return true;
                }
            }
            return false;
        };
        auto shouldDeleteOutput = [=](const MNN::OpT* op) {
            if (op->type == OpType_Extra) {
                return op->main.AsExtra()->type == "Assert";
            }
            return false;
        };
        std::set<int> uselessIndex;
        for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
            auto& op          = *iter;
            bool shouldDelete = shouldDeleteJudge(op.get());
            if (!shouldDelete) {
                iter++;
                continue;
            }
            bool deleteOutput = shouldDeleteOutput(op.get());
            // Find the next op
            if (op->outputIndexes.empty() || op->inputIndexes.empty()) {
                iter = net->oplists.erase(iter);
                continue;
            }

            auto originInput  = op->inputIndexes[0];
            auto originOutputs = op->outputIndexes;
            for (auto subIter = net->oplists.begin(); subIter != net->oplists.end(); subIter++) {
                auto& subOp = *subIter;
                if (deleteOutput) {
                    for (auto iter=subOp->inputIndexes.begin(); iter != subOp->inputIndexes.end();) {
                        if (std::find(originOutputs.begin(), originOutputs.end(), *iter) != originOutputs.end()) {
                            iter = subOp->inputIndexes.erase(iter);
                            continue;
                        }
                        iter++;
                    }
                } else {
                    for (int v = 0; v < subOp->inputIndexes.size(); ++v) {
                        if (std::find(originOutputs.begin(), originOutputs.end(), subOp->inputIndexes[v]) != originOutputs.end()) {
                            subOp->inputIndexes[v] = originInput;
                        }
                    }
                }
            }
            for (int index = 0; index < op->inputIndexes.size(); ++index) {
                uselessIndex.insert(op->inputIndexes[index]);
            }
            iter = net->oplists.erase(iter);
        }
        for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
            for (auto index : (*iter)->inputIndexes) {
                if (uselessIndex.find(index) != uselessIndex.end()) {
                    uselessIndex.erase(index);
                }
            }
        }

        for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
            auto& op     = *iter;
            bool useless = true;
            for (auto index : op->outputIndexes) {
                if (uselessIndex.find(index) == uselessIndex.end()) {
                    useless = false;
                    break;
                }
            }
            if (!useless) {
                iter++;
                continue;
            }
            iter = net->oplists.erase(iter);
        }

        return true;
    }
};
static PostConverterRegister<RemoveUnusefulOp> __l("RemoveUnusefulOp");
