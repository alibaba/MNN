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
        auto shouldDeleteJudge = [=](const MNN::OpT* op) {
            static auto unuseOpType = std::vector<OpType>({OpType_Seq2Out, OpType_Dropout});
            static auto unuseExtraOpType = std::vector<std::string>({"Identity", "NoOp", "Dropout"});
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
            return false;
        };
        for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
            auto& op          = *iter;
            bool shouldDelete = shouldDeleteJudge(op.get());
            if (!shouldDelete) {
                iter++;
                continue;
            }
            // Find the next op
            if (op->outputIndexes.empty() || op->inputIndexes.empty()) {
                iter = net->oplists.erase(iter);
                continue;
            }

            auto originInput  = op->inputIndexes[0];
            auto originOutputs = op->outputIndexes;
            for (auto subIter = net->oplists.begin(); subIter != net->oplists.end(); subIter++) {
                auto& subOp = *subIter;
                for (int v = 0; v < subOp->inputIndexes.size(); ++v) {
                    if (std::find(originOutputs.begin(), originOutputs.end(), subOp->inputIndexes[v]) != originOutputs.end()) {
                        subOp->inputIndexes[v] = originInput;
                    }
                }
            }
            iter = net->oplists.erase(iter);
        }
        return true;
    }
};
static PostConverterRegister<RemoveUnusefulOp> __l("RemoveUnusefulOp");
