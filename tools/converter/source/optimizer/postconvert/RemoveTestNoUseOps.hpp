//
//  RemoveTestNoUseOps.hpp
//  MNNConverter
//
//  Created by MNN on 2019/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <vector>
#include <string>
#include <algorithm>
#include "../PostTreatUtils.hpp"
using namespace MNN;

class RemoveTestNoUseOps : public PostConverter {
public:
    /* The Op's output set as input */
    virtual bool shouldDeleteJudge(const MNN::OpT* op, const MNN::NetT* const netPtr) const = 0;

    virtual bool shouldRemoveUnusefulInputs(const MNN::OpT* op) const = 0;

    virtual bool shouldDeleteOutput(const MNN::OpT* op) const = 0;

    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {

        const MNN::NetT* const netPtr = net.get();

        std::set<int> uselessIndex;
        for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
            auto& op          = *iter;
            bool shouldDelete = shouldDeleteJudge(op.get(), netPtr);
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
            bool removeUselessInput = shouldRemoveUnusefulInputs(op.get());
            if (removeUselessInput) {
                for (int index = 0; index < op->inputIndexes.size(); ++index) {
                    uselessIndex.insert(op->inputIndexes[index]);
                }
            }
            iter = net->oplists.erase(iter);
        }

        bool needIteration = false;
        do {
            needIteration = false;
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
                if (!op->inputIndexes.empty()) {
                    for (auto index : op->inputIndexes) {
                        uselessIndex.insert(index);
                    }
                    needIteration = true;
                }
                iter = net->oplists.erase(iter);
            }
        } while (needIteration);

        return true;
    }
};
