//
//  RemoveCopy.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
#include "config.hpp"
#include "../Global.hpp"
class RemoveCopy : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        std::set<std::string> netOutputNames;
        for (auto& t : net->outputName) {
            netOutputNames.insert(t);
        }
        for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
            auto& op          = *iter;
            if (op->type == MNN::OpType_Input) {
                for (auto o : op->outputIndexes) {
                    netOutputNames.insert(net->tensorName[o]);
                }
            }
        }
        std::map<int, std::unique_ptr<MNN::TensorDescribeT>> desmap;
        for (auto&& iter : net->extraTensorDescribe) {
            desmap.insert(std::make_pair(iter->index, std::move(iter)));
        }
        auto config = Global<modelConfig>::Get();
        for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
            auto& op          = *iter;
            if (op->type != MNN::OpType_Identity || op->inputIndexes.size() != op->outputIndexes.size()) {
                iter++;
                continue;
            }
            
            bool hasOutputName = false;
            for (auto o : op->outputIndexes) {
                if (netOutputNames.find(net->tensorName[o]) != netOutputNames.end()) {
                    hasOutputName = true;
                    break;
                }
            }
            bool hasOutputFromInput = false;
            for (auto o : op->inputIndexes) {
                if (netOutputNames.find(net->tensorName[o]) != netOutputNames.end()) {
                    hasOutputFromInput = true;
                    break;
                }
            }
            if (hasOutputFromInput && hasOutputName) {
                iter++;
                continue;
            }
            auto originInput  = op->inputIndexes;
            auto originOutputs = op->outputIndexes;
            MNN_ASSERT(originInput.size() == originOutputs.size());
            if (hasOutputName) {
                bool valid = true;
                for (int i=0; i<op->inputIndexes.size(); ++i) {
                    auto o = op->outputIndexes[i];
                    auto originInput = op->inputIndexes[i];
                    if (netOutputNames.find(net->tensorName[o]) != netOutputNames.end()) {
                        if (netOutputNames.find(net->tensorName[originInput]) != netOutputNames.end()) {
                            valid = false;
                            break;
                        }
                        auto originName = net->tensorName[originInput];
                        net->tensorName[originInput] = net->tensorName[o];
                        net->tensorName[o] = originName;
                    }
                }
                if (!valid) {
                    continue;
                }
            }

            std::map<int, int> replaceIndexes;
            for (int i=0; i<op->inputIndexes.size();++i) {
                replaceIndexes.insert(std::make_pair(op->outputIndexes[i], op->inputIndexes[i]));
            }
            for (auto& replaceIter : replaceIndexes) {
                auto desIter = desmap.find(replaceIter.first);
                if (desIter != desmap.end()) {
                    desIter->second->index = replaceIter.second;
                    desIter->second->name = net->tensorName[replaceIter.second];
                    desmap[replaceIter.second] = std::move(desIter->second);
                    desmap.erase(desIter);
                }
            }
            for (auto subIter = net->oplists.begin(); subIter != net->oplists.end(); subIter++) {
                auto& subOp = *subIter;
                for (int v = 0; v < subOp->inputIndexes.size(); ++v) {
                    if (replaceIndexes.find(subOp->inputIndexes[v]) != replaceIndexes.end()) {
                        subOp->inputIndexes[v] = replaceIndexes[subOp->inputIndexes[v]];
                    }
                }
            }
            iter = net->oplists.erase(iter);
        }
        net->extraTensorDescribe.clear();
        for (auto&& iter : desmap) {
            net->extraTensorDescribe.emplace_back(std::move(iter.second));
        }
        return true;
    }
};
static PostConverterRegister<RemoveCopy> __l("RemoveCopy");
