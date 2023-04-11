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
        auto config = Global<modelConfig>::Get();
        if (config->optimizeLevel < 1 || config->inSubGraph) {
            return true;
        }
        for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
            auto& op          = *iter;
            if (op->type != MNN::OpType_Identity) {
                iter++;
                continue;
            }
            std::map<int, int> replaceIndexes;
            for (int i=0; i<op->inputIndexes.size();++i) {
                replaceIndexes.insert(std::make_pair(op->outputIndexes[i], op->inputIndexes[i]));
                net->tensorName[op->inputIndexes[i]] = net->tensorName[op->outputIndexes[i]];
            }
            for (auto subIter = net->oplists.begin(); subIter != net->oplists.end(); subIter++) {
                auto& subOp = *subIter;
                for (int v = 0; v < subOp->inputIndexes.size(); ++v) {
                    if (replaceIndexes.find(subOp->inputIndexes[v]) != replaceIndexes.end()) {
                        subOp->inputIndexes[v] = replaceIndexes[subOp->inputIndexes[v]];
                    }
                }
            }
            for (int v=0; v<op->inputIndexes.size(); ++v) {
                for (auto& o : net->outputName) {
                    if (o == net->tensorName[op->inputIndexes[v]]) {
                        o = net->tensorName[op->outputIndexes[v]];
                        break;
                    }
                }
            }
            iter = net->oplists.erase(iter);
        }
        return true;
    }
};
static PostConverterRegister<RemoveCopy> __l("RemoveCopy");
