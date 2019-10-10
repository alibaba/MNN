//
//  RemoveInplace.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"

class RemoveInplace : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
            auto& op = *iter;
            if (!PostTreatUtils::_isSingleInputOutput(op.get())) {
                continue;
            }
            if (op->inputIndexes[0] != op->outputIndexes[0]) {
                continue;
            }
            auto originIndex = op->inputIndexes[0];
            net->tensorName.push_back(op->name);
            int newIndex         = net->tensorName.size() - 1;
            op->outputIndexes[0] = newIndex;
            for (auto subIter = iter + 1; subIter != net->oplists.end(); subIter++) {
                auto& subOp = *subIter;
                for (int i = 0; i < subOp->inputIndexes.size(); ++i) {
                    if (subOp->inputIndexes[i] == originIndex) {
                        subOp->inputIndexes[i] = newIndex;
                    }
                }
                for (int i = 0; i < subOp->outputIndexes.size(); ++i) {
                    if (subOp->outputIndexes[i] == originIndex) {
                        subOp->outputIndexes[i] = newIndex;
                    }
                }
            }
            net->tensorNumber = net->tensorName.size();
        }
        return true;
    }
};
static PostConverterRegister<RemoveInplace> __l("RemoveInplace");
