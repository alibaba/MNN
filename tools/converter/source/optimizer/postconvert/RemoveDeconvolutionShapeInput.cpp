//
//  RemoveDeconvolutionShapeInput.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
MNN::OpT* ensureOpInNet(std::unique_ptr<MNN::NetT>& net, MNN::OpT* op) {
    for (auto& _op : net->oplists) {
        if (_op.get() == op) {
            return op;
        }
    }
    return nullptr;
}
class RemoveDeconvolutionShapeInput : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        std::set<MNN::OpT*> shapeOps;
        auto& mNet = net;
        for (auto& op : mNet->oplists) {
            if (op->type == MNN::OpType_Deconvolution || op->type == MNN::OpType_DeconvolutionDepthwise) {
                if (op->inputIndexes.size() == 1) {
                    continue;
                }
                int firstInputIndex = op->inputIndexes[0];
                op->inputIndexes.erase(op->inputIndexes.begin());
                MNN::OpT* shapeOp = PostTreatUtils::_findOpByOutputIndex(firstInputIndex, mNet.get());
                if (shapeOp) {
                    shapeOps.insert(shapeOp);
                }
            }
        }
        for (auto& op : shapeOps) {
            std::vector<MNN::OpT*> opsToBeChecked;
            opsToBeChecked.push_back(op);
            while (!opsToBeChecked.empty()) {
                bool hasRemoved = false;
                std::vector<MNN::OpT*> addedToBeChecked;
                for (auto iter = opsToBeChecked.begin(); iter != opsToBeChecked.end();) {
                    MNN::OpT* op = *iter;
                    if (!ensureOpInNet(mNet, op)) {
                        hasRemoved = true;
                        iter       = opsToBeChecked.erase(iter);
                        continue;
                    }
                    if (PostTreatUtils::_getOpDecestorCount(op, mNet.get()) == 0) {
                        for (int inputIndex : op->inputIndexes) {
                            addedToBeChecked.push_back(PostTreatUtils::_findOpByOutputIndex(inputIndex, mNet.get()));
                        }
                        hasRemoved = true;
                        PostTreatUtils::_removeOpInNet(op, mNet.get());
                        iter = opsToBeChecked.erase(iter);
                        continue;
                    }
                    iter++;
                }
                if (!hasRemoved)
                    break;
                opsToBeChecked.insert(opsToBeChecked.end(), addedToBeChecked.begin(), addedToBeChecked.end());
            }
        }
        return true;
    }
};
static PostConverterRegister<RemoveDeconvolutionShapeInput> __l("RemoveDeconvolutionShapeInput");
