//
//  MoveUnaryOpBeforeReshape.cpp
//  MNNConverter
//
//  Created by MNN on 2023/06/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
#include "config.hpp"
#include "../Global.hpp"
class MoveUnaryOpBeforeReshape : public PostConverter {
public:
    static bool _isUnaryOp(const MNN::OpT* op) {
        switch (op->type) {
            case MNN::OpType_ReLU:
            case MNN::OpType_ReLU6:
            case MNN::OpType_UnaryOp:
                return true;
            default:
                break;
        }
        return false;
    }
    static bool _isFullReshapeOp(const MNN::OpT* op) {
        switch (op->type) {
            case MNN::OpType_ConvertTensor:
            case MNN::OpType_Reshape:
            case MNN::OpType_Squeeze:
            case MNN::OpType_Unsqueeze:
            case MNN::OpType_ExpandDims:
            case MNN::OpType_Flatten:
                return true;
            default:
                break;
        }
        return false;
    }

    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto config = Global<modelConfig>::Get();
        std::vector<int> tensorUseCount(net->tensorName.size());
        std::vector<int> tensorFromOp(net->tensorName.size());
        bool find;
        do {
            find = false;
            for (int i=0; i<net->tensorName.size(); ++i) {
                tensorUseCount[i] = 0;
                tensorFromOp[i] = -1;
            }
            for (int i=0; i<net->oplists.size(); ++i) {
                auto& op          = net->oplists[i];
                bool valid = true;
                for (auto index : op->inputIndexes) {
                    if (index < 0) {
                        valid = false;
                        break   ;
                    }
                    tensorUseCount[index]++;
                }
                for (auto index : op->outputIndexes) {
                    if (index < 0) {
                        valid = false;
                        break;
                    }
                    tensorFromOp[index] = i;
                }
            }
            for (int i=0; i<net->oplists.size(); ++i) {
                auto& op          = net->oplists[i];
                bool valid = true;
                for (auto index : op->inputIndexes) {
                    if (index < 0) {
                        valid = false;
                        break;
                    }
                }
                for (auto index : op->outputIndexes) {
                    if (index < 0) {
                        valid = false;
                        break;
                    }
                }
                if (!valid) {
                    continue;
                }
                if (!_isUnaryOp(op.get())) {
                    continue;
                }
                if (op->inputIndexes.empty()) {
                    continue;
                }
                int inputIndex = op->inputIndexes[0];
                auto inputOpIndex = tensorFromOp[inputIndex];
                if (inputOpIndex == -1) {
                    continue;
                }
                auto inputOp = net->oplists[inputOpIndex].get();
                if (!_isFullReshapeOp(inputOp)) {
                    continue;
                }
                if (inputOp->outputIndexes.empty()) {
                    // Should not go here
                    continue;
                }
                if (tensorUseCount[inputOp->outputIndexes[0]] > 1) {
                    // The result is use for other op, can't swap
                    continue;
                }
                // Swap unary and reshape
                find = true;
                std::swap(op->inputIndexes[0], inputOp->inputIndexes[0]);
                std::swap(op->outputIndexes[0], inputOp->outputIndexes[0]);
                auto t = std::move(net->oplists[i]);
                net->oplists[i] = std::move(net->oplists[inputOpIndex]);
                net->oplists[inputOpIndex] = std::move(t);
                break;
            }
        } while (find);
        return true;
    }
};
static PostConverterRegister<MoveUnaryOpBeforeReshape> __l("MoveUnaryOpBeforeReshape");
