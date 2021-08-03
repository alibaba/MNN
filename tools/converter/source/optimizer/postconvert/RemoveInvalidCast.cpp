//
//  RemoveInvalidCast.cpp
//  MNNConverter
//
//  Created by MNN on 2021/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <algorithm>
#include "../PostTreatUtils.hpp"

class RemoveInvalidCast : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        if (net->sourceType == MNN::NetSource_TENSORFLOW || net->sourceType == MNN::NetSource_TFLITE) {
            // The two framework has valid src type for cast, don't need treat
            return true;
        }
        bool needTreat = false;
        for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
            auto& op = *iter;
            if (op->type == MNN::OpType_Cast) {
                needTreat = true;
                break;
            }
        }
        if (!needTreat) {
            return true;
        }
        // Infer DataType for All Tensor
        std::vector<MNN::DataType> types(net->tensorName.size(), MNN::DataType_DT_INVALID);
        for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
            auto& op = *iter;
            switch (op->type) {
                case MNN::OpType_Input:
                    types[op->outputIndexes[0]] = op->main.AsInput()->dtype;
                    break;
                case MNN::OpType_Cast:
                    types[op->outputIndexes[0]] = op->main.AsCastParam()->dstT;
                    break;
                case MNN::OpType_Const:
                case MNN::OpType_TrainableParam:
                    types[op->outputIndexes[0]] = op->main.AsBlob()->dataType;
                    break;
                case MNN::OpType_Fill:
                    types[op->outputIndexes[0]] = types[op->inputIndexes[1]];
                    break;
                case MNN::OpType_Shape:
                case MNN::OpType_Size:
                case MNN::OpType_Rank:
                case MNN::OpType_UnravelIndex:
                    types[op->outputIndexes[0]] = MNN::DataType_DT_INT32;
                    break;
                case MNN::OpType_RandomUniform:
                    types[op->outputIndexes[0]] = op->main.AsRandomUniform()->type;
                    break;
                case MNN::OpType_TopKV2:
                    types[op->outputIndexes[0]] = types[op->inputIndexes[0]];
                    if (op->outputIndexes.size() > 1) {
                        types[op->outputIndexes[1]] = MNN::DataType_DT_INT32;
                    }
                    break;
                case MNN::OpType_ScatterNd:
                case MNN::OpType_Select:
                    types[op->outputIndexes[0]] = types[op->inputIndexes[1]];
                    break;
                case MNN::OpType_OneHot:
                    types[op->outputIndexes[0]] = types[op->inputIndexes[2]];
                    break;
                default:
                    break;
            }
        }
        // Remove Useless Cast
        const MNN::NetT* const netPtr = net.get();
        for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
            auto& op          = *iter;
            if (op->type != MNN::OpType_Cast) {
                iter++;
                continue;
            }
            if (types[op->inputIndexes[0]] == MNN::DataType_DT_INVALID) {
                iter++;
                continue;
            }
            if (types[op->inputIndexes[0]] != types[op->outputIndexes[0]]) {
                iter++;
                break;
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
static PostConverterRegister<RemoveInvalidCast> __l("RemoveInvalidCast");
