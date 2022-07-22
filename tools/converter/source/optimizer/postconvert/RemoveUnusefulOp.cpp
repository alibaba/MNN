//
//  RemoveUnusefulOp.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <string>
#include <vector>
#include "../PostTreatUtils.hpp"
#include "RemoveTestNoUseOps.hpp"

using namespace MNN;

class RemoveUnusefulOp : public RemoveTestNoUseOps {
public:
    /* The Op's output set as input */
    bool shouldDeleteJudge(const MNN::OpT* op, const MNN::NetT* const netPtr) const override {
        static auto unuseOpType = std::vector<OpType>({OpType_Seq2Out});
        static auto unuseExtraOpType =
            std::vector<std::string>({"Identity", "IdentityN", "NoOp", "Assign", "Print", "Assert", "StopGradient", "Enter", "NextIteration", "AliasWithName"});
        if (std::find(unuseOpType.begin(), unuseOpType.end(), op->type) != unuseOpType.end()) {
            return true;
        }
        if (op->type == OpType_Extra) {
            if (std::find(unuseExtraOpType.begin(), unuseExtraOpType.end(), op->main.AsExtra()->type) !=
                unuseExtraOpType.end()) {
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
            if (op->main.AsCastParam()->dstT == MNN::DataType_DT_INT32 &&
                op->main.AsCastParam()->srcT == MNN::DataType_DT_INT64) {
                return true;
            }
            if (op->main.AsCastParam()->srcT == MNN::DataType_DT_INT32 &&
                op->main.AsCastParam()->dstT == MNN::DataType_DT_INT64) {
                return true;
            }
        }
        if (op->type == OpType_Crop) {
            if (op->main.AsCrop()->offset.empty()) {
                return true;
            }
        }
        if (op->type == OpType_Concat) {
            if (op->inputIndexes.size() == 1) {
                return true;
            }
        }
        if (op->type == OpType_Slice) {
            auto slice = op->main.AsSlice();
            if (slice->sourceType != NetSource_TENSORFLOW &&
                op->outputIndexes.size() == 1) {
                return true;
            }
            if (slice->slicePoints.empty() &&
                op->outputIndexes.size() == 1) {
                return true;
            }
            if (slice->slicePoints.size() == 1 &&
                slice->slicePoints[0] == 1 &&
                op->outputIndexes.size() == 1) {
                return true;
            }
        }
        return false;
    };
    bool shouldRemoveUnusefulInputs(const MNN::OpT* op) const override {
        if (op->type == OpType_Extra) {
            if (op->main.AsExtra()->type == "Assert") {
                return true;
            }
            if (op->main.AsExtra()->type == "NoOp") {
                return true;
            }
            if (op->main.AsExtra()->type == "Print") {
                return true;
            }
            // StopGradient should be replaced by Identity.
            // if (op->main.AsExtra()->type == "StopGradient") {
            //     return true;
            // }
        }
        return false;
    };
    bool shouldDeleteOutput(const MNN::OpT* op) const override {
        if (op->type == OpType_Extra) {
            return op->main.AsExtra()->type == "Assert";
        }
        return false;
    };
};
static PostConverterRegister<RemoveUnusefulOp> __l("RemoveUnusefulOp");
