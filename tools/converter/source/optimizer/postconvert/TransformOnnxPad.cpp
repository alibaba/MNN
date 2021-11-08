//
//  TransformOnnxPad.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../PostTreatUtils.hpp"
using namespace MNN;
class TransformOnnxPad : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
            auto op = iter->get();
            if (OpType_Padding == op->type && op->main.type == OpParameter_Blob && op->inputIndexes.size() == 1) {
                std::unique_ptr<OpT> paddingConst(new OpT);
                paddingConst->type          = OpType_Const;
                paddingConst->main.type     = OpParameter_Blob;
                paddingConst->main.value    = new BlobT(*op->main.AsBlob());
                paddingConst->name          = op->name + "padding";
                paddingConst->outputIndexes = {(int)net->tensorName.size()};
                net->tensorName.emplace_back(paddingConst->name);
                op->inputIndexes = {op->inputIndexes[0], paddingConst->outputIndexes[0]};
                op->main.Reset();
                iter = net->oplists.insert(iter, std::move(paddingConst));
                iter++;
                iter++;
                continue;
            }
            iter++;
        }
        return true;
    }
};
static PostConverterRegister<TransformOnnxPad> __l("TransformOnnxPad");
