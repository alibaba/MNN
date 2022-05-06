//
//  ReIndexOnnxIfAlias.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNDefine.h>
#include "../PostTreatUtils.hpp"
#include "../Global.hpp"
#include "../SubGraphComplete.hpp"
using namespace MNN;
class ReIndexOnnxIfAlias : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        if (net->sourceType != NetSource_ONNX) {
            return true;
        }
        auto* ctx = Global<MNN::Express::OptimizeContext>::Get();
        auto updateNameFromNewSubGraph = [=](const std::string& graphName, std::string& aliasName) {
            for (auto gOld : ctx->subgraphs) {
                if (gOld->name != graphName) {
                    continue;
                }
                int idx = -1;
                for (int i = 0; i < gOld->outputs.size(); ++i) {
                    if (gOld->tensors[gOld->outputs[i]] == aliasName) {
                        idx = i;
                        break;
                    }
                }
                if (idx < 0) {
                    break;
                }
                for (auto gNew : ctx->completed_subgraphs) {
                    if (gNew->name != graphName) {
                        continue;
                    }
                    aliasName = gNew->tensors[gNew->outputs[idx]];
                    break;
                }
            }
        };
        for (auto& op : net->oplists) {
            if (op->type != OpType_If) {
                continue;
            }
            auto param = op->main.AsIfParam();
            for (auto& pair : param->aliases_outputs) {
                updateNameFromNewSubGraph(param->then_graph, pair->data[0]);
                updateNameFromNewSubGraph(param->else_graph, pair->data[1]);
            }
        }
        return true;
    }
};
static PostConverterRegister<ReIndexOnnxIfAlias> __l("ReIndexOnnxIfAlias");
