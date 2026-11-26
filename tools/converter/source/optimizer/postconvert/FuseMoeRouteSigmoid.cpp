//
//  FuseMoeRouteSigmoid.cpp
//  MNNConverter
//
//  Created by MNN on 2026/11/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <set>
#include <string>
#include <vector>
#include <MNN/MNNDefine.h>
#include "../PostTreatUtils.hpp"

using namespace MNN;

class FuseMoeRouteSigmoid : public PostConverter {
public:
    bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto& ops = net->oplists;
        const int n = (int)ops.size();
        const int tensorCount = (int)net->tensorName.size();
        std::vector<int> producer(tensorCount, -1);
        std::vector<std::vector<int>> consumers(tensorCount);
        for (int i = 0; i < n; ++i) {
            for (int out : ops[i]->outputIndexes) {
                if (out >= 0 && out < tensorCount) {
                    producer[out] = i;
                }
            }
            for (int in : ops[i]->inputIndexes) {
                if (in >= 0 && in < tensorCount) {
                    consumers[in].push_back(i);
                }
            }
        }
        std::set<std::string> outputNames(net->outputName.begin(), net->outputName.end());
        std::set<int> graphOutputs;
        for (int t = 0; t < tensorCount; ++t) {
            if (outputNames.count(net->tensorName[t]) > 0) {
                graphOutputs.insert(t);
            }
        }
        auto isConst = [&](int t) {
            return producer[t] >= 0 && ops[producer[t]]->type == OpType_Const;
        };

        std::set<MNN::OpT*> dead;
        int fused = 0;
        for (int i = 0; i < n; ++i) {
            auto il = ops[i].get();
            if (il->type != OpType_IndexedLinear || il->main.type != OpParameter_IndexedLinearParam ||
                il->inputIndexes.size() != 3) {
                continue;
            }
            auto param = il->main.AsIndexedLinearParam();
            if (param->routeWeight != 2) {
                continue;
            }
            const int probs = il->inputIndexes[2];
            if (graphOutputs.count(probs) > 0 || consumers[probs].size() != 2 || producer[probs] < 0) {
                continue;
            }
            auto sig = ops[producer[probs]].get();
            if (sig->type != OpType_UnaryOp || sig->main.type != OpParameter_UnaryOp ||
                sig->main.AsUnaryOp()->opType != UnaryOpOperation_SIGMOID ||
                sig->inputIndexes.size() != 1 || sig->outputIndexes.size() != 1 ||
                sig->outputIndexes[0] != probs) {
                continue;
            }
            int addIdx = -1, topkIdx = -1;
            bool ok = true;
            for (int c : consumers[probs]) {
                if (c == i) {
                    continue;
                }
                auto co = ops[c].get();
                if (co->type == OpType_BinaryOp && co->main.type == OpParameter_BinaryOp &&
                    co->main.AsBinaryOp()->opType == BinaryOpOperation_ADD && co->inputIndexes.size() == 2 &&
                    co->outputIndexes.size() == 1 && consumers[co->outputIndexes[0]].size() == 1 &&
                    graphOutputs.count(co->outputIndexes[0]) == 0) {
                    addIdx = c;
                } else {
                    ok = false;
                }
            }
            if (!ok || addIdx < 0) {
                continue;
            }
            auto add = ops[addIdx].get();
            topkIdx = consumers[add->outputIndexes[0]][0];
            auto topk = ops[topkIdx].get();
            if (topkIdx >= i || topk->type != OpType_TopKV2 || topk->main.type != OpParameter_TopKV2 ||
                topk->inputIndexes.size() != 3 || topk->inputIndexes[0] != add->outputIndexes[0] ||
                topk->outputIndexes.size() != 2 || !consumers[topk->outputIndexes[0]].empty() ||
                graphOutputs.count(topk->outputIndexes[0]) > 0 || !topk->main.AsTopKV2()->largest) {
                continue;
            }
            const int bias = add->inputIndexes[0] == probs ? add->inputIndexes[1] : add->inputIndexes[0];
            if (!isConst(bias) || !isConst(topk->inputIndexes[1]) || !isConst(topk->inputIndexes[2])) {
                continue;
            }
            auto biasBlob = ops[producer[bias]]->main.AsBlob();
            if (biasBlob->dims.size() != 1) {
                continue;
            }
            auto axisBlob = ops[producer[topk->inputIndexes[2]]]->main.AsBlob();
            if (axisBlob->int32s.size() != 1 || axisBlob->int32s[0] != -1) {
                continue;
            }
            const int logits = sig->inputIndexes[0];
            topk->inputIndexes = {logits, topk->inputIndexes[1], topk->inputIndexes[2], bias};
            topk->main.AsTopKV2()->routeSigmoid = true;
            topk->outputIndexes[0] = probs;
            dead.insert(sig);
            dead.insert(add);
            ++fused;
        }
        if (fused == 0) {
            return true;
        }
        std::vector<std::unique_ptr<MNN::OpT>> newOps;
        newOps.reserve(ops.size());
        for (auto& op : ops) {
            if (dead.count(op.get()) > 0) {
                continue;
            }
            newOps.push_back(std::move(op));
        }
        ops = std::move(newOps);
        MNN_PRINT("FuseMoeRouteSigmoid: folded %d routing sigmoid chain(s)\n", fused);
        return true;
    }
};

static PostConverterRegister<FuseMoeRouteSigmoid> __l("FuseMoeRouteSigmoid");