//
//  FuseMoeRouteSoftmax.cpp
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

class FuseMoeRouteSoftmax : public PostConverter {
public:
    bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        if (!net->subgraphs.empty()) {
            return true;
        }
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

        std::set<MNN::OpT*> dead;
        auto plainTopKLastAxis = [&](const MNN::OpT* co) -> bool {
            if (co->main.type != OpParameter_TopKV2 || co->main.AsTopKV2()->routeSigmoid) {
                return false;
            }
            if (co->inputIndexes.size() == 2) {
                return true;
            }
            if (co->inputIndexes.size() != 3) {
                return false;
            }
            const int ax = co->inputIndexes[2];
            if (ax < 0 || ax >= tensorCount || producer[ax] < 0) {
                return false;
            }
            auto ao = ops[producer[ax]].get();
            if (ao->type != OpType_Const || ao->main.type != OpParameter_Blob) {
                return false;
            }
            auto blob = ao->main.AsBlob();
            return blob != nullptr && blob->int32s.size() == 1 && blob->int32s[0] == -1;
        };
        int fused = 0;
        for (int i = 0; i < n; ++i) {
            auto il = ops[i].get();
            if (il->type != OpType_IndexedLinear || il->main.type != OpParameter_IndexedLinearParam ||
                il->inputIndexes.size() != 3) {
                continue;
            }
            auto param = il->main.AsIndexedLinearParam();
            if (param->routeWeight != 2 || param->routeEps != 0.0f) {
                continue;
            }
            const int probs = il->inputIndexes[2];
            if (graphOutputs.count(probs) > 0 || consumers[probs].size() != 2) {
                continue;
            }
            const int smIdx = producer[probs];
            if (smIdx < 0) {
                continue;
            }
            auto sm = ops[smIdx].get();
            if (sm->type != OpType_Softmax || sm->main.type != OpParameter_Axis ||
                sm->main.AsAxis()->axis != -1 || sm->inputIndexes.size() != 1 ||
                sm->outputIndexes.size() != 1 || sm->outputIndexes[0] != probs) {
                continue;
            }
            int topkIdx = -1;
            bool ok = true;
            for (int c : consumers[probs]) {
                if (c == i) {
                    continue;
                }
                auto co = ops[c].get();
                if (co->type == OpType_TopKV2 && !co->inputIndexes.empty() && co->inputIndexes[0] == probs &&
                    co->outputIndexes.size() == 2 && consumers[co->outputIndexes[0]].empty() &&
                    graphOutputs.count(co->outputIndexes[0]) == 0 && plainTopKLastAxis(co)) {
                    topkIdx = c;
                } else {
                    ok = false;
                }
            }
            if (!ok || topkIdx < 0) {
                continue;
            }
            const int logits = sm->inputIndexes[0];
            ops[topkIdx]->inputIndexes[0] = logits;
            il->inputIndexes[2] = logits;
            param->routeWeight = 3;
            dead.insert(sm);
            ++fused;
        }
        if (fused == 0) {
            return true;
        }
        std::vector<std::unique_ptr<MNN::OpT>> newOps;
        newOps.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (dead.count(ops[i].get()) > 0) {
                continue;
            }
            newOps.push_back(std::move(ops[i]));
        }
        ops = std::move(newOps);
        MNN_PRINT("FuseMoeRouteSoftmax: folded %d routing softmax(es)\n", fused);
        return true;
    }
};

static PostConverterRegister<FuseMoeRouteSoftmax> __l("FuseMoeRouteSoftmax");