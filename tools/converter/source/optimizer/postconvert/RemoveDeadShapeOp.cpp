//
//  RemoveDeadShapeOp.cpp
//  MNNConverter
//
//  Created by MNN on 2026/07/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <set>
#include <vector>
#include "../PostTreatUtils.hpp"

using namespace MNN;

// After transformer fusion (RoPE / Attention), the shape-computation subgraphs
// that originally fed the fused ops' Reshape targets are left with no consumer.
// The generic reference-count DCE in RemoveTestNoUseOps treats any unconsumed
// output as a network output (refcount 1), so these orphaned subgraphs survive.
//
// This pass performs a reachability-based dead-code elimination: starting from
// the real network outputs, it walks the producer graph backwards and marks all
// reachable ops. Unreachable ops are removed, but ONLY when their type belongs
// to a conservative whitelist of pure shape / index arithmetic ops, so no op
// carrying tensor computation or side effects can ever be deleted.
class RemoveDeadShapeOp : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        // Conservative whitelist: only pure shape / index arithmetic ops.
        // Ops like Reshape, BinaryOp, Concat, StridedSlice, Gather etc. are
        // excluded because they also participate in data computation in many
        // models (especially TF), and would be wrongly deleted if the
        // reachability analysis misses any output tensor name.
        static const std::set<OpType> kShapeOpWhitelist = {
            OpType_Shape,       OpType_Rank,        OpType_Size,
        };

        const int tensorCount = (int)net->tensorName.size();

        // producer[tensorIndex] = op index that writes it (-1 if none)
        std::vector<int> producer(tensorCount, -1);
        for (int i = 0; i < (int)net->oplists.size(); ++i) {
            for (auto out : net->oplists[i]->outputIndexes) {
                if (out >= 0 && out < tensorCount) {
                    producer[out] = i;
                }
            }
        }

        // Roots: declared network outputs.
        std::set<std::string> outputNames(net->outputName.begin(), net->outputName.end());
        std::vector<int> stack;
        for (int t = 0; t < tensorCount; ++t) {
            if (outputNames.find(net->tensorName[t]) != outputNames.end()) {
                stack.push_back(t);
            }
        }

        // Safety: if no output roots were resolved, the reachability analysis
        // would mark everything as dead. Skip to avoid incorrect deletions.
        if (stack.empty()) {
            return true;
        }

        // Backward reachability over the producer graph.
        std::vector<bool> reachable(net->oplists.size(), false);
        while (!stack.empty()) {
            int t = stack.back();
            stack.pop_back();
            int op = (t >= 0 && t < tensorCount) ? producer[t] : -1;
            if (op < 0 || reachable[op]) {
                continue;
            }
            reachable[op] = true;
            for (auto in : net->oplists[op]->inputIndexes) {
                stack.push_back(in);
            }
        }

        // Decide deletion per original op index to avoid index drift while erasing.
        // Input ops have no inputs and are never in the whitelist, so graph inputs
        // are always preserved.
        std::vector<bool> deleteOp(net->oplists.size(), false);
        for (int i = 0; i < (int)net->oplists.size(); ++i) {
            deleteOp[i] = !reachable[i] && kShapeOpWhitelist.count(net->oplists[i]->type) > 0;
        }
        int removed = 0;
        int cursor  = 0;
        for (auto iter = net->oplists.begin(); iter != net->oplists.end(); ++cursor) {
            if (deleteOp[cursor]) {
                iter = net->oplists.erase(iter);
                ++removed;
            } else {
                ++iter;
            }
        }
        if (removed > 0) {
            LOG(INFO) << "[RemoveDeadShapeOp] removed " << removed << " dead shape ops";
        }
        return true;
    }
};

static PostConverterRegister<RemoveDeadShapeOp> __l("RemoveDeadShapeOp");