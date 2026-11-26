//
//  GraphRewrite.hpp
//  MNNConverter
//

#ifndef MNN_CONVERTER_GRAPH_REWRITE_HPP
#define MNN_CONVERTER_GRAPH_REWRITE_HPP

#include <memory>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "MNN_generated.h"

struct GraphView {
    MNN::NetT* net;
    std::vector<int> producer;
    std::vector<std::vector<int>> consumers;
    GraphView(MNN::NetT* n) : net(n) {
        const int tensorCount = (int)net->tensorName.size();
        producer.assign(tensorCount, -1);
        consumers.assign(tensorCount, {});
        for (int i = 0; i < (int)net->oplists.size(); ++i) {
            const auto& op = net->oplists[i];
            for (int t : op->outputIndexes) {
                if (t >= 0 && t < tensorCount) {
                    producer[t] = producer[t] == -1 ? i : -2;
                }
            }
            for (int t : op->inputIndexes) {
                if (t >= 0 && t < tensorCount) {
                    consumers[t].push_back(i);
                }
            }
        }
    }
    MNN::OpT* opAt(int i) const { return (i >= 0 && i < (int)net->oplists.size()) ? net->oplists[i].get() : nullptr; }
    int producerOf(int tensor) const { return (tensor >= 0 && tensor < (int)producer.size()) ? producer[tensor] : -1; }
    MNN::OpT* producerOp(int tensor) const { return opAt(producerOf(tensor)); }
};

struct OpReplacement {
    std::unique_ptr<MNN::OpT> op;
    int insertPos;
    std::vector<int> removedOps;
};

inline bool applyReplacements(MNN::NetT* net, const GraphView& g, std::vector<OpReplacement>& replacements) {
    const int opCount = (int)net->oplists.size();
    std::map<std::string, int> tensorIndex;
    for (int t = 0; t < (int)net->tensorName.size(); ++t) {
        tensorIndex[net->tensorName[t]] = t;
    }
    std::set<int> graphOutputs;
    for (const auto& name : net->outputName) {
        auto iter = tensorIndex.find(name);
        if (iter != tensorIndex.end()) {
            graphOutputs.insert(iter->second);
        }
    }
    std::set<int> removedAll;
    std::set<int> producedAll;
    for (auto& r : replacements) {
        if (r.op == nullptr || r.insertPos < 0 || r.insertPos > opCount) {
            return false;
        }
        for (int idx : r.removedOps) {
            if (idx < 0 || idx >= opCount || !removedAll.insert(idx).second) {
                return false;
            }
        }
        for (int t : r.op->outputIndexes) {
            if (!producedAll.insert(t).second) {
                return false;
            }
        }
    }
    for (auto& r : replacements) {
        std::set<int> removedHere(r.removedOps.begin(), r.removedOps.end());
        for (int t : r.op->outputIndexes) {
            if (removedHere.count(g.producerOf(t)) == 0) {
                return false;
            }
            for (int c : g.consumers[t]) {
                if (removedAll.count(c) == 0 && c < r.insertPos) {
                    return false;
                }
            }
        }
        for (int t : r.op->inputIndexes) {
            int p = g.producerOf(t);
            if (p == -2 || (p >= 0 && (p >= r.insertPos || removedAll.count(p) > 0))) {
                return false;
            }
        }
        for (int idx : r.removedOps) {
            for (int t : net->oplists[idx]->outputIndexes) {
                if (producedAll.count(t) > 0) {
                    continue;
                }
                if (graphOutputs.count(t) > 0) {
                    return false;
                }
                for (int c : g.consumers[t]) {
                    if (removedAll.count(c) == 0) {
                        return false;
                    }
                }
            }
        }
    }
    std::vector<std::unique_ptr<MNN::OpT>> newOps;
    newOps.reserve(opCount + replacements.size());
    for (int i = 0; i < opCount; ++i) {
        for (auto& r : replacements) {
            if (r.op != nullptr && r.insertPos == i) {
                newOps.push_back(std::move(r.op));
            }
        }
        if (removedAll.count(i) == 0) {
            newOps.push_back(std::move(net->oplists[i]));
        }
    }
    for (auto& r : replacements) {
        if (r.op != nullptr && r.insertPos == opCount) {
            newOps.push_back(std::move(r.op));
        }
    }
    net->oplists = std::move(newOps);
    return true;
}

inline int sweepUnreachable(MNN::NetT* net) {
    std::set<std::string> outputNames(net->outputName.begin(), net->outputName.end());
    const int tensorCount = (int)net->tensorName.size();
    std::vector<int> producer(tensorCount, -1);
    for (int i = 0; i < (int)net->oplists.size(); ++i) {
        for (int t : net->oplists[i]->outputIndexes) {
            if (t >= 0 && t < tensorCount) {
                producer[t] = i;
            }
        }
    }
    std::vector<bool> reachable(net->oplists.size(), false);
    std::vector<int> stack;
    for (int t = 0; t < tensorCount; ++t) {
        if (outputNames.find(net->tensorName[t]) != outputNames.end()) {
            stack.push_back(t);
        }
    }
    if (stack.empty()) {
        return 0;
    }
    for (int i = 0; i < (int)net->oplists.size(); ++i) {
        const auto& op = net->oplists[i];
        if (op->type == MNN::OpType_Input || op->outputIndexes.empty()) {
            if (!reachable[i]) {
                reachable[i] = true;
                for (int t : op->inputIndexes) {
                    stack.push_back(t);
                }
            }
        }
    }
    while (!stack.empty()) {
        int t = stack.back();
        stack.pop_back();
        int p = (t >= 0 && t < tensorCount) ? producer[t] : -1;
        if (p < 0 || reachable[p]) {
            continue;
        }
        reachable[p] = true;
        for (int in : net->oplists[p]->inputIndexes) {
            stack.push_back(in);
        }
    }
    int removed = 0;
    int cursor = 0;
    for (auto iter = net->oplists.begin(); iter != net->oplists.end();) {
        if (!reachable[cursor]) {
            iter = net->oplists.erase(iter);
            ++removed;
        } else {
            ++iter;
        }
        ++cursor;
    }
    return removed;
}

#endif