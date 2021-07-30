//
//  TFGraphResolverHelpers.hpp
//  MNNConverter
//
//  Created by MNN on 2020/06/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TF_GRAPH_RESOLVER_HELPERS_HPP_
#define TF_GRAPH_RESOLVER_HELPERS_HPP_

#include "TFGraphResolver.hpp"

#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "MNN_generated.h"
#include "../compression/quantization.hpp"

inline bool IsControlInput(const std::string& name) {
    return name.size() && name.substr(0, 1) == "^";
}

inline bool IsInWhileLoop(const std::string& name,
                          const std::unordered_map<std::string,
                                                   std::string>& types) {
    return types.at(name) == "WhileLoop";
}

inline std::vector<std::string> RSplitString(const std::string& name,
                                             const std::string& sp) {
    std::vector<std::string> splits;
    size_t pos = name.rfind(sp);
    if (pos != std::string::npos) {
        splits.push_back(name.substr(0, pos));
        splits.push_back(name.substr(pos + 1));
    } else {
        splits.push_back(name);
    }
    return std::move(splits);
}

inline bool IsControlFlowNode(const TFNode* node) {
    static std::unordered_set<std::string> control_flow_ops{
        "Enter", "Merge", "LoopCond", "Switch", "NextIteration", "Exit",
    };
    return control_flow_ops.count(node->op);
}

template <typename T>
inline T NodeAttr(const TFNode* node, const std::string& attr_name) {
    MNN_ERROR("This function should not be called.\n");
    return T();
}

template <>
inline std::string NodeAttr<std::string>(const TFNode* node,
                                         const std::string& attr_name) {
    if (!node->node_def->attr().count(attr_name)) {
        MNN_ERROR("Can not find attribute named %s.\n", attr_name.c_str());
    }
    return node->node_def->attr().at(attr_name).s();
}

inline void EraseInput(TFNode* node, TFEdge* edge) {
    auto it = std::remove_if(node->inputs.begin(), node->inputs.end(),
                             [edge](const TFEdge* input) {
                                 return input == edge;
                             });
    node->inputs.erase(it, node->inputs.end());
}

inline void AddInput(TFNode* node, TFEdge* edge) {
    node->inputs.push_back(edge);
}

inline void EraseOutput(TFNode* node, TFEdge* edge) {
    auto it = std::remove_if(node->outputs.begin(), node->outputs.end(),
                             [edge](const TFEdge* output) {
                                 return output == edge;
                             });
    node->outputs.erase(it, node->outputs.end());
}

inline void AddOutput(TFNode* node, TFEdge* edge) {
    node->outputs.push_back(edge);
}

template <typename NodeT, typename StopFunc>
inline std::vector<NodeT*> ReverseVisit(const std::vector<NodeT*>& final_nodes,
                                        StopFunc stop_fn) {
    std::vector<NodeT*> nodes;
    std::unordered_set<NodeT*> visited;
    std::queue<NodeT*> queue;
    for (NodeT* node : final_nodes) {
        queue.push(node);
        visited.insert(node);
    }

    while (!queue.empty()) {
        NodeT* node = queue.front();
        queue.pop();
        
        if (stop_fn(node)) {
            continue;
        }
        nodes.push_back(node);
        for (auto* edge : node->inputs) {
            NodeT* start = edge->start;
            if (visited.insert(start).second) {
                queue.push(start);
            }
        }
    }
    return std::move(nodes);
}

#endif  // TF_GRAPH_RESOLVER_HELPERS_HPP_
