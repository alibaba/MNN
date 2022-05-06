//
//  SubGraphComplete.hpp
//  MNNConverter
//
//  Created by MNN on 2020/06/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_CONVERTER_OPTIMIZER_SUBGRAPH_COMPLETE_HPP_
#define MNN_CONVERTER_OPTIMIZER_SUBGRAPH_COMPLETE_HPP_

#include "MNN/expr/Expr.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

namespace MNN {
namespace Express {

struct OptimizeContext {
    std::vector<SubGraphProtoT*> subgraphs;
    bool is_training;
    bool verbose;
    bool first_run = true;
    NetSource source;

    std::vector<SubGraphProtoT*> completed_subgraphs;

    using NetTPtr = std::unique_ptr<MNN::NetT>;
    template <typename K, typename V>
    using HashMap = std::unordered_map<K, V>;

    // NetTPtr (*RunOptimize)(NetTPtr&, const HashMap<std::string, VARP>&);
    std::function<NetTPtr(NetTPtr&,  // NOLINT
                          const HashMap<std::string, VARP>&)> RunOptimize;
};

SubGraphProtoT* FindSubGraphByName(
                      const std::vector<SubGraphProtoT*>& subgraphs,
                      const std::string& subgraph_name);

bool CompleteSubGraph(const std::unordered_map<std::string, VARP>& inputs,
                      const SubGraphProtoT* subgraph);

}  // namespace Express
}  // namespace MNN

#endif  // MNN_CONVERTER_OPTIMIZER_SUBGRAPH_COMPLETE_HPP_


