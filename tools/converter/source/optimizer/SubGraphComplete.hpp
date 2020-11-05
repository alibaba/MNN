//
//  SubGraphComplete.hpp
//  MNN
//
//  Created by MNN on 2020/06/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_CONVERTER_OPTIMIZER_SUBGRAPH_COMPLETE_HPP_
#define MNN_CONVERTER_OPTIMIZER_SUBGRAPH_COMPLETE_HPP_

#include "MNN/expr/Expr.hpp"

namespace MNN {
namespace Express {

struct OptimizeContext {
    std::vector<SubGraphProtoT*> subgraphs;
    bool is_train;
    NetSource source;

    std::vector<SubGraphProtoT*> completed_subgraphs;
};

SubGraphProtoT* FindSubGraphByName(
                      const std::vector<SubGraphProtoT*>& subgraphs,
                      const std::string& subgraph_name);

bool CompleteSubGraph(const std::unordered_map<std::string, VARP>& inputs,
                      const SubGraphProtoT* subgraph);

}  // namespace Express
}  // namespace MNN

#endif  // MNN_CONVERTER_OPTIMIZER_SUBGRAPH_COMPLETE_HPP_


