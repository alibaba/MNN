//
//  TFGraphResolver.hpp
//  MNNConverter
//
//  Created by MNN on 2020/06/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TF_GRAPH_RESOLVER_HPP_
#define TF_GRAPH_RESOLVER_HPP_

#include <vector>
#include <unordered_map>

#include "MNN/MNNDefine.h"
#include "graph.pb.h"
#include "MNN_generated.h"

#include "../compression/quantization.hpp"

struct TFNode;
typedef tensorflow::NodeDef NodeDef;

struct TFEdge {
    std::string name;
    TFNode* start;
    TFNode* end;
};

struct TFNode {
    const NodeDef* node_def;
    std::string name;
    std::string op;
    std::vector<TFEdge*> inputs;
    std::vector<TFEdge*> outputs;
};

class TFGraph {
 public:
     TFGraph() = default;
     virtual ~TFGraph() = default;

     void AddNode(const NodeDef* node);
     void Finalize();

     std::unique_ptr<MNN::SubGraphProtoT> ToProto() const;

 private:
    friend class TFGraphResolver;
    std::string name_ = "main";

    std::vector<std::unique_ptr<TFNode>> nodes_;
    std::vector<std::unique_ptr<TFEdge>> edges_;
    std::vector<std::unique_ptr<NodeDef>> allocated_nodes_;
    // Output nodes.
    std::vector<TFNode*> final_nodes_;
};

class TFGraphResolver {
 public:
    explicit TFGraphResolver(const tensorflow::GraphDef& graph_def);
    virtual ~TFGraphResolver() = default;

    TFGraph* graph(const int graph_index);

    const TFGraph* graph(const int graph_index) const;

    int graph_size() const { return graphs_.size(); }

 private:
    TFGraph* main_graph();

    void ResolveQuantization(TFGraph* graph,
                             const compression::Quantization& int8_calibration);

    std::unique_ptr<TFNode> BuildQuantOrDequantNode(
        const std::string& name,
        const std::string& op,
        const int& nbit,
        const std::vector<float>& scales,
        const float& zero_point,
        const float& clamp_min,
        const float& clamp_max,
        const MNN::Compression::LayerQuantizeParams_QuantMethod& method);

   std::unique_ptr<TFEdge> BuildEdge(
        const std::string& name, TFNode* start, TFNode* end);

 private:
    std::vector<std::unique_ptr<TFGraph>> graphs_;

    struct StringComp {
        bool operator()(const std::string& lhs, const std::string& rhs) const {
            if (lhs.size() != rhs.size()) {
                return lhs.size() > rhs.size();
            } else {
                return lhs > rhs;
            }
        }
    };
};

#endif  // TF_GRAPH_RESOLVER_HPP_
