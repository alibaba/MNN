//
//  TFGraphResolver.cpp
//  MNNConverter
//
//  Created by MNN on 2020/06/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TFGraphResolver.hpp"
#include "TFGraphResolverHelpers.hpp"

#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "graph.pb.h"
#include "tfOpConverter.hpp"
#include "MNN_generated.h"
#include "../compression/quantization.hpp"
#include <flatbuffers/util.h>

void TFGraph::AddNode(const NodeDef* node) {
    std::unique_ptr<TFNode> tf_node(new TFNode);
    tf_node->node_def = node;
    tf_node->name = node->name();
    tf_node->op = node->op();
    nodes_.push_back(std::move(tf_node));
}

void TFGraph::Finalize() {
    std::unordered_map<std::string, TFNode*> nodes;
    for (auto& node : nodes_) {
        nodes.emplace(node->name, node.get());
    }
    for (auto& node : nodes_) {
        const NodeDef* node_def = node->node_def;
        for (int i = 0; i < node_def->input_size(); ++i) {
            const std::string& input = node_def->input(i);
            if (IsControlInput(input)) {
                continue;
            }
            std::string input_op = input;
            auto splits = RSplitString(input, ":");
            if (splits.size() == 2) {
                input_op = splits.at(0);
            }
            TFNode* start = nodes.at(input_op);
            std::unique_ptr<TFEdge> edge(new TFEdge);
            *edge = TFEdge{input, start, node.get()};
            node->inputs.push_back(edge.get());
            start->outputs.push_back(edge.get());
            edges_.push_back(std::move(edge));
        }
    }
    for (auto& node : nodes_) {
        if (node->outputs.empty()) {
            final_nodes_.push_back(node.get());
        }
    }
}

std::unique_ptr<MNN::SubGraphProtoT> TFGraph::ToProto() const {
    std::unique_ptr<MNN::SubGraphProtoT> graph_proto(new MNN::SubGraphProtoT);
    graph_proto->name = name_;
    std::vector<const TFNode*> entry_nodes;

    std::unordered_map<std::string, int> tensor_indices;
    // Add normal nodes.
    for (int i = 0; i < nodes_.size(); ++i) {
        TFNode* node = nodes_[i].get();
        std::shared_ptr<TmpNode> tempNode(new TmpNode());
        tempNode->opName = node->name;
        tempNode->opType = node->op;
        tempNode->tfNode = node->node_def;

        MNN::OpT *op = new MNN::OpT;
        auto creator = tfOpConverterSuit::get()->search(tempNode->opType);
        DCHECK(creator) << "MNN Converter NOT_SUPPORTED_OP: [ "
                        << tempNode->opType << " ]";
        op->name = tempNode->opName;
        op->type = creator->opType();
        op->main.type = creator->type();

        // resize the inputIndexes and outputIndexes
        int input_size = node->inputs.size();
        op->inputIndexes.resize(input_size);

        // -1 is placeholder value, and the number of -1 is the number of
        // output tensors.
        // defalut: every op output one tensor, if the number of the output
        // tensors is bigger than 1, set the outputIndexes in the op
        // converter(void run(MNN::OpT *dstOp, TmpNode *srcNode))
        op->outputIndexes = {-1};
        creator->run(op, tempNode.get());

        for (int j = 0; j < input_size; j++) {
            std::string input = node->inputs[j]->name;
            auto it = tensor_indices.find(input);
            if (it == tensor_indices.end()) {
                int index = tensor_indices.size();
                it = tensor_indices.emplace(input, index).first;
                graph_proto->tensors.push_back(input);
            }
            op->inputIndexes[j] = it->second;
        }

        int output_size = node->outputs.size();
        for (int j = 0; j < node->outputs.size(); ++j) {
            std::string output = node->outputs[j]->name;
            auto it = tensor_indices.find(output);
            if (it == tensor_indices.end()) {
                int index = tensor_indices.size();
                it = tensor_indices.emplace(output, index).first;
                graph_proto->tensors.push_back(output);
            }
            int index = 0;
            auto splits = RSplitString(output, ":");
            if (splits.size() == 2) {
                index = atoi(splits[1].c_str());
            }
            if (op->outputIndexes.size() <= index) {
                int origin_size = op->outputIndexes.size();
                op->outputIndexes.resize(index + 1);
                for (int p = origin_size; p <= index; ++p) {
                    op->outputIndexes[p] = -1;
                }
            }
            op->outputIndexes[index] = it->second;
        }
        graph_proto->nodes.emplace_back(op);
    }

    for (auto &op : graph_proto->nodes) {
        for (int i = 0; i < op->outputIndexes.size(); ++i) {
            if (op->outputIndexes[i] == -1) {
                int index = graph_proto->tensors.size();
                op->outputIndexes[i] = index;
                std::string output = op->name;
                if (i != 0) {
                    output += ":" + flatbuffers::NumToString(i);
                }
                graph_proto->tensors.emplace_back(output);
            }
        }
    }
    return std::move(graph_proto);
}

std::unique_ptr<TFEdge> TFGraphResolver::BuildEdge(
        const std::string& name, TFNode* start, TFNode* end) {
    std::unique_ptr<TFEdge> edge(new TFEdge);
    *edge = TFEdge{name, start, end};
    return std::move(edge);
}

std::unique_ptr<TFNode> TFGraphResolver::BuildQuantOrDequantNode(
                            const std::string& name,
                            const std::string& op,
                            const int& nbit,
                            const std::vector<float>& scales,
                            const float& zero_point, const float& clamp_min, const float& clamp_max,
                            const MNN::Compression::LayerQuantizeParams_QuantMethod& method) {
    std::unique_ptr<NodeDef> node_def(new NodeDef);
    *(node_def->mutable_name()) = name;
    *(node_def->mutable_op()) = op;
    (*node_def->mutable_attr())["nbit"].set_i(nbit);
    auto* list = (*node_def->mutable_attr())["scale"].mutable_list();
    for (int i = 0; i < scales.size(); ++i) {
        if (op == "CustomQuantize") {
            list->mutable_f()->Add(1.f / scales[i]);
        } else {
            list->mutable_f()->Add(scales[i]);
        }
    }
    (*node_def->mutable_attr())["zero_point"].set_f(zero_point);
    (*node_def->mutable_attr())["clamp_min"].set_f(clamp_min);
    (*node_def->mutable_attr())["clamp_max"].set_f(clamp_max);
    (*node_def->mutable_attr())["method"].set_i(int(method));
    std::unique_ptr<TFNode> quant_node(new TFNode);
    quant_node->name = name;
    quant_node->op = op;
    quant_node->node_def = node_def.get();

    main_graph()->allocated_nodes_.push_back(std::move(node_def));
    return std::move(quant_node);
}

void TFGraphResolver::ResolveQuantization(
        TFGraph* graph,
        const compression::Quantization& int8_calibration) {
    std::vector<std::unique_ptr<TFNode>> append_nodes;
    std::vector<std::unique_ptr<TFEdge>> append_edges;

    static int64_t uuid = 0;
    auto AddQuantizeAndDequantizeNodes =
            [&, this](const std::vector<TFEdge*> edges,
                      const compression::Quantization::TensorParams& params) {
        TFNode* start_node = edges.at(0)->start;
        for (TFEdge* edge : edges) {
            EraseOutput(start_node, edge);
        }
        auto splits = RSplitString(edges.at(0)->name, ":");
        const std::string& op_name = splits.at(0);
        // Add quantize node.
        std::string quant_name = op_name + "_quant_" + flatbuffers::NumToString(uuid);
        std::unique_ptr<TFNode> quant_node = BuildQuantOrDequantNode(
            quant_name, "CustomQuantize", params.nbit, params.scale,
            params.zero_point, params.clamp_min, params.clamp_max, params.method);
        // Add dequantize node.
        std::string dequant_name = quant_name + "_dequant_" + flatbuffers::NumToString(uuid);
        std::unique_ptr<TFNode> dequant_node = BuildQuantOrDequantNode(
            dequant_name, "CustomDequantize", params.nbit, params.scale,
            params.zero_point, params.clamp_min, params.clamp_max, params.method);

        // Update UUID.
        ++uuid;

        // Connect quantize and dequantize node.
        std::unique_ptr<TFEdge> quant_edge =
            BuildEdge(edges.at(0)->name, start_node, quant_node.get());
        // Connect dequantize and the next node.
        std::unique_ptr<TFEdge> dequant_edge =
            BuildEdge(quant_node->name, quant_node.get(), dequant_node.get());

        AddOutput(start_node, quant_edge.get());

        quant_node->inputs = {quant_edge.get()};
        quant_node->outputs = {dequant_edge.get()};
        dequant_node->inputs = {dequant_edge.get()};
        dequant_node->outputs = edges;
        for (TFEdge* edge : edges) {
            edge->name = dequant_node->name;
            edge->start = dequant_node.get();
        }
        append_nodes.push_back(std::move(quant_node));
        append_nodes.push_back(std::move(dequant_node));
        append_edges.push_back(std::move(quant_edge));
        append_edges.push_back(std::move(dequant_edge));

        // Return dequant edge.
        return append_edges.back().get();
    };

    const auto& tensor_params = int8_calibration.tensors;
    for (auto& node : graph->nodes_) {
        std::unordered_map<std::string, std::vector<TFEdge*>> quant_edges;
        for (TFEdge* output : node->outputs) {
            std::string tensor_name = output->name;
            if (node->op == "Enter" || node->op == "Switch") {
                // The input names of the node maybe replaced by the quantize
                // and dequantize op, so here we use the input name from the
                // `node_def` since it should not be modified at any time.
                // tensor_name = node->inputs.at(0)->name;
                tensor_name = node->node_def->input(0);
            }
            quant_edges[tensor_name].push_back(output);
        }
        for (const auto& it : quant_edges) {
            auto p = tensor_params.find(it.first);
            if (p == tensor_params.end()) {
                continue;
            }
            const auto& params = p->second.at(0);
            AddQuantizeAndDequantizeNodes(it.second, params);
        }
    }
    for (auto& node : graph->nodes_) {
        std::unordered_map<std::string, std::vector<TFEdge*>> quant_edges;
        for (int i = 0; i < node->inputs.size(); ++i) {
            TFEdge* edge = node->inputs[i];
            quant_edges[edge->name].push_back(edge);
        }
        for (const auto& it : quant_edges) {
            auto p = tensor_params.find(it.first);
            if (p == tensor_params.end()) {
                continue;
            }
            const auto& params = p->second.at(0);
            AddQuantizeAndDequantizeNodes(it.second, params);
        }
    }
    // Append nodes and edges to root graph.
    for (auto& node : append_nodes) {
        main_graph()->nodes_.push_back(std::move(node));
    }
    for (auto& edge : append_edges) {
        main_graph()->edges_.push_back(std::move(edge));
    }
}

TFGraphResolver::TFGraphResolver(const tensorflow::GraphDef& graph_def) {
    std::unique_ptr<TFGraph> tf_graph(new TFGraph);
    const int count = graph_def.node_size();
    for (int i = 0; i < count; ++i) {
        const NodeDef& node_def = graph_def.node(i);
        tf_graph->AddNode(&node_def);
    }
    tf_graph->Finalize();
    graphs_.push_back(std::move(tf_graph));

    TFGraph* main_graph = graphs_.back().get();
}

const TFGraph* TFGraphResolver::graph(const int graph_index) const {
    return graphs_.at(graph_index).get();
}

TFGraph* TFGraphResolver::graph(const int graph_index) {
    return graphs_.at(graph_index).get();
}

TFGraph* TFGraphResolver::main_graph() {
    return this->graph(0);
}
