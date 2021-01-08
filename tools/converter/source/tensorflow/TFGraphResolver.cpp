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
#include "TmpGraph.hpp"
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
    std::vector<TFNode*> nodes = ReverseVisit(
            final_nodes_, [&entry_nodes](const TFNode* node) {
        if (IsControlFlowNode(node)) {
            entry_nodes.push_back(node);
            return true;
        }
        return false;
    });

    std::unordered_map<std::string, int> tensor_indices;
    // Add control flow entry nodes.
    std::unordered_set<std::string> entry_names;
    for (const TFNode* node : entry_nodes) {
        for (const TFEdge* edge : node->outputs) {
            entry_names.insert(edge->name);
        }
    }
    for (const auto& entry_name : entry_names) {
        MNN::OpT* entry_op = new MNN::OpT;
        entry_op->type = MNN::OpType_Input;
        entry_op->name = entry_name;
        entry_op->main.type = MNN::OpParameter_Input;
        entry_op->main.value = new MNN::InputT;
        entry_op->main.AsInput()->dtype = MNN::DataType_DT_FLOAT;
        entry_op->main.AsInput()->dims = {-1};
        entry_op->main.AsInput()->dformat = MNN::MNN_DATA_FORMAT_NCHW;

        int idx = tensor_indices.size();
        tensor_indices.emplace(entry_name, idx);
        entry_op->outputIndexes = {idx};
        graph_proto->nodes.emplace_back(entry_op);
        graph_proto->tensors.push_back(entry_name);
    }
    // Add normal nodes.
    for (int i = nodes.size() - 1; i >= 0; --i) {
        TFNode* node = nodes[i];
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

TFGraphResolver::TFGraphResolver(const tensorflow::GraphDef& graph_def,
                                 const common::Options& options) {
    std::unique_ptr<TFGraph> tf_graph(new TFGraph);
    const int count = graph_def.node_size();
    for (int i = 0; i < count; ++i) {
        const NodeDef& node_def = graph_def.node(i);
        tf_graph->AddNode(&node_def);
    }
    tf_graph->Finalize();
    graphs_.push_back(std::move(tf_graph));

    TFGraph* main_graph = graphs_.back().get();
    // Resolve quantization.
    if (options.doCompress) {
        const auto& pipeline = options.compressionPipeline;
        for (const auto& progress : pipeline.progress()) {
            if (progress.type != CompressionAlgo::QUANTIZE) {
                continue;
            }
            ResolveQuantization(main_graph, progress.quant_params);
        }
    }

    using NodeVector = std::vector<TFNode*>;
    std::unordered_map<std::string, NodeVector> node_clusters;
    for (auto& node : main_graph->nodes_) {
        std::string name = RSplitString(node->name, "/").at(0);
        auto it = node_clusters.find(name);
        if (it == node_clusters.end()) {
            it = node_clusters.emplace(name, NodeVector()).first;
        }
        it->second.push_back(node.get());
    }

    // We broadly divided all nodes into clusters by the prefix of the node
    // name, and each cluster belongs to one of the tree categories,
    // Normal, Condition or WhileLoop.
    // The nodes which have the same name prefix maybe belong to the same
    // cluster. The nodes that type is `Condition` maybe belong to a condition
    // subgraph. The nodes that type is `WhileLoop` maybe belong to a while loop
    // subgraph.
    std::unordered_map<std::string, std::string> cluster_types;
    for (const auto& cluster : node_clusters) {
        std::string type = "Normal";
        const NodeVector& nodes = cluster.second;
        for (TFNode* node : nodes) {
            if (node->op == "Switch") {
                type = "Condition";
                continue;
            }
            if (node->op == "LoopCond") {
                type = "WhileLoop";
                break;
            }
        }
        cluster_types.emplace(cluster.first, type);
    }

    for (auto& node : main_graph->nodes_) {
        std::string name_prefix = RSplitString(node->name, "/").at(0);
        if (node->op == "Enter") {
            std::string frame_name =
                NodeAttr<std::string>(node.get(), "frame_name");
            frame_name = RSplitString(frame_name, "/").at(0);
            loops_[frame_name].enter.push_back(node->inputs[0]);
        }
        if (node->op == "Merge") {
            if (IsInWhileLoop(name_prefix, cluster_types)) {
                // TODO().
            } else {
                conditions_[name_prefix].branch_else.push_back(node->inputs[0]);
                conditions_[name_prefix].branch_then.push_back(node->inputs[1]);
            }
        }
        if (node->op == "LoopCond") {
            loops_[name_prefix].name = name_prefix;
            loops_[name_prefix].cond = node->inputs[0];
        }
        if (node->op == "Switch") {
            if (IsInWhileLoop(name_prefix, cluster_types)) {
                loops_[name_prefix].loop_vars.push_back(node->inputs[0]);
                for (TFEdge* output : node->outputs) {
                    output->name = node->inputs[0]->name;
                }
            } else {
                conditions_[name_prefix].name = name_prefix;
                conditions_[name_prefix].cond = node->inputs[1];
            }
        }
        if (node->op == "NextIteration") {
            loops_[name_prefix].body.push_back(node->inputs[0]);
            for (TFEdge* output : node->outputs) {
                output->name = node->inputs[0]->name;
            }
        }
        if (node->op == "Exit") {
            loops_[name_prefix].exit.push_back(node->inputs[0]);
        }
    }

    for (auto& loop : loops_) {
        ResolveWhileLoop(loop.second);
    }
    for (auto& cond : conditions_) {
        ResolveCondition(cond.second);
    }
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

void TFGraphResolver::ResolveWhileLoop(const WhileLoop& loop) {
    std::unique_ptr<TFGraph> cond_subgraph(new TFGraph);
    std::unique_ptr<TFGraph> body_subgraph(new TFGraph);
    cond_subgraph->name_ = loop.name + "/cond";
    body_subgraph->name_ = loop.name + "/body";
    cond_subgraph->final_nodes_.push_back(loop.cond->start);
    for (TFEdge* body : loop.body) {
        body_subgraph->final_nodes_.push_back(body->start);
    }
    std::unique_ptr<NodeDef> node_def(new NodeDef);
    *(node_def->mutable_name()) = loop.name;
    *(node_def->mutable_op()) = "CustomWhileLoop";
    (*node_def->mutable_attr())["cond_graph"].set_s(cond_subgraph->name_);
    (*node_def->mutable_attr())["body_graph"].set_s(body_subgraph->name_);
    main_graph()->AddNode(node_def.get());
    TFNode* loop_node = main_graph()->nodes_.back().get();

    std::unordered_map<std::string, int> foreign_outputs;
    for (int idx = 0; idx < loop.exit.size(); ++idx) {
        TFEdge* exit = loop.exit[idx];
        TFNode* exit_node = exit->end;
        for (TFEdge* exit_edge : exit_node->outputs) {
            exit_edge->start = loop_node;
            exit_edge->name = loop.name + ":" + flatbuffers::NumToString(idx);
            loop_node->outputs.push_back(exit_edge);
        }
        exit_node->outputs.clear();
        foreign_outputs.emplace(exit->name, idx);
    }

    using StringSet = std::unordered_set<std::string>;
    std::unordered_map<std::string, StringSet> foreign_inputs;
    for (int i = 0; i < foreign_outputs.size(); ++i) {
        *(node_def->mutable_input()->Add()) = "";
    }
    for (int idx = 0; idx < loop.enter.size(); ++idx) {
        TFEdge* enter = loop.enter[idx];
        int input_idx = -1;
        for (TFEdge* merge : enter->end->outputs) {
            std::string mapping_name = merge->name;
            if (merge->end->op == "Merge") {
                mapping_name = merge->end->name;
            }
            auto it = foreign_inputs.find(enter->name);
            if (it == foreign_inputs.end()) {
                foreign_inputs.emplace(enter->name, StringSet{mapping_name});
            } else {
                it->second.insert(mapping_name);
            }
            if (foreign_outputs.count(mapping_name)) {
                int output_idx = foreign_outputs.at(mapping_name);
                if (input_idx != -1) {
                    MNN_ASSERT(input_idx == output_idx);
                } else {
                    input_idx = output_idx;
                }
            }
        }
        // Clear enter node's input edges.
        enter->end->inputs.clear();
        enter->end = loop_node;
        loop_node->inputs.push_back(enter);
        if (input_idx != -1) {
            *(node_def->mutable_input(input_idx)) = enter->name;
        } else {
            *(node_def->mutable_input()->Add()) = enter->name;
        }
    }

    std::unordered_map<std::string, std::string> updates;
    for (TFEdge* body : loop.body) {
        TFNode* next_iteration = body->end;
        for (int i = 0; i < next_iteration->outputs.size(); ++i) {
            TFNode* merge = next_iteration->outputs[i]->end;
            updates.emplace(body->name, merge->name);
        }
    }
    // Set aliases inputs.
    auto* aliases_inputs =
        (*node_def->mutable_attr())["aliases_inputs"].mutable_func();
    for (int idx = 0; idx < loop_node->inputs.size(); ++idx) {
        const TFEdge* input = loop_node->inputs[idx];
        const auto& it = foreign_inputs.find(input->name);
        MNN_ASSERT(it != foreign_inputs.end());
        auto* mapping_inputs = aliases_inputs->mutable_attr();
        std::string string_idx = flatbuffers::NumToString(idx);

        auto* list = (*mapping_inputs)[string_idx].mutable_list();
        for (const auto& mapping_name : it->second) {
            *(list->mutable_s()->Add()) = mapping_name;
        }
    }

    // Set aliases outputs.
    auto* aliases_outputs =
        (*node_def->mutable_attr())["aliases_outputs"].mutable_func();
    for (int idx = 0; idx < loop.exit.size(); ++idx) {
        const TFEdge* output = loop.exit[idx];
        const auto& it = foreign_outputs.find(output->name);
        MNN_ASSERT(it != foreign_outputs.end());

        auto* mapping_outputs = aliases_outputs->mutable_attr();
        std::string string_idx = flatbuffers::NumToString(idx);
        auto* list = (*mapping_outputs)[string_idx].mutable_list();
        *(list->mutable_s()->Add()) = output->name;
    }

    // Set aliases updates.
    auto* aliases_updates =
        (*node_def->mutable_attr())["aliases_updates"].mutable_func();
    for (const auto& it : updates) {
        auto* mapping_updates = aliases_updates->mutable_attr();
        auto* list = (*mapping_updates)[it.first].mutable_list();
        *(list->mutable_s()->Add()) = it.second;
    }

    graphs_.push_back(std::move(cond_subgraph));
    graphs_.push_back(std::move(body_subgraph));
    // Keep allocated `node_def` and ensure that it will not be released.
    main_graph()->allocated_nodes_.push_back(std::move(node_def));
}

TFEdge* TFGraphResolver::AddIdentityNode(TFEdge* origin_edge) {
    std::unique_ptr<NodeDef> identity_def(new NodeDef);
    int output_idx = 0;
    auto splits = RSplitString(origin_edge->name, ":");
    if (splits.size() > 1) {
        output_idx = atoi(splits[1].c_str());
    }
    MNN_ASSERT(splits.size() >= 1);
    *(identity_def->mutable_name()) =
        splits[0] + "/Identity_" + flatbuffers::NumToString(output_idx);
    *(identity_def->mutable_op()) = "Identity";
    *(identity_def->mutable_input()->Add()) = origin_edge->name;

    main_graph()->AddNode(identity_def.get());
    TFNode* identity_node = main_graph()->nodes_.back().get();

    TFNode* end_node = origin_edge->end;
    origin_edge->end = identity_node;
    std::unique_ptr<TFEdge> identity_edge(new TFEdge);
    *identity_edge = TFEdge{identity_def->name(), identity_node, end_node};
    identity_node->inputs = {origin_edge};
    identity_node->outputs = {identity_edge.get()};
    
    for (int idx = 0; idx < end_node->inputs.size(); ++idx) {
        if (end_node->inputs[idx] == origin_edge) {
            end_node->inputs[idx] = identity_edge.get();
        }
    }
    // Keep identity edge.
    main_graph()->edges_.push_back(std::move(identity_edge));
    // Keep identity node def.
    main_graph()->allocated_nodes_.push_back(std::move(identity_def));
    return main_graph()->edges_.back().get();
}

void TFGraphResolver::ResolveCondition(const Condition& cond) {
    MNN_ASSERT(cond.branch_then.size() == cond.branch_else.size());
    if (!cond.branch_then.size()) {
        return;
    }
    std::unique_ptr<TFGraph> then_subgraph(new TFGraph);
    std::unique_ptr<TFGraph> else_subgraph(new TFGraph);
    then_subgraph->name_ = cond.name + "/then";
    else_subgraph->name_ = cond.name + "/else";
    for (TFEdge* branch_then : cond.branch_then) {
        then_subgraph->final_nodes_.push_back(branch_then->start);
    }
    for (TFEdge* branch_else : cond.branch_else) {
        else_subgraph->final_nodes_.push_back(branch_else->start);
    }

    std::unique_ptr<NodeDef> node_def(new NodeDef);
    *(node_def->mutable_name()) = cond.name;
    *(node_def->mutable_op()) = "CustomCondition";
    (*node_def->mutable_attr())["then_graph"].set_s(then_subgraph->name_);
    (*node_def->mutable_attr())["else_graph"].set_s(else_subgraph->name_);
    main_graph()->AddNode(node_def.get());
    TFNode* if_node = main_graph()->nodes_.back().get();
    // Add condition input.
    if_node->inputs.push_back(cond.cond);

    using StringSet = std::unordered_set<std::string>;
    std::unordered_map<std::string, StringSet> foreign_inputs;
    for (TFGraph* subgraph : {then_subgraph.get(), else_subgraph.get()}) {
        std::vector<TFNode*> switch_nodes;
        std::vector<TFNode*> nodes = ReverseVisit(
            subgraph->final_nodes_, [&switch_nodes](TFNode* node) {
            if (IsControlFlowNode(node)) {
                MNN_ASSERT(node->op == "Switch");
                switch_nodes.push_back(node);
                return true;
            } else {
                return false;
            }
        });
        for (int idx = 0; idx < switch_nodes.size(); ++idx) {
            TFNode* switch_node = switch_nodes.at(idx);
            TFEdge* input_edge = switch_node->inputs.at(0);
            auto it = foreign_inputs.find(input_edge->name);
            if (it == foreign_inputs.end()) {
                if_node->inputs.push_back(input_edge);
                *(node_def->mutable_input()->Add()) = input_edge->name;
                it = foreign_inputs.emplace(input_edge->name,
                                            StringSet{}).first;
            }
            input_edge->end = if_node;
            for (TFEdge* output_edge : switch_node->outputs) {
                it->second.insert(output_edge->name);
            }
        }
    }

    using StringVec = std::vector<std::string>;
    std::unordered_map<int, StringVec> foreign_outputs;
    for (int idx = 0; idx < cond.branch_then.size(); ++idx) {
        TFEdge* branch_then = cond.branch_then.at(idx);
        TFNode* merge = branch_then->end;

        // Create `Identity` node as the output of the `If` node since
        // that the final `Merge` node maybe has no outputs.
        std::unique_ptr<NodeDef> identity_def(new NodeDef);
        *(identity_def->mutable_name()) = merge->name;
        *(identity_def->mutable_op()) = "Identity";
        std::string input = if_node->name + ":" + flatbuffers::NumToString(idx);
        *(identity_def->mutable_input()->Add()) = input;
        main_graph()->AddNode(identity_def.get());
        TFNode* identity_node = main_graph()->nodes_.back().get();

        // Create edge between `If` node and `Identity` node.
        std::unique_ptr<TFEdge> identity_edge(new TFEdge);
        *identity_edge = TFEdge{input, if_node, identity_node};
        identity_node->inputs = {identity_edge.get()};

        for (TFEdge* output : merge->outputs) {
            identity_node->outputs.push_back(output);
            output->start = identity_node;
            output->name = identity_node->name;
        }
        if_node->outputs.push_back(identity_edge.get());
        // TODO(): Add identity node uniformly on original graph.
        // Only the root graph need add final nodes if merge has no outputs.
        if (!merge->outputs.size()) {
            main_graph()->final_nodes_.push_back(identity_node);
        }

        // Keep identity node def.
        main_graph()->allocated_nodes_.push_back(std::move(identity_def));
        // Keep identity edge.
        main_graph()->edges_.push_back(std::move(identity_edge));

        TFEdge* branch_else = cond.branch_else.at(idx);
        StringVec output_names{branch_then->name, branch_else->name};
        foreign_outputs.emplace(idx, output_names);
    }

    // Set aliases inputs.
    auto* aliases_inputs =
        (*node_def->mutable_attr())["aliases_inputs"].mutable_func();
    for (int idx = 0; idx < if_node->inputs.size(); ++idx) {
        const TFEdge* input = if_node->inputs.at(idx);
        const auto& it = foreign_inputs.find(input->name);

        auto* mapping_inputs = aliases_inputs->mutable_attr();
        std::string string_idx = flatbuffers::NumToString(idx);
        auto* list = (*mapping_inputs)[string_idx].mutable_list();
        if (it != foreign_inputs.end()) {
            for (const auto& mapping_name : it->second) {
                *(list->mutable_s()->Add()) = mapping_name;
            }
        }
    }

    // Set aliases outputs.
    auto* aliases_outputs =
        (*node_def->mutable_attr())["aliases_outputs"].mutable_func();
    for (int idx = 0; idx < if_node->outputs.size(); ++idx) {
        const auto& it = foreign_outputs.find(idx);
        MNN_ASSERT(it != foreign_outputs.end());

        auto* mapping_outputs = aliases_outputs->mutable_attr();
        std::string string_idx = flatbuffers::NumToString(idx);
        auto* list = (*mapping_outputs)[string_idx].mutable_list();
        for (const auto& mapping_name : it->second) {
            *(list->mutable_s()->Add()) = mapping_name;
        }
    }

    graphs_.push_back(std::move(then_subgraph));
    graphs_.push_back(std::move(else_subgraph));
    // Keep allocated `node_def` and ensure that it will not be released.
    main_graph()->allocated_nodes_.push_back(std::move(node_def));
}
