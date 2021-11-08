//
//  TfUtils.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <limits>
#include <set>

#include "TfUtils.hpp"
#include "logkit.h"

bool tf_read_proto_from_binary(const char* filepath, google::protobuf::Message* message) {
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open()) {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);
#if GOOGLE_PROTOBUF_VERSION >= 3011000
    codedstr.SetTotalBytesLimit(INT_MAX);
#else
    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX/2);
#endif
    bool success = message->ParseFromCodedStream(&codedstr);

    fs.close();

    return success;
}

bool find_attr_value(const tensorflow::NodeDef* node, const char* key, tensorflow::AttrValue& value) {
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node->attr();

    const google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.find(key);
    if (it != attr.end()) {
        value = it->second;
        return true;
    }

    return false;
}

bool convertDataFormat(const float* src, float* dst, int planeNumber, int CI, int CO) {
    // H W CI CO --> CO CI H W
    assert(planeNumber > 0);
    assert(CI > 0);
    assert(CO > 0);
    assert(src != nullptr);
    for (int coi = 0; coi < CO; coi++) {
        for (int cii = 0; cii < CI; cii++) {
            for (int i = 0; i < planeNumber; ++i) {
                dst[(coi * CI + cii) * planeNumber + i] = src[(i * CI + cii) * CO + coi];
            }
        }
    }

    return true;
}

namespace TFModelOptimizer {
static std::vector<std::string> strSplit(const std::string input_str, std::string pattern) {
    std::string::size_type pos;
    std::vector<std::string> result;
    std::string str = input_str;
    str += pattern;
    const int size = str.size();

    for (int i = 0; i < size; i++) {
        pos = str.find(pattern, i);
        if (pos < size) {
            std::string s = str.substr(i, pos - i);
            result.push_back(s);
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}

static bool strEndsWith(std::string text, std::string suffix) {
    return suffix.empty() || (text.size() >= suffix.size() &&
                              memcmp(text.data() + (text.size() - suffix.size()), suffix.data(), suffix.size()) == 0);
}

void NodeNamePartsFromInput(const std::string& input_name, std::string* prefix, std::string* node_name,
                            std::string* suffix) {
    auto input_parts = strSplit(input_name, ":");
    if (input_parts.size() < 2) {
        *suffix = "";
    } else {
        *suffix = ":" + input_parts[1];
    }
    *node_name = input_parts[0];
    if ((*node_name)[0] == '^') {
        *prefix = "^";
        node_name->erase(node_name->begin());
    } else {
        *prefix = "";
    }
}

std::string NodeNameFromInput(const std::string& input_name) {
    std::string prefix;
    std::string node_name;
    std::string suffix;
    NodeNamePartsFromInput(input_name, &prefix, &node_name, &suffix);
    return node_name;
}

std::string CanonicalInputName(const std::string& input_name) {
    std::string prefix;
    std::string node_name;
    std::string suffix;
    NodeNamePartsFromInput(input_name, &prefix, &node_name, &suffix);
    if (suffix.empty()) {
        suffix = ":0";
    }
    return prefix + node_name + suffix;
}

void MatchedNodesAsArray(const NodeMatch& match, std::vector<tensorflow::NodeDef>* result) {
    std::set<std::string> found_nodes;
    std::vector<NodeMatch> current_matches = {match};
    while (!current_matches.empty()) {
        std::vector<NodeMatch> next_matches;
        for (const NodeMatch& current_match : current_matches) {
            if (found_nodes.count(current_match.node.name())) {
                continue;
            }
            found_nodes.insert(current_match.node.name());
            result->push_back(current_match.node);
            for (const NodeMatch& input_match : current_match.inputs) {
                next_matches.push_back(input_match);
            }
        }
        current_matches = next_matches;
    }
}

void RecordMatchedNodes(const NodeMatch& match, std::set<std::string>* matchedNodes) {
    matchedNodes->insert(match.node.name());
    for (const NodeMatch& input_match : match.inputs) {
        RecordMatchedNodes(input_match, matchedNodes);
    }
}

std::string OpTypePattern::DebugString() const {
    std::string result = "{" + op + ", {";
    for (const OpTypePattern& input : inputs) {
        result += input.DebugString() + ",";
    }
    result += "}}";
    return result;
}

std::string NodeMatch::DebugString() const {
    std::string result = "{";
    result += node.DebugString();
    result += ", {";
    for (const NodeMatch& input : inputs) {
        result += input.DebugString() + ",";
    }
    result += "}}";
    return result;
}

void MapNamesToNodes(const GraphDef& graph_def, std::map<std::string, const NodeDef*>* result) {
    for (const NodeDef& node : graph_def.node()) {
        (*result)[node.name()] = &node;
    }
}

void MapNodesToOutputs(const GraphDef& graph_def, std::map<std::string, std::vector<const NodeDef*>>* result) {
    // std::map<std::string, const NodeDef*> node_map;
    // MapNamesToNodes(graph_def, &node_map);
    for (const NodeDef& node : graph_def.node()) {
        for (const std::string& input : node.input()) {
            std::string input_node_name = NodeNameFromInput(input);
            (*result)[input_node_name].push_back(&node);
        }
    }
}

int SortByExecutionOrder(const GraphDef& input_graph_def, GraphDef* output_graph_def) {
    const int num_nodes = input_graph_def.node_size();
    std::vector<int> ready;
    std::vector<int> pending_count;
    pending_count.reserve(num_nodes);
    std::vector<std::vector<int>> outputs(num_nodes);
    std::map<std::string, int> name_index;
    for (int i = 0; i < input_graph_def.node_size(); ++i) {
        const NodeDef& node(input_graph_def.node(i));
        name_index[node.name()] = i;
    }

    for (int n = 0; n < num_nodes; ++n) {
        const NodeDef& node_def(input_graph_def.node(n));
        if (IsMerge(node_def)) {
            int num_control_edges = 0;
            for (int i = 0; i < node_def.input_size(); ++i) {
                if (node_def.input(i)[0] == '^') {
                    num_control_edges++;
                }
            }
            pending_count.push_back(num_control_edges + 1);
        } else {
            pending_count.push_back(node_def.input_size());
        }
        if (node_def.input_size() == 0) {
            ready.push_back(n);
            continue;
        }
        for (int i = 0; i < node_def.input_size(); ++i) {
            const std::string input_name      = node_def.input(i);
            const std::string input_node_name = NodeNameFromInput(input_name);
            if (!name_index.count(input_node_name)) {
                LOG(FATAL) << "Node '" << node_def.name() << "': Unknown input node ==> '" << node_def.input(i) << "'";
            }
            outputs[name_index[input_node_name]].push_back(n);
        }
    }

    int processed = 0;
    output_graph_def->Clear();
    while (!ready.empty()) {
        int o = ready.back();
        ready.pop_back();
        ++processed;
        const NodeDef& node_def(input_graph_def.node(o));
        *output_graph_def->mutable_node()->Add() = node_def;

        for (size_t i = 0; i < outputs[o].size(); ++i) {
            const int output = outputs[o][i];
            pending_count[output]--;
            if (pending_count[output] == 0) {
                ready.push_back(output);
            }
        }
    }
    if (processed < num_nodes) {
        LOG(FATAL) << "IN " << (num_nodes - processed) << " NODES IN A CYCLE";
        return -1;
    }
    return 0;
}

GraphMatcher::GraphMatcher(const tensorflow::GraphDef& graph_def, match_constraint_fun func) {
    SortByExecutionOrder(graph_def, &graph_def_);
    MapNamesToNodes(graph_def_, &node_map_);
    match_constraint_ = std::move(func);
    matched_nodes_    = std::set<std::string>();
}

int GraphMatcher::GetOpTypeMatches(const OpTypePattern& pattern, std::vector<NodeMatch>* matches) {
    std::set<std::string> matched_nodes = matched_nodes_;
    for (const NodeDef& node : graph_def_.node()) {
        // Skip any nodes that are already part of a match.
        if (matched_nodes.count(node.name())) {
            continue;
        }
        NodeMatch match;
        if (DoesOpTypeMatch(node, pattern, matched_nodes, &match)) {
            RecordMatchedNodes(match, &matched_nodes);
            matches->push_back(match);
        }
    }
    return 0;
}

bool GraphMatcher::DoesOpTypeMatch(const NodeDef& node, const OpTypePattern& pattern,
                                   const std::set<std::string>& previously_matched_nodes, NodeMatch* match) {
    // LOG(INFO) << "Looking at node " << node->DebugString();
    // LOG(INFO) << "Pattern=" << pattern.DebugString();

    if (previously_matched_nodes.count(node.name())) {
        return false;
    }
    bool pattern_matched = false;
    if (pattern.op == "*") {
        pattern_matched = true;
    } else {
        auto pattern_ops = strSplit(pattern.op, "|");
        for (const auto& pattern_op : pattern_ops) {
            if (node.op() == pattern_op) {
                // default
                pattern_matched = true;
                // call match constraint function
                pattern_matched = match_constraint_(node, pattern, match);
            }
        }
    }
    if (!pattern_matched) {
        return false;
    }

    match->node = node;
    // LOG(INFO) << "Match=" << match->DebugString();
    std::vector<std::string> non_control_inputs;
    for (const std::string& input : node.input()) {
        if (!input.empty() && (input[0] != '^')) {
            non_control_inputs.push_back(input);
        }
    }

    if (pattern.inputs.empty()) {
        // the last one pattern
        return true;
    }
    if (non_control_inputs.size() != pattern.inputs.size()) {
        return false;
    }
    for (int i = 0; i < pattern.inputs.size(); ++i) {
        const auto& input_node_name = NodeNameFromInput(non_control_inputs[i]);
        const NodeDef& input_node   = *(node_map_[input_node_name]);
        const auto input_pattern    = pattern.inputs[i];
        match->inputs.push_back(NodeMatch());
        NodeMatch* input_match = &(match->inputs.back());
        if (!DoesOpTypeMatch(input_node, input_pattern, previously_matched_nodes, input_match)) {
            return false;
        }
    }
    return true;
}

void GraphMatcher::SetMatchConstraintFunction(match_constraint_fun func) {
    match_constraint_ = std::move(func);
}

void GraphMatcher::SetMatchedNodes(std::set<std::string>& matched_nodes) {
    matched_nodes_ = matched_nodes;
}

int ReplaceMatchingOpTypes(
    const GraphDef& input_graph_def, const OpTypePattern& pattern,
    const std::function<int(const NodeMatch&, const std::set<std::string>&, const std::set<std::string>&,
                            std::vector<NodeDef>*)>& node_generator,
    GraphDef* output_graph_def) {
    GraphMatcher matcher(input_graph_def, [](const NodeDef& node, const OpTypePattern& pattern,
                                             const NodeMatch* match) { return true; });
    std::vector<NodeMatch> matches;
    matcher.GetOpTypeMatches(pattern, &matches);

    std::set<std::string> matched_nodes;
    std::map<std::string, const NodeMatch*> matches_by_headName;
    for (const NodeMatch& match : matches) {
        matches_by_headName[match.node.name()] = &match;
        RecordMatchedNodes(match, &matched_nodes);
    }
    std::map<std::string, std::vector<const NodeDef*>> outputs_map;
    MapNodesToOutputs(input_graph_def, &outputs_map);
    output_graph_def->Clear();
    for (const NodeDef& input_node : input_graph_def.node()) {
        if (matches_by_headName.count(input_node.name())) {
            const NodeMatch* match = matches_by_headName[input_node.name()];
            std::vector<NodeDef> matched_nodes_array;
            MatchedNodesAsArray(*match, &matched_nodes_array);
            std::set<std::string> matched_nodes_lookup;
            for (const auto matched_node : matched_nodes_array) {
                matched_nodes_lookup.insert(matched_node.name());
            }
            std::set<std::string> input_nodes;
            std::set<std::string> output_nodes;
            for (const auto matched_node : matched_nodes_array) {
                for (const auto& input_name : matched_node.input()) {
                    if (!matched_nodes_lookup.count(input_name)) {
                        input_nodes.insert(matched_node.name());
                    }
                }

                if (outputs_map.count(matched_node.name())) {
                    for (const auto& dependent_node : outputs_map[matched_node.name()]) {
                        if (!matched_nodes_lookup.count(dependent_node->name())) {
                            output_nodes.insert(matched_node.name());
                        }
                    }
                }
            }

            std::vector<NodeDef> new_nodes;
            node_generator(*match, input_nodes, output_nodes, &new_nodes);

            std::set<std::string> new_node_names;
            for (const auto& new_node : new_nodes) {
                new_node_names.insert(new_node.name());
            }
            bool abort_replacement = false;
            if (false) {
                for (const auto& expected_output : output_nodes) {
                    if (!new_node_names.count(expected_output)) {
                        LOG(ERROR) << "Expected " << expected_output << " to be preserved.";
                        abort_replacement = true;
                    }
                }
            }
            if (abort_replacement) {
                LOG(ERROR) << "Generator function didn't preserve needed nodes.";
                std::vector<NodeDef> old_nodes;
                MatchedNodesAsArray(*match, &old_nodes);
                for (const NodeDef& old_node : old_nodes) {
                    NodeDef* added_node = output_graph_def->mutable_node()->Add();
                    *added_node         = old_node;
                }
            } else {
                for (const NodeDef& new_node : new_nodes) {
                    NodeDef* added_node = output_graph_def->mutable_node()->Add();
                    *added_node         = new_node;
                }
            }
        } else if (!matched_nodes.count(input_node.name())) {
            NodeDef* added_node = output_graph_def->mutable_node()->Add();
            *added_node         = input_node;
        } else {
        }
    }
    return 0;
}

int RenameNodeInputs(const GraphDef& input_graph_def, const std::map<std::string, std::string>& inputs_to_rename,
                     const std::unordered_set<std::string>& nodes_to_ignore, GraphDef* output_graph_def) {
    std::map<std::string, std::vector<std::pair<std::string, std::string>>> canonical_inputs_to_rename;
    for (const auto& input_to_rename : inputs_to_rename) {
        canonical_inputs_to_rename[NodeNameFromInput(input_to_rename.first)].push_back(
            {input_to_rename.first, input_to_rename.second});
    }

    output_graph_def->Clear();
    for (const NodeDef& node : input_graph_def.node()) {
        NodeDef* new_node = output_graph_def->mutable_node()->Add();
        *new_node         = node;
        new_node->mutable_input()->Clear();
        for (const std::string& input_name : node.input()) {
            std::set<std::string> already_visited;
            std::string new_input_name = input_name;
            while (canonical_inputs_to_rename.count(NodeNameFromInput(new_input_name))) {
                std::string input_node_name = NodeNameFromInput(new_input_name);
                if (already_visited.count(input_node_name)) {
                    LOG(FATAL) << "RenameNodeInputs argument contains a cycle for " << input_node_name;
                }
                already_visited.insert(input_node_name);
                if (nodes_to_ignore.count(node.name())) {
                    break;
                }
                bool any_match_found = false;
                for (const std::pair<std::string, std::string>& input_to_rename :
                     canonical_inputs_to_rename.at(input_node_name)) {
                    const std::string& source_name = input_to_rename.first;
                    const std::string& dest_name   = input_to_rename.second;
                    bool is_match;
                    std::string match_name;
                    if (strEndsWith(source_name, ":*")) {
                        is_match = true;
                        std::string prefix;
                        std::string unused_node_name;
                        std::string suffix;
                        NodeNamePartsFromInput(new_input_name, &prefix, &unused_node_name, &suffix);
                        match_name = prefix + dest_name + suffix;
                    } else {
                        is_match   = (CanonicalInputName(source_name) == CanonicalInputName(new_input_name));
                        match_name = dest_name;
                    }
                    if (is_match) {
                        new_input_name  = match_name;
                        any_match_found = true;
                    }
                }
                if (!any_match_found) {
                    break;
                }
            }
            *(new_node->mutable_input()->Add()) = new_input_name;
        }
    }
    return 0;
}

void CollectSubGraphNodes(const std::vector<std::string>& input_nodes, const std::vector<std::string>& output_nodes,
                          std::map<std::string, const NodeDef*>& node_map, std::set<std::string>& sub_graph_nodes) {
    if (input_nodes.empty() || output_nodes.empty()) {
        return;
    }

    std::vector<std::string> visit_nodes = output_nodes;
    while (!visit_nodes.empty()) {
        sub_graph_nodes.insert(visit_nodes.back());
        const NodeDef* cur_output_node = node_map[visit_nodes.back()];

        visit_nodes.pop_back();
        const int input_node_size = cur_output_node->input_size();
        if (input_node_size == 0) {
            continue;
        }

        for (int j = 0; j < input_node_size; ++j) {
            const auto& input_node_name = NodeNameFromInput(cur_output_node->input(j));
            if (std::find(input_nodes.begin(), input_nodes.end(), input_node_name) != input_nodes.end()) {
                continue;
            }
            if (std::find(visit_nodes.begin(), visit_nodes.end(), input_node_name) == visit_nodes.end() &&
                !sub_graph_nodes.count(input_node_name)) {
                visit_nodes.push_back(input_node_name);
            }
        }
    }
}

void CopyOriginalMatch(const NodeMatch& match, std::vector<NodeDef>* new_nodes) {
    std::vector<NodeDef> old_nodes;
    MatchedNodesAsArray(match, &old_nodes);
    for (const NodeDef& old_node : old_nodes) {
        new_nodes->push_back(old_node);
    }
}

void SetAttrValue(tensorflow::DataType type, AttrValue* out) {
    out->set_type(type);
}

void SetAttrValue(bool value, AttrValue* out) {
    out->set_b(value);
}

void SetAttrValue(const std::string& value, AttrValue* out) {
    out->set_s(value.c_str());
}

void SetAttrValue(float value, AttrValue* out) {
    out->set_f(value);
}

void AddNodeInput(const std::string& input_name, NodeDef* node) {
    *(node->mutable_input()->Add()) = input_name;
}
void CopyNodeAttr(const NodeDef& source, const std::string& source_key, const std::string& dest_key, NodeDef* dest) {
    CHECK_NE(0, source.attr().count(source_key)) << "No key '" << source_key << "' found in " << source.DebugString();
    (*(dest->mutable_attr()))[dest_key] = source.attr().at(source_key);
}
TransformRegistry* GetTransformRegistry() {
    static TransformRegistry transformRegistry;
    return &transformRegistry;
}

} // namespace TFModelOptimizer
