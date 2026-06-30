//
//  TfModelOptimizer.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <fstream>
#include "TfUtils.hpp"
#include "logkit.h"
#include <flatbuffers/util.h>

namespace TFModelOptimizer {
int FoldMoments(const tensorflow::GraphDef& input_graph_def, const TransformFuncContext& context,
                tensorflow::GraphDef* output_graph_def) {
    std::map<std::string, std::string> inputs_to_rename;
    GraphDef replaced_graph_def;
    ReplaceMatchingOpTypes(
        input_graph_def, // clang-format off
        {"Mean",
          {
            {"Mul",
              {
                {"Sub",
                  {
                    {"*"},
                    {"Mean",
                      {
                        {"*"},
                        {"Const"}
                      }
                    }
                  }
                },
                {"*"}
              }
            },
            {"Const"}
          }
        }, // clang-format on
        [&inputs_to_rename](const NodeMatch& match, const std::set<std::string>& input_nodes,
                            const std::set<std::string>& output_nodes, std::vector<NodeDef>* new_nodes) {
            // Find all the nodes we expect in the subgraph.
            const NodeDef& variance_mean_node          = match.node;
            const NodeDef& mul_node                    = match.inputs[0].node;
            const NodeDef& sub_node                    = match.inputs[0].inputs[0].node;
            const NodeDef& mean_node                   = match.inputs[0].inputs[0].inputs[1].node;
            const NodeDef& mean_node_input_node        = match.inputs[0].inputs[0].inputs[1].inputs[0].node;
            const NodeDef& mean_reduction_indices_node = match.inputs[0].inputs[0].inputs[1].inputs[1].node;
            CHECK_EQ(sub_node.input(0), mean_node.input(0)) << "sub and mean should have the same input!";

            NodeDef moments_node;
            moments_node.set_op("Moments");
            moments_node.set_name(mean_node.name() + "__moments");
            SetNodeAttr<DataType>("T", DT_FLOAT, &moments_node);
            CopyNodeAttr(mean_node, "keep_dims", "keep_dims", &moments_node);
            CopyNodeAttr(mean_node, "Tidx", "Tidx", &moments_node);

            NodeDef moments_axes_node;
            moments_axes_node.set_op("Const");
            moments_axes_node.set_name(mean_node.name() + "_axes");
            CopyNodeAttr(mean_reduction_indices_node, "dtype", "dtype", &moments_axes_node);
            CopyNodeAttr(mean_reduction_indices_node, "value", "value", &moments_axes_node);

            AddNodeInput(mean_node.input(0), &moments_node);
            AddNodeInput(moments_axes_node.name(), &moments_node);

            inputs_to_rename[mean_node.name()]          = moments_node.name() + ":0";
            inputs_to_rename[variance_mean_node.name()] = moments_node.name() + ":1";

            new_nodes->push_back(moments_node);
            new_nodes->push_back(moments_axes_node);
            new_nodes->push_back(mean_node_input_node);
            return 0;
        },
        &replaced_graph_def);

    // Change the input_name of the nodes that use mean and variance.
    RenameNodeInputs(replaced_graph_def, inputs_to_rename, std::unordered_set<std::string>(), output_graph_def);

    return 0;
}

REGISTER_GRAPH_TRANSFORM("fold_moments", FoldMoments);

// Fold batchnorm which is unfolded into series of algebraic expressions
// For example:(x - mean) * rsqrt(variance + epsilon) * gamma + beta
// gamma and beta are Const nodes, mean and variance may be Const nodes, or come
// from the outputs of nn.moments()
int FoldBatchNormsAlgebraic(const GraphDef& input_graph_def, const TransformFuncContext& context,
                            GraphDef* output_graph_def) {
    std::map<std::string, std::string> inputs_to_rename;
    GraphDef replaced_graph_def;
    ReplaceMatchingOpTypes(
        input_graph_def, // clang-format off
        {"Add",
          {
            {"Mul",                   // mul_1-->x * (rsqrt(variance + epsilon) * gamma)
              {
                {"*"},
                {"Mul",               // mul-->rsqrt(variance + epsilon) * gamma
                  {
                    {"Rsqrt",
                      {
                        {"Add",       // add-->variance + epsilon
                          {
                            {"*"},    // variance node
                            {"Const"} // epsilon
                          }
                        }
                      }
                    },
                    {"Const"}         // gamma const value
                  }
                }
              }
            },
            {"Sub",                   // sub-->beta - (rsqrt(variance + epsilon) * gamma) * mean
              {
                {"Const"},            // beta const value
                {"Mul",               // mul_2-->(rsqrt(variance + epsilon) * gamma) * mean
                  {
                    {"*"},            // mean node
                    {"Mul"}           // mul
                  }
                }
              }
            }
          }
        }, // clang-format on
        [&inputs_to_rename](const NodeMatch& match, const std::set<std::string>& input_nodes,
                            const std::set<std::string>& output_nodes, std::vector<NodeDef>* new_nodes) {
            // Find all the nodes we expect in the subgraph.
            const NodeDef& add_node         = match.node;
            const NodeDef& mul1_node        = match.inputs[0].node;
            const NodeDef& sub_node         = match.inputs[1].node;
            const NodeDef& mul1_input0_node = match.inputs[0].inputs[0].node;
            const NodeDef& mul_node         = match.inputs[0].inputs[1].node;
            const NodeDef& add_epsilon_node = match.inputs[0].inputs[1].inputs[0].inputs[0].node;
            const NodeDef& epsilon_node     = match.inputs[0].inputs[1].inputs[0].inputs[0].inputs[1].node;
            const NodeDef& gamma_node       = match.inputs[0].inputs[1].inputs[1].node;
            const NodeDef& beta_node        = match.inputs[1].inputs[0].node;
            const NodeDef& mul2_node        = match.inputs[1].inputs[1].node;
            const NodeDef& mul_node_alias   = match.inputs[1].inputs[1].inputs[1].node;

            const NodeDef& mean_node     = match.inputs[1].inputs[1].inputs[0].node;
            const NodeDef& variance_node = match.inputs[0].inputs[1].inputs[0].inputs[0].inputs[0].node;
            CHECK_EQ(mul_node.name(), mul_node_alias.name()) << "Sub graph not matched!";

            CHECK_EQ("Const", epsilon_node.op()) << "Sub graph not matched!";
            CHECK_EQ("Const", gamma_node.op()) << "You Should Apply remove_nodes(op=Identity) first!";
            CHECK_EQ("Const", beta_node.op()) << "You Should Apply remove_nodes(op=Identity) first!";

            NodeDef instance_norms_node;
            if (mean_node.op() == "Const" && variance_node.op() == "Const") {
                instance_norms_node.set_op("FusedBatchNorm");
                instance_norms_node.set_name(add_node.name() + "__FusedBatchNorm");
            } else {
                instance_norms_node.set_op("InstanceNorm");
                instance_norms_node.set_name(add_node.name() + "__InstanceNorm");
            }
            SetNodeAttr<DataType>("T", DT_FLOAT, &instance_norms_node);
            // CopyNodeAttr(epsilon_node, "value", "epsilon", &instance_norms_node);
            float epsilon = 0.001;
            tensorflow::AttrValue value;
            if (find_attr_value(&epsilon_node, "value", value)) {
                epsilon = value.tensor().float_val(0);
            }
            SetNodeAttr<float>("epsilon", epsilon, &instance_norms_node);
            AddNodeInput(mul1_input0_node.name(), &instance_norms_node);
            AddNodeInput(gamma_node.name(), &instance_norms_node);
            AddNodeInput(beta_node.name(), &instance_norms_node);
            AddNodeInput(mul2_node.input(0), &instance_norms_node);
            AddNodeInput(add_epsilon_node.input(0), &instance_norms_node);

            new_nodes->push_back(instance_norms_node);
            new_nodes->push_back(gamma_node);
            new_nodes->push_back(beta_node);
            new_nodes->push_back(mean_node);
            new_nodes->push_back(variance_node);
            new_nodes->push_back(mul1_input0_node);

            inputs_to_rename[add_node.name()] = instance_norms_node.name();
            return 0;
        },
        &replaced_graph_def);

    // Chang the input_name which use nodes in this sub graph
    RenameNodeInputs(replaced_graph_def, inputs_to_rename, std::unordered_set<std::string>(), output_graph_def);
    return 0;
}

REGISTER_GRAPH_TRANSFORM("fold_batch_norms_algebraic", FoldBatchNormsAlgebraic);

// Deletes any specified types of nodes, unless they're necessary for the
// graph's inputs or outputs.
int RemoveNodes(const GraphDef& input_graph_def, const TransformFuncContext& context, GraphDef* output_graph_def) {
    if (!context.params.count("op")) {
        LOG(FATAL) << "remove_nodes expects at least one 'op' argument, e.g. remove_nodes(op=Identity)";
    }
    int32_t max_inputs = 1;

    // Make sure we don't get rid of any nodes used as graph inputs or outputs.
    std::set<std::string> required_nodes;
    for (const std::string& input : context.input_names) {
        required_nodes.insert(NodeNameFromInput(input));
    }
    for (const std::string& output : context.output_names) {
        required_nodes.insert(NodeNameFromInput(output));
    }

    std::vector<std::string> ops_to_remove = context.params.at("op");
    GraphDef current_graph_def             = input_graph_def;
    for (const std::string& op : ops_to_remove) {
        for (int num_inputs = 1; num_inputs <= max_inputs; ++num_inputs) {
            // Look for a variable number of inputs.
            OpTypePattern pattern = {op};
            pattern.inputs.resize(num_inputs);
            for (int i = 0; i < num_inputs; ++i) {
                pattern.inputs[i] = {"*"};
            }
            // Keep looking for nodes to remove until there are no more changes.
            bool any_nodes_removed;
            do {
                any_nodes_removed = false;
                std::map<std::string, std::string> inputs_to_rename;
                GraphDef replaced_graph_def;
                ReplaceMatchingOpTypes(
                    current_graph_def, pattern,
                    [&inputs_to_rename, &required_nodes, &any_nodes_removed](
                        const NodeMatch& match, const std::set<std::string>& input_nodes,
                        const std::set<std::string>& output_nodes, std::vector<NodeDef>* new_nodes) {
                        const NodeDef& replace_node = match.node;
                        // If this node is needed in the inputs or outputs don't replace
                        // it.
                        if (required_nodes.count(replace_node.name())) {
                            LOG(INFO) << "Skipping replacement for " << replace_node.name();
                            CopyOriginalMatch(match, new_nodes);
                            return 0;
                        }
                        const NodeDef& input_node = match.inputs[0].node;
                        std::string target_name   = input_node.name();
                        for (const std::string& input : replace_node.input()) {
                            if (!input.compare(0, target_name.size(), target_name)) {
                                if (input.size() == target_name.size() || input[target_name.size()] == ':') {
                                    target_name = input;
                                    break;
                                }
                            }
                        }
                        inputs_to_rename[replace_node.name()]       = target_name;
                        inputs_to_rename["^" + replace_node.name()] = "^" + input_node.name();
                        new_nodes->push_back(input_node);
                        any_nodes_removed = true;
                        return 0;
                    },
                    &replaced_graph_def);
                // Make sure all references to removed nodes now point to their inputs.
                RenameNodeInputs(replaced_graph_def, inputs_to_rename, std::unordered_set<std::string>(),
                                 &current_graph_def);
            } while (any_nodes_removed);
        }
    }
    *output_graph_def = current_graph_def;
    return 0;
}

REGISTER_GRAPH_TRANSFORM("remove_nodes", RemoveNodes);

int ResolveRNNGRUCell(const tensorflow::GraphDef& input_graph_def, const TransformFuncContext& context,
                      tensorflow::GraphDef* output_graph_def) {
    // clang-format off
    const OpTypePattern gru_cell_pattern =
      {"Add",                                       // Cell State at time (t)
        {
          {"Mul",
              {
                {"Split",                           // the same node as below the Split node, ref: r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
                    {
                      {"Const"},
                      {"Sigmoid",
                        {
                          {"BiasAdd",
                            {
                              {"MatMul",
                                {
                                  {"ConcatV2|Concat",
                                    {
                                      {"*"},         // inputs at time (t)
                                      {"*"},         // state
                                      {"Const"}      // axis
                                    }
                                  },
                                  {"Const"}
                                }
                              },
                              {"Const"}
                            }
                          }
                        }
                      }
                    }
                },
                {"Add"}                              // Cell State at time (t - 1)
              }
          },
          {"Mul",
            {
              {"Sub",
                {
                  {"Const"},
                  {"Split"}                         
                }
              },
              {"Tanh",
                {
                  {"BiasAdd",
                    {
                      {"MatMul",
                        {
                          {"ConcatV2|Concat",
                            {
                              {"*"},                  // inputs at time (t)
                              {"*"},                  // state
                              {"Const"}               // axis
                            }
                          },
                          {"Const"}
                        }
                      },
                      {"Const"}
                    }
                  }
                }
              }
            }
          }
        }
      };
    // clang-format on

    std::map<std::string, std::vector<const NodeDef*>> outputs_map;
    MapNodesToOutputs(input_graph_def, &outputs_map);
    // gru match constraint function
    std::set<std::string> rnn_outputs; // this is rnn mid outputs
    match_constraint_fun gru_match_constraint = [&outputs_map, &rnn_outputs](const NodeDef& node,
                                                                             const OpTypePattern& pattern,
                                                                             const NodeMatch* match) {
        if (node.op() == "Add") {
            const auto& add_output_nodes    = outputs_map[node.name()];
            const int add_output_nodes_size = add_output_nodes.size();
            if (add_output_nodes_size >= 3 && pattern.inputs.size() == 2) {
                // when Add output 3 outputs(Mul,Mul,Concat|ConcatV2,[output]), set pattern_matched to be false
                std::map<std::string, int> op_types;
                for (const auto output : add_output_nodes) {
                    if (op_types.count(output->op())) {
                        op_types[output->op()] += 1;
                    } else {
                        op_types[output->op()] = 1;
                    }
                }
                bool rnn_mid_state_output = op_types.size() >= 2 && op_types["Mul"] == 2 && op_types["ConcatV2"] == 1;
                if (rnn_mid_state_output && add_output_nodes_size > 3) {
                    rnn_outputs.insert(add_output_nodes.back()->name());
                }
                if (rnn_mid_state_output) {
                    return false;
                }
            }
        }
        return true;
    };

    // search this pattern in the tensorflow Graph
    GraphMatcher matcher(input_graph_def, gru_match_constraint);
    std::vector<NodeMatch> matches;
    matcher.GetOpTypeMatches(gru_cell_pattern, &matches);

    const int matchedSize      = matches.size();
    bool keep_all_outputs      = false;
    const int rnn_outputs_size = rnn_outputs.size();
    DCHECK(rnn_outputs_size <= matchedSize) << "RNN GRU Cell Output ERROR!";
    if (rnn_outputs_size >= 1) {
        keep_all_outputs = true;
    }

    if (matchedSize >= 1) {
        // get one node from matched node to mark the matched-pattern, incase search thoes nodes again
        std::set<std::string> matched_nodes{};
        for (int i = 0; i < matchedSize; ++i) {
            matched_nodes.insert(matches[i].node.name());
        }

        std::vector<NodeDef> rnn_nodes;
        // from the very last node(Add), collect all the nodes in the GRU cell sub-graph,
        // and delete the nodes but keep the gate and candidate kernel(weight)
        std::set<std::string> ready_to_detele;
        // replace the input node name
        std::map<std::string, std::string> inputs_to_rename;

        for (int i = 0; i < matchedSize; ++i) {
            // this model has gru cell
            // replace the GRU cell with RNNSequenceGRU
            // DCHECK(matches.size() == 1) << "Now only recognise the static_rnn()";
            // [TODO] check the matches according to the same node in the matches(Split, inputs)

            const auto& the_very_last_node = matches[i].node; // this is the GRU Cell output node
            const auto& gru_input_node =
                matches[i].inputs[0].inputs[0].inputs[1].inputs[0].inputs[0].inputs[0].inputs[0].node;
            const auto& gate_kernel_node = matches[i].inputs[0].inputs[0].inputs[1].inputs[0].inputs[0].inputs[1].node;
            const auto& gate_bias_node   = matches[i].inputs[0].inputs[0].inputs[1].inputs[0].inputs[1].node;
            const auto& candidate_kernel_node = matches[i].inputs[1].inputs[1].inputs[0].inputs[0].inputs[1].node;
            const auto& candidate_bias_node   = matches[i].inputs[1].inputs[1].inputs[0].inputs[1].node;
            const auto& gate_concat_node = matches[i].inputs[0].inputs[0].inputs[1].inputs[0].inputs[0].inputs[0].node;

            const auto& rnn_next_nodes = outputs_map[the_very_last_node.name()];
            // if keep all outputs(rnn mid output), the rnn next op is Pack[tf.stack], then delete this op
            if (keep_all_outputs) {
                DCHECK(rnn_next_nodes.size() == 1 && rnn_next_nodes[0]->op() == "Pack");
                auto pack_node = rnn_next_nodes[0];
                ready_to_detele.insert(pack_node->name());
            }

            DCHECK(gru_input_node.op() == "Unpack")
                << "Now only support for getting input from one node, like form [tf.unstack]";
            DCHECK(gate_kernel_node.op() == "Const") << "Get RNN weight error";
            DCHECK(gate_bias_node.op() == "Const") << "Get RNN weight error";
            DCHECK(candidate_kernel_node.op() == "Const") << "Get RNN weight error";
            DCHECK(candidate_bias_node.op() == "Const") << "Get RNN weight error";
            std::map<std::string, const NodeDef*> node_map;
            MapNamesToNodes(input_graph_def, &node_map);
            const std::vector<std::string> input_nodes  = {gru_input_node.name()};
            const std::vector<std::string> output_nodes = {the_very_last_node.name()};
            CollectSubGraphNodes(input_nodes, output_nodes, node_map, ready_to_detele);
            // keep kernel node
            ready_to_detele.erase(gate_kernel_node.name());
            ready_to_detele.erase(gate_bias_node.name());
            ready_to_detele.erase(candidate_kernel_node.name());
            ready_to_detele.erase(candidate_bias_node.name());

            // construct rnn gru node
            NodeDef rnn_sequence_gru_node;
            rnn_sequence_gru_node.set_op("RNNSequenceGRU");
            rnn_sequence_gru_node.set_name(the_very_last_node.name() + "__RNNSequenceGRU_" + flatbuffers::NumToString(i));
            SetNodeAttr<DataType>("T", DT_FLOAT, &rnn_sequence_gru_node);
            if (keep_all_outputs) {
                SetNodeAttr<bool>("keep_all_outputs", true, &rnn_sequence_gru_node);
            } else {
                SetNodeAttr<bool>("keep_all_outputs", false, &rnn_sequence_gru_node);
            }
            // AddNodeInput(gru_input_node.name(), &rnn_sequence_gru_node);

            if (keep_all_outputs) {
                inputs_to_rename[rnn_next_nodes[0]->name()] = rnn_sequence_gru_node.name();
            }

            bool is_bidirectional_rnn = false;
            // check whether this RNN-GRU is bidirectional_rnn
            {
                // only one gru cell matched, then go on searching to check bidirectional_rnn
                matcher.SetMatchedNodes(matched_nodes);
                std::vector<NodeMatch> bid_matches;
                matcher.GetOpTypeMatches(gru_cell_pattern, &bid_matches);
                if (bid_matches.size() == 1) {
                    // this is bidirectional_rnn
                    is_bidirectional_rnn               = true;
                    const auto& the_very_last_node_bid = bid_matches[0].node; // this is the GRU Cell output node
                    const auto& gru_input_node_bid =
                        bid_matches[0].inputs[0].inputs[0].inputs[1].inputs[0].inputs[0].inputs[0].inputs[0].node;
                    DCHECK(gru_input_node.name() == gru_input_node_bid.name())
                        << "bidirectional_rnn fw and bw should share one input!";
                    const auto& gate_kernel_node_bid =
                        bid_matches[0].inputs[0].inputs[0].inputs[1].inputs[0].inputs[0].inputs[1].node;
                    const auto& gate_bias_node_bid =
                        bid_matches[0].inputs[0].inputs[0].inputs[1].inputs[0].inputs[1].node;
                    const auto& candidate_kernel_node_bid =
                        bid_matches[0].inputs[1].inputs[1].inputs[0].inputs[0].inputs[1].node;
                    const auto& candidate_bias_node_bid = bid_matches[0].inputs[1].inputs[1].inputs[0].inputs[1].node;

                    const auto& gate_concat_node_bid =
                        bid_matches[0].inputs[0].inputs[0].inputs[1].inputs[0].inputs[0].inputs[0].node;

                    const std::vector<std::string> input_nodes  = {gru_input_node_bid.name()};
                    const std::vector<std::string> output_nodes = {the_very_last_node_bid.name()};
                    CollectSubGraphNodes(input_nodes, output_nodes, node_map, ready_to_detele);

                    // keep kernel node
                    ready_to_detele.erase(gate_kernel_node_bid.name());
                    ready_to_detele.erase(gate_bias_node_bid.name());
                    ready_to_detele.erase(candidate_kernel_node_bid.name());
                    ready_to_detele.erase(candidate_bias_node_bid.name());

                    // delete the rnn's input node when input_node is Unpack(tf.unstack)
                    DCHECK(gru_input_node.input_size() == 1) << "Error";
                    AddNodeInput(NodeNameFromInput(gru_input_node.input(0)), &rnn_sequence_gru_node);
                    ready_to_detele.insert(gru_input_node.name());

                    // check fw or bw?
                    std::string prefix;
                    std::string node_name;
                    std::string suffix;
                    DCHECK(gate_concat_node.input_size() == 3) << "Error!";
                    NodeNamePartsFromInput(gate_concat_node.input(0), &prefix, &node_name, &suffix);

                    std::string prefix_bid;
                    std::string node_name_bid;
                    std::string suffix_bid;
                    DCHECK(gate_concat_node_bid.input_size() == 3) << "Error!";
                    NodeNamePartsFromInput(gate_concat_node_bid.input(0), &prefix_bid, &node_name_bid, &suffix_bid);
                    if (suffix != "" && suffix_bid == "") {
                        // the second match is bw
                        // fw
                        AddNodeInput(gate_kernel_node.name(), &rnn_sequence_gru_node);
                        AddNodeInput(gate_bias_node.name(), &rnn_sequence_gru_node);
                        AddNodeInput(candidate_kernel_node.name(), &rnn_sequence_gru_node);
                        AddNodeInput(candidate_bias_node.name(), &rnn_sequence_gru_node);
                        // bw weight
                        AddNodeInput(gate_kernel_node_bid.name(), &rnn_sequence_gru_node);
                        AddNodeInput(gate_bias_node_bid.name(), &rnn_sequence_gru_node);
                        AddNodeInput(candidate_kernel_node_bid.name(), &rnn_sequence_gru_node);
                        AddNodeInput(candidate_bias_node_bid.name(), &rnn_sequence_gru_node);

                        inputs_to_rename[the_very_last_node.name()]     = rnn_sequence_gru_node.name();
                        inputs_to_rename[the_very_last_node_bid.name()] = rnn_sequence_gru_node.name() + ":1";
                    } else if (suffix == "" && suffix_bid != "") {
                        AddNodeInput(gate_kernel_node_bid.name(), &rnn_sequence_gru_node);
                        AddNodeInput(gate_bias_node_bid.name(), &rnn_sequence_gru_node);
                        AddNodeInput(candidate_kernel_node_bid.name(), &rnn_sequence_gru_node);
                        AddNodeInput(candidate_bias_node_bid.name(), &rnn_sequence_gru_node);
                        AddNodeInput(gate_kernel_node.name(), &rnn_sequence_gru_node);
                        AddNodeInput(gate_bias_node.name(), &rnn_sequence_gru_node);
                        AddNodeInput(candidate_kernel_node.name(), &rnn_sequence_gru_node);
                        AddNodeInput(candidate_bias_node.name(), &rnn_sequence_gru_node);

                        inputs_to_rename[the_very_last_node.name()]     = rnn_sequence_gru_node.name() + ":1";
                        inputs_to_rename[the_very_last_node_bid.name()] = rnn_sequence_gru_node.name();
                    } else {
                        DLOG(FATAL) << "Now only support for getting input from one node, like form [tf.unstack]";
                    }
                } else {
                    // not bidirectional_rnn
                    {
                        // delete the rnn's input node when input_node is Unpack(tf.unstack)
                        DCHECK(gru_input_node.input_size() == 1) << "Error";
                        AddNodeInput(NodeNameFromInput(gru_input_node.input(0)), &rnn_sequence_gru_node);
                        ready_to_detele.insert(gru_input_node.name());

                        AddNodeInput(gate_kernel_node.name(), &rnn_sequence_gru_node);
                        AddNodeInput(gate_bias_node.name(), &rnn_sequence_gru_node);
                        AddNodeInput(candidate_kernel_node.name(), &rnn_sequence_gru_node);
                        AddNodeInput(candidate_bias_node.name(), &rnn_sequence_gru_node);

                        inputs_to_rename[the_very_last_node.name()] = rnn_sequence_gru_node.name();
                    }
                }
            }
            if (is_bidirectional_rnn) {
                SetNodeAttr<bool>("is_bidirectional_rnn", true, &rnn_sequence_gru_node);
            } else {
                SetNodeAttr<bool>("is_bidirectional_rnn", false, &rnn_sequence_gru_node);
            }

            rnn_nodes.push_back(rnn_sequence_gru_node);
        }

        // construct new graph
        GraphDef replaced_graph_def;
        replaced_graph_def.Clear();
        for (auto& node : rnn_nodes) {
            NodeDef* gru_node = replaced_graph_def.mutable_node()->Add();
            *gru_node         = node;
        }
        for (const NodeDef& input_node : input_graph_def.node()) {
            if (ready_to_detele.count(input_node.name())) {
                continue;
            }
            NodeDef* keep_node = replaced_graph_def.mutable_node()->Add();
            *keep_node         = input_node;
        }

        RenameNodeInputs(replaced_graph_def, inputs_to_rename, std::unordered_set<std::string>(), output_graph_def);

        std::ofstream out("rnn_gru.pb");
        std::string outStr;
        output_graph_def->SerializeToString(&outStr);
        out << outStr;
        out.close();

    } else {
        // keep all the graph
        *output_graph_def = input_graph_def;
    }

    return 0;
}

REGISTER_GRAPH_TRANSFORM("ResolveRNNGRUCell", ResolveRNNGRUCell);

int FuseConvPad(const tensorflow::GraphDef& input_graph_def, const TransformFuncContext& context,
                tensorflow::GraphDef* output_graph_def) {
    GraphDef replaced_graph_def;
    ReplaceMatchingOpTypes(
        input_graph_def, // clang-format off
    {"Conv2D|DepthwiseConv2dNative",
      {
        {"Pad",
          {
            {"*"},
            {"*"}
          }
        },
        {"*"}
      }
    }, // clang-format on
        [](const NodeMatch& match, const std::set<std::string>& input_nodes, const std::set<std::string>& output_nodes,
           std::vector<NodeDef>* new_nodes) {
            const NodeDef& conv_node     = match.node;
            const NodeDef& pad_node      = match.inputs[0].node;
            const NodeDef& weight_node   = match.inputs[1].node;
            const NodeDef& input_node    = match.inputs[0].inputs[0].node;
            const NodeDef& pad_dims_node = match.inputs[0].inputs[1].node;

            new_nodes->push_back(weight_node);
            new_nodes->push_back(input_node);
            NodeDef fused_conv_pad;
            const auto& originalOpType = conv_node.op();
            fused_conv_pad.set_op(originalOpType);
            fused_conv_pad.set_name(conv_node.name());
            AddNodeInput(input_node.name(), &fused_conv_pad);
            AddNodeInput(weight_node.name(), &fused_conv_pad);
            CopyNodeAttr(conv_node, "T", "T", &fused_conv_pad);
            CopyNodeAttr(conv_node, "data_format", "data_format", &fused_conv_pad);
            CopyNodeAttr(conv_node, "strides", "strides", &fused_conv_pad);
            CopyNodeAttr(conv_node, "dilations", "dilations", &fused_conv_pad);
            SetNodeAttr<std::string>("padding", "Symmetric", &fused_conv_pad);
            new_nodes->push_back(fused_conv_pad);

            return 0;
        },
        &replaced_graph_def);
    *output_graph_def = replaced_graph_def;
    return 0;
}

REGISTER_GRAPH_TRANSFORM("FuseConvPad", FuseConvPad);

int FuseRelu6(const tensorflow::GraphDef& input_graph_def, const TransformFuncContext& context,
              tensorflow::GraphDef* output_graph_def) {
    std::map<std::string, std::string> inputs_to_rename;
    GraphDef replaced_graph_def;
    ReplaceMatchingOpTypes(
        input_graph_def, // clang-format off
    {"Minimum",
      {
        {"Relu"},
        {"Const"}
      }
    }, // clang-format on
        [&inputs_to_rename](const NodeMatch& match, const std::set<std::string>& input_nodes,
                            const std::set<std::string>& output_nodes, std::vector<NodeDef>* new_nodes) {
            const auto& minimun_node = match.node;
            const auto& relu_node    = match.inputs[0].node;
            const auto& const_node   = match.inputs[1].node;

            tensorflow::AttrValue value;
            if (find_attr_value(&const_node, "value", value)) {
                const float minimun_value = value.tensor().float_val(0);
                DCHECK(6.0f == minimun_value) << "fuse relu6 failed!";
            } else {
                DLOG(FATAL) << "fuse relu6 failed!";
            }
            NodeDef relu6;
            relu6.set_op("Relu6");
            relu6.set_name(relu_node.name());
            AddNodeInput(relu_node.input(0), &relu6);
            new_nodes->push_back(relu6);
            inputs_to_rename[minimun_node.name()] = relu6.name();
            return 0;
        },
        &replaced_graph_def);

    RenameNodeInputs(replaced_graph_def, inputs_to_rename, std::unordered_set<std::string>(), output_graph_def);

    return 0;
}

} // namespace TFModelOptimizer
