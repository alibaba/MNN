//
//  TfUtils.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TFUTILS_HPP
#define TFUTILS_HPP

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <functional>
#include "TmpGraph.hpp"
#include "graph.pb.h"

// import tensorflow GraphDef from file
bool tf_read_proto_from_binary(const char* filepath, google::protobuf::Message* message);

// get node's attribute according to the key
bool find_attr_value(const tensorflow::NodeDef* node, const char* key, tensorflow::AttrValue& value);

// Convert weight format from [KH,KW,CI,CO] to [CO,CI,KH,KW]
bool convertDataFormat(const float* src, float* dst, int planeNumber, int CI, int CO);

namespace TFModelOptimizer {
// namespace TFModelOptimizer comes from tensorflow transform graph tools
using namespace tensorflow;

inline bool IsMerge(const NodeDef& node_def) {
    return node_def.op() == "Merge" || node_def.op() == "RefMerge";
}

struct OpTypePattern {
    std::string op;
    std::vector<OpTypePattern> inputs;
    std::string DebugString() const;
};

struct NodeMatch {
    NodeMatch() : node() {
    }
    NodeDef node;
    std::vector<NodeMatch> inputs;
    std::string DebugString() const;
};

void NodeNamePartsFromInput(const std::string& input_name, std::string* prefix, std::string* node_name,
                            std::string* suffix);
std::string NodeNameFromInput(const std::string& input_name);
std::string CanonicalInputName(const std::string& input_name);
void MapNamesToNodes(const GraphDef& graph_def, std::map<std::string, const NodeDef*>* result);
void MapNodesToOutputs(const GraphDef& graph_def, std::map<std::string, std::vector<const NodeDef*>>* result);
int SortByExecutionOrder(const GraphDef& input_graph_def, GraphDef* output_graph_def);

typedef std::function<bool(const NodeDef& node, const OpTypePattern& pattern, const NodeMatch* match)>
    match_constraint_fun;

class GraphMatcher {
public:
    GraphMatcher(const GraphDef& graph_def, match_constraint_fun func);
    int GetOpTypeMatches(const OpTypePattern& pattern, std::vector<NodeMatch>* matches);
    void SetMatchConstraintFunction(match_constraint_fun func);
    void SetMatchedNodes(std::set<std::string>& matched_nodes);

private:
    bool DoesOpTypeMatch(const NodeDef& node, const OpTypePattern& pattern,
                         const std::set<std::string>& previously_matched_nodes, NodeMatch* match);

    GraphDef graph_def_;
    std::map<std::string, const NodeDef*> node_map_;
    // std::map<std::string, std::vector<const NodeDef*>> outputs_map_;
    match_constraint_fun match_constraint_;
    std::set<std::string> matched_nodes_;
};

int ReplaceMatchingOpTypes(
    const GraphDef& input_graph_def, const OpTypePattern& pattern,
    const std::function<int(const NodeMatch&, const std::set<std::string>&, const std::set<std::string>&,
                            std::vector<NodeDef>*)>& node_generator,
    GraphDef* output_graph_def);

int RenameNodeInputs(const GraphDef& input_graph_def, const std::map<std::string, std::string>& inputs_to_rename,
                     const std::unordered_set<std::string>& nodes_to_ignore, GraphDef* output_graph_def);

// Collect the sub-graph nodes(include output_nodes, but not input_nodes), from the output_nodes to input_nodes
void CollectSubGraphNodes(const std::vector<std::string>& input_nodes, const std::vector<std::string>& output_nodes,
                          std::map<std::string, const NodeDef*>& node_map, std::set<std::string>& sub_graph_nodes);

// set attr value function
void SetAttrValue(tensorflow::DataType type, AttrValue* out);
void SetAttrValue(bool value, AttrValue* out);
void SetAttrValue(const std::string& value, AttrValue* out);
void SetAttrValue(float value, AttrValue* out);

void AddNodeInput(const std::string& input_name, NodeDef* node);
// Copies an attribute from one NodeDef to another.
void CopyNodeAttr(const NodeDef& source, const std::string& source_key, const std::string& dest_key, NodeDef* dest);

// Inserts a value into a NodeDef's map of attributes.
// This is a bit different than AddNodeAttr in node_def_util.h because it
// overwrites any existing attributes with the same key.
template <class T>
inline void SetNodeAttr(const std::string& key, const T& value, NodeDef* node) {
    AttrValue attr_value;
    SetAttrValue(value, &attr_value);
    auto* attr_map   = node->mutable_attr();
    (*attr_map)[key] = attr_value;
}

void CopyOriginalMatch(const NodeMatch& match, std::vector<NodeDef>* new_nodes);

// Holds information that's needed for transform functions.
typedef std::map<std::string, std::vector<std::string>> TransformFuncParameters;
struct TransformFuncContext {
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    TransformFuncParameters params;

    // Returns how many occurrences of the given parameter are present.
    int CountParameters(const std::string& name) const;

    // Gets a single instance of a parameter, using a default if it's not present.
    void GetOneStringParameter(const std::string& name, const std::string& default_value, std::string* result) const;

    // Gets a single occurrence of a parameter as a 32-bit integer, falling back
    // to a default if it isn't present and returning an error if it isn't
    // convertible to a number.
    void GetOneInt32Parameter(const std::string& name, int32_t default_value, int32_t* result) const;

    // Gets a single occurrence of a parameter as a 64-bit integer, falling back
    // to a default if it isn't present and returning an error if it isn't
    // convertible to a number.
    void GetOneInt64Parameter(const std::string& name, int64_t default_value, int64_t* result) const;

    // Gets a single occurrence of a parameter as a floating point number, falling
    // back to a default if it isn't present and returning an error if it isn't
    // convertible to a number.
    void GetOneFloatParameter(const std::string& name, float default_value, float* result) const;

    // Gets a single occurrence of a parameter as a boolean, falling back to a
    // default if it isn't present and returning an error if it's not one of
    // "true", "1", "false", or "0".
    void GetOneBoolParameter(const std::string& name, bool default_value, bool* result) const;
};

typedef std::function<int(const GraphDef&, const TransformFuncContext& context, GraphDef*)> TransformFunc;

typedef std::map<std::string, TransformFunc> TransformRegistry;
TransformRegistry* GetTransformRegistry();
class TransformRegistrar {
public:
    TransformRegistrar(const std::string& name, TransformFunc transform_func) {
        auto transform_registry     = GetTransformRegistry();
        (*transform_registry)[name] = transform_func;
    }
};

#define REGISTER_GRAPH_TRANSFORM(name, func) REGISTER_GRAPH_TRANSFORM_UNIQ_HELPER(__COUNTER__, name, func)
#define REGISTER_GRAPH_TRANSFORM_UNIQ_HELPER(ctr, name, func) REGISTER_GRAPH_TRANSFORM_UNIQ(ctr, name, func)
#define REGISTER_GRAPH_TRANSFORM_UNIQ(ctr, name, func) \
    static TransformRegistrar registrar__body__##ctr##__object(name, func);

// tensorflow mode optimizerss
int FoldBatchNormsAlgebraic(const GraphDef& input_graph_def, const TransformFuncContext& context,
                            GraphDef* output_graph_def);
int FoldMoments(const GraphDef& input_graph_def, const TransformFuncContext& context, GraphDef* output_graph_def);
int RemoveNodes(const GraphDef& input_graph_def, const TransformFuncContext& context, GraphDef* output_graph_def);
int ResolveRNNGRUCell(const GraphDef& input_graph_def, const TransformFuncContext& context, GraphDef* output_graph_def);
int FuseConvPad(const tensorflow::GraphDef& input_graph_def, const TransformFuncContext& context,
                tensorflow::GraphDef* output_graph_def);
int FuseRelu6(const tensorflow::GraphDef& input_graph_def, const TransformFuncContext& context,
              tensorflow::GraphDef* output_graph_def);

} // namespace TFModelOptimizer

#endif // TFUTILS_HPP
