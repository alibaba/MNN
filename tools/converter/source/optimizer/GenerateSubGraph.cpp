//
//  GenerateSubGraph.cpp
//  MNNConverter
//
//  Created by MNN on b'2021/01/17'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GenerateSubGraph.hpp"
#include "PostTreatUtils.hpp"
#include <MNN/MNNDefine.h>
#include "Program.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <sstream>
namespace MNN {
using NodeVector = std::vector<std::unique_ptr<OpT>>;

struct ClusterNode {
    std::string name;
    NodeVector nodes;
    bool hasLoop = false;
    bool hasSwitch = false;
    bool hasMerge = false;
    std::vector<std::shared_ptr<ClusterNode>> children;
    ClusterNode* parent = nullptr;
};

static inline std::vector<std::string> RSplitString(const std::string& name,
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

static void _makeClusterNode(const std::string& name, std::map<std::string, std::shared_ptr<ClusterNode>>& clusters, std::vector<std::shared_ptr<ClusterNode>>& rootClusters) {
    if (clusters.find(name) != clusters.end()) {
        return;
    }
    std::shared_ptr<ClusterNode> newNode(new ClusterNode);
    newNode->name = name;
    clusters.emplace(name, newNode);
    auto parent = RSplitString(name, "/").at(0);
    if (parent == name) {
        rootClusters.emplace_back(newNode);
        return;
    }
    _makeClusterNode(parent, clusters, rootClusters);
    newNode->parent = clusters[parent].get();
    clusters[parent]->children.emplace_back(newNode);
    return;
}

static void _mergeSubGraph(std::shared_ptr<ClusterNode> node) {
    for (auto c : node->children) {
        _mergeSubGraph(c);
    }
    bool merge = false;
    auto children = std::move(node->children);
    node->children.clear();
    for (auto c : children) {
        if (c->hasLoop || c->hasMerge) {
            // Can't merge
            node->children.emplace_back(c);
            continue;
        }
        for (auto& o : c->nodes) {
            node->nodes.emplace_back(std::move(o));
        }
        node->children.insert(node->children.end(), c->children.begin(), c->children.end());
    }
}

static void _printSubGraph(std::shared_ptr<ClusterNode> node, int indent = 0) {
    for (int v=0; v<indent; ++v) {
        MNN_PRINT(" ");
    }
    MNN_PRINT("%s\n", node->name.c_str());
    for (auto c : node->children) {
        _printSubGraph(c, indent+4);
    }
}
static bool _isControlOp(const OpT* op) {
    std::set<std::string> controlOps{"Merge", "Switch", "LoopCond", "Enter", "Exit", "NextIteration"};
    return op->type == OpType_Extra && controlOps.find(op->main.AsExtra()->type) != controlOps.end();
}

std::vector<std::unique_ptr<OpT>> _makeCond(std::shared_ptr<ClusterNode> cNode, MNN::NetT* netT, const std::map<std::string, int>& originTensorIndexes) {
    std::vector<std::unique_ptr<OpT>> res;
    std::unique_ptr<OpT> condOp(new OpT);
    condOp->type = OpType_If;
    condOp->main.type = OpParameter_IfParam;
    condOp->main.value = new IfParamT;
    condOp->name = cNode->name;

    // Find cond tensor
    std::set<int> condTensorIndexes;
    for (int i=0; i<cNode->nodes.size(); ++i) {
        auto& op = cNode->nodes[i];
        if (op->type == OpType_Extra && op->main.AsExtra()->type == "Switch") {
            // Find outside condIndex
            auto originIndex = op->inputIndexes[1];
            bool find = false;
            do {
                for (auto& subop : cNode->nodes) {
                    for (auto out : subop->outputIndexes) {
                        if (out == originIndex) {
                            find = true;
                            break;
                        }
                    }
                    if (find) {
                        break;
                    }
                }
            } while (false);
            if (!find) {
                condTensorIndexes.insert(originIndex);
            }
        }
    }
    MNN_ASSERT(condTensorIndexes.size() > 0);
    int condTensorIndex = *condTensorIndexes.begin();
    // Find dependency for condTensors
    if (condTensorIndexes.size() > 1) {
        MNN_ASSERT(cNode->parent != nullptr);
        for (auto index : condTensorIndexes) {
            bool valid = true;
            for (auto& op : cNode->parent->nodes) {
                if (op->inputIndexes.size() > 1 && op->inputIndexes[1] == index) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                condTensorIndex = index;
            }
        }
        // Remove Switch For Parent Switch
        bool needCheck = true;
        std::map<int, int> replaceTensor;
        needCheck = true;
        while (needCheck) {
            needCheck = false;
            auto nodes = std::move(cNode->nodes);
            for (int i = 0; i < nodes.size(); ++i) {
                if ((!needCheck) && nodes[i]->type == OpType_Extra && nodes[i]->main.AsExtra()->type == "Switch") {
                    if (nodes[i]->inputIndexes[1] != condTensorIndex) {
                        // Once Time remove only one switch
                        for (auto output : nodes[i]->outputIndexes) {
                            replaceTensor.insert(std::make_pair(output, nodes[i]->inputIndexes[0]));
                        }
                        needCheck = true;
                        continue;
                    }
                }
                cNode->nodes.emplace_back(std::move(nodes[i]));
            }
            for (auto& op : cNode->nodes) {
                for (int i = 0; i < op->inputIndexes.size(); ++i) {
                    if (replaceTensor.find(op->inputIndexes[i]) != replaceTensor.end()) {
                        op->inputIndexes[i] = replaceTensor[op->inputIndexes[i]];
                    }
                }
            }
        }
    }

    //0: no use, 1: left, 2: right, -1: switch, -2: merge
    std::vector<int> opMask(cNode->nodes.size(), 0);
    std::vector<int> tensorMask(netT->tensorName.size(), 0);
    for (int i=0; i<cNode->nodes.size(); ++i) {
        if (opMask[i] != 0) {
            continue;
        }
        auto& op = cNode->nodes[i];
        if (op->type == OpType_Extra && op->main.AsExtra()->type == "Switch") {
            tensorMask[op->outputIndexes[0]] = 2;
            if (op->outputIndexes.size() > 1) {
                tensorMask[op->outputIndexes[1]] = 1;
            }
            opMask[i] = -1;
            continue;
        }
        if (op->type == OpType_Extra && op->main.AsExtra()->type == "Merge") {
            tensorMask[op->outputIndexes[0]] = -2;
            opMask[i] = -2;
            condOp->outputIndexes.emplace_back(op->outputIndexes[0]);
            continue;
        }
        bool valid = false;
        for (auto index : op->inputIndexes) {
            if (tensorMask[index] > 0) {
                opMask[i] = tensorMask[index];
                valid = true;
            }
        }
        for (auto index : op->outputIndexes) {
            if (tensorMask[index] > 0) {
                MNN_ASSERT(opMask[i] <= 0 || opMask[i] == tensorMask[index]);
                opMask[i] = tensorMask[index];
                valid = true;
            }
        }
        if (valid) {
            for (auto index : op->inputIndexes) {
                tensorMask[index] = opMask[i];
            }
            for (auto index : op->outputIndexes) {
                tensorMask[index] = opMask[i];
            }
        }
    }
    // Remove Switch
    bool needCheck = true;
    std::map<int, int> replaceTensor;
    while (needCheck) {
        needCheck = false;
        auto nodes = std::move(cNode->nodes);
        for (int i = 0; i < nodes.size(); ++i) {
            if (nodes[i]->type == OpType_Extra && nodes[i]->main.AsExtra()->type == "Switch" && (!needCheck)) {
                // Once Time remove only one switch
                for (auto output : nodes[i]->outputIndexes) {
                    replaceTensor.insert(std::make_pair(output, nodes[i]->inputIndexes[0]));
                }
                needCheck = true;
                continue;
            }
            cNode->nodes.emplace_back(std::move(nodes[i]));
        }
        for (auto& op : cNode->nodes) {
            for (int i = 0; i < op->inputIndexes.size(); ++i) {
                if (replaceTensor.find(op->inputIndexes[i]) != replaceTensor.end()) {
                    op->inputIndexes[i] = replaceTensor[op->inputIndexes[i]];
                }
            }
        }
    }

    std::map<int, Express::VARP> varMap;
    std::set<OpT*> invalidSet;
    std::vector<int> inputIndexes;
    std::set<int> extraInputIndexes;
    std::vector<int> leftOutputs;
    std::vector<int> rightOutputs;
    std::vector<std::string> mergeNames;
    for (auto& node : cNode->nodes) {
        if (node->type == OpType_Extra && node->main.AsExtra()->type == "Merge") {
            mergeNames.emplace_back(node->name);
            if (tensorMask[node->inputIndexes[0]] == 1) {
                leftOutputs.emplace_back(node->inputIndexes[0]);
                rightOutputs.emplace_back(node->inputIndexes[1]);
            } else {
                leftOutputs.emplace_back(node->inputIndexes[1]);
                rightOutputs.emplace_back(node->inputIndexes[0]);
            }
            continue;
        }
        Express::Program::createUnit(varMap, inputIndexes, cNode->nodes, node.get(), netT, invalidSet, extraInputIndexes);
    }
    auto makeSubGraph = [&](const std::vector<int>& index) {
        std::vector<Express::VARP> out;
        for (auto l : index) {
            auto iter = varMap.find(l);
            if (iter != varMap.end()) {
                out.emplace_back(iter->second);
            } else {
                auto tempInput = Express::_Input();
                tempInput->setName(netT->tensorName[l]);
                out.emplace_back(tempInput);
                extraInputIndexes.insert(l);
            }
        }
        std::unique_ptr<NetT> newT(new NetT);
        Express::Variable::save(out, newT.get());
        std::unique_ptr<SubGraphProtoT> subGraph(new SubGraphProtoT);
        subGraph->tensors = std::move(newT->tensorName);
        subGraph->nodes = std::move(newT->oplists);
        for (int i = 0; i < subGraph->nodes.size(); ++i) {
            if (subGraph->nodes[i]->type == OpType_Input) {
                subGraph->inputs.emplace_back(i);
            }
        }
        for (auto l : index) {
            auto& outputName = netT->tensorName[l];
            for (int i = 0; i < subGraph->tensors.size(); ++i) {
                if (subGraph->tensors[i] == outputName) {
                    subGraph->outputs.emplace_back(i);
                    break;
                }
            }
        }
        return subGraph;
    };
    {
        auto leftGraph = makeSubGraph(leftOutputs);
        leftGraph->name = cNode->name + "/then";
        condOp->main.AsIfParam()->then_graph = leftGraph->name;
        netT->subgraphs.emplace_back(std::move(leftGraph));

        auto rightGraph = makeSubGraph(rightOutputs);
        rightGraph->name = cNode->name + "/else";
        condOp->main.AsIfParam()->else_graph = rightGraph->name;
        netT->subgraphs.emplace_back(std::move(rightGraph));
    }
    condOp->inputIndexes.emplace_back(condTensorIndex);
    std::unique_ptr<StringVecT> inputT(new StringVecT);
    inputT->data.emplace_back(netT->tensorName[condTensorIndex]);
    condOp->main.AsIfParam()->aliases_inputs.emplace_back(std::move(inputT));
    extraInputIndexes.erase(condTensorIndex);
    for (auto index : extraInputIndexes) {
        condOp->inputIndexes.emplace_back(index);
        std::unique_ptr<StringVecT> inputT(new StringVecT);
        inputT->data.emplace_back(netT->tensorName[index]);
        condOp->main.AsIfParam()->aliases_inputs.emplace_back(std::move(inputT));
    }
    for (int i = 0; i < leftOutputs.size(); ++i) {
        std::unique_ptr<StringVecT> outputPari(new StringVecT);
        outputPari->data.emplace_back(netT->tensorName[leftOutputs[i]]);
        outputPari->data.emplace_back(netT->tensorName[rightOutputs[i]]);
        condOp->main.AsIfParam()->aliases_outputs.emplace_back(std::move(outputPari));
    }
    // Compability for old usage
    for (int i = 0; i < condOp->outputIndexes.size(); ++i) {
        std::ostringstream newName;
        newName << condOp->name << ":" << i;
        netT->tensorName[condOp->outputIndexes[i]] = newName.str();
    }
    res.emplace_back(std::move(condOp));
    cNode->nodes.clear();
    return res;
}


std::vector<std::unique_ptr<OpT>> _makeWhile(std::shared_ptr<ClusterNode> cNode, MNN::NetT* netT, const std::map<std::string, int>& originTensorIndexes) {
    std::vector<std::unique_ptr<OpT>> res;
    // Remove switch and find LoopCond
    int loopCond = -1;
    {
        std::map<int, int> replaceTensor;
        auto childs = std::move(cNode->nodes);
        for (auto& op : childs) {
            if (op->type == OpType_Extra && op->main.AsExtra()->type == "Switch") {
                for (auto o : op->outputIndexes) {
                    replaceTensor.insert(std::make_pair(o, op->inputIndexes[0]));
                }
                continue;
            }
            if (op->type == OpType_Extra && op->main.AsExtra()->type == "LoopCond") {
                loopCond = op->outputIndexes[0];
            }
            cNode->nodes.emplace_back(std::move(op));
        }
        for (auto& op : cNode->nodes) {
            for (int i = 0; i < op->inputIndexes.size(); ++i) {
                if (replaceTensor.find(op->inputIndexes[i]) != replaceTensor.end()) {
                    op->inputIndexes[i] = replaceTensor[op->inputIndexes[i]];
                }
            }
        }
    }
    MNN_ASSERT(loopCond != -1);

    // Generate Condition Graph
    std::map<int, Express::VARP> varMap;

    // While Op
    std::unique_ptr<SubGraphProtoT> condGraph(new SubGraphProtoT);
    condGraph->name = cNode->name + "/cond";
    std::unique_ptr<SubGraphProtoT> bodyGraph(new SubGraphProtoT);
    bodyGraph->name = cNode->name + "/body";

    std::unique_ptr<OpT> whileOpU(new OpT);
    auto whileOp = whileOpU.get();// For easy to debug
    whileOp->type = OpType_While;
    whileOp->main.type = OpParameter_WhileParam;
    whileOp->main.value = new WhileParamT;
    whileOp->name = cNode->name;
    auto whileParam = whileOp->main.AsWhileParam();
    whileParam->cond_graph = condGraph->name;
    whileParam->body_graph = bodyGraph->name;

    std::set<int> extraInputIndexes;
    // Remove Merge and find body
    std::vector<int> bodyUpdate;
    std::set<std::string> bodyOutputNames;
    {
        std::vector<std::pair<int, int>> updateIndexes;
        auto childs = std::move(cNode->nodes);
        std::map<int, int> replaceTensor;
        std::set<int> updateToTensors;
        std::set<int> inputTensors;
        int copy_idx = 0;
        char idx_buffer[128];
        for (auto& op : childs) {
            if (op->type == OpType_Extra && op->main.AsExtra()->type == "Merge") {
                continue;
            }
            for (auto idx : op->inputIndexes) {
                inputTensors.insert(idx);
            }
        }
        for (auto& op : childs) {
            if (op->type == OpType_Extra && op->main.AsExtra()->type == "Merge") {
                int updateFromIdx = op->inputIndexes[1], updateToIdx = op->inputIndexes[0];
                // if tensor_x is at outside of loop and used by two op, and these two op
                // has one update data, so need copy tensor_x to tensor_x_copy.
                if (updateToTensors.find(updateToIdx) != updateToTensors.end() || inputTensors.find(updateToIdx) != inputTensors.end()) {
                    std::unique_ptr<OpT> copyOp(new OpT);
                    copyOp->type = OpType_Concat;
                    copyOp->inputIndexes.push_back(updateToIdx);
                    sprintf(idx_buffer, "%d", copy_idx++);
                    auto opName = netT->tensorName[updateToIdx] + "_copy_" + idx_buffer;
                    updateToIdx = netT->tensorName.size();
                    copyOp->outputIndexes.push_back(updateToIdx);
                    netT->tensorName.push_back(opName);
                    netT->tensorNumber++;
                    res.emplace_back(std::move(copyOp));
                    extraInputIndexes.insert(updateToIdx);
                }
                updateToTensors.insert(updateToIdx);
                updateIndexes.emplace_back(std::make_pair(updateFromIdx, updateToIdx));
                replaceTensor.insert(std::make_pair(op->outputIndexes[0], updateToIdx));
                bodyUpdate.emplace_back(updateFromIdx);
                bodyOutputNames.insert(netT->tensorName[updateFromIdx]);
                continue;
            }
            cNode->nodes.emplace_back(std::move(op));
        }
        for (auto& op : cNode->nodes) {
            for (int i = 0; i < op->inputIndexes.size(); ++i) {
                if (replaceTensor.find(op->inputIndexes[i]) != replaceTensor.end()) {
                    op->inputIndexes[i] = replaceTensor[op->inputIndexes[i]];
                }
            }
        }
        for (auto& p : updateIndexes) {
            if (replaceTensor.find(p.first) != replaceTensor.end()) {
                p.first = replaceTensor[p.first];
            }
            if (replaceTensor.find(p.second) != replaceTensor.end()) {
                p.second = replaceTensor[p.second];
            }
        }
        for (auto& p : updateIndexes) {
            std::unique_ptr<StringVecT> updateName(new StringVecT);
            updateName->data.emplace_back(netT->tensorName[p.first]);
            updateName->data.emplace_back(netT->tensorName[p.second]);
            whileParam->aliases_updates.emplace_back(std::move(updateName));
        }
    }

    // Get output
    for (auto& op : cNode->nodes) {
        if (op->type != OpType_Extra) {
            continue;
        }
        if (op->main.AsExtra()->type == "Exit") {
            whileOp->outputIndexes.emplace_back(op->outputIndexes[0]);
            whileParam->aliases_outputs.emplace_back(netT->tensorName[op->inputIndexes[0]]);
            bodyOutputNames.insert(netT->tensorName[op->inputIndexes[0]]);
        }
    }

    // Create Loop Cond
    std::set<OpT*> invalidSet;
    std::vector<int> inputIndexes;
    for (auto& node : cNode->nodes) {
        Express::Program::createUnit(varMap, inputIndexes, cNode->nodes, node.get(), netT, invalidSet, extraInputIndexes);
    }
    for (auto index : extraInputIndexes) {
        std::unique_ptr<StringVecT> inputNames(new StringVecT);
        inputNames->data.emplace_back(netT->tensorName[index]);
        whileParam->aliases_inputs.emplace_back(std::move(inputNames));
        whileOp->inputIndexes.emplace_back(index);
    }
    {
        std::unique_ptr<NetT> condNet(new NetT);
        Express::Variable::save({varMap[loopCond]}, condNet.get());
        for (auto& op : condNet->oplists) {
            if (op->type == OpType_Extra && op->main.AsExtra()->type == "LoopCond") {
                condGraph->outputs.emplace_back(op->inputIndexes[0]);
                continue;
            }
            if (op->type == OpType_Input) {
                condGraph->inputs.emplace_back(op->outputIndexes[0]);
            }
            condGraph->nodes.emplace_back(std::move(op));
        }
        condGraph->tensors = std::move(condNet->tensorName);
        MNN_ASSERT(condGraph->outputs.size() > 0);
    }
    {
        std::unique_ptr<NetT> bodyNet(new NetT);
        std::vector<Express::VARP> bodyOutputs;
        for (auto b : bodyUpdate) {
            if (varMap.find(b) != varMap.end()) {
                bodyOutputs.emplace_back(varMap[b]);
            }
        }
        Express::Variable::save(bodyOutputs, bodyNet.get());
        for (auto& op : bodyNet->oplists) {
            if (op->type == OpType_Input) {
                bodyGraph->inputs.emplace_back(op->outputIndexes[0]);
            }
            for (auto o : op->outputIndexes) {
                if (bodyOutputNames.find(bodyNet->tensorName[o]) != bodyOutputNames.end()) {
                    bodyGraph->outputs.emplace_back(o);
                }
            }
            bodyGraph->nodes.emplace_back(std::move(op));
        }
        bodyGraph->tensors = std::move(bodyNet->tensorName);
    }
    {
        // Const op needed update turn to Input
        auto turnConst = [&](SubGraphProtoT* subGraph) {
            for (auto& s : whileParam->aliases_updates) {
                auto& second = s->data[1];
                for (int i = 0; i < subGraph->nodes.size(); ++i) {
                    auto& op = subGraph->nodes[i];
                    if (OpType_Const != op->type) {
                        continue;
                    }
                    if (subGraph->tensors[op->outputIndexes[0]] == second) {
                        // Const move outside
                        auto opPtr = op.get();
                        res.emplace_back(std::move(op));
                        subGraph->nodes[i].reset(new OpT);
                        subGraph->nodes[i]->type = OpType_Input;
                        subGraph->nodes[i]->main.type = OpParameter_Input;
                        subGraph->nodes[i]->main.value = new InputT;
                        subGraph->nodes[i]->main.AsInput()->dims = opPtr->main.AsBlob()->dims;
                        subGraph->nodes[i]->main.AsInput()->dtype = opPtr->main.AsBlob()->dataType;
                        subGraph->nodes[i]->main.AsInput()->dformat = opPtr->main.AsBlob()->dataFormat;
                        subGraph->nodes[i]->outputIndexes = opPtr->outputIndexes;
                        opPtr->outputIndexes[0] = originTensorIndexes.find(second)->second;
                        std::unique_ptr<StringVecT> newVecT(new StringVecT);
                        newVecT->data.emplace_back(second);
                        whileParam->aliases_inputs.emplace_back(std::move(newVecT));
                        whileOp->inputIndexes.emplace_back(opPtr->outputIndexes[0]);
                    }
                }
            }
        };
        turnConst(condGraph.get());
        turnConst(bodyGraph.get());
    }
    //FUNC_PRINT_ALL(whileOp->name.c_str(), s);
    netT->subgraphs.emplace_back(std::move(condGraph));
    netT->subgraphs.emplace_back(std::move(bodyGraph));
    res.emplace_back(std::move(whileOpU));
    cNode->nodes.clear();
    return res;
}

static std::vector<std::unique_ptr<OpT>> _makeSubGraph(std::shared_ptr<ClusterNode> cNode, MNN::NetT* netT, const std::map<std::string, int>& t) {
    // Make Subgraph In order, first make children, second make parent
    for (auto c : cNode->children) {
        auto opList = std::move(_makeSubGraph(c, netT, t));
        for (auto&& op : opList) {
            cNode->nodes.emplace_back(std::move(op));
        }
    }
    if (cNode->hasLoop) {
        return _makeWhile(cNode, netT, t);
    }
    if (cNode->hasMerge) {
        return _makeCond(cNode, netT, t);
    }
    return {};
}

int GenerateSubGraph(std::unique_ptr<MNN::NetT>& netT) {
    // Remove unuseful op before cluster
    std::vector<std::string> passes = {
        "RemoveUnusefulOp",
    };
    for (auto pass : passes) {
        auto convert = PostConverter::get(pass);
        if (nullptr == convert) {
            continue;
        }
        convert->onExecute(netT);
    }
    bool hasControlFlow = false;
    for (auto& op : netT->oplists) {
        if (_isControlOp(op.get())) {
            hasControlFlow = true;
            break;
        }
    }
    if (!hasControlFlow) {
        return 0;
    }
    // We broadly divided all nodes into clusters by the prefix of the node
    // name, and each cluster belongs to one of the tree categories,
    // Normal, Condition or WhileLoop.
    // The nodes which have the same name prefix maybe belong to the same
    // cluster. The nodes that type is `Condition` maybe belong to a condition
    // subgraph. The nodes that type is `WhileLoop` maybe belong to a while loop
    // subgraph.
    std::map<std::string, std::shared_ptr<ClusterNode>> clusters;
    std::vector<std::shared_ptr<ClusterNode>> rootClusters;
    bool hasControlflow = false;
    for (auto& node : netT->oplists) {
        std::string name = RSplitString(node->name, "/").at(0);
        _makeClusterNode(name, clusters, rootClusters);
        auto it = clusters.find(name);
        if (node->type == OpType_Extra) {
            auto type = node->main.AsExtra()->type;
            if (type == "LoopCond") {
                hasControlflow = true;
                it->second->hasLoop = true;
            }
            else if (type == "Switch") {
                hasControlflow = true;
                it->second->hasSwitch = true;
            }
            else if (type == "Merge") {
                hasControlflow = true;
                it->second->hasMerge = true;
            }
        }
        it->second->nodes.emplace_back(std::move(node));
    }
    netT->oplists.clear();
    std::map<std::string, int> tensorNameMap;
    for (int i=0; i<netT->tensorName.size(); ++i) {
        tensorNameMap[netT->tensorName[i]] = i;
    }
    for (auto n : rootClusters) {
        _mergeSubGraph(n);
    }
#ifdef MNN_PRINT_SUBGRAPH
    for (auto n : rootClusters) {
        _printSubGraph(n);
    }
#endif
    for (auto n : rootClusters) {
        auto controlOp = _makeSubGraph(n, netT.get(), tensorNameMap);
        for (auto& c : n->nodes) {
            netT->oplists.emplace_back(std::move(c));
        }
        for (auto& op : controlOp) {
            netT->oplists.emplace_back(std::move(op));
        }
    }
    return 0;
}
}
