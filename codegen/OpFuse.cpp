//
//  OpFuse.cpp
//  MNN
//
//  Created by MNN on 2020/9/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpFuse.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "PluginModule.hpp"
#include <queue>
#include <unordered_map>
#include "cpu/CPUAst.hpp"
#include "jit/LLVMJit.hpp"

#if !defined(_MSC_VER)
#include <dlfcn.h>
#endif
/**
    OpFuse
 */
namespace MNN {
static void dumpOp(const Op* op) {
    if (op->name()) printf("name: %s, ", op->name()->c_str());
    printf("Type: %s,\n", MNN::EnumNameOpType(op->type()));
    if (op->type() == OpType_BinaryOp)  {
        auto binary = op->main_as_BinaryOp();
        auto type = binary->opType();
        printf("Op: %s\n", MNN::EnumNamesBinaryOpOperation()[type]);
    } else if (op->type() == OpType_UnaryOp){
        auto unary = op->main_as_UnaryOp();
        auto type = unary->opType();
        printf("Op: %s\n", MNN::EnumNamesUnaryOpOperation()[type]);
    }
}
static void dumpRegion(Tensor::InsideDescribe::Region& reg) {
    printf("\n{\nsize: [%d, %d, %d], origin: %p\n", reg.size[0], reg.size[1], reg.size[2], reg.origin);
    printf("src: { stride: [%d, %d, %d], offset: %d }\n", reg.src.stride[0],reg.src.stride[1],reg.src.stride[2],reg.src.offset);
    printf("dst: { stride: [%d, %d, %d], offset: %d }\n}\n", reg.dst.stride[0],reg.dst.stride[1],reg.dst.stride[2],reg.dst.offset);
}
static void dumpTensor(const Tensor* t) {
    printf("\t%p [", t);
    for (int d : t->shape())
        printf("%d,", d);
    printf("],\n");
    auto des = TensorUtils::getDescribe(t);
    if (des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
        printf("Regions:");
        for (auto reg : des->regions) {
            dumpRegion(reg);
        }
    }
}
static void dumpCmd(const Command* cmd) {
    printf("\n{\n");
    dumpOp(cmd->op);
    printf("output: \n");
    dumpTensor(cmd->outputs[0]);
    printf("input: \n");
    for (auto input : cmd->inputs) {
        dumpTensor(input);
    }
    printf("}\n");
}

// is legal fused type
bool isLegal(const Command* cmd) {
    auto type = cmd->op->type();
    bool elemWise = type == OpType_BinaryOp
           || type == OpType_UnaryOp
           || type == OpType_ReLU
           || type == OpType_ReLU6
           || type == OpType_Eltwise;
    if (elemWise) {
        return true;
    }
#define fuse_raster
#ifdef fuse_raster
    if (type == OpType_Raster) {
        auto outputFormat = TensorUtils::getDescribe(cmd->outputs[0])->dimensionFormat;
        bool legalFormat = outputFormat != MNN_DATA_FORMAT_NC4HW4;
        if (TensorUtils::getDescribe(cmd->inputs[0])->regions.size() > 1) return false;
        for (auto reg : TensorUtils::getDescribe(cmd->inputs[0])->regions) {
            legalFormat &= TensorUtils::getDescribe(reg.origin)->dimensionFormat == outputFormat;
        }
        return legalFormat;
    }
#endif
    return false;
}

Node* LCA(Node* x, Node* y) {
    while (x != y) {
        if (!x || !y) {
            return nullptr;
        }
        if (x->topoIndex < y->topoIndex) {
            x = x->domainatePred;
        } else {
            y = y->domainatePred;
        }
    }
    return x;
}
bool allPathLegal(Node* s, Node* t) {
    bool legal = true;
    std::queue<Node*> q;
    q.push(s);
    while (!q.empty()) {
        auto node = q.front();
        q.pop();
        legal &= isLegal(node->cmd);
        for (auto succ : node->succ) {
            if (succ != t) {
                q.push(succ);
            }
        }
    }
    return legal;
}
std::vector<Node*> fuseNode(Node* root, std::vector<Node*>& edges) {
    std::vector<Node*> fuseSet;
    std::queue<Node*> q;
    q.push(root);
    while (!q.empty()) {
        auto node = q.front();
        fuseSet.insert(fuseSet.begin(), node);
        q.pop();
        for (auto child : node->domainateSucc) {
            if (isLegal(child->cmd) && allPathLegal(child, root)) {
                q.push(child);
            } else {
                edges.push_back(child);
            }
        }
    }
    return fuseSet;
}

void codegen(CommandBuffer& cmd, std::vector<std::vector<Node*>>& fuseSets) {
    // generate Kernel
    CPUPluginModule plugin("codegen_demo");
    for (auto compSet : fuseSets) {
        // printf("set size: %lu \n", compSet.size());
        InOutTensors tensors = plugin.addFunction(compSet);
        auto inputs = tensors.first;
        auto outputs = tensors.second;
        // build Plugin Op
        Command cmdPlugin;
        {
            std::unique_ptr<OpT> pluginOp(new OpT);
            pluginOp->type = OpType_Plugin;
            pluginOp->name = "PluginWrapper";
            PluginT* plugin_param = new PluginT;
            plugin_param->type    = "PluginWrapper";
            plugin_param->attr.resize(1);
            plugin_param->attr[0].reset(new AttributeT);
            plugin_param->attr[0]->key = "kernel";
            plugin_param->attr[0]->i = plugin.getFunctionNum()-1;
            pluginOp->main.type  = OpParameter_Plugin;
            pluginOp->main.value = plugin_param;
            flatbuffers::FlatBufferBuilder builder;
            auto lastOffset = Op::Pack(builder, pluginOp.get());
            builder.Finish(lastOffset);
            cmdPlugin = GeometryComputerUtils::makeCommand(builder, inputs, outputs);
        }
        for (int i = 0; i < compSet.size(); i++) {
            auto cmd = const_cast<Command*>(compSet[i]->cmd);
            if (i == compSet.size()-1) {
                cmd->op = cmdPlugin.op;
                cmd->inputs = cmdPlugin.inputs;
                cmd->outputs = cmdPlugin.outputs;
                cmd->buffer = cmdPlugin.buffer;
            } else {
                cmd->op = nullptr;
                cmd->buffer.clear();
            }
        }
    }
    // printf("total: %d\n", idx);
    plugin.codegen();
    // printf("cmd num: %lu \n", cmd.command.size());
    for (auto iter = cmd.command.begin(); iter != cmd.command.end();) {
        if (iter->op == nullptr) {
            iter = cmd.command.erase(iter);
        } else {
            ++iter;
        }
    }
#if !defined(_MSC_VER)
    // printf("cmd num: %lu \n", cmd.command.size());
    dlopen("./libplugin_fuse.so", RTLD_NOW | RTLD_LOCAL);
#endif
}

void jit(CommandBuffer& cmd, std::vector<std::vector<Node*>>& fuseSets) {
    LLVMJIT* theJit = LLVMJIT::createLLVMJIT();
    CPUPluginModule plugin("jit_demo");
    std::string kernelStr;
    for (auto compSet : fuseSets) {
        /*
        // printf("set size: %lu \n", compSet.size());
        if (true) {
            for (auto com : compSet) {
                // json :
                // { fusedOps: [ { idx:int, srcOps: [name: string], inputs:[name:string], outputs:[name:string] } ], dynlib:string, jitObj:string, module:string }
                dumpCmd(com->cmd);
            }
        }
        */
        kernelStr += "[";
        for (auto com : compSet) {
            kernelStr += com->cmd->op->name()->str();
        }
        kernelStr += "]";
        InOutTensors tensors = plugin.addFunction(compSet);
        auto inputs = tensors.first;
        auto outputs = tensors.second;
        // build Plugin Op
        Command cmdPlugin;
        {
            std::unique_ptr<OpT> pluginOp(new OpT);
            pluginOp->type = OpType_Plugin;
            pluginOp->name = "JitPluginWrapper";
            PluginT* plugin_param = new PluginT;
            plugin_param->type    = "JitPluginWrapper";
            plugin_param->attr.resize(1);
            plugin_param->attr[0].reset(new AttributeT);
            plugin_param->attr[0]->key = "kernel";
            plugin_param->attr[0]->i = plugin.getFunctionNum() - 1;
            pluginOp->main.type  = OpParameter_Plugin;
            pluginOp->main.value = plugin_param;
            flatbuffers::FlatBufferBuilder builder;
            auto lastOffset = Op::Pack(builder, pluginOp.get());
            builder.Finish(lastOffset);
            cmdPlugin = GeometryComputerUtils::makeCommand(builder, inputs, outputs);
        }
        for (int i = 0; i < compSet.size(); i++) {
            auto cmd = const_cast<Command*>(compSet[i]->cmd);
            if (i == compSet.size()-1) {
                cmd->op = cmdPlugin.op;
                cmd->inputs = cmdPlugin.inputs;
                cmd->outputs = cmdPlugin.outputs;
                cmd->buffer = cmdPlugin.buffer;
            } else {
                cmd->op = nullptr;
                cmd->buffer.clear();
            }
        }
    }
    for (auto iter = cmd.command.begin(); iter != cmd.command.end();) {
        if (iter->op == nullptr) {
            iter = cmd.command.erase(iter);
        } else {
            ++iter;
        }
    }
    size_t id = std::hash<std::string>()(kernelStr);
    std::unique_ptr<LLVMTarget> target(new LLVMTarget("jit-kenerl-" + std::to_string(id)));
    target->getModule()->setDataLayout(theJit->getDataLayout());
    plugin.codegen(target.get());
    // add module to JIT and compile
    auto m = target->getThreadSafeModule();
    auto resourceTracker = theJit->getMainJITDylib().createResourceTracker();
    theJit->addModule(std::move(m), resourceTracker);
    theJit->compileAllFunction(plugin.getFunctionNum());
}

bool opFuse(CommandBuffer& cmd) {
    std::unordered_map<const Tensor*, Node*> outputTensor;
    // build graph
    std::vector<std::unique_ptr<Node>> graph;
    auto insertEdge = [&outputTensor](const Tensor* inputTensor, Node* succNode) {
        if (outputTensor.find(inputTensor) != outputTensor.end()) {
            auto preNode = outputTensor[inputTensor];
            succNode->pred.push_back(preNode);
            preNode->succ.push_back(succNode);
        }
    };
    for (int i = 0; i < cmd.command.size(); i++) {
        auto& iter = cmd.command[i];
        if (!iter.buffer.empty()) {
            iter.op = flatbuffers::GetMutableRoot<Op>((void*)iter.buffer.data());
        }
        std::unique_ptr<Node> node(new Node);
        node->cmd = &iter;
        node->topoIndex = i;
        for (auto input : iter.inputs) {
            if (TensorUtils::getDescribe(input)->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                for (auto& region : TensorUtils::getDescribe(input)->regions) {
                    insertEdge(region.origin, node.get());
                }
            } else {
                insertEdge(input, node.get());
            }
        }
        for (auto output : iter.outputs) {
            outputTensor[output] = node.get();
        }
        graph.push_back(std::move(node));
    }
    std::queue<Node*> postDominateNodeQueue;
    // build dominate tree
    for (int i = graph.size()-1; i >= 0; i--) {
        auto node = graph[i].get();
        if (!node->succ.empty()) {
            auto parent = node->succ[0];
            for (int j = 1; j < node->succ.size(); j++) {
                parent = LCA(parent, node->succ[j]);
            }
            node->domainatePred = parent;
            if (parent) {
                parent->domainateSucc.push_back(node);
            } else {
                postDominateNodeQueue.push(node);
            }
        } else {
            node->domainatePred = nullptr;
            postDominateNodeQueue.push(node);
        }
    }
    // bfs find subgraph
    std::vector<std::vector<Node*>> fuseSets;
    while (!postDominateNodeQueue.empty()) {
        auto root = postDominateNodeQueue.front();
        postDominateNodeQueue.pop();
        if (root->domainateSucc.empty()) {
            continue;
        }
        std::vector<Node*> childs;
        if (isLegal(root->cmd)) {
            auto fuseSet = fuseNode(root, childs);
            if (fuseSet.size() > 1) {
                fuseSets.emplace_back(std::move(fuseSet));
            }
        } else {
            childs = root->domainateSucc;
        }
        for (auto child : childs) {
            postDominateNodeQueue.push(child);
        }
    }
    jit(cmd, fuseSets);
    return true;
}
} // namespace MNN

