//
//  OpFuse.cpp
//  MNN
//
//  Created by MNN on 2020/9/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpFuse.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "SourceModule.hpp"
#include "opencl/OpenCLTarget.hpp"
#include "metal/MetalTarget.hpp"
#ifdef MNN_CODEGEN_CUDA
#include "cuda/CUDATarget.hpp"
#endif
#include <queue>
#include <unordered_map>
#include "core/OpCommonUtils.hpp"

namespace MNN {
static void dumpOp(const Op* op) {
    if (op->name()) MNN_PRINT("name: %s, ", op->name()->c_str());
    MNN_PRINT("Type: %s,\n", MNN::EnumNameOpType(op->type()));
    if (op->type() == OpType_BinaryOp)  {
        auto binary = op->main_as_BinaryOp();
        auto type = binary->opType();
        MNN_PRINT("Op: %s\n", MNN::EnumNamesBinaryOpOperation()[type]);
    } else if (op->type() == OpType_UnaryOp){
        auto unary = op->main_as_UnaryOp();
        auto type = unary->opType();
        MNN_PRINT("Op: %s\n", MNN::EnumNamesUnaryOpOperation()[type]);
    }
}
static void dumpRegion(Tensor::InsideDescribe::Region& reg) {
    MNN_PRINT("\n{\nsize: [%d, %d, %d], origin: %p\n", reg.size[0], reg.size[1], reg.size[2], reg.origin);
    MNN_PRINT("src: { stride: [%d, %d, %d], offset: %d }\n", reg.src.stride[0],reg.src.stride[1],reg.src.stride[2],reg.src.offset);
    MNN_PRINT("dst: { stride: [%d, %d, %d], offset: %d }\n}\n", reg.dst.stride[0],reg.dst.stride[1],reg.dst.stride[2],reg.dst.offset);
}
static void dumpTensor(const Tensor* t) {
    MNN_PRINT("\t%p [", t);
    for (int d : t->shape())
        MNN_PRINT("%d,", d);
    MNN_PRINT("], format:%d\n", TensorUtils::getDescribe(t)->dimensionFormat);
    auto des = TensorUtils::getDescribe(t);
    if (des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
        MNN_PRINT("Regions:");
        for (auto reg : des->regions) {
            dumpRegion(reg);
        }
    }
}
static void dumpCmd(const Command* cmd) {
    MNN_PRINT("\n{\n");
    dumpOp(cmd->op);
    MNN_PRINT("output: \n");
    dumpTensor(cmd->outputs[0]);
    MNN_PRINT("input: \n");
    for (auto input : cmd->inputs) {
        dumpTensor(input);
    }
    MNN_PRINT("}\n");
}

// is legal fused type
bool isLegal(Command* cmd, MNNForwardType forwardType) {
    auto type = cmd->op->type();
    bool elemWise = type == OpType_BinaryOp
           || type == OpType_UnaryOp
           || type == OpType_ReLU
           || type == OpType_ReLU6
           || type == OpType_Eltwise;
    if (elemWise) {
        if(forwardType == MNN_FORWARD_OPENCL) {
            for (auto t : cmd->inputs) {
                if (t->width() * UP_DIV(t->channel(), 4) > 16384) {
                    return false;
                }
                auto des = TensorUtils::getDescribe(t)->regions;
                for(auto region : des)
                {
                    auto tensor = region.origin;
                    if (tensor->width() * UP_DIV(tensor->channel(), 4) > 16384) {
                        return false;
                    }
                }
            }
        }
        if(TensorUtils::getDescribe(cmd->outputs[0])->dimensionFormat ==  MNN_DATA_FORMAT_NC4HW4) {
            cmd->canVectorize = true;
        } else {
            int count = 1;
            for(int i = 0; i < cmd->outputs[0]->dimensions(); i++) {
                count *= cmd->outputs[0]->length(i);
            }
            if(count % 4 == 0) {
                cmd->canVectorize = true;
            } else {
                cmd->canVectorize = false;
            }
        }
        return true;
    }

    if (forwardType == MNN_FORWARD_CUDA && type == OpType_Raster) {
        // Fuse NC4HW4 -> NCHW/HHWC
        OpCommonUtils::TensorConvertParameter singleConvert;
        auto input = cmd->outputs[0];
        OpCommonUtils::rasterInputReset(cmd->inputs, cmd->outputs[0]);
        singleConvert.type = 0;
        auto des = TensorUtils::getDescribe(input);
        if(des->regions.size() == 1) {
            OpCommonUtils::turnRegion2Convert(des->regions[0], cmd->outputs[0], singleConvert);
            if (singleConvert.type > 0){
                auto realInput = TensorUtils::getDescribe(input)->regions[0].origin;
                auto sourceFormat = TensorUtils::getDescribe(realInput)->dimensionFormat;
                if (MNN_DATA_FORMAT_NC4HW4 == sourceFormat) { // NC4HW4 -> NCHW/NHWC is Supported!
                    if(singleConvert.type == 1) { // output NCHW
                        if(singleConvert.batch != cmd->outputs[0]->length(0)) {
                            return false;
                        }
                        if(cmd->outputs[0]->dimensions() < 3 || singleConvert.channel != cmd->outputs[0]->length(1)) {
                            return false;
                        }
                        int area = 1;
                        for(int i = 2; i < cmd->outputs[0]->dimensions(); i++) {
                            area *= cmd->outputs[0]->length(i);
                        }
                        if(singleConvert.area != area) {
                            return false;
                        }
                        return true;
                    }
                    if(singleConvert.type == 2) { // output NHWC
                        if(singleConvert.batch != cmd->outputs[0]->length(0)) {
                            return false;
                        }
                        int dims = cmd->outputs[0]->dimensions();
                        if(dims < 3 || singleConvert.channel != cmd->outputs[0]->length(dims-1)) {
                            return false;
                        }
                        int area = 1;
                        for(int i = 1; i < dims-1; i++) {
                            area *= cmd->outputs[0]->length(i);
                        }
                        if(singleConvert.area != area) {
                            return false;
                        }
                        if(singleConvert.channel % 4 == 0) {
                            cmd->canVectorize = true;
                        }
                        return true;
                    }
                    return false;
                }
            }
        }
    }
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
bool allPathLegal(Node* s, Node* t, MNNForwardType type) {
    bool legal = true;
    std::queue<Node*> q;
    q.push(s);
    while (!q.empty()) {
        auto node = q.front();
        q.pop();
        legal &= isLegal(node->cmd, type);
        if(!legal) {
            return false;
        }
        for (auto succ : node->succ) {
            if (succ != t) {
                q.push(succ);
            }
        }
    }
    return legal;
}
std::vector<Node*> fuseNode(Node* root, std::vector<Node*>& edges, MNNForwardType type) {
    std::vector<Node*> fuseSet;
    std::queue<Node*> q;
    q.push(root);
    int rasterCount = 0;
    bool insert = false;
    while (!q.empty()) {
        insert = false;
        auto node = q.front();
        if(node->cmd->op->type() == OpType_Raster) {
            // Current only fuse single raster
            rasterCount++;
            if(rasterCount < 2) {
                fuseSet.insert(fuseSet.begin(), node);
                insert = true;
            }
        } else {
            fuseSet.insert(fuseSet.begin(), node);
            insert = true;
        }

        q.pop();
        if(insert) {
            for (auto child : node->domainateSucc) {
                if (isLegal(child->cmd, type) && allPathLegal(child, root, type)) {
                    q.push(child);
                } else {
                    edges.push_back(child);
                }
            }
        }
    }
    return fuseSet;
}

bool codegen(std::vector<Schedule::OpCacheInfo>& infos, std::vector<std::vector<Node*>>& fuseSets, MNNForwardType type, BackendConfig::PrecisionMode precision) {
    // generate Kernel
    std::unique_ptr<Target> target;
    switch (type) {
#ifdef MNN_CODEGEN_OPENCL
        case MNN_FORWARD_OPENCL:
            target.reset(new OpenCLTarget(precision));
            break;
#endif
#ifdef MNN_CODEGEN_METAL
        case MNN_FORWARD_METAL:
            target.reset(new MetalTarget(precision));
            break;
#endif
#ifdef MNN_CODEGEN_CUDA
        case MNN_FORWARD_CUDA:
            target.reset(new CUDATarget(precision));
            break;
#endif
        default:
            return false;
    }
#if 0
    if (fuseSets.size() > 0) {
        MNN_PRINT(">>>>>>>>>>>>> fuseSets.size = %lu\n", fuseSets.size());
    }
#endif
    std::map<std::string, int> mapKernelSources;
    for (int i = 0; i < fuseSets.size(); i++) {
        auto& compSet = fuseSets[i];
        /*
        for (auto comp : compSet) {
            dumpCmd(comp->cmd);
        }
        */
        bool fuseKernelVectorize = true;
        for (auto& node : compSet) {
            auto cmd = node->cmd;
            if(!cmd->canVectorize) {
                fuseKernelVectorize = false;
                break;
            }
        }
        target->setFuseKernelVectorize(fuseKernelVectorize);
        SourceModule fuseModule(target.get());
        InOutTensors tensors = fuseModule.buildKernel(compSet, i);
        auto inputs = tensors.first;
        auto outputs = tensors.second;
        // build Plugin Op
        SharedPtr<Command> cmdPlugin;
        {
            auto sourceCode = fuseModule.codegen();
            if(mapKernelSources.find(sourceCode) == mapKernelSources.end()) {
                int kernelCount = mapKernelSources.size();
                mapKernelSources.insert(std::pair<std::string, int>(sourceCode, kernelCount));
            }
            std::string kernelName = "kernel_" + std::to_string(mapKernelSources[sourceCode]);
            sourceCode.insert(fuseModule.strIndexForKernelNum(), kernelName);

            std::unique_ptr<OpT> fuseOp(new OpT);
            fuseOp->type = OpType_Extra;
            fuseOp->name = fuseModule.opName();
            ExtraT* extra_param = new ExtraT;
            extra_param->type = kernelName;
            extra_param->info.resize(sourceCode.size() + 1);
            memcpy(extra_param->info.data(), sourceCode.data(), sourceCode.size() + 1);
            extra_param->vector = fuseKernelVectorize;
            fuseOp->main.type  = OpParameter_Extra;
            fuseOp->main.value = extra_param;
            flatbuffers::FlatBufferBuilder builder;
            auto lastOffset = Op::Pack(builder, fuseOp.get());
            builder.Finish(lastOffset);
            cmdPlugin = GeometryComputerUtils::makeCommand(builder, inputs, outputs);
        }
        for (int i = 0; i < compSet.size(); i++) {
            auto cmd = const_cast<Command*>(compSet[i]->cmd);
            if (i == compSet.size()-1) {
                cmd->op = cmdPlugin->op;
                cmd->inputs = cmdPlugin->inputs;
                cmd->outputs = cmdPlugin->outputs;
                cmd->buffer = cmdPlugin->buffer;
            } else {
                cmd->op = nullptr;
                cmd->buffer.reset();
            }
        }
    }
    // printf(">>> fuse Kernel num: %lu\n", fuseSets.size());
    for (auto& info : infos) {
        for (auto iter = info.executeBuffer.command.begin(); iter != info.executeBuffer.command.end();) {
            if (iter->get()->op == nullptr) {
                iter = info.executeBuffer.command.erase(iter);
            } else {
                ++iter;
            }
        }
    }
    // Clear useless cacheBuffer
    for (auto& info : infos) {
        for (auto iter = info.cacheBuffer.command.begin(); iter != info.cacheBuffer.command.end();) {
            if (iter->get()->op == nullptr) {
                iter = info.cacheBuffer.command.erase(iter);
            } else {
                ++iter;
            }
        }
    }    
    return true;
}

bool opFuse(std::vector<Schedule::OpCacheInfo>& infos, MNNForwardType type, BackendConfig::PrecisionMode precision) {
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
    for (int i = 0; i < infos.size(); i++) {
        auto& info = infos[i];
        auto& cmdBuffer = info.executeBuffer;
        for (int j = 0; j < cmdBuffer.command.size(); j++) {
            auto iter = cmdBuffer.command[j];
            /*
            if (iter->buffer.get()) {
                iter->op = flatbuffers::GetMutableRoot<Op>((void*)iter->buffer);
            }
            */
            std::unique_ptr<Node> node(new Node);
            node->cmd = iter.get();
            node->topoIndex = i;
            for (auto input : iter->inputs) {
                insertEdge(input, node.get());
            }
            for (auto output : iter->outputs) {
                outputTensor[output] = node.get();
            }
            graph.push_back(std::move(node));
        }
    }
    std::queue<Node*> postDominateNodeQueue;
    // build dominate tree
    for (int i = static_cast<int>(graph.size()) - 1; i >= 0; i--) {
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
        if (isLegal(root->cmd, type)) {
            auto fuseSet = fuseNode(root, childs, type);
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

#if 0
    MNN_PRINT("fuse total number: %lu \n", fuseSets.size());
    for (auto compSet : fuseSets) {
        MNN_PRINT("set size: %lu \n", compSet.size());
        if (true) {
            for (auto com : compSet) {
                // json :
                // { fusedOps: [ { idx:int, srcOps: [name: string], inputs:[name:string], outputs:[name:string] } ], dynlib:string, jitObj:string, module:string }
                dumpCmd(com->cmd);
            }
        }
    }
#endif
    return codegen(infos, fuseSets, type, precision);
}
} // namespace MNN

