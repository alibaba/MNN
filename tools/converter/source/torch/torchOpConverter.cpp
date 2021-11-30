//
//  torchOpConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2021/04/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "torchOpConverter.hpp"
using namespace MNN;

class defaultTorchOpConverter : public torchOpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) override {
        auto extra        = new ExtraT;
        dstOp->main.type  = OpParameter_Extra;
        dstOp->main.value = extra;
        extra->engine     = "Torch";
        extra->type       = getRealOpType(node);
    }
    virtual MNN::OpParameter type() override {
        return OpParameter_Extra;
    }
    virtual MNN::OpType opType() override {
        return OpType_Extra;
    }
};

torchOpConverterSuit* torchOpConverterSuit::global = nullptr;

torchOpConverter* torchOpConverterSuit::search(const std::string& name) {
    auto iter = mConverterContainer.find(name);
    if (iter == mConverterContainer.end()) {
        static defaultTorchOpConverter defaultConverter;
        return &defaultConverter;
    }
    return iter->second;
}

torchOpConverterSuit* torchOpConverterSuit::get() {
    if (global == nullptr) {
        global = new torchOpConverterSuit;
    }
    return global;
}

torchOpConverterSuit::~torchOpConverterSuit() {
    for (auto& it : mConverterContainer) {
        delete it.second;
    }
    mConverterContainer.clear();
}

void torchOpConverterSuit::insert(torchOpConverter* t, const char* name) {
    mConverterContainer.insert(std::make_pair(name, t));
}

void TorchScope::buildMNNOp(const torch::jit::Node *node) {
    std::unique_ptr<MNN::OpT> op(new MNN::OpT);
    const auto opType = getRealOpType(node);
    op->name = node->output(0)->debugName();
    auto opConverter = torchOpConverterSuit::get()->search(opType);
    op->defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
    op->type      = opConverter->opType();
    op->main.type = opConverter->type();
    for (int inputIdx : opConverter->inputTensorIdx()) {
        if (inputIdx < 0) {
            for (const auto input : node->inputs()) {
                op->inputIndexes.push_back(lookupTensor(input->debugName()));
            }
            break;
        }
        op->inputIndexes.push_back(lookupTensor(node->input(inputIdx)->debugName()));
    }
    for (const auto output : node->outputs()) {
        op->outputIndexes.push_back(declareTensor(output->debugName()));
    }
    opConverter->run(op.get(), node, this);
    oplists().emplace_back(std::move(op));
}

bool TorchScope::dealPrime(const torch::jit::Node *node) {
    std::string opType = getRealOpType(node);
    switch (node->kind()) {
        case at::prim::Constant:
        case at::prim::ListConstruct:
        case at::prim::ListUnpack:
        case at::prim::TupleConstruct:
        case at::prim::Uninitialized:
            for (const auto output : node->outputs()) {
                declareVar(output->debugName(), node);
            }
            return true;
        default:
            break;
    }
    if (opType == "If") {
        if (!node->outputs().empty()) {
            return false;
        }
        return true;
    }
    if (opType == "Loop") {
        return false;
    }
    return true;
}

int TorchScope::lookupTensor(std::string name) {
    const auto iter = mTensorIdx.find(name);
    if (iter != mTensorIdx.end()) {
        return iter->second;
    }
    const auto iterVar = varTable.find(name);
    if (iterVar != varTable.end()) {
        buildMNNOp(iterVar->second);
        return lookupTensor(name);
    }
    return -1;
}

void TorchScope::declareVar(std::string name, const torch::jit::Node* var) {
    if (varTable.count(name)) {
        return;
    }
    varTable[name] = var;
}

const torch::jit::Node* TorchScope::lookupVar(std::string name) const {
    const auto iter = varTable.find(name);
    if (iter != varTable.end()) {
        return iter->second;
    }
    return nullptr;
}


void TorchScope::buildSubGraph(const torch::jit::Block* block,
                               const std::string& name, bool increment) {
    std::unique_ptr<MNN::SubGraphProtoT> subgraph(new MNN::SubGraphProtoT);
    subgraph->name = name;
    std::unique_ptr<TorchScope> scope(new TorchScope(subgraph.get(), mNet, this));
    for (const auto& node : block->nodes()) {
        const auto& kind = node->kind();
        const auto opType = getRealOpType(node);
        if (kind.is_prim() && dealPrime(node)) {
            continue;
        }
        const auto& output = node->output(0);
        const auto& outputName = output->debugName();
        const std::string& type = output->type()->str();
        auto opConverter = torchOpConverterSuit::get()->search(opType);
        MNN::OpT* MNNOp  = new MNN::OpT;
        MNNOp->defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
        MNNOp->name      = outputName;
        MNNOp->type      = opConverter->opType();
        MNNOp->main.type = opConverter->type();
        for (int inputIdx : opConverter->inputTensorIdx()) {
            if (inputIdx < 0) {
                for (const auto input : node->inputs()) {
                    scope->addInputForOp(MNNOp, input->debugName());
                }
                break;
            }
            const auto inputName = node->input(inputIdx)->debugName();
            scope->addInputForOp(MNNOp, inputName, true);
        }
        for (const auto output : node->outputs()) {
            MNNOp->outputIndexes.push_back(scope->declareTensor(output->debugName()));
        }

        opConverter->run(MNNOp, node, scope.get());
        subgraph->nodes.emplace_back(MNNOp);
    }
    for (const auto output : block->outputs()) {
        int idx = scope->lookupTensor(output->debugName());
        if (idx < 0) {
            idx = scope->buildIntInputOp(output->debugName());
            scope->deps().push_back(output->debugName());
        }
        if (idx >= 0) {
            subgraph->outputs.push_back(idx);
        }
    }
    if (increment) {
        scope->buildIncrement(name, block->inputs().at(0)->debugName());
    }
    mNet->subgraphs.emplace_back(std::move(subgraph));
}
