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
    virtual void run(MNN::OpT* dstOp, const torch::jit::Node* node, torchContext* context) override {
        auto extra        = new ExtraT;
        dstOp->main.type  = OpParameter_Extra;
        dstOp->main.value = extra;
        extra->engine     = "Torch";
        extra->type       = node->kind().toUnqualString();
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

void torchContext::buildOp(const torch::jit::Node *node) {
    std::unique_ptr<MNN::OpT> op(new MNN::OpT);
    const auto opType = getRealOpType(node->kind().toUnqualString());
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
    auto& oplists = mSubNet ? mSubNet->nodes : mNet->oplists;
    oplists.emplace_back(std::move(op));
}

bool torchContext::dealPrime(const torch::jit::Node *node) {
    std::string opType = node->kind().toUnqualString();
    switch (node->kind()) {
        case at::prim::Constant:
        case at::prim::ListConstruct:
        case at::prim::ListUnpack:
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
        return true;
    }
    return true;
}

int torchContext::declareTensor(std::string name) {
    if (tensorIdx.count(name)) {
        return tensorIdx[name];
    }
    auto& tensors = mSubNet ? mSubNet->tensors : mNet->tensorName;
    tensors.push_back(name);
    int idx = tensorIdx.size();
    tensorIdx[name] = idx;
    return idx;
}

int torchContext::lookupTensor(std::string name) {
    const auto iter = tensorIdx.find(name);
    if (iter != tensorIdx.end()) {
        return iter->second;
    }
    const auto iterVar = varTable.find(name);
    if (iterVar != varTable.end()) {
        buildOp(iterVar->second);
        return lookupTensor(name);
    }
    return -1;
}

std::string torchContext::lookupTensor(int idx) const {
    auto& tensors = mSubNet ? mSubNet->tensors : mNet->tensorName;
    if (idx < tensors.size()) {
        return tensors[idx];
    }
    MNN_ASSERT(false);
    return "NaN";
}

void torchContext::declareVar(std::string name, const torch::jit::Node* var) {
    if (varTable.count(name)) {
        return;
    }
    varTable[name] = var;
}

const torch::jit::Node* torchContext::lookupVar(std::string name) const {
    const auto iter = varTable.find(name);
    if (iter != varTable.end()) {
        return iter->second;
    }
    return nullptr;
}

std::vector<int> torchContext::addSubGraph(const torch::jit::Block* block, const std::string& name) {
    std::vector<int> outsideInputs;
    std::unique_ptr<MNN::SubGraphProtoT> subgraph(new MNN::SubGraphProtoT);
    subgraph->name = name;
    std::unique_ptr<torchContext> context(new torchContext(mNet, subgraph.get()));
    for (const auto& node : block->nodes()) {
        const auto& kind = node->kind();
        const auto opType = getRealOpType(kind.toUnqualString());
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
            const auto inputName = node->input(inputIdx)->debugName();
            int idx = context->lookupTensor(inputName);
            if (idx < 0) {
                idx = context->declareTensor(inputName);
                MNN::OpT* inputOp  = new MNN::OpT;
                inputOp->name      = inputName;
                inputOp->type      = MNN::OpType_Input;
                inputOp->main.type = MNN::OpParameter_Input;
                auto param  = new MNN::InputT;
                param->dtype = MNN::DataType_DT_FLOAT;
                param->dformat = MNN::MNN_DATA_FORMAT_NCHW;
                inputOp->main.value = param;
                subgraph->inputs.push_back(idx);
                inputOp->outputIndexes.push_back(idx);
                subgraph->nodes.emplace_back(inputOp);
                outsideInputs.push_back(this->lookupTensor(inputName));
            }
            MNNOp->inputIndexes.push_back(idx);
        }
        for (const auto output : node->outputs()) {
            MNNOp->outputIndexes.push_back(context->declareTensor(output->debugName()));
        }
        opConverter->run(MNNOp, node, this);
        subgraph->nodes.emplace_back(MNNOp);
    }
    for (const auto output : block->outputs()) {
        subgraph->outputs.push_back(context->lookupTensor(output->debugName()));
    }
    mNet->subgraphs.emplace_back(std::move(subgraph));
    return outsideInputs;
}
