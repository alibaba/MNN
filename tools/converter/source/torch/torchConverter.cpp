//
//  torchConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2020/11/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "torchOpConverter.hpp"
#include "torchOptimize.hpp"
#include <torch/csrc/jit/passes/freeze_module.h>
#if !defined(_MSC_VER)
#include <dlfcn.h>
#endif

void loadCustomOp(std::string customTorchOps) {
    if (customTorchOps.empty()) {
        return;
    }
#if !defined(_MSC_VER)
    constexpr char delimiter = ';';
    std::string::size_type lastPos = customTorchOps.find_first_not_of(delimiter, 0);
    std::string::size_type pos = customTorchOps.find_first_of(delimiter, lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos) {
        auto custom_lib = customTorchOps.substr(lastPos, pos - lastPos);
        dlopen(custom_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        lastPos = customTorchOps.find_first_not_of(delimiter, pos);
        pos = customTorchOps.find_first_of(delimiter, lastPos);
    }
#endif
}

MNN_PUBLIC int torch2MNNNet(const std::string inputModel, const std::string bizCode,
                            std::unique_ptr<MNN::NetT>& netT, std::string customTorchOps) {
    loadCustomOp(customTorchOps);
    // Deserialize the ScriptModule from a file, set to eval mode and freeze
    c10::Device device("cpu");
    torch::jit::Module module;
    try {
        module = torch::jit::load(inputModel.c_str(), device);
    } catch (std::exception e) {
        MNN_ERROR("[ERROR] TorchScript model can't load. Please using `torch.jit.script` or `torch.jit.trace` save model.\n");
        return 1;
    }
    auto graph = torch::jit::torchOptPass(module);
    std::unique_ptr<TorchScope> scope(new TorchScope(netT.get()));
    for (const auto input : graph->inputs()) {
        auto type = input->type()->cast<at::TensorType>();
        if (!type) {
            continue;
        }
        auto scalarType = type->scalarType().value_or(at::ScalarType::Float);
        auto inputName = input->debugName();
        scope->declareTensor(inputName);
        MNN::OpT* MNNOp  = new MNN::OpT;
        MNNOp->name      = inputName;
        MNNOp->type      = MNN::OpType_Input;
        MNNOp->main.type = MNN::OpParameter_Input;
        auto param  = new MNN::InputT;
        param->dtype = ScalarType2Dtype(scalarType);
        param->dformat = MNN::MNN_DATA_FORMAT_NCHW;
        MNNOp->main.value = param;
        netT->oplists.emplace_back(MNNOp);
        MNNOp->outputIndexes.push_back(scope->lookupTensor(inputName));
    }
    for (const auto &output : graph->outputs()) {
        netT->outputName.push_back(output->debugName());
    }
    for (const auto &node : graph->nodes()) {
        const auto& kind = node->kind();
        bool isOutputNode = false;
        for (const auto output : node->outputs()) {
            isOutputNode |= std::find(netT->outputName.begin(), netT->outputName.end(), output->debugName()) != netT->outputName.end();
        }
        // python prim ops
        if (!isOutputNode && kind.is_prim() && scope->dealPrime(node)) {
            continue;
        }
        scope->buildMNNOp(node);
    }
    netT->sourceType = MNN::NetSource_TORCH;
    netT->bizCode = bizCode;
    return 0;
}
