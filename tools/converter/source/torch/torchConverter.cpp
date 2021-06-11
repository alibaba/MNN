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

MNN_PUBLIC int torch2MNNNet(const std::string inputModel, const std::string bizCode,
                 std::unique_ptr<MNN::NetT>& netT) {
    const auto graph = torch::jit::torchOptPass(inputModel.c_str());
    std::unique_ptr<torchContext> context(new torchContext(netT.get()));
    for (const auto input : graph->inputs()) {
        if (input->type()->str() != "Tensor") {
            continue;
        }
        auto inputName = input->debugName();
        context->declareTensor(inputName);
        MNN::OpT* MNNOp  = new MNN::OpT;
        MNNOp->name      = inputName;
        MNNOp->type      = MNN::OpType_Input;
        MNNOp->main.type = MNN::OpParameter_Input;
        auto param  = new MNN::InputT;
        param->dtype = MNN::DataType_DT_FLOAT;
        param->dformat = MNN::MNN_DATA_FORMAT_NCHW;
        MNNOp->main.value = param;
        netT->oplists.emplace_back(MNNOp);
        MNNOp->outputIndexes.push_back(context->lookupTensor(inputName));
    }
    for (const auto &node : graph->nodes()) {
        const auto& kind = node->kind();
        const auto& opType = kind.toUnqualString();
        // python prim ops
        if (kind.is_prim() && context->dealPrime(node)) {
            continue;
        }
        context->buildOp(node);
    }
    for (const auto &output : graph->outputs()) {
        netT->outputName.push_back(output->debugName());
    }
    netT->sourceType = MNN::NetSource_TORCH;
    netT->bizCode = bizCode;
    return 0;
}
