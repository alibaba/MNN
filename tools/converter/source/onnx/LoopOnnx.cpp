//
//  LoopOnnx.cpp
//  MNN
//
//  Created by MNN on 2021/06/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"
#include <MNN/MNNDefine.h>
DECLARE_OP_CONVERTER(LoopOnnx);

MNN::OpType LoopOnnx::opType() {
    return MNN::OpType_While;
}
MNN::OpParameter LoopOnnx::type() {
    return MNN::OpParameter_WhileParam;
}

void LoopOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                   OnnxScope* scope) {
    auto param = new MNN::WhileParamT;
    dstOp->name += "/Loop";
    param->body_graph = dstOp->name +  "/body";
    auto body = &onnxNode->attribute(0).g();
    // build body
    std::string empty;
    int N = onnxNode->input_size() - 2;
    int K = onnxNode->output_size() - N;
    MNN_ASSERT(body->input_size() == N+2);
    MNN_ASSERT(body->output_size() == N+K+1);
    auto ousideInputs = scope->buildSubGraph(body, param->body_graph, true);
    std::vector<int> outsideIndex(ousideInputs.size());
    std::vector<int> outsideIndexOutside(ousideInputs.size());
    for (int i=0; i<ousideInputs.size(); ++i) {
        auto idx = scope->lookupTensor(ousideInputs[i]);
        MNN_ASSERT(idx >= 0);
        outsideIndex[i] = idx;
        outsideIndexOutside[i] = scope->declareTensor(dstOp->name + "_extra_unused_" + ousideInputs[i]);
    }
    dstOp->inputIndexes.insert(dstOp->inputIndexes.end(), outsideIndex.begin(), outsideIndex.end());
    dstOp->outputIndexes.insert(dstOp->outputIndexes.begin()+N, outsideIndexOutside.begin(), outsideIndexOutside.end());
    // update i
    dstOp->main.value = param;
}

REGISTER_CONVERTER(LoopOnnx, Loop);
