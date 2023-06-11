//
//  NNAPICast.cpp
//  MNN
//
//  Created by MNN on 2023/04/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NNAPICast.hpp"

namespace MNN {


NNAPICast::NNAPICast(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NNAPICommonExecution(b, op) {
}

ErrorCode NNAPICast::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return buildOperation(ANEURALNETWORKS_CAST, getTensorIdxs(inputs), getTensorIdxs(outputs));
}

REGISTER_NNAPI_OP_CREATOR(NNAPICast, OpType_Cast)
} // namespace MNN