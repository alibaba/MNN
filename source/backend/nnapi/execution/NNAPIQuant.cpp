//
//  NNAPIQuant.cpp
//  MNN
//
//  Created by MNN on 2023/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NNAPIQuant.hpp"

namespace MNN {


NNAPIQuant::NNAPIQuant(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NNAPICommonExecution(b, op) {
}

ErrorCode NNAPIQuant::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    outputs[0]->buffer().type = halide_type_of<int8_t>();
    return mNNAPIBackend->buildQuantOperation(inputs[0], outputs[0]);
    // return buildOperation(ANEURALNETWORKS_QUANTIZE, getTensorIdxs(inputs), getTensorIdxs(outputs));
}

NNAPIDequant::NNAPIDequant(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NNAPICommonExecution(b, op) {
}

ErrorCode NNAPIDequant::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return buildOperation(ANEURALNETWORKS_DEQUANTIZE, getTensorIdxs(inputs), getTensorIdxs(outputs));
}

REGISTER_NNAPI_OP_CREATOR(NNAPIQuant, OpType_FloatToInt8)
REGISTER_NNAPI_OP_CREATOR(NNAPIDequant, OpType_Int8ToFloat)
} // namespace MNN