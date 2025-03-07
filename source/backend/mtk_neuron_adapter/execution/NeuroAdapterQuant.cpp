#include "NeuronAdapterQuant.hpp"

namespace MNN {


NeuronAdapterQuant::NeuronAdapterQuant(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NeuronAdapterCommonExecution(b, op) {
}

ErrorCode NeuronAdapterQuant::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    outputs[0]->buffer().type = halide_type_of<int8_t>();
    return mNeuronAdapterBackend->buildQuantOperation(inputs[0], outputs[0]);
    // return buildOperation(ANEURALNETWORKS_QUANTIZE, getTensorIdxs(inputs), getTensorIdxs(outputs));
}

NeuronAdapterDequant::NeuronAdapterDequant(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NeuronAdapterCommonExecution(b, op) {
}

ErrorCode NeuronAdapterDequant::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return buildOperation(NEURON_DEQUANTIZE, getTensorIdxs(inputs), getTensorIdxs(outputs));
}

REGISTER_NeuronAdapter_OP_CREATOR(NeuronAdapterQuant, OpType_FloatToInt8)
REGISTER_NeuronAdapter_OP_CREATOR(NeuronAdapterDequant, OpType_Int8ToFloat)
} // namespace MNN