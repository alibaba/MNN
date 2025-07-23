#include "QNNBroadcastTo.hpp"

namespace MNN {
namespace QNN {

ErrorCode QNNBroadcastTo::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    int inputDims = input->dimensions();

    std::vector<uint32_t> multiplesData(inputDims, 0);
    for (int i = 0; i < inputDims; i++) {
        MNN_ASSERT((output->length(i) % input->length(i)) == 0);
        multiplesData[getNHWCAxis(i, inputDims, TensorUtils::getDimType(input))] = output->length(i) / input->length(i);
    }

    this->createParamTensor("multiples", QNN_DATATYPE_UINT_32, {(uint32_t)inputDims}, (void *) multiplesData.data());

    // add Node "Tile"
    mNodeType = "Tile";
    mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
    mParams.push_back(*(mParamTensorWrappers.back()->getNativeParam()));
    mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));

    mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);

    return NO_ERROR;
}

class QNNBroadcastToCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        MNN_ASSERT(inputs.size() == 2);
        MNN_ASSERT(outputs.size() == 1);

        auto input  = inputs[0];
        auto shape  = inputs[1];
        int inputDims = input->dimensions();
        int shapeDims = shape->elementSize();
        MNN_ASSERT(inputDims == shapeDims);

        if (inputDims > 5) {
            return nullptr;
        }

        if (op->main() && op->main_as_Axis()->axis()) {
            return nullptr;
        }

        return new QNNBroadcastTo(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNBroadcastToCreator, OpType_BroadcastTo)

} // end namespace QNN
} // end namespace MNN
