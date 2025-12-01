#include "QNNPermute.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNPermute::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Tensor * input = inputs[0];
    int dim = input->dimensions();
    Tensor::DimensionType inputDimType = inputs[0]->getDimensionType();
    Tensor::DimensionType outputDimType = outputs[0]->getDimensionType();
    MNN_ASSERT(inputDimType == outputDimType);

    #ifdef QNN_VERBOSE
    MNN_PRINT("QNN Permute: %s input0:", mNodeName.c_str());
    auto shape0 = inputs[0]->shape();
    for(int i = 0; i < shape0.size(); i++) {
        MNN_PRINT("%d x ", shape0[i]);
    }

    MNN_PRINT("\noutput:");
    auto outShape = outputs[0]->shape();
    for(int i = 0; i < outShape.size(); i++) {
        MNN_PRINT("%d x ", outShape[i]);
    }
    MNN_PRINT("\n");

    #endif

    mNodeType = "Transpose";
    auto param = mOp->main_as_Permute();
    auto axis = param->dims();
    int size = (int) param->dims()->size();
    MNN_ASSERT(size == dim);
    std::vector<uint32_t> mapRaw(dim, 0);
    for (int i = 0; i < dim; i++) {
        int index = axis->Get(i);
        mapRaw[i] = (uint32_t) index;
    }
    
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) dim}, mapRaw.data());
    this->addNodeCommon(inputs, outputs);
    return NO_ERROR;
}

class QNNPermuteCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);\

        if (op->main_as_Permute()->dims() == nullptr) {
            return nullptr;
        }

        return new QNNPermute(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNPermuteCreator, OpType_Permute)
#endif
} // end namespace QNN
} // end namespace MNN
