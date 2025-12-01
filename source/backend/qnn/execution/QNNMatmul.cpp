#include "QNNMatMul.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNMatMul::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto param = mOp->main_as_MatMul();

    mNodeType = "MatMul";

    bool transpose0 = param->transposeA();
    bool transpose1 = param->transposeB();

    #ifdef QNN_VERBOSE
    MNN_PRINT("QNN MatMul\ninput0:");
    auto shape0 = inputs[0]->shape();
    for(int i = 0; i < shape0.size(); i++) {
        MNN_PRINT("%d x ", shape0[i]);
    }
    MNN_PRINT("\ninput1:");
    auto shape1 = inputs[1]->shape();
    for(int i = 0; i < shape1.size(); i++) {
        MNN_PRINT("%d x ", shape1[i]);
    }
    MNN_PRINT("\noutput:");
    auto outShape = outputs[0]->shape();
    for(int i = 0; i < outShape.size(); i++) {
        MNN_PRINT("%d x ", outShape[i]);
    }
    MNN_PRINT("\n");
    #endif
    
    this->createParamScalar("transpose_in0", transpose0); // mParamScalarWrappers[0], transpose_in0
    this->createParamScalar("transpose_in1", transpose1); // mParamScalarWrappers[1], transpose_in1

    // Add nodes.
    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}

class QNNMatMulCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        // Currently, GEMV is not allowed.
        if (inputs[0]->dimensions() == 1 || inputs[1]->dimensions() == 1) {
            return nullptr;
        }

        // Currently, the broadcast case is not allowed.
        if (inputs[0]->dimensions()!= inputs[1]->dimensions()) {
            return nullptr;
        }

        return new QNNMatMul(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNMatMulCreator, OpType_MatMul)
#endif
} // end namespace QNN
} // end namespace MNN
