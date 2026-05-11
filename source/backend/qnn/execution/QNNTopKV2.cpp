#include "QNNTopKV2.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNTopKV2::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    auto topk = mOp->main_as_TopKV2();
    int k = inputs[0]->shape()[inputs[0]->dimensions() - 1];
    bool largest = topk->largest();

    this->createParamScalar("k", (uint32_t)k);
    this->createParamScalar("largest", (bool)largest);

    mNodeType = "TopK";
    {
        CLEAR_BEFORE_ADDING_NODE;

        mNodeType = "TopK";

        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
        mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // k
        mParams.push_back(*(mParamScalarWrappers[1]->getNativeParam())); // largest
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[1])));

        mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams,
                                 mInputs, mOutputs);
    }

    return NO_ERROR;
}

class QNNTopKV2Creator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                         const MNN::Op* op, Backend* backend) const override {
        return new QNNTopKV2(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNTopKV2Creator, OpType_TopKV2)
#endif
} // end namespace QNN
} // end namespace MNN
