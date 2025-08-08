#include "QNNMatMul.hpp"

namespace MNN {
namespace QNN {

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

    #define QNN_MATMUL_OPT_3D
    #ifdef QNN_MATMUL_OPT_3D
    if(inputs[0]->dimensions() == 3) {
        int batch = outputs[0]->length(0);
        int e = outputs[0]->length(1);
        int h = outputs[0]->length(2);
        Qnn_DataType_t dataType = mBackend->getNativeTensor(outputs[0])->v1.dataType;

        this->createStageTensor("Stage_0", dataType, std::vector<int>({batch, e, h})); // mTempTensorWrappers[0], stage_0

        {
            if(TensorUtils::getDimType(inputs[0]) != Tensor::DimensionType::TENSORFLOW) {
                if(TensorUtils::getDescribe(inputs[0])->usage != Tensor::InsideDescribe::Usage::CONSTANT) {
                    transpose0 = !transpose0;
                }
                if(TensorUtils::getDescribe(inputs[1])->usage != Tensor::InsideDescribe::Usage::CONSTANT) {
                    transpose1 = !transpose1;
                }            
            }
            #ifdef QNN_VERBOSE
            MNN_PRINT("QNN MatMul tr0:%d tr0:%d, input0 const: %d, input1 const: %d\n", transpose0, transpose1, \
                TensorUtils::getDescribe(inputs[0])->usage == Tensor::InsideDescribe::Usage::CONSTANT, \
                TensorUtils::getDescribe(inputs[1])->usage == Tensor::InsideDescribe::Usage::CONSTANT);
            #endif
            
            this->createParamScalar("transpose_in0", transpose0); // mParamScalarWrappers[0], transpose_in0
            this->createParamScalar("transpose_in1", transpose1); // mParamScalarWrappers[1], transpose_in1

            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            std::string name = mNodeName + "_MatMul";
            mNodeType = "MatMul";
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); //input0
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[1]))); // input1
            mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam()));  // transpose0
            mParams.push_back(*(mParamScalarWrappers[1]->getNativeParam()));  // transpose1
            mOutputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // stage_0

            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        // Transpose
        {
            std::vector<uint32_t> mapReal{0, 2, 1};
            this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) 3}, mapReal.data());
            std::string name = mNodeName + "_Transpose";
            mNodeType = "Transpose";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // QKV
            mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // perm
            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // output
        
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        return NO_ERROR;
    }
    #endif
    
    this->createParamScalar("transpose_in0", transpose0); // mParamScalarWrappers[0], transpose_in0
    this->createParamScalar("transpose_in1", transpose1); // mParamScalarWrappers[1], transpose_in1

    if ((inputs[0]->dimensions() > 3) &&
        (TensorUtils::getDimType(inputs[0]) != Tensor::DimensionType::TENSORFLOW)) {
        return this->onEncodePermute(inputs, outputs);
    }

    // Add nodes.
    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}

ErrorCode QNNMatMul::onEncodePermute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
    int dim = inputs[0]->dimensions();

    std::vector<int> stageInput0Shape = inputs[0]->shape();
    std::vector<int> stageInput1Shape = inputs[1]->shape();
    std::vector<int> stageOutputShape = outputs[0]->shape();

    this->createStageTensor("stageInput0", dataType, stageInput0Shape); // mTempTensorWrappers[0], stage input0
    this->createStageTensor("stageInput1", dataType, stageInput1Shape); // mTempTensorWrappers[1], stage input1
    this->createStageTensor("stageOutput", dataType, stageOutputShape); // mTempTensorWrappers[2], stage output

    std::vector<uint32_t> permBeforeData(dim, 0);
    std::vector<uint32_t> permAfterData(dim, 0);

    for (int i = 0; i < dim; i++) {
        permBeforeData[i] = getNHWCAxis(i, dim, Tensor::DimensionType::CAFFE);
        permAfterData[i] = getNCHWAxis(i, dim, Tensor::DimensionType::TENSORFLOW);
    }

    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) dim}, (void *) permBeforeData.data(), "before");        // mParamTensorWrappers[0], perm before
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) dim}, (void *) permAfterData.data(), "after");          // mParamTensorWrappers[1], perm after

    this->addNodeCommonPermute("permInput0", *(mBackend->getNativeTensor(inputs[0])), *(mParamTensorWrappers[0]->getNativeParam()), *(mTempTensorWrappers[0]->getNativeTensor()));
    this->addNodeCommonPermute("permInput1", *(mBackend->getNativeTensor(inputs[1])), *(mParamTensorWrappers[0]->getNativeParam()), *(mTempTensorWrappers[1]->getNativeTensor()));

    {
        CLEAR_BEFORE_ADDING_NODE;

        std::string name = mNodeName + "_" + "MatMul";
        mNodeType = "MatMul";
        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // stage input0
        mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // stage input1
        if (inputs.size() == 3) {
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[2])));  // bias
        }
        mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam()));  // transpose0
        mParams.push_back(*(mParamScalarWrappers[1]->getNativeParam()));  // transpose1
        mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // stage output

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    this->addNodeCommonPermute("permOutput", *(mTempTensorWrappers[2]->getNativeTensor()), *(mParamTensorWrappers[1]->getNativeParam()), *(mBackend->getNativeTensor(outputs[0])));

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

} // end namespace QNN
} // end namespace MNN
