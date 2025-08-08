#include "QNNAttention.hpp"

namespace MNN {
namespace QNN {

/*
seqLenQ == seqLenKV
query : [Batch, seqLenQ,  headNum, headDim] -> (real layout) [Batch, headNum, headDim, seqLenQ]
key   : [Batch, seqLenKV, headNum, headDim] -> (real layout) [Batch, headNum, headDim, seqLenKV]
value : [Batch, seqLenKV, headNum, headDim] -> (real layout) [Batch, headNum, headDim, seqLenKV]
ouput : [Batch, seqLenQ, headNum * headDim] -> (real layout) [Batch, headNum * headDim, seqLenQ]
*/
ErrorCode QNNAttention::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef QNN_VERBOSE
    MNN_PRINT("QNN Attention inputs shape:\n");
    for(int i = 0; i < inputs.size(); i++) {
        auto shape = inputs[i]->shape();
        for(int j = 0; j < shape.size(); j++) {
            MNN_PRINT("%d ", shape[j]);
        }
        MNN_PRINT("\n");
    }
    MNN_PRINT("QNN Attention outputs shape:\n");
    for(int i = 0; i < outputs.size(); i++) {
        auto shape = outputs[i]->shape();
        for(int j = 0; j < shape.size(); j++) {
            MNN_PRINT("%d ", shape[j]);
        }
        MNN_PRINT("\n");
    }
#endif
    auto shape = inputs[0]->shape();
    int batch = shape[0];
    int seqLen = shape[1];
    int headNum = shape[2];
    int headDim = shape[3];
    int seqLenQ = seqLen, seqLenKV = seqLen;
    float scale = 1.0 / sqrt(headDim);

    Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
    this->createStageTensor("QK", dataType, std::vector<int>({batch, headNum, seqLenQ, seqLenKV})); // mTempTensorWrappers[0], stage QK

    #define SCALE_FIRST
    #ifdef SCALE_FIRST
    this->createStageTensor("Scale", dataType, std::vector<int>({batch, headNum, headDim, seqLenQ})); // mTempTensorWrappers[1], stage Scale
    #else
    this->createStageTensor("Scale", dataType, std::vector<int>({batch, headNum, seqLenQ, seqLenKV})); // mTempTensorWrappers[1], stage Scale
    #endif
    this->createStageTensor("Softmax", dataType, std::vector<int>({batch, headNum, seqLenQ, seqLenKV})); // mTempTensorWrappers[2], stage Softmax
    this->createStageTensor("QKV", dataType, std::vector<int>({batch, headNum, seqLenQ, headDim})); // mTempTensorWrappers[3], stage QKV
    this->createStageTensor("Transpose", dataType, std::vector<int>({batch, headNum, headDim, seqLenQ})); // mTempTensorWrappers[4], stage QKV

    #ifdef SCALE_FIRST
    // scale
    {
        size_t totalSize = batch * headNum * seqLenQ * headDim;
        std::vector<float> scaleVec(totalSize, scale);
        this->createStaticFloatTensor("coef", dataType, std::vector<uint32_t>({(uint32_t)batch, (uint32_t)headNum, (uint32_t)headDim, (uint32_t)seqLenQ}), scaleVec.data());

        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_Scale";
        mNodeType = "ElementWiseMultiply";
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); //QK
        mInputs.push_back(*(mTempTensorWrappers[5]->getNativeTensor())); // coef
        mOutputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // Scale

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    // Q * K
    {
        bool transpose0 = true;
        bool transpose1 = false;
        this->createParamScalar("transpose_in0", transpose0); // mParamScalarWrappers[0], transpose_in0
        this->createParamScalar("transpose_in1", transpose1); // mParamScalarWrappers[1], transpose_in1

        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_MatMul_QK";
        mNodeType = "MatMul";
        mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); //input0
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[1]))); // input1
        mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam()));  // transpose0
        mParams.push_back(*(mParamScalarWrappers[1]->getNativeParam()));  // transpose1
        mOutputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // QK

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    // softmax
    {
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_Softmax";
        mNodeType = "Softmax";
        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));
        mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));
    
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    #else
    // Q * K
    {
        bool transpose0 = true;
        bool transpose1 = false;
        this->createParamScalar("transpose_in0", transpose0); // mParamScalarWrappers[0], transpose_in0
        this->createParamScalar("transpose_in1", transpose1); // mParamScalarWrappers[1], transpose_in1

        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_MatMul_QK";
        mNodeType = "MatMul";
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); //input0
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[1]))); // input1
        mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam()));  // transpose0
        mParams.push_back(*(mParamScalarWrappers[1]->getNativeParam()));  // transpose1
        mOutputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // QK

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // scale
    {
        size_t totalSize = batch * headNum * seqLenQ * seqLenKV;
        std::vector<float> scaleVec(totalSize, scale);
        this->createStaticFloatTensor("coef", dataType, std::vector<uint32_t>({(uint32_t)batch, (uint32_t)headNum, (uint32_t)seqLenQ, (uint32_t)seqLenKV}), scaleVec.data());

        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_Scale";
        mNodeType = "ElementWiseMultiply";
        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); //QK
        mInputs.push_back(*(mTempTensorWrappers[5]->getNativeTensor())); // coef
        mOutputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // Scale

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // softmax
    {
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_Softmax";
        mNodeType = "Softmax";
        mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));
        mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));
    
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    #endif
    // QK * V
    {
        bool transpose0 = false;
        bool transpose1 = true;
        this->createParamScalar("transpose_in0", transpose0); // mParamScalarWrappers[0], transpose_in0
        this->createParamScalar("transpose_in1", transpose1); // mParamScalarWrappers[1], transpose_in1

        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_MatMul_QKV";
        mNodeType = "MatMul";
        mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); //input0
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[2]))); // input1
        mParams.push_back(*(mParamScalarWrappers[2]->getNativeParam()));  // transpose0
        mParams.push_back(*(mParamScalarWrappers[3]->getNativeParam()));  // transpose1
        mOutputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // QKV

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Transpose
    {
        std::vector<uint32_t> mapReal{0, 1, 3, 2};
        this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) 4}, mapReal.data());
        std::string name = mNodeName + "_Transpose";
        mNodeType = "Transpose";
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mInputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // QKV
        mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // perm
        mOutputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor())); // output
    
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    // Reshape
    {
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_Reshape";
        mNodeType = "Reshape";

        mInputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor()));
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    return NO_ERROR;
}

class QNNAttentionCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        auto param = op->main_as_AttentionParam();
        if(param->kv_cache()) {
            MNN_ERROR("MNN QNN not support attention op with kv_cache\n");
            return nullptr;
        }
        if(inputs.size() != 3 || outputs.size() != 1) {
            MNN_ERROR("MNN QNN not support attention op with inputs size:%d outputs size:%d\n", (int)inputs.size(), (int)outputs.size());
            return nullptr;
        }
        if(inputs[0]->dimensions() != 4 || inputs[1]->dimensions() != 4 || inputs[2]->dimensions() != 4 || outputs[0]->dimensions() != 3) {
            MNN_ERROR("MNN QNN not support attention op with inputs/outputs dimensions\n");
            return nullptr;
        }

        return new QNNAttention(backend, op); 
    }
};

REGISTER_QNN_OP_CREATOR(QNNAttentionCreator, OpType_Attention)

} // end namespace QNN
} // end namespace MNN
