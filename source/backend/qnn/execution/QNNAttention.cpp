#include "QNNAttention.hpp"
namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

// #define GQA_USE_GATHER
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
    int seqLenQ = seqLen;
    int kvHeadNum = inputs[1]->length(2);
    int seqLenKV = inputs[1]->length(1);
    float scale = 1.0 / sqrt(headDim);

    Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
    this->createStageTensor("ScaleQ", dataType, std::vector<int>({batch, headNum, headDim, seqLenQ})); // mTempTensorWrappers[0], stage Scale
    this->createStageTensor("QK", dataType, std::vector<int>({batch, headNum, seqLenQ, seqLenKV})); // mTempTensorWrappers[1], stage QK
    this->createStageTensor("Softmax", dataType, std::vector<int>({batch, headNum, seqLenQ, seqLenKV})); // mTempTensorWrappers[2], stage Softmax
    this->createStageTensor("QKV", dataType, std::vector<int>({batch, headNum, seqLenQ, headDim})); // mTempTensorWrappers[3], stage QKV
    this->createStageTensor("Transpose", dataType, std::vector<int>({batch, headNum, headDim, seqLenQ})); // mTempTensorWrappers[4], stage Transpose

    size_t totalSize = batch * headNum * seqLenQ * headDim;
    std::vector<float> scaleVec(totalSize, scale);
    // mTempTensorWrappers[5], static coef
    this->createStaticFloatTensor("coef", dataType, std::vector<uint32_t>({(uint32_t)batch, (uint32_t)headNum, (uint32_t)headDim, (uint32_t)seqLenQ}), scaleVec.data());

    std::vector<uint32_t> mapReal{0, 1, 3, 2};
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) 4}, mapReal.data(), "output_trans"); // mParamTensorWrappers[0]
    // GQA
    bool isGQA = (headNum != kvHeadNum);
    int tensorNumGQA = 0;
    int group = headNum / kvHeadNum;
    if(isGQA) {
        this->createStageTensor("RepeatedKey", dataType, std::vector<int>({batch, headNum, headDim, seqLenKV})); // mTempTensorWrappers[6], stage RepeatedKey
        this->createStageTensor("RepeatedValue", dataType, std::vector<int>({batch, headNum, headDim, seqLenKV})); // mTempTensorWrappers[7], stage RepeatedValue

        #ifdef GQA_USE_GATHER
        // index: fill in Key and Value to shape of Query
        // [a0, a1, ..., a(kvHeadNum-1)] -> [a0 ... a0, a1 ... a1, a(kvHeadNum-1) ... a(kvHeadNum-1)]
        std::vector<int32_t> index(totalSize);
        for(int b = 0; b < batch; b++) {
            int base_index = 0;
            for(int h = 0; h < kvHeadNum; h++) {
                for(int a = 0; a < group * headDim * seqLenKV; a++) {
                    index[(b * kvHeadNum + h) * group * headDim * seqLenKV + a] = base_index;
                }
                base_index++;
            }
        }
        this->createStaticTensor("gather_index", QNN_DATATYPE_INT_32, {(uint32_t)batch, (uint32_t)headNum, (uint32_t)headDim, (uint32_t)seqLenKV}, index.data());
        tensorNumGQA = 3;
        #else

        std::vector<uint32_t> splitIndex(kvHeadNum-1);
        for(int i = 0; i < splitIndex.size(); i++) {
            splitIndex[i] = i + 1;
        }
        // mParamTensorWrappers[1]
        this->createParamTensor("split_index", QNN_DATATYPE_UINT_32, {(uint32_t)kvHeadNum-1}, (void *)splitIndex.data(), "K_Split");
        // mTempTensorWrappers[8] .. [7+kvHeadNum] stage SplitKV_Temp
        for(int i = 0; i < kvHeadNum; i++) {
            this->createStageTensor("SplitK_Temp" + std::to_string(i), dataType, std::vector<int>({batch, 1, headDim, seqLenKV})); 
        }
        // mParamTensorWrappers[2]
        this->createParamTensor("split_index", QNN_DATATYPE_UINT_32, {(uint32_t)kvHeadNum-1}, (void *)splitIndex.data(), "V_Split");
        // mTempTensorWrappers[8+kvHeadNum] .. [7+2*kvHeadNum] stage SplitKV_Temp
        for(int i = 0; i < kvHeadNum; i++) {
            this->createStageTensor("SplitV_Temp" + std::to_string(i), dataType, std::vector<int>({batch, 1, headDim, seqLenKV})); 
        }
        tensorNumGQA = 2 + 2*kvHeadNum;
        #endif

        this->createParamScalar("axis", (uint32_t)1);
    }
    bool hasMask = (inputs.size() > 3);
    int maskPosIndex = 6 + tensorNumGQA;
    int scalarBaseIndex = isGQA ? 1 : 0;
    if(hasMask) {
        this->createStageTensor("tempMask", dataType, std::vector<int>({batch, 1, seqLenQ, seqLenKV})); // mTempTensorWrappers[maskPosIndex], stage Mask
        this->createStageTensor("maskResult", dataType, std::vector<int>({batch, headNum, seqLenQ, seqLenKV})); // mTempTensorWrappers[maskPosIndex+1], stage Mask
    }

    // scale
    {
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_Scale";
        mNodeType = "ElementWiseMultiply";
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); //Q
        mInputs.push_back(*(mTempTensorWrappers[5]->getNativeTensor())); // coef
        mOutputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // ScaleQ

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Q * K
    {
        auto tempK = *(mBackend->getNativeTensor(inputs[1]));
        if(isGQA) {
            #ifdef GQA_USE_GATHER
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            std::string name = mNodeName + "_K_Repeat";
            mNodeType = "GatherElements";

            mInputs.push_back(*(mBackend->getNativeTensor(inputs[1]))); // input1
            mInputs.push_back(*(mTempTensorWrappers[8]->getNativeTensor())); // gather_index
            mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // axis
            mOutputs.push_back(*(mTempTensorWrappers[6]->getNativeTensor())); // stage RepeatedKey

            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            #else
            {
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                std::string name = mNodeName + "_K_Split";
                mNodeType = "Split";

                mInputs.push_back(*(mBackend->getNativeTensor(inputs[1]))); // input1
                mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // axis
                mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // split_index
                for(int i = 0; i < kvHeadNum; i++) {
                    mOutputs.push_back(*(mTempTensorWrappers[8+i]->getNativeTensor())); // stage TempKey
                }
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }
            {
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                std::string name = mNodeName + "_K_Concat";
                mNodeType = "Concat";

                for(int i = 0; i < kvHeadNum; i++) {
                    for(int j = 0; j < group; j++) {
                        mInputs.push_back(*(mTempTensorWrappers[8+i]->getNativeTensor())); // stage TempKey
                    }
                }
                mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // axis
                mOutputs.push_back(*(mTempTensorWrappers[6]->getNativeTensor())); // stage TempKey
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }
            #endif
            tempK = *(mTempTensorWrappers[6]->getNativeTensor());
        }
        bool transpose0 = true;
        bool transpose1 = false;
        this->createParamScalar("transpose_in0", transpose0); // mParamScalarWrappers[scalarBaseIndex + 0], transpose_in0
        this->createParamScalar("transpose_in1", transpose1); // mParamScalarWrappers[scalarBaseIndex + 1], transpose_in1

        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_MatMul_QK";
        mNodeType = "MatMul";
        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); //ScaleQ
        mInputs.push_back(tempK); // input1
        mParams.push_back(*(mParamScalarWrappers[scalarBaseIndex + 0]->getNativeParam()));  // transpose0
        mParams.push_back(*(mParamScalarWrappers[scalarBaseIndex + 1]->getNativeParam()));  // transpose1
        mOutputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // QK

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    auto softmax_in = *(mTempTensorWrappers[1]->getNativeTensor());

    // mask 
    if(hasMask)
    {
        if(inputs[3]->getType() != halide_type_of<float>()) {
            MNN_ERROR("Qnn attention only support float mask currently\n");
        }
        // mask reshape
        {
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            std::string name = mNodeName + "_Mask_Reshape";
            mNodeType = "Reshape";
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[3]))); // stage mask
            mOutputs.push_back(*(mTempTensorWrappers[maskPosIndex]->getNativeTensor())); // tempMask

            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }

        // mask compute
        {
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            std::string name = mNodeName + "_Mask_Add";
            mNodeType = "ElementWiseAdd";
            mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // QK
            mInputs.push_back(*(mTempTensorWrappers[maskPosIndex]->getNativeTensor())); // stage tempMask
            mOutputs.push_back(*(mTempTensorWrappers[maskPosIndex + 1]->getNativeTensor())); // 

            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        softmax_in = *(mTempTensorWrappers[maskPosIndex + 1]->getNativeTensor());
    }

    // softmax
    {
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_Softmax";
        mNodeType = "Softmax";
        mInputs.push_back(softmax_in);
        mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));// Stage Softmax
    
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    // QK * V
    {
        auto tempV = *(mBackend->getNativeTensor(inputs[2]));
        if(isGQA) {
            #ifdef GQA_USE_GATHER
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            std::string name = mNodeName + "_V_Repeat";
            mNodeType = "GatherElements";

            mInputs.push_back(*(mBackend->getNativeTensor(inputs[2]))); // input2
            mInputs.push_back(*(mTempTensorWrappers[8]->getNativeTensor())); // gather_index
            mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // axis
            mOutputs.push_back(*(mTempTensorWrappers[7]->getNativeTensor())); // stage RepeatedValue

            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            #else
            {
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                std::string name = mNodeName + "_V_Split";
                mNodeType = "Split";

                mInputs.push_back(*(mBackend->getNativeTensor(inputs[2]))); // input2
                mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // axis
                mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // split_index
                for(int i = 0; i < kvHeadNum; i++) {
                    mOutputs.push_back(*(mTempTensorWrappers[8+kvHeadNum+i]->getNativeTensor())); // stage TempValue
                }
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }
            {
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                std::string name = mNodeName + "_V_Concat";
                mNodeType = "Concat";

                for(int i = 0; i < kvHeadNum; i++) {
                    for(int j = 0; j < group; j++) {
                        mInputs.push_back(*(mTempTensorWrappers[8+kvHeadNum+i]->getNativeTensor())); // stage TempKey
                    }
                }
                mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // axis
                mOutputs.push_back(*(mTempTensorWrappers[7]->getNativeTensor())); // stage TempKey
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }
            #endif
            tempV = *(mTempTensorWrappers[7]->getNativeTensor());
        }
        bool transpose0 = false;
        bool transpose1 = true;
        this->createParamScalar("transpose_in0", transpose0); // mParamScalarWrappers[scalarBaseIndex + 2], transpose_in0
        this->createParamScalar("transpose_in1", transpose1); // mParamScalarWrappers[scalarBaseIndex + 3], transpose_in1

        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_MatMul_QKV";
        mNodeType = "MatMul";
        mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); //Softmax
        mInputs.push_back(tempV); // input2
        mParams.push_back(*(mParamScalarWrappers[scalarBaseIndex + 2]->getNativeParam()));  // transpose0
        mParams.push_back(*(mParamScalarWrappers[scalarBaseIndex + 3]->getNativeParam()));  // transpose1
        mOutputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // QKV

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Transpose
    {
        std::string name = mNodeName + "_Transpose";
        mNodeType = "Transpose";
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mInputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // QKV
        mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // perm
        mOutputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor())); // Transpose
    
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    // Reshape
    {
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_Reshape";
        mNodeType = "Reshape";

        mInputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor())); // Transpose
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
            MNN_PRINT("Warning: Attention param kv_cache is true, QNN Backend not really save kv_cache\n");
        }
        if(inputs.size() < 3 || inputs.size() > 4 || outputs.size() != 1) {
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
#endif
} // end namespace QNN
} // end namespace MNN
