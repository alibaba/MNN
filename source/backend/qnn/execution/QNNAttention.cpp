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
    auto kvMaxSize = mBackend->getRuntime()->hint().kvcacheSizeLimit;
    bool needState = false;
    auto attn = mOp->main_as_AttentionParam();
    if (nullptr != attn && attn->kv_cache()) {
        needState = true;
    }

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
    auto Query_perm = this->createStageTensor("Query_perm", dataType, std::vector<int>({batch, headNum, seqLenQ, headDim})); // [0], stage query
    Qnn_Tensor_t* keyperm;
    Qnn_Tensor_t* valueperm;
    if (needState) {
        std::shared_ptr<Tensor> t(Tensor::createDevice<float>(std::vector<int>({batch, kvHeadNum, seqLenKV, headDim})));
        keyperm = mBackend->addExtraOutput(t.get());
        valueperm = mBackend->addExtraOutput(t.get());
    } else {
        keyperm = this->createStageTensor("Key_perm", dataType, std::vector<int>({batch, kvHeadNum, seqLenKV, headDim}))->getNativeTensor();
        valueperm = this->createStageTensor("Value_perm", dataType, std::vector<int>({batch, kvHeadNum, seqLenKV, headDim}))->getNativeTensor();
    }
    auto scaleQ = this->createStageTensor("ScaleQ", dataType, std::vector<int>({batch, headNum, seqLenQ, headDim})); // [3], stage Scale
    auto QK = this->createStageTensor("QK", dataType, std::vector<int>({batch, headNum, seqLenQ, seqLenKV})); // [4], stage QK
    std::shared_ptr<QNNTensorWrapper> Softmax;
    if (needState) {
        Softmax = this->createStageTensor("Softmax", dataType, std::vector<int>({batch, headNum, seqLenQ, seqLenKV + kvMaxSize}));
    } else {
        Softmax = this->createStageTensor("Softmax", dataType, std::vector<int>({batch, headNum, seqLenQ, seqLenKV}));
    }
    auto QKV = this->createStageTensor("QKV", dataType, std::vector<int>({batch, headNum, seqLenQ, headDim})); // [6], stage QKV
    auto Transpose = this->createStageTensor("Transpose", dataType, std::vector<int>({batch, seqLenQ, headNum, headDim})); // [7], stage Transpose

    size_t totalSize = batch * headNum * seqLenQ * headDim;
    std::vector<float> scaleVec(totalSize, scale);
    // [5], static coef
    auto coef = this->createStaticFloatTensor("coef", dataType, std::vector<uint32_t>({(uint32_t)1, (uint32_t)1, (uint32_t)1, (uint32_t)1}), scaleVec.data());

    std::vector<uint32_t> mapReal{0, 2, 1, 3};
    std::vector<uint32_t> mapOutputReal{0, 2, 1, 3};
    auto input_perm_query = this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) 4}, mapReal.data(), "input_query"); // [0]
    auto input_perm_key = this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) 4}, mapReal.data(), "input_key"); // [0]
    auto input_perm_value = this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) 4}, mapReal.data(), "input_value"); // [0]
    auto output_perm = this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) 4}, mapOutputReal.data(), "output_trans"); // [3]
    Qnn_Tensor_t* stateMask = nullptr;
    Qnn_Tensor_t* pastK = nullptr;
    Qnn_Tensor_t* pastV = nullptr;
    if (needState) {
        stateMask = mBackend->getMaskTensor(kvMaxSize);
        // Create pk, pv
        std::shared_ptr<Tensor> pastKWrap(Tensor::createDevice<float>({1, kvHeadNum, kvMaxSize, headDim}));
        pastK = mBackend->addExtraInput(pastKWrap.get());
        std::shared_ptr<Tensor> pastVWrap(Tensor::createDevice<float>({1, kvHeadNum, kvMaxSize, headDim}));
        pastV = mBackend->addExtraInput(pastVWrap.get());
    }
    // transpose input
    {
        // transpose query
        {
            std::string name = mNodeName + "_Transpose_query";
            mNodeType = "Transpose";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // input0
            mParams.push_back(*(input_perm_query->getNativeParam())); // perm_query
            mOutputs.push_back(*(Query_perm->getNativeTensor())); // stage query
        
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        // transpose key
        {
            std::string name = mNodeName + "_Transpose_key";
            mNodeType = "Transpose";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[1]))); // input1
            mParams.push_back(*(input_perm_key->getNativeParam())); // perm_key
            mOutputs.push_back(*(keyperm)); // stage key
        
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        // transpose value
        {
            std::string name = mNodeName + "_Transpose_value";
            mNodeType = "Transpose";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[2]))); // input2
            mParams.push_back(*(input_perm_value->getNativeParam())); // perm_value
            mOutputs.push_back(*(valueperm)); // stage value
        
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
    }
    
    // GQA
    bool isGQA = (headNum != kvHeadNum);
    int tensorNumGQA = 0;
    int group = headNum / kvHeadNum;
    bool hasMask = (inputs.size() > 3);
    int scalarBaseIndex = isGQA ? 1 : 0;
    std::shared_ptr<QNNTensorWrapper> tempMask;
    std::shared_ptr<QNNTensorWrapper> maskResult;
    if(hasMask) {
        tempMask = this->createStageTensor("tempMask", dataType, std::vector<int>({batch, 1, seqLenQ, seqLenKV})); // [maskPosIndex], stage Mask
        maskResult = this->createStageTensor("maskResult", dataType, std::vector<int>({batch, headNum, seqLenQ, seqLenKV})); // [maskPosIndex+1], stage Mask
    }

    // scale
    {
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_Scale";
        mNodeType = "ElementWiseMultiply";
        mInputs.push_back(*(Query_perm->getNativeTensor())); //stage query
        mInputs.push_back(*(coef->getNativeTensor())); // coef
        mOutputs.push_back(*(scaleQ->getNativeTensor())); // ScaleQ

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Q * K
    {
        auto tempK = *(keyperm);
        if(isGQA) {
            {
                std::vector<std::shared_ptr<QNNTensorWrapper>> splits(kvHeadNum);
                auto axisParam = this->createParamScalar("axis", (uint32_t)1);
                auto repeatKey = this->createStageTensor("RepeatedKey", dataType, std::vector<int>({batch, headNum, seqLenKV, headDim}));
                {
                    mParams.clear();
                    mInputs.clear();
                    mOutputs.clear();
                    std::string name = mNodeName + "_K_Split";
                    mNodeType = "Split";
                    std::vector<uint32_t> splitIndex(kvHeadNum-1);
                    for(int i = 0; i < splitIndex.size(); i++) {
                        splitIndex[i] = i + 1;
                    }
                    auto split_index = this->createParamTensor("split_index", QNN_DATATYPE_UINT_32, {(uint32_t)kvHeadNum-1}, (void *)splitIndex.data(), "K_Split");
                    for(int i = 0; i < kvHeadNum; i++) {
                        auto o = this->createStageTensor("SplitK_Temp" + std::to_string(i), dataType, std::vector<int>({batch, 1, seqLenKV, headDim}));
                        splits[i] = o;
                        mOutputs.push_back(*o->getNativeTensor());
                    }

                    mInputs.push_back(*(keyperm)); // stage key
                    mParams.push_back(*(axisParam->getNativeParam())); // axis
                    mParams.push_back(*(split_index->getNativeParam())); // split_index
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
                            mInputs.push_back(*(splits[i]->getNativeTensor())); // stage TempKey
                        }
                    }
                    mParams.push_back(*(axisParam->getNativeParam())); // axis
                    mOutputs.push_back(*(repeatKey->getNativeTensor())); // stage TempKey
                    mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
                }
                tempK = *(repeatKey->getNativeTensor());
            }
            if (needState) {
                // Repeat PastK
                std::vector<std::shared_ptr<QNNTensorWrapper>> splits(kvHeadNum);
                auto axisParam = this->createParamScalar("axis", (uint32_t)1);
                auto repeatKey = this->createStageTensor("RepeatedKeyPast", dataType, std::vector<int>({batch, headNum, kvMaxSize, headDim}));
                {
                    mParams.clear();
                    mInputs.clear();
                    mOutputs.clear();
                    std::string name = mNodeName + "_K_Split_Past";
                    mNodeType = "Split";
                    std::vector<uint32_t> splitIndex(kvHeadNum-1);
                    for(int i = 0; i < splitIndex.size(); i++) {
                        splitIndex[i] = i + 1;
                    }
                    auto split_index = this->createParamTensor("split_index", QNN_DATATYPE_UINT_32, {(uint32_t)kvHeadNum-1}, (void *)splitIndex.data(), "K_Split_Past");
                    for(int i = 0; i < kvHeadNum; i++) {
                        auto o = this->createStageTensor("SplitK_Temp_Past" + std::to_string(i), dataType, std::vector<int>({batch, 1, kvMaxSize, headDim}));
                        splits[i] = o;
                        mOutputs.push_back(*o->getNativeTensor());
                    }

                    mInputs.push_back(*(pastK)); // stage key
                    mParams.push_back(*(axisParam->getNativeParam())); // axis
                    mParams.push_back(*(split_index->getNativeParam())); // split_index
                    mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
                }
                {
                    mParams.clear();
                    mInputs.clear();
                    mOutputs.clear();
                    std::string name = mNodeName + "_KPast_Concat";
                    mNodeType = "Concat";

                    for(int i = 0; i < kvHeadNum; i++) {
                        for(int j = 0; j < group; j++) {
                            mInputs.push_back(*(splits[i]->getNativeTensor())); // stage TempKey
                        }
                    }
                    mParams.push_back(*(axisParam->getNativeParam())); // axis
                    mOutputs.push_back(*(repeatKey->getNativeTensor())); // stage TempKey
                    mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
                }
                pastK = repeatKey->getNativeTensor();
            }
        }
        bool transpose0 = false;
        bool transpose1 = true;
        auto tr0 = this->createParamScalar("transpose_in0", transpose0); // [scalarBaseIndex + 0], transpose_in0
        auto tr1 = this->createParamScalar("transpose_in1", transpose1); // [scalarBaseIndex + 1], transpose_in1

        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_MatMul_QK";
        mNodeType = "MatMul";
        mInputs.push_back(*(scaleQ->getNativeTensor())); //ScaleQ
        mInputs.push_back(tempK); // input1
        mParams.push_back(*(tr0->getNativeParam()));  // transpose0
        mParams.push_back(*(tr1->getNativeParam()));  // transpose1
        mOutputs.push_back(*(QK->getNativeTensor())); // QK

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
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
                mOutputs.push_back(*(tempMask->getNativeTensor())); // tempMask

                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }

            // mask compute
            {
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                std::string name = mNodeName + "_Mask_Add";
                mNodeType = "ElementWiseAdd";
                mInputs.push_back(*(QK->getNativeTensor())); // QK stage
                mInputs.push_back(*(tempMask->getNativeTensor())); // stage tempMask
                mOutputs.push_back(*(maskResult->getNativeTensor())); //

                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }
            QK = maskResult;
        }
        if (needState) {
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*(scaleQ->getNativeTensor())); //ScaleQ
            mInputs.push_back(*pastK); // input1
            mParams.push_back(*(tr0->getNativeParam()));  // transpose0
            mParams.push_back(*(tr1->getNativeParam()));  // transpose1
            auto QKPast = this->createStageTensor("QKPast", dataType, std::vector<int>({batch, headNum, seqLenQ, kvMaxSize}));
            mOutputs.push_back(*(QKPast->getNativeTensor())); // QK

            mBackend->addNodeToGraph(mOpConfigVersion, (mNodeName + "_MatMulQKPast").c_str(), mPackageName.c_str(), "MatMul", mParams, mInputs, mOutputs);
            // BroadCast Mask
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            auto qkPastAdd = this->createStageTensor("QKPastMask", dataType, std::vector<int>({batch, headNum, seqLenQ, kvMaxSize}));
            auto broadcastMask = this->createStageTensor("MaskBroadCast", dataType, std::vector<int>({batch, headNum, seqLenQ, kvMaxSize}));
            std::vector<int> multiData = {batch, headNum, seqLenQ, 1};
            auto multi = this->createParamTensor("multiples", QNN_DATATYPE_UINT_32, {(uint32_t)multiData.size()}, multiData.data());
            mNodeType = "Tile";

            mInputs.push_back(*(stateMask)); // stage 0
            mParams.push_back(*(multi->getNativeParam())); // multiples
            mOutputs.push_back(*(broadcastMask->getNativeTensor())); // stage 1

            mBackend->addNodeToGraph(mOpConfigVersion, (mNodeName + "_Tile").c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);

            // Add
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.emplace_back(*(QKPast->getNativeTensor()));
            mInputs.emplace_back(*(broadcastMask->getNativeTensor()));
            mOutputs.emplace_back(*(qkPastAdd->getNativeTensor()));
            mBackend->addNodeToGraph(mOpConfigVersion, (mNodeName + "_MatMulQKPast_Mask").c_str(), mPackageName.c_str(), "ElementWiseAdd", mParams, mInputs, mOutputs);
            // Concat
            auto axisParam = this->createParamScalar("axis", (uint32_t)3);
            auto qkFuse = this->createStageTensor("QKFuse", dataType, std::vector<int>({batch, headNum, seqLenQ, kvMaxSize + seqLenKV}));
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*QK->getNativeTensor());
            mInputs.push_back(*qkPastAdd->getNativeTensor());
            mOutputs.push_back(*qkFuse->getNativeTensor());
            mParams.push_back(*axisParam->getNativeParam());
            mBackend->addNodeToGraph(mOpConfigVersion, (mNodeName + "_Concat_QK").c_str(), mPackageName.c_str(), "Concat", mParams, mInputs, mOutputs);
            QK = qkFuse;
        }
    }
    auto softmax_in = *(QK->getNativeTensor());

    // softmax
    {
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_Softmax";
        mNodeType = "Softmax";
        mInputs.push_back(softmax_in);
        mOutputs.push_back(*(Softmax->getNativeTensor()));// Stage Softmax
    
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    // QK * V
    {
        auto tempV = *(valueperm);
        int vSeqLen = seqLenKV;
        if (needState) {
            // Concat V
            auto axisParam = this->createParamScalar("axis", (uint32_t)2);
            auto vFuse = this->createStageTensor("VFuse", dataType, std::vector<int>({batch, kvHeadNum, seqLenKV + kvMaxSize, headDim}));
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*valueperm);
            mInputs.push_back(*pastV);
            mOutputs.push_back(*vFuse->getNativeTensor());
            mParams.push_back(*axisParam->getNativeParam());
            mBackend->addNodeToGraph(mOpConfigVersion, (mNodeName + "_Concat_V").c_str(), mPackageName.c_str(), "Concat", mParams, mInputs, mOutputs);
            tempV = *vFuse->getNativeTensor();
            vSeqLen = seqLenKV + kvMaxSize;
        }
        if(isGQA) {
            std::vector<std::shared_ptr<QNNTensorWrapper>> splits(kvHeadNum);
            auto axisParam = this->createParamScalar("axis", (uint32_t)1);
            auto RepeatedValue = this->createStageTensor("RepeatedValue", dataType, std::vector<int>({batch, headNum, vSeqLen, headDim}));
            {
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                std::string name = mNodeName + "_V_Split";
                mNodeType = "Split";

                std::vector<uint32_t> splitIndex(kvHeadNum-1);
                for(int i = 0; i < splitIndex.size(); i++) {
                    splitIndex[i] = i + 1;
                }
                auto split_index = this->createParamTensor("split_index", QNN_DATATYPE_UINT_32, {(uint32_t)kvHeadNum-1}, (void *)splitIndex.data(), "V_Split");
                for(int i = 0; i < kvHeadNum; i++) {
                    auto o = this->createStageTensor("SplitV_Temp" + std::to_string(i), dataType, std::vector<int>({batch, 1, vSeqLen, headDim}));
                    splits[i] = o;
                    mOutputs.push_back(*o->getNativeTensor());
                }

                mInputs.push_back(tempV); // stage value
                mParams.push_back(*(axisParam->getNativeParam())); // axis
                mParams.push_back(*(split_index->getNativeParam())); // split_index
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
                        mInputs.push_back(*(splits[i]->getNativeTensor()));
                    }
                }
                mParams.push_back(*(axisParam->getNativeParam())); // axis
                mOutputs.push_back(*(RepeatedValue->getNativeTensor())); // stage TempKey
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }
            tempV = *(RepeatedValue->getNativeTensor());
        }

        bool transpose0 = false;
        bool transpose1 = false;
        auto tr0 = this->createParamScalar("transpose_in0", transpose0);
        auto tr1 = this->createParamScalar("transpose_in1", transpose1);

        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_MatMul_QKV";
        mNodeType = "MatMul";
        mInputs.push_back(*(Softmax->getNativeTensor())); //Softmax
        mInputs.push_back(tempV); // input2
        mParams.push_back(*(tr0->getNativeParam()));  // transpose0
        mParams.push_back(*(tr1->getNativeParam()));  // transpose1
        mOutputs.push_back(*(QKV->getNativeTensor())); // QKV

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Transpose
    {
        std::string name = mNodeName + "_Transpose";
        mNodeType = "Transpose";
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mInputs.push_back(*(QKV->getNativeTensor())); // QKV
        mParams.push_back(*(output_perm->getNativeParam())); // perm
        mOutputs.push_back(*(Transpose->getNativeTensor())); // Transpose
    
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    // Reshape
    {
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        std::string name = mNodeName + "_Reshape";
        mNodeType = "Reshape";

        mInputs.push_back(*(Transpose->getNativeTensor())); // Transpose
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
