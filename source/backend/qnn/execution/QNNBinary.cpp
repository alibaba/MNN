//
//  QNNBinary.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNBinary.hpp"
// #define QNN_VORBOSE
namespace MNN {
namespace QNN {

ErrorCode QNNBinary::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    std::map<BinaryOpOperation, std::string> binaryMap {
        {BinaryOpOperation_ADD, "ElementWiseAdd"},
        {BinaryOpOperation_SUB, "ElementWiseSubtract"},
        {BinaryOpOperation_MUL, "ElementWiseMultiply"},
        {BinaryOpOperation_DIV, "ElementWiseDivide"},
        {BinaryOpOperation_POW, "ElementWisePower"},
        {BinaryOpOperation_REALDIV, "ElementWiseDivide"},
        {BinaryOpOperation_MINIMUM, "ElementWiseMinimum"},
        {BinaryOpOperation_MAXIMUM, "ElementWiseMaximum"},
        {BinaryOpOperation_GREATER, "ElementWiseGreater"},
        {BinaryOpOperation_GREATER_EQUAL, "ElementWiseGreaterEqual"},
        {BinaryOpOperation_LESS, "ElementWiseLess"},
        {BinaryOpOperation_FLOORDIV, ""},
        {BinaryOpOperation_SquaredDifference, ""},
        {BinaryOpOperation_LESS_EQUAL, "ElementWiseLessEqual"},
        {BinaryOpOperation_FLOORMOD, ""},
        {BinaryOpOperation_EQUAL, "ElementWiseEqual"},
        {BinaryOpOperation_MOD, ""},
        {BinaryOpOperation_ATAN2, ""},
        {BinaryOpOperation_LOGICALOR, "ElementWiseOr"},
        {BinaryOpOperation_NOTEQUAL, "ElementWiseNotEqual"},
        {BinaryOpOperation_BITWISE_AND, ""},
        {BinaryOpOperation_BITWISE_OR, ""},
        {BinaryOpOperation_BITWISE_XOR, ""},
        {BinaryOpOperation_LOGICALXOR, ""},
        {BinaryOpOperation_LEFTSHIFT, ""},
        {BinaryOpOperation_RIGHTSHIFT, ""}
    };

    BinaryOpOperation binaryType;
    if (mOp->type() == OpType_BinaryOp) {
        binaryType = static_cast<BinaryOpOperation>(mOp->main_as_BinaryOp()->opType());
    } else {
        auto elewiseType = mOp->main_as_Eltwise()->type();
        switch (elewiseType) {
            case EltwiseType_PROD:
                binaryType = BinaryOpOperation_MUL;
                break;
            case EltwiseType_SUM:
                binaryType = BinaryOpOperation_ADD;
                break;
            case EltwiseType_SUB:
                binaryType = BinaryOpOperation_SUB;
                break;
            case EltwiseType_MAXIMUM:
                binaryType = BinaryOpOperation_MAXIMUM;
                break;
            default:
                MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
        }
    }
    auto iter = binaryMap.find(binaryType);
    if (iter == binaryMap.end() || iter->second.empty()) {
        MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
    }
    std::string binaryTypeName = iter->second;

    int dim0 = inputs[0]->dimensions();
    int dim1 = inputs[1]->dimensions();
    int minDim = dim0 > dim1 ? dim1 : dim0;

    #ifdef QNN_VORBOSE
    MNN_PRINT("QNN Binary dim0:%d dim1:%d, \ninput0:", dim0, dim1);
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

    // broadcast binary with scalar
    if(dim0 != dim1 && minDim == 0) {
        std::vector<uint32_t> shape;
        int fullIndex = 0;
        if(dim0 > dim1) {
            shape = getNHWCShape(inputs[0]);
        } else {
            shape = getNHWCShape(inputs[1]);
            fullIndex = 1;
        }

        int idleIndex = 1 - fullIndex;
        Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[fullIndex])->v1.dataType;
        std::vector<uint32_t> dim(inputs[fullIndex]->dimensions(), 1);

        this->createStageTensor("stage_0", dataType, dim); // mTempTensorWrappers[0]
        this->createStageTensor("stage_1", dataType, shape); // mTempTensorWrappers[1]

        // Reshape
        {
            std::string name = mNodeName + "_Reshape";
            mNodeType = "Reshape";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[idleIndex]))); // input
            mOutputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // stage 0
    
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }

        // Tile
        {
            this->createParamTensor("multiples", QNN_DATATYPE_UINT_32, {(uint32_t)shape.size()}, shape.data());
            std::string name = mNodeName + "_Tile";
            mNodeType = "Tile";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // stage 0
            mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // multiples
            mOutputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // stage 0
    
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        // binary
        {
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mNodeType = binaryTypeName;

            if(idleIndex == 0) {
                mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));
                mInputs.push_back(*(mBackend->getNativeTensor(inputs[1]))); // input0
            } else {
                mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // input0
                mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));
            }
            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));

            mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        return NO_ERROR;
    }

    if (dim0 != dim1 && TensorUtils::getDimType(inputs[0]) != gQnnTensorDimType) {
        return this->onEncodeBroadcast(inputs, outputs, binaryTypeName);
    }

    mNodeType = binaryTypeName;
    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}

ErrorCode QNNBinary::onEncodeBroadcast(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const std::string & binaryTypeName) {
    // Create resources.
    int dim0 = inputs[0]->dimensions();
    int dim1 = inputs[1]->dimensions();
    int fullIndex = dim0 > dim1 ? 0 : 1;

    int idleIndex = 1 - fullIndex;

    int minDim = inputs[idleIndex]->dimensions();
    bool singleTranspose = minDim <= 2;

    if(!singleTranspose) {
        std::vector<int> oriShape = inputs[idleIndex]->shape();
        std::vector<uint32_t> permShape = getNHWCShape(inputs[idleIndex]);
        if(oriShape.size() == permShape.size()) {
            singleTranspose = true;
            for(int i = 0; i < oriShape.size(); i++) {
                if(oriShape[i] != permShape[i]) {
                    singleTranspose = false;
                    break;
                }
            }
        }
    }
    if(!singleTranspose) {
        MNN_ERROR("MNN QNN not support binary broadcast with dim0:%d dim1:%d\n", dim0, dim1);
        return NOT_SUPPORT;
    }

    Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[fullIndex])->v1.dataType;
    {
        int dim = inputs[fullIndex]->dimensions();
        std::vector<int> shapeTempInput = inputs[fullIndex]->shape();
        std::vector<int> shapeTempOutput = outputs[0]->shape();
        std::vector<uint32_t> permBeforeData(dim, 0);
        std::vector<uint32_t> permAfterData(dim, 0);

        for (int i = 0; i < dim; i++) {
            permBeforeData[i] = getNHWCAxis(i, dim, Tensor::DimensionType::CAFFE);
            permAfterData[i] = getNCHWAxis(i, dim, Tensor::DimensionType::TENSORFLOW);
        }
        #ifdef QNN_VORBOSE
        MNN_PRINT("QNN Binary perm_data:");
        for(int i = 0; i < permBeforeData.size(); i++) {
            MNN_PRINT("%d ", permBeforeData[i]);
        }
        MNN_PRINT(",");
        for(int i = 0; i < permAfterData.size(); i++) {
            MNN_PRINT("%d ", permAfterData[i]);
        }
        MNN_PRINT("\n");
        #endif
        this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) dim}, (void *) permBeforeData.data(), "before"); // mParamTensorWrappers[0]
        this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) dim}, (void *) permAfterData.data(), "after"); // mParamTensorWrappers[1]

        this->createStageTensor("tempInput", dataType, shapeTempOutput); // mTempTensorWrappers[0]
        this->createStageTensor("tempOutput", dataType, shapeTempOutput); // mTempTensorWrappers[1]
    }

    // Add nodes.
    // Permute before.
    {
        std::string name = mNodeName + "_perm_before";
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "Transpose";

        mInputs.push_back(*(mBackend->getNativeTensor(inputs[fullIndex]))); // input0
        mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // perm before
        mOutputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // temp input
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }



    bool idleNeedReshape = minDim > 2;
    if(idleNeedReshape) {
        // need reshape
        std::vector<int> shapeTempInput = inputs[idleIndex]->shape();
        this->createStageTensor("tempIdleInput", dataType, shapeTempInput); // mTempTensorWrappers[2]

        std::string name = mNodeName + "_idle_reshape";
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "Reshape";

        mInputs.push_back(*(mBackend->getNativeTensor(inputs[idleIndex]))); // idle input
        mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // temp input
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }


    // Binary broadcast.
    {
        Qnn_Tensor_t idleTensor;
        if(idleNeedReshape) {
            idleTensor = *(mTempTensorWrappers[2]->getNativeTensor());
        } else {
            idleTensor = *(mBackend->getNativeTensor(inputs[idleIndex]));
        }
        std::string name = mNodeName + "_binary_broadcast";
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = binaryTypeName;

        if(idleIndex == 1) {
            mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // temp input
            mInputs.push_back(idleTensor); // input1
        } else {
            mInputs.push_back(idleTensor); // input1
            mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // temp input
        }

        mOutputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // temp output
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Permute after.
    {
        std::string name = mNodeName + "_perm_after";
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "Transpose";

        mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // temp output
        mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // perm after
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // output0
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    return NO_ERROR;
}

class QNNBinaryCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        MNN_ASSERT(inputs.size() == 2 && outputs.size() == 1);
        
        if (op->type() == OpType_BinaryOp) {
            std::set<BinaryOpOperation> supportedBinaryTypes = {
                BinaryOpOperation_ADD,
                BinaryOpOperation_SUB,
                BinaryOpOperation_MUL,
                BinaryOpOperation_DIV,
                BinaryOpOperation_POW,
                BinaryOpOperation_REALDIV,
                BinaryOpOperation_MINIMUM,
                BinaryOpOperation_MAXIMUM,
                BinaryOpOperation_LOGICALOR,
                // BinaryOpOperation_GREATER,
                // BinaryOpOperation_GREATER_EQUAL,
                // BinaryOpOperation_LESS,
                // BinaryOpOperation_LESS_EQUAL,
                // BinaryOpOperation_EQUAL,
                // BinaryOpOperation_NOTEQUAL
            };
            if (supportedBinaryTypes.find(static_cast<BinaryOpOperation>(op->main_as_BinaryOp()->opType())) == supportedBinaryTypes.end()) {
                return nullptr;
            }
        }
        
        #ifdef QNN_VORBOSE
        auto input0 = inputs[0];
        auto input1 = inputs[1];
        uint32_t dimension0 = input0->dimensions();
        uint32_t dimension1 = input1->dimensions();
        MNN_PRINT("Info MNN QNN Broadcast binary dim0:%d dim1:%d\n", dimension0, dimension1);
        #endif
        
        return new QNNBinary(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNBinaryCreator, OpType_BinaryOp)
REGISTER_QNN_OP_CREATOR(QNNBinaryCreator, OpType_Eltwise)

} // end namespace QNN
} // end namespace MNN
