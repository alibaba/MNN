//
//  QNNBinary.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNBinary.hpp"
#include <algorithm>

namespace MNN {
namespace QNN {

ErrorCode QNNBinary::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    int dim0 = inputs[0]->dimensions();
    int dim1 = inputs[1]->dimensions();
    int minDim = dim0 > dim1 ? dim1 : dim0;
    int fullIndex = dim0 > dim1 ? 0 : 1;

    // Broadcast binary with scalar.
    // By our experiments, this branch is faster than using Qnn binary operations directly, although Qnn binary operations supports scalar broadcasting.
    if(dim0 != dim1 && minDim == 0) {
        return this->onEncodeScalarOptimize(inputs, outputs, fullIndex);
    }

    if (dim0 != dim1 && TensorUtils::getDimType(inputs[0]) != gQnnTensorDimType) {
        return this->onEncodeBroadcast(inputs, outputs, fullIndex);
    }

    mNodeType = mBinaryTypeName;
    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}

ErrorCode QNNBinary::onEncodeScalarOptimize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, int fullIndex) {
    std::vector<uint32_t> shape = getNHWCShape(inputs[fullIndex]);
    int idleIndex = 1 - fullIndex;
    Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[fullIndex])->v1.dataType;

    std::vector<uint32_t> dim(inputs[fullIndex]->dimensions(), 1);

    this->createStageTensor("stage_0", dataType, dim); // mTempTensorWrappers[0]
    this->createStageTensor("stage_1", dataType, shape); // mTempTensorWrappers[1]

    this->createParamTensor("multiples", QNN_DATATYPE_UINT_32, {(uint32_t)shape.size()}, shape.data()); // mParamTensorWrappers[0]

    // Reshape
    {
        CLEAR_BEFORE_ADDING_NODE;

        std::string name = mNodeName + "_Reshape";
        mNodeType = "Reshape";

        mInputs.push_back(*(mBackend->getNativeTensor(inputs[idleIndex]))); // idle input
        mOutputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // stage 0

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Tile
    {
        CLEAR_BEFORE_ADDING_NODE;

        std::string name = mNodeName + "_Tile";
        mNodeType = "Tile";

        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // stage 0
        mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // multiples
        mOutputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // stage 1

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Binary
    {
        CLEAR_BEFORE_ADDING_NODE;

        mNodeType = mBinaryTypeName;

        mInputs.push_back(*(mBackend->getNativeTensor(inputs[fullIndex]))); // full input
        mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // stage 1
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));

        mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    return NO_ERROR;
}

ErrorCode QNNBinary::onEncodeBroadcast(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, int fullIndex) {
    // Create resources.
    int idleIndex = 1 - fullIndex;
    int fullDim = inputs[fullIndex]->dimensions();
    int idleDim = inputs[idleIndex]->dimensions();
    std::vector<uint32_t> idleNHWCShape = getNHWCShape(inputs[idleIndex]);
    int offset = fullDim - idleDim;
    Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[fullIndex])->v1.dataType;

    std::vector<uint32_t> order(idleDim);
    for (int i = 0; i < order.size(); i++) {
        order[i] = (uint32_t) getNHWCAxis(getNCHWAxis(i, idleDim, Tensor::TENSORFLOW) + offset, fullDim, Tensor::CAFFE);
    }

    std::vector<uint32_t> permData(idleDim);
    for (int i = 0; i < idleDim; i++) {permData[i] = i;}
    std::sort(permData.begin(), permData.end(), [&order](uint32_t a, uint32_t b) {return order[a] < order[b];});
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) idleDim}, (void *) permData.data()); // mParamTensorWrappers[0]

    std::vector<uint32_t> shapeStagePerm(idleDim);
    for (int i = 0; i < idleDim; i++) {shapeStagePerm[i] = idleNHWCShape[permData[i]];}
    this->createStageTensor("stage_perm", dataType, shapeStagePerm); // mTempTensorWrappers[0]

    std::vector<uint32_t> shapeStageReshape(fullDim, 1);
    for (int i = 0; i < idleDim; i++) {shapeStageReshape[order[i]] = idleNHWCShape[i];}
    this->createStageTensor("stage_reshape", dataType, shapeStageReshape); // mTempTensorWrappers[1]

    // Permute.
    this->addNodeCommonPermute("Permute",
                               *(mBackend->getNativeTensor(inputs[idleIndex])),
                               *(mParamTensorWrappers[0]->getNativeParam()),
                               *(mTempTensorWrappers[0]->getNativeTensor()));

    // Reshape.
    this->addNodeCommonReshape("Reshape",
                               *(mTempTensorWrappers[0]->getNativeTensor()),
                               *(mTempTensorWrappers[1]->getNativeTensor()));

    // Binary broadcast.
    {
        CLEAR_BEFORE_ADDING_NODE;

        mNodeType = mBinaryTypeName;
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[fullIndex])));
        mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
        mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    return NO_ERROR;
}

class QNNBinaryCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        MNN_ASSERT(inputs.size() == 2 && outputs.size() == 1);

        std::map<BinaryOpOperation, std::string> binaryMap {
            {BinaryOpOperation_ADD, "ElementWiseAdd"},
            {BinaryOpOperation_SUB, "ElementWiseSubtract"},
            {BinaryOpOperation_MUL, "ElementWiseMultiply"},
            {BinaryOpOperation_DIV, "ElementWiseDivide"},
            {BinaryOpOperation_POW, "ElementWisePower"},
            {BinaryOpOperation_REALDIV, "ElementWiseDivide"},
            {BinaryOpOperation_MINIMUM, "ElementWiseMinimum"},
            {BinaryOpOperation_MAXIMUM, "ElementWiseMaximum"}
            // {BinaryOpOperation_GREATER, ""},
            // {BinaryOpOperation_GREATER_EQUAL, ""},
            // {BinaryOpOperation_LESS, ""},
            // {BinaryOpOperation_FLOORDIV, ""},
            // {BinaryOpOperation_SquaredDifference, ""},
            // {BinaryOpOperation_LESS_EQUAL, ""},
            // {BinaryOpOperation_FLOORMOD, ""},
            // {BinaryOpOperation_EQUAL, ""},
            // {BinaryOpOperation_MOD, ""},
            // {BinaryOpOperation_ATAN2, ""},
            // {BinaryOpOperation_LOGICALOR, ""},
            // {BinaryOpOperation_NOTEQUAL, ""},
            // {BinaryOpOperation_BITWISE_AND, ""},
            // {BinaryOpOperation_BITWISE_OR, ""},
            // {BinaryOpOperation_BITWISE_XOR, ""},
            // {BinaryOpOperation_LOGICALXOR, ""},
            // {BinaryOpOperation_LEFTSHIFT, ""},
            // {BinaryOpOperation_RIGHTSHIFT, ""}
        };

        std::map<EltwiseType, std::string> eltwiseMap {
            {EltwiseType_PROD, "ElementWiseMultiply"},
            {EltwiseType_SUM, "ElementWiseAdd"},
            {EltwiseType_SUB, "ElementWiseSubtract"},
            {EltwiseType_MAXIMUM, "ElementWiseMaximum"}
        };

        std::string binaryTypeName;
        if (op->type() == OpType_BinaryOp) {
            auto iter = binaryMap.find(static_cast<BinaryOpOperation>(op->main_as_BinaryOp()->opType()));
            if (iter == binaryMap.end()) {
                MNN_ERROR("MNN_QNN: Not supported Binary type.\n");
                return nullptr;
            }
            binaryTypeName = iter->second;
        } else {
            auto iter = eltwiseMap.find(op->main_as_Eltwise()->type());
            if (iter == eltwiseMap.end()) {
                MNN_ERROR("MNN_QNN: Not supported Eltwise type.\n");
                return nullptr;
            }
            binaryTypeName = iter->second;
        }

        return new QNNBinary(backend, op, binaryTypeName);
    }
};

REGISTER_QNN_OP_CREATOR(QNNBinaryCreator, OpType_BinaryOp)
REGISTER_QNN_OP_CREATOR(QNNBinaryCreator, OpType_Eltwise)

} // end namespace QNN
} // end namespace MNN
