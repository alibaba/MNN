//
//  QNNReduce.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNReduce.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNReduce::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 2 || inputs.size() == 1);
    std::map<ReductionType, std::string> reduceMap {
        {ReductionType_SUM, "ReduceSum"},
        {ReductionType_MEAN, "ReduceMean"},
        {ReductionType_MAXIMUM, "ReduceMax"},
        {ReductionType_MINIMUM, "ReduceMin"},
        {ReductionType_PROD, "ReduceProd"},
    };

    auto param = mOp->main_as_ReductionParam();
    auto operation = param->operation();
    bool keepDims = param->keepDims();
    auto iter = reduceMap.find(operation);
    if (iter == reduceMap.end()) {
        MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
    }
    mNodeType = iter->second;

    std::vector<uint32_t> axesData;
    int inputDim = inputs[0]->dimensions();
    int positiveAxis;
    Tensor::DimensionType inputDimType = inputs[0]->getDimensionType();
    if (inputs.size() == 2) {
        int32_t * reduceAxes = inputs[1]->host<int32_t>();
        for (int i = 0; i < inputs[1]->elementSize(); ++i) {
            positiveAxis = (reduceAxes[i] < 0) ? (inputDim + reduceAxes[i]) : (reduceAxes[i]);
            axesData.push_back((uint32_t) positiveAxis);
        }
    } else {
        MNN_ASSERT(param->dim() != nullptr);
        const int32_t * reduceAxes = param->dim()->data();
        for (int i = 0; i < param->dim()->size(); i++) {
            positiveAxis = (reduceAxes[i] < 0) ? (inputDim + reduceAxes[i]) : (reduceAxes[i]);
            axesData.push_back((uint32_t) positiveAxis);
        }
    }

    // The HTP graph runs in FLOAT16 mode and its fp16 reduce kernel (q::reduce_sum.fp16 etc.) cannot
    // be lowered when the reduction covers the innermost axis (fails at graph_prepare). Note that a
    // plain reshape that appends a trailing singleton does NOT help -- HTP squeezes trailing 1s and
    // still sees an innermost reduction. Instead, when reducing the single last axis, physically
    // transpose it to the second-to-last position (so the innermost axis is a real, non-1 dim), run
    // a native reduce there, then reshape back. This keeps exact reduce semantics for every reduce
    // type (Sum/Mean/Max/Min/Prod). Reductions over non-innermost axes already lower fine and take
    // the normal path below.
    bool singleLastAxis = (axesData.size() == 1 && (int)axesData[0] == inputDim - 1);
    // The transpose trick needs a real second-to-last axis; a rank < 2 tensor has none. Check at
    // runtime (MNN_ASSERT is stripped in release builds) and fall through to the normal path when
    // rank < 2 -- otherwise perm[inRank-2] / tDims[inRank-2] would underflow (uint32) and corrupt memory.
    if (singleLastAxis && mBackend->getNativeTensor(inputs[0])->v1.rank >= 2) {
        Qnn_Tensor_t* inNative = mBackend->getNativeTensor(inputs[0]);
        Qnn_Tensor_t* outNative = mBackend->getNativeTensor(outputs[0]);
        uint32_t inRank = inNative->v1.rank;
        uint32_t* inDims = inNative->v1.dimensions;
        Qnn_DataType_t dtype = inNative->v1.dataType;

        // Transpose the reduced (last) axis to the second-to-last position, so the innermost axis is
        // a real (non-1) dim: [.., M, K] -> [.., K, M].
        std::vector<uint32_t> perm(inRank);
        for (uint32_t i = 0; i < inRank; i++)
            perm[i] = i;
        std::swap(perm[inRank - 1], perm[inRank - 2]);

        std::vector<uint32_t> tDims(inDims, inDims + inRank);
        std::swap(tDims[inRank - 1], tDims[inRank - 2]); // [.., K, M]
        auto tStage = this->createStageTensor("reduce_tin", dtype, tDims);

        std::vector<uint32_t> rDims = tDims;
        rDims[inRank - 2] = 1; // [.., 1, M]
        auto rStage = this->createStageTensor("reduce_tout", dtype, rDims);

        auto permParam = this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {inRank}, perm.data());
        this->addNodeCommonPermute("reduce_transpose_in", *inNative, *(permParam->getNativeParam()),
                                   *(tStage->getNativeTensor()));

        uint32_t reduceAxis = inRank - 2;
        auto axesParam = this->createParamTensor("axes", QNN_DATATYPE_UINT_32, {1}, &reduceAxis);
        auto keepParam = this->createParamScalar("keep_dims", true);
        {
            CLEAR_BEFORE_ADDING_NODE;
            mNodeType = iter->second;
            std::string rName = mNodeName + "_reduce_t";
            mParams.push_back(*(axesParam->getNativeParam()));
            mParams.push_back(*(keepParam->getNativeParam()));
            mInputs.push_back(*(tStage->getNativeTensor()));
            mOutputs.push_back(*(rStage->getNativeTensor()));
            mBackend->addNodeToGraph(mOpConfigVersion, rName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams,
                                     mInputs, mOutputs);
        }
        this->addNodeCommonReshape("reduce_reshape_t", *(rStage->getNativeTensor()), *outNative);
        return NO_ERROR;
    }

    this->createParamTensor("axes", QNN_DATATYPE_UINT_32, {(uint32_t) axesData.size()}, (void *) axesData.data());
    this->createParamScalar("keep_dims", keepDims);

    if (inputs.size() == 2) {
        mParams.push_back(*(mParamTensorWrappers.back()->getNativeParam()));
        mParams.push_back(*(mParamScalarWrappers.back()->getNativeParam()));
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
        mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    } else {
        this->addNodeCommon(inputs, outputs);
    }

    return NO_ERROR;
}


class QNNReduceCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNReduce(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNReduceCreator, OpType_Reduction)
#endif
} // end namespace QNN
} // end namespace MNN
