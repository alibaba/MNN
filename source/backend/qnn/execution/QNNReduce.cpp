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
    Tensor::DimensionType inputDimType = inputs[0]->getDimensionType();
    if (inputs.size() == 2) {
        int32_t * reduceAxes = inputs[1]->host<int32_t>();
        for (int i = 0; i < inputs[1]->elementSize(); ++i) {
            axesData.push_back((uint32_t) getNHWCAxis(reduceAxes[i], inputDim, inputDimType));
        }
    } else {
        MNN_ASSERT(param->dim() != nullptr);
        const int32_t * reduceAxes = param->dim()->data();
        for (int i = 0; i < param->dim()->size(); i++) {
            axesData.push_back((uint32_t) getNHWCAxis(reduceAxes[i], inputDim, inputDimType));
        }
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

} // end namespace QNN
} // end namespace MNN
