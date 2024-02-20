//
//  NPUReduction.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUReduction.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUReduction::NPUReduction(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {
}

ErrorCode NPUReduction::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();

    auto type = mOp->main_as_ReductionParam()->operation();

    auto xOp = mNpuBackend->getInputOps(mOp);

    vector<int32_t> origAxis;
    auto reduct = mOp->main_as_ReductionParam();

    if (inputs.size() >= 2) {
        for (int i = 0; i < inputs[1]->elementSize(); ++i) {
            int32_t *reduce_dim = inputs[1]->host<int32_t>();
            origAxis.push_back(reduce_dim[i]);
        }
    } else if (nullptr != reduct->dim()) {
        for (int i = 0; i < reduct->dim()->size(); ++i) {
            origAxis.push_back(reduct->dim()->data()[i]);
        }
    } else {
        MNN_ASSERT(false);
    }
    mConstAxis = hiai::op::Const(opName + "_axis");
    {
        ge::TensorDesc fdesc(ge::Shape({static_cast<long>(origAxis.size())}), ge::FORMAT_ND, ge::DT_INT32);
        ge::TensorPtr constTensor = std::make_shared<ge::Tensor>();
        constTensor->SetTensorDesc(fdesc);
        constTensor->SetData((uint8_t *)(origAxis.data()), origAxis.size()*sizeof(int32_t));
        mConstAxis.set_attr_value(constTensor);
    }
    vector<int64_t> dims;
    for (int32_t i = 0; i < outputs[0]->buffer().dimensions; i++) {
        dims.push_back(outputs[0]->buffer().dim[i].extent);
    } 
    shapeConst = hiai::op::Const(opName + "_shape_const");
    {
        ge::TensorDesc fdesc(ge::Shape({static_cast<int64_t>(dims.size())}), 
            ge::FORMAT_NCHW,  ge::DT_INT32);
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)dims.data(), dims.size() * sizeof(int32_t));
        shapeConst.set_attr_value(filter);
    }

    if(type == ReductionType_MAXIMUM) {
        shared_ptr<hiai::op::ReduceMax> reduction(new hiai::op::ReduceMax(opName));
        (*reduction)
            .set_input_x(*xOp.get()).set_input_axes(mConstAxis)
            .set_attr_keep_dims(mOp->main_as_ReductionParam()->keepDims());
        mNpuBackend->setOutputOps(mOp, {reduction}, outputs);
    }else if(type == ReductionType_SUM) {
        shared_ptr<hiai::op::ReduceSum> reduction(new hiai::op::ReduceSum(opName));
        (*reduction)
            .set_input_x(*xOp.get()).set_input_axes(mConstAxis)
            .set_attr_keep_dims(mOp->main_as_ReductionParam()->keepDims());
        mNpuBackend->setOutputOps(mOp, {reduction}, outputs);
    }else if(type == ReductionType_MEAN) {
        shared_ptr<hiai::op::ReduceMean> reduction(new hiai::op::ReduceMean(opName));
        (*reduction)
            .set_input_x(*xOp.get()).set_input_axes(mConstAxis)
            .set_attr_keep_dims(reduct->keepDims());
        if(reduct->keepDims() == false) {
            shared_ptr<hiai::op::Reshape> reshape1(new hiai::op::Reshape(opName+"reshape1"));
            (*reshape1).set_input_x(*reduction.get()).set_input_shape(shapeConst);
            mNpuBackend->setOutputOps(mOp, {reduction,reshape1}, outputs);
        } else {
            mNpuBackend->setOutputOps(mOp, {reduction}, outputs);
        }
    } else if(type == ReductionType_ANY) {
        shared_ptr<ge::op::ReduceAll> reduction(new ge::op::ReduceAll(opName));
        vector<int64_t> axis;
        for (int32_t j = 0; j < origAxis.size(); j++) {
            axis.push_back(static_cast<int64_t>(origAxis[j]));
        }
        (*reduction)
            .set_input_x(*xOp.get()).set_attr_axes(axis)
            .set_attr_keep_dims(mOp->main_as_ReductionParam()->keepDims());
        mNpuBackend->setOutputOps(mOp, {reduction}, outputs);
    }else{
        MNN_ERROR("npu reducton not support type : %d \n", type);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUReduction>> __reduction_op(OpType_Reduction);

} // namespace MNN