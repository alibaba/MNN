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

vector<int64_t> NPUReduction::convertAxis(vector<int64_t> origAxis, Tensor * input)
{
    vector<int64_t> newAxis(origAxis.size(),0);
    int step = TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NCHW ? 0 : 1; 
    int index = step + (input->buffer().dimensions-1)*2;
    for (size_t i = 0; i < origAxis.size(); i++) {
        newAxis[i] = axisMap[index][origAxis[i]];
        MNN_PRINT("i = %d, newAxis[i] = %ld\n",i,newAxis[i]);
    }
    return newAxis;
}

ErrorCode NPUReduction::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();

    auto type = mOp->main_as_ReductionParam()->operation();

    auto xOp = mNpuBackend->getInputOps(mOp);

    vector<int64_t> origAxis;
    vector<int64_t> axis;
    auto reduct = mOp->main_as_ReductionParam();
    if (nullptr != reduct->dim()) {
        for (int i = 0; i < reduct->dim()->size(); ++i) {
            origAxis.push_back(reduct->dim()->data()[i]);
        }
    }else if(inputs.size() == 2){
        for (int i = 0; i < inputs[1]->length(0);++i) {
            int32_t *reduce_dim = inputs[1]->host<int32_t>();
            origAxis.push_back(reduce_dim[i]);
        }
    }else{
        MNN_ASSERT(false);
    }

    axis = convertAxis(origAxis,inputs[0]);

    mConstAxis = ge::op::Const(opName + "_axis");
    {
        ge::TensorDesc fdesc(ge::Shape({static_cast<long>(axis.size())}), ge::FORMAT_ND, ge::DT_INT32);
        ge::TensorPtr constTensor = std::make_shared<ge::Tensor>();
        constTensor->SetTensorDesc(fdesc);
        constTensor->SetData((uint8_t *)(axis.data()), axis.size()*sizeof(float));
        mConstAxis.set_attr_value(constTensor);
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
            auto  shapeDims = tensorShapeFormat(outputs[0]);
            shared_ptr<ge::op::Reshape> reshape1(new ge::op::Reshape(opName+"reshape1"));
            (*reshape1).set_input_tensor(*reduction.get()).set_attr_shape(ge::AttrValue::LIST_INT(shapeDims));
            mNpuBackend->setOutputOps(mOp, {reduction,reshape1}, outputs);
        } else {
            mNpuBackend->setOutputOps(mOp, {reduction}, outputs);
        }
    } else if(type == ReductionType_ANY) {
        shared_ptr<ge::op::ReduceAll> reduction(new ge::op::ReduceAll(opName));
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