//
//  NPUGatherV2.cpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUGatherV2.hpp"

using namespace std;

namespace MNN {

NPUGatherV2::NPUGatherV2(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {

    auto opName = mOp->name()->str();

    bool isConst0 = TensorUtils::getDescribe(inputs[0])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    bool isConst1 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    if(!isConst0 && isConst1){
        auto input1 = inputs[1];
        // om input weight const op
        mConst = ge::op::Const(opName + "_w_const");
        {
            ge::TensorDesc fdesc(ge::Shape({input1->batch(), input1->channel(), input1->height(), input1->width()}), ge::FORMAT_NCHW, ge::DT_FLOAT); // in o h w ?
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t *)input1->host<float>(), input1->elementSize() * sizeof(float));
            mConst.set_attr_value(filter);
        }

    }else if(isConst0 && !isConst1){
        auto input0 = inputs[0];
        // om input weight const op
        mConst = ge::op::Const(opName + "_w_const");
        {
            ge::TensorDesc fdesc(ge::Shape({input0->batch(), input0->channel(), input0->height(), input0->width()}), ge::FORMAT_NCHW, ge::DT_FLOAT); // in o h w ?
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t *)input0->host<float>(), input0->elementSize() * sizeof(float));
            mConst.set_attr_value(filter);
        } 
    }
}

ErrorCode NPUGatherV2::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto params  = inputs[0];
    auto indices = inputs[1];

    auto opName = mOp->name()->str();
    auto param  = mOp->main_as_GatherV2();

    shared_ptr<ge::op::Gather> prob(new ge::op::Gather(opName));
    vector<pair<shared_ptr<ge::Operator>, string>> ops;

    bool isConst0 = TensorUtils::getDescribe(inputs[0])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    bool isConst1 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    int axis     = 0;
    if (inputs.size() == 3) {
        const Tensor *axisTensor = inputs[2];
        axis                     = axisTensor->host<int32_t>()[0];
    }

    if (axis < 0) {
        axis = params->buffer().dimensions + axis;
    }

    if(!isConst0 && isConst1){
        // 
        auto inputIndex0 = mOp->inputIndexes()->data()[0];
        auto iops0       = mNpuBackend->mGrapMap[inputIndex0]; // x
        auto xOp0        = iops0.back().first;
        (*prob)
            .set_input_indices(*xOp0.get())
            .set_input_params(mConst)
            .set_attr_axis(axis);
    }else if(isConst0 && !isConst1){
        // 
        auto inputIndex1 = mOp->inputIndexes()->data()[1];
        auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; // x
        auto xOp1        = iops1.back().first;

        (*prob)
            .set_input_indices(mConst)
            .set_input_params(*xOp1.get())
            .set_attr_axis(axis);        
    }else{
        auto inputIndex = mOp->inputIndexes()->data()[0];
        auto iops       = mNpuBackend->mGrapMap[inputIndex]; // x
        auto xOp        = iops.back().first;

        auto inputIndex1 = mOp->inputIndexes()->data()[1];
        auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; // x
        auto xOp1        = iops1.back().first;

        (*prob)
            .set_input_indices(*xOp.get())
            .set_input_params(*xOp1.get())
            .set_attr_axis(axis);
    }

    mNpuBackend->setOutputOps(mOp, {prob}, outputs);

    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUGatherV2>> __gatherV2_op(OpType_GatherV2);

} // namespace MNN
