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

    if (isConst0 && !isConst1) {
        auto input = inputs[0];
        // om input weight const op
        mConst = hiai::op::Const(opName + "_x_const");
        vector<int64_t> dims;
        for (int32_t i = 0; i < input->buffer().dimensions; i++) {
            dims.push_back(input->buffer().dim[i].extent);
        }
        ge::TensorDesc fdesc(ge::Shape(dims), ge::FORMAT_NCHW, ge::DT_FLOAT); // in o h w ?
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        if (input->getType().code == halide_type_int && input->getType().bits == 32) {
            fdesc.SetDataType(ge::DT_INT32);
            filter->SetData((uint8_t *)input->host<int32_t>(), input->elementSize() * sizeof(int32_t));
        } else {
            filter->SetData((uint8_t *)input->host<float>(), input->elementSize() * sizeof(float));
        }
        filter->SetTensorDesc(fdesc);
        mConst.set_attr_value(filter);
    } else if (!isConst0 && isConst1) {
        auto input = inputs[1];
        // om input weight const op
        vector<int64_t> dims;
        for (int32_t i = 0; i < input->buffer().dimensions; i++) {
            dims.push_back(input->buffer().dim[i].extent);
        }
        mConst = hiai::op::Const(opName + "_i_const");
        ge::TensorDesc fdesc(ge::Shape(dims), ge::FORMAT_NCHW, ge::DT_INT32); // in o h w ?
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)input->host<int32_t>(), input->elementSize() * sizeof(int32_t));
        mConst.set_attr_value(filter);
    }
}

ErrorCode NPUGatherV2::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto params  = inputs[0];
    auto indices = inputs[1];

    auto opName = mOp->name()->str();
    auto param  = mOp->main_as_GatherV2();

    shared_ptr<hiai::op::GatherV2D> prob(new hiai::op::GatherV2D(opName));
    shared_ptr<hiai::op::CastT> castOp(new hiai::op::CastT(opName + "_cast"));
    bool isConst0 = TensorUtils::getDescribe(inputs[0])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    bool isConst1 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    bool isConst2 = TensorUtils::getDescribe(inputs[2])->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    int axis     = 0;
    if (isConst2 && inputs.size() == 3) {
        const Tensor *axisTensor = inputs[2];
        axis                     = axisTensor->host<int32_t>()[0];
    }
    if (axis < 0) {
        axis = params->buffer().dimensions + axis;
    }
    auto xOp = mNpuBackend->getInputOps(mOp);
    if (!isConst0 && isConst1) {
        auto inputIndex0 = mOp->inputIndexes()->data()[0];
        auto iops0       = mNpuBackend->mGrapMap[inputIndex0]; // x
        auto xOp0        = iops0.back().first;
        (*prob)
            .set_input_x(*xOp0.get())
            .set_input_indices(mConst)
            .set_attr_axis(axis);
        mNpuBackend->setOutputOps(mOp, {prob}, outputs);
    } else if (isConst0 && !isConst1){
        auto inputIndex1 = mOp->inputIndexes()->data()[1];
        auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; // x
        auto xOp1        = iops1.back().first;
        (*castOp).set_input_x(*xOp1.get()).set_attr_dst_dtype(ge::DataType::DT_INT32);
        (*prob)
            .set_input_x(mConst)
            .set_input_indices(*castOp.get())
            .set_attr_axis(axis);
        mNpuBackend->setOutputOps(mOp, {castOp, prob}, outputs); 
    } else {
        auto inputIndex = mOp->inputIndexes()->data()[0];
        auto iops       = mNpuBackend->mGrapMap[inputIndex]; // x
        xOp        = iops.back().first;

        auto inputIndex1 = mOp->inputIndexes()->data()[1];
        auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; // x
        auto xOp1        = iops1.back().first;
        (*castOp).set_input_x(*xOp1.get()).set_attr_dst_dtype(ge::DataType::DT_INT32);
        (*prob)
            .set_input_x(*xOp.get())
            .set_input_indices(*castOp.get())
            .set_attr_axis(axis);
        mNpuBackend->setOutputOps(mOp, {castOp, prob}, outputs);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUGatherV2>> __gatherV2_op(OpType_GatherV2);

} // namespace MNN
