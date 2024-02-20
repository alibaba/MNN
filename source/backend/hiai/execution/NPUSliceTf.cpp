//
//  NPUSliceTf.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUSliceTf.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUSliceTf::NPUSliceTf(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {
    auto opName = mOp->name()->str();

    bool isConst = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT && TensorUtils::getDescribe(inputs[2])->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    if(!isConst){
        MNN_ERROR("slice tf not support input != const now !!! \n");
    }

    mConst_start = hiai::op::Const(opName + "_start_const");
    {
        auto input1 = inputs[1];
        ge::TensorDesc fdesc(ge::Shape({input1->elementSize()}), ge::FORMAT_NCHW, ge::DT_INT32); // in o h w ?
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)input1->host<int32_t>(), input1->elementSize() * sizeof(int32_t));
        mConst_start.set_attr_value(filter);
    }

    mConst_size = hiai::op::Const(opName + "_size_const");
    {
        auto input1 = inputs[2];
        ge::TensorDesc fdesc(ge::Shape({input1->elementSize()}), ge::FORMAT_NCHW, ge::DT_INT32); // in o h w ?
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)input1->host<int32_t>(), input1->elementSize() * sizeof(int32_t));
        mConst_size.set_attr_value(filter);
    }
}

ErrorCode NPUSliceTf::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();
    shared_ptr<hiai::op::Slice> slice(new hiai::op::Slice(opName));
    auto xOp = mNpuBackend->getInputOps(mOp);

    (*slice).set_input_x(*xOp)
            .set_input_offsets(mConst_start)
            .set_input_size(mConst_size);
    mNpuBackend->setOutputOps(mOp, {slice}, outputs);

    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUSliceTf>> __slicetf_op(OpType_SliceTf);

} // namespace MNN