//
//  NPUPadding.cpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUPadding.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUPadding::NPUPadding(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {
    auto opName = mOp->name()->str();
    auto input1 = inputs[1];
    bool isConst1 = TensorUtils::getDescribe(input1)->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    MNN_ASSERT(isConst1 == true);
    auto data = input1->host<int>();
    //MNN_PRINT("Padding input1->buffer().dim[0].extent=%d\n",input1->buffer().dim[0].extent);
    if (input1->buffer().dim[0].extent == 3) {
        mPadData = {0, 0, data[4], data[5], data[0], data[1], data[2], data[3]};
    } else if ((input1->buffer().dim[0].extent == 4) || (input1->buffer().dim[0].extent == 8)) {
        mPadData = {data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]};
    }
    // om input weight const op
    mConst = hiai::op::Const(opName + "_w_const");
    {
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        ge::TensorDesc fdesc(ge::Shape(ge::Shape({4, 2})), ge::FORMAT_NCHW, ge::DT_INT32);
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)mPadData.data(), mPadData.size() * sizeof(int32_t));
        mConst.set_attr_value(filter);
    }
}

ErrorCode NPUPadding::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    
    auto opName = mOp->name()->str();
    auto xOp = mNpuBackend->getInputOps(mOp);
    shared_ptr<hiai::op::Pad> padding(new hiai::op::Pad(opName));
    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops       = mNpuBackend->mGrapMap[inputIndex]; // x
    xOp        = iops.back().first;
    (*padding).set_input_x(*xOp.get()).set_input_paddings(mConst);
    mNpuBackend->setOutputOps(mOp, {padding}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUPadding>> __padding_op(OpType_Padding);

} // namespace MNN
