//
//  NPUArgMax.cpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUArgMax.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUArgMax::NPUArgMax(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {
}

ErrorCode NPUArgMax::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();

    shared_ptr<hiai::op::ArgMaxExt2> argMax(new hiai::op::ArgMaxExt2(opName));

    auto xOp = mNpuBackend->getInputOps(mOp);
    auto argMaxParam = mOp->main_as_ArgMax();

    // om input weight const op
    mConst_axis = hiai::op::Const(opName + "_w_const");
    {
        auto aixs = argMaxParam->axis();
        ge::TensorDesc fdesc(ge::Shape({1}),ge::DT_INT32); 
        ge::TensorPtr axis = std::make_shared<ge::Tensor>();
        axis->SetTensorDesc(fdesc);
        axis->SetData((uint8_t *)&aixs, sizeof(int32_t));
        mConst_axis.set_attr_value(axis);
    }

    (*argMax)
        .set_input_x(*xOp.get())
        .set_input_axis(mConst_axis);
    mNpuBackend->setOutputOps(mOp, {argMax}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUArgMax>> __argmax_op(OpType_ArgMax);

} // namespace MNN