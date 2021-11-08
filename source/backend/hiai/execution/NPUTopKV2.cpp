//
//  NPUTopKV2.cpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUTopKV2.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUTopKV2::NPUTopKV2(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {}

ErrorCode NPUTopKV2::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();
    auto param  = mOp->main_as_Axis();

    shared_ptr<ge::op::TopK> prob(new ge::op::TopK(opName));

    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops       = mNpuBackend->mGrapMap[inputIndex]; // x
    auto xOp        = iops.back().first;

    mConst_w = ge::op::Const(opName + "_w_const");
    {
        ge::TensorDesc fdesc(ge::Shape({1, 1, 1, 1}), ge::FORMAT_NCHW,
                             ge::DT_FLOAT); // in o h w ?
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)inputs[1]->host<float>(), sizeof(float));

        mConst_w.set_attr_value(filter);
    }

    (*prob)
        .set_input_x(*xOp.get())
        .set_input_k(mConst_w);

    mNpuBackend->setOutputOps(mOp, {prob}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUTopKV2>> __topkv2_op(OpType_TopKV2);

} // namespace MNN
