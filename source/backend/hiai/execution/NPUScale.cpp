//
//  NPUScale.cpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUScale.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUScale::NPUScale(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {}

ErrorCode NPUScale::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();

    auto param     = mOp->main_as_Scale();
    auto scaleData = param->scaleData();
    auto biasData  = param->biasData();

    shared_ptr<ge::op::Scale> scale(new ge::op::Scale(opName + "_scale"));

    auto xOp = mNpuBackend->getInputOps(mOp);

    // om input filter const op
    mConst_fliter = ge::op::Const(opName + "_filter_const");
    {
        ge::TensorDesc fdesc(ge::Shape({1, scaleData->size(), 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT); // in o h w ?
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)scaleData->data(), scaleData->size() * sizeof(float));

        mConst_fliter.set_attr_value(filter);
    }
    // om input bias const op
    mConst_bias = ge::op::Const(opName + "_bias_const");
    {
        ge::TensorDesc fdesc(ge::Shape({1, biasData->size(), 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)biasData->data(), biasData->size() * sizeof(float));

        mConst_bias.set_attr_value(filter);
    }

    (*scale).set_input_x(*xOp.get()).set_input_filter(mConst_fliter).set_input_bias(mConst_bias).set_attr_has_bias_value(true);

    mNpuBackend->setOutputOps(mOp, {scale}, outputs);

    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUScale>> __scale_op(OpType_Scale);

} // namespace MNN