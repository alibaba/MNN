//
//  NPUInstanceNorm.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUInstanceNorm.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUInstanceNorm::NPUInstanceNorm(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {}

ErrorCode NPUInstanceNorm::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto xOp = mNpuBackend->getInputOps(mOp);
    auto opName = mOp->name()->str();

    auto slope = mOp->main_as_BatchNorm()->slopeData();
    mScale = hiai::op::Const(opName + "_scale");
    {
        ge::TensorDesc fdesc(ge::Shape({1,slope->size(),1,1}),ge::DT_FLOAT);
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)slope->data(), slope->size() * sizeof(float));
        mScale.set_attr_value(filter);
    }

    auto bias = mOp->main_as_BatchNorm()->biasData();
    mBias = hiai::op::Const(opName + "_bias");
    {
        ge::TensorDesc fdesc(ge::Shape({1,bias->size(),1,1}),ge::DT_FLOAT); 
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)bias->data(), bias->size() * sizeof(float));
        mBias.set_attr_value(filter);
    }
    shared_ptr<hiai::op::InstanceNorm> insNorm(new hiai::op::InstanceNorm(opName));
    (*insNorm).set_input_x(*xOp.get())
              .set_input_gamma(mScale)
              .set_input_beta(mBias);

    mNpuBackend->setOutputOps(mOp, {insNorm}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUInstanceNorm>> __instanceNorm_op(OpType_InstanceNorm);

} // namespace MNN