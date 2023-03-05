//
//  NPUActivation.cpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUActivation.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUActivation::NPUActivation(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, int type) : MNN::NPUCommonExecution(b,op) {
    mType = type;
}


ErrorCode NPUActivation::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();

    

    auto xOp = mNpuBackend->getInputOps(mOp);

    if(mType == 5){
        shared_ptr<hiai::op::PRelu> prelu(new hiai::op::PRelu(opName + "_prelu"));
        auto slopePtr = mOp->main_as_PRelu()->slope()->data();
        auto slopeSize = mOp->main_as_PRelu()->slope()->size();

        mConst_w = ge::op::Const(opName + "_w_const");
        {
            ge::TensorDesc fdesc(ge::Shape({1, slopeSize, 1, 1}), ge::FORMAT_NCHW,
                                ge::DT_FLOAT); // in o h w ?
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t *)slopePtr, slopeSize * sizeof(float));
            mConst_w.set_attr_value(filter);
        }

        (*prelu)
            .set_input_x(*xOp.get()).set_input_weight(mConst_w);
        mNpuBackend->setOutputOps(mOp, {prelu}, outputs);
    }else{
        float slope = 0.0;
        if (mOp->type() == OpType_ReLU) {
            slope = mOp->main_as_Relu()->slope();
            mType = 5;
        }

        shared_ptr<ge::op::Activation> relu(new ge::op::Activation(opName + "_relu"));
        (*relu)
            .set_input_x(*xOp.get())
            .set_attr_coef(.000000)
            .set_attr_negative_slope(slope)
            .set_attr_mode(mType);
        mNpuBackend->setOutputOps(mOp, {relu}, outputs);
    }

    return NO_ERROR;
}

class ActivationCreator : public NPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {

        if (op->type() == OpType_ReLU) {
            return new NPUActivation(backend, op, inputs, outputs, 1);
        }else if (op->type() == OpType_ReLU6) {
            return new NPUActivation(backend, op, inputs, outputs, 14);
        }else if (op->type() == OpType_Sigmoid) {
            return new NPUActivation(backend, op, inputs, outputs, 0);
        }else if (op->type() == OpType_PReLU) {
            return new NPUActivation(backend, op, inputs, outputs, 5);
        }else if (op->type() == OpType_TanH) {
            return new NPUActivation(backend, op, inputs, outputs, 2);
        }else{
            MNN_ERROR("Activation not support this case %d \n", op->type());
            return nullptr;
        }
    }
};

NPUCreatorRegister<ActivationCreator> __relu_op(OpType_ReLU);
NPUCreatorRegister<ActivationCreator> __relu6_op(OpType_ReLU6);
NPUCreatorRegister<ActivationCreator> __sigmoid_op(OpType_Sigmoid);
NPUCreatorRegister<ActivationCreator> __prelu_op(OpType_PReLU);
NPUCreatorRegister<ActivationCreator> __tanh_op(OpType_TanH);

} // namespace MNN