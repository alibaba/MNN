//
//  NPUSoftmax.cpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUSoftmax.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUSoftmax::NPUSoftmax(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {}

ErrorCode NPUSoftmax::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();
    auto param  = mOp->main_as_Axis();

    auto xOp = mNpuBackend->getInputOps(mOp);

    auto shape = tensorShapeFormat(inputs[0]);
    if(shape[1] > 10000 && shape[0] == shape[2] == shape[3] == 1){
        mConstSub = ge::op::Const(opName + "_sub_n");
        {
            ge::TensorDesc fdesc(ge::Shape({1,shape[1],1,1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
            ge::TensorPtr constTensor = std::make_shared<ge::Tensor>();
            constTensor->SetTensorDesc(fdesc);
            vector<float> x(shape[1], 50);
            constTensor->SetData((uint8_t *)(x.data()), x.size()*sizeof(float));
            mConstSub.set_attr_value(constTensor);
        }

        shared_ptr<ge::op::Sub> sub(new ge::op::Sub(opName + "_sub"));
        (*sub).set_input_x1(*xOp.get()).set_input_x2(mConstSub);

        shared_ptr<ge::op::Exp> exp(new ge::op::Exp(opName + "_exp"));
        (*exp).set_input_x(*sub.get());

        mConstAxis = ge::op::Const(opName + "_axis");
        {
            ge::TensorDesc fdesc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
            ge::TensorPtr constTensor = std::make_shared<ge::Tensor>();
            constTensor->SetTensorDesc(fdesc);
            int x = axisFormat(inputs[0], param->axis());
            constTensor->SetData((uint8_t *)(&x), sizeof(int));
            mConstAxis.set_attr_value(constTensor);
        }
        shared_ptr<ge::op::ReduceSum> sum(new ge::op::ReduceSum(opName + "_sum"));
        (*sum).set_input_x(*exp.get()).set_input_w(mConstAxis).set_attr_keep_dims(true);

        shared_ptr<ge::op::Reciprocal> rec(new ge::op::Reciprocal(opName + "_rec"));
        (*rec).set_input_x(*sum.get());

        shared_ptr<ge::op::Mul> mul(new ge::op::Mul(opName + "_mul"));
        (*mul).set_input_x(*exp.get()).set_input_y(*rec.get());

        mNpuBackend->setOutputOps(mOp, {sub, exp, sum, rec, mul}, outputs);

    }else{
        shared_ptr<ge::op::Softmax> softmax(new ge::op::Softmax(opName));

        (*softmax).set_input_x(*xOp.get()).set_attr_axis(axisFormat(inputs[0], param->axis())).set_attr_algo(1);

        mNpuBackend->setOutputOps(mOp, {softmax}, outputs);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUSoftmax>> __softmax_op(OpType_Softmax);

} // namespace MNN
