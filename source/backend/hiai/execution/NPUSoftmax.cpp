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
    shared_ptr<hiai::op::Softmax> softmax(new hiai::op::Softmax(opName));

    (*softmax).set_input_x(*xOp.get()).set_attr_axis(param->axis());

    mNpuBackend->setOutputOps(mOp, {softmax}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUSoftmax>> __softmax_op(OpType_Softmax);

} // namespace MNN
