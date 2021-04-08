//
//  NPUSqueeze.cpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUSqueeze.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUSqueeze::NPUSqueeze(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {
}

ErrorCode NPUSqueeze::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();

    shared_ptr<ge::op::Reshape> prob(new ge::op::Reshape(opName));

    auto xOp = mNpuBackend->getInputOps(mOp);

    auto shape = tensorShapeFormat(outputs[0]);

    (*prob).set_input_tensor(*xOp.get()).set_attr_shape(ge::AttrValue::LIST_INT(shape));
    
    mNpuBackend->setOutputOps(mOp, {prob}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUSqueeze>> __squeeze_op(OpType_Squeeze);
NPUCreatorRegister<TypedCreator<NPUSqueeze>> __unsqueeze_op(OpType_Unsqueeze);

} // namespace MNN
