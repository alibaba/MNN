//
//  NPUExpandDims.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUExpandDims.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUExpandDims::NPUExpandDims(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {
}

ErrorCode NPUExpandDims::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    
    auto opName = mOp->name()->str();
    auto xOp = mNpuBackend->getInputOps(mOp);

    shared_ptr<ge::op::Reshape> prob(new ge::op::Reshape(opName));

    auto output = outputs[0];

    auto shape = tensorShapeFormat(outputs[0]);
    
    (*prob).set_input_tensor(*xOp.get()).set_attr_shape(ge::AttrValue::LIST_INT(shape));
    
    mNpuBackend->setOutputOps(mOp, {prob}, outputs);

    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUExpandDims>> __expand_dims_op(OpType_ExpandDims);

} // namespace MNN