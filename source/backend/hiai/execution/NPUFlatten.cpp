//
//  NPUFlatten.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUFlatten.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUFlatten::NPUFlatten(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {
}

ErrorCode NPUFlatten::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    
    auto opName = mOp->name()->str();
    auto xOp = mNpuBackend->getInputOps(mOp);
    
    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops = mNpuBackend->mGrapMap[inputIndex]; // x
    xOp = iops.back().first;
    shared_ptr<hiai::op::Flatten> flatten(new hiai::op::Flatten(opName));
    (*flatten).set_input_x(*xOp.get());
    mNpuBackend->setOutputOps(mOp, {flatten}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUFlatten>> __flatten_op(OpType_Flatten);

} // namespace MNN