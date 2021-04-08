//
//  NPUPack.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUPack.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUPack::NPUPack(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {}

ErrorCode NPUPack::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();
    shared_ptr<ge::op::Pack> pack(new ge::op::Pack(opName));

    auto param = mOp->main_as_PackParam();
    
    auto xOp = mNpuBackend->getInputOps(mOp);
    (*pack)
        .set_dynamic_input_values(0, *xOp.get())
        .set_attr_axis(axisFormat(inputs[0], param->axis()));

    mNpuBackend->setOutputOps(mOp, {pack}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUPack>> __pack_op(OpType_Pack);

} // namespace MNN