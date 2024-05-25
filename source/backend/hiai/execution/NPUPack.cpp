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
    auto xOp = mNpuBackend->getInputOps(mOp);
    auto opName = mOp->name()->str();
    shared_ptr<hiai::op::Pack> pack(new hiai::op::Pack(opName));
    auto param = mOp->main_as_PackParam();
    int64_t N = inputs.size();
    (*pack).create_dynamic_input_x(N).set_attr_axis(param->axis()).set_attr_N(N);
    for (int32_t i = 0; i < inputs.size(); i++) {
        auto inputIndex = mOp->inputIndexes()->data()[i];
        auto iops = mNpuBackend->mGrapMap[inputIndex]; // x
        xOp = iops.back().first;
        (*pack).set_dynamic_input_x(i+1, *xOp.get());
    }
    mNpuBackend->setOutputOps(mOp, {pack}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUPack>> __pack_op(OpType_Pack);

} // namespace MNN