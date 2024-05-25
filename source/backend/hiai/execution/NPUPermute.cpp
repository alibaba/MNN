//
//  NPUPermute.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUPermute.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUPermute::NPUPermute(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {
}

ErrorCode NPUPermute::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();
    auto xOp = mNpuBackend->getInputOps(mOp);
    shared_ptr<hiai::op::Permute> permute(new hiai::op::Permute(opName));

    auto param = mOp->main_as_Permute();
    auto axis = param->dims();
    int32_t size = param->dims()->size();
    vector<int64_t> dims;
    for (int32_t i = 0; i < size; i++) {
        int32_t index = axis->Get(i);
        dims.push_back(index);
    }
    int index = mOp->inputIndexes()->data()[0];
    (*permute).set_input_x(*xOp.get()).set_attr_order(dims);
    mNpuBackend->setOutputOps(mOp, {permute}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUPermute>> __permute_op(OpType_Permute);

} // namespace MNN