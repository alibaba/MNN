//
//  NPUConcat.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUConcat.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUConcat::NPUConcat(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {
}

ErrorCode NPUConcat::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();
    auto param  = mOp->main_as_Axis();

    shared_ptr<hiai::op::ConcatD> concatD(new hiai::op::ConcatD(opName));
    auto xOp = mNpuBackend->getInputOps(mOp);
    auto inputSize = mOp->inputIndexes()->size();
    int32_t axis = param->axis();
    (*concatD).create_dynamic_input_x(inputSize).set_attr_concat_dim(axis);

    for (int i = 0; i < inputSize; ++i) {
        auto inputIndex = mOp->inputIndexes()->data()[i];
        auto iops       = mNpuBackend->mGrapMap[inputIndex]; // x
        xOp        = iops.back().first;
        if (mNpuBackend->mSclipMap.find(inputIndex) == mNpuBackend->mSclipMap.end()) {
            (*concatD).set_dynamic_input_x(i + 1, *xOp.get());
        } else {
            (*concatD).set_dynamic_input_x(i + 1, xOp->GetOutput(mNpuBackend->mSclipMap[inputIndex]));
        }
    }
    mNpuBackend->setOutputOps(mOp, {concatD}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUConcat>> __concatD_op(OpType_Concat);

} // namespace MNN
