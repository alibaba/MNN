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

    shared_ptr<ge::op::Concat> concat(new ge::op::Concat(opName));

    auto inputSize = mOp->inputIndexes()->size();
    (*concat).create_dynamic_input_x(inputSize).set_attr_axis(axisFormat(inputs[0], param->axis()));

    for (int i = 0; i < inputSize; ++i) {
        auto inputIndex = mOp->inputIndexes()->data()[i];
        auto iops       = mNpuBackend->mGrapMap[inputIndex]; // x
        auto xOp        = iops.back().first;

        ge::Operator *px = (ge::Operator *)xOp.get();
        (*concat).set_dynamic_input_x(i + 1, *px);
    }

    mNpuBackend->setOutputOps(mOp, {concat}, outputs);

    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUConcat>> __concat_op(OpType_Concat);

} // namespace MNN
