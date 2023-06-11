//
//  NPUEltwise.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUEltwise.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUEltwise::NPUEltwise(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {
    
}

ErrorCode NPUEltwise::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();
    shared_ptr<hiai::op::Eltwise> eltwise(new hiai::op::Eltwise(opName));

    /*
     * set om op
     * */
    auto inputIndex1 = mOp->inputIndexes()->data()[0];
    auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; // x
    auto xOp1        = iops1.back().first;

    
    auto inputIndex2 = mOp->inputIndexes()->data()[1];
    auto iops2       = mNpuBackend->mGrapMap[inputIndex2]; // x
    auto xOp2        = iops2.back().first;

    auto inputSize = mOp->inputIndexes()->size();
    auto param = mOp->main_as_Eltwise();
    (*eltwise).create_dynamic_input_x(inputSize)
     .set_attr_N(inputSize)
     .set_attr_mode(param->type()); // 0:product,1:sum,2:max;default is CC_ELTWISE_SUM.  TODO SUB  Weight

    for (int i = 0; i < inputSize; ++i) {
        auto inputIndex = mOp->inputIndexes()->data()[i];
        auto iops       = mNpuBackend->mGrapMap[inputIndex]; // x
        auto xOp        = iops.back().first;

        hiai::Operator *px = (hiai::Operator *)xOp.get();
        (* eltwise).set_dynamic_input_x(i + 1, *px);
    }

    if(param->type()==EltwiseType_SUB) {
        shared_ptr<hiai::op::Sub> sub(new hiai::op::Sub(opName));
        (*sub)
            .set_input_x1(*xOp1.get())
            .set_input_x2(*xOp2.get());
        mNpuBackend->setOutputOps(mOp, {sub}, outputs);
    } else {
        mNpuBackend->setOutputOps(mOp, {eltwise}, outputs);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUEltwise>> __elewise_op(OpType_Eltwise);

} // namespace MNN