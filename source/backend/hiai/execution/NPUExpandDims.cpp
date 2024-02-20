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
    
    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops = mNpuBackend->mGrapMap[inputIndex]; // x
    xOp = iops.back().first;
    
    auto param = mOp->main_as_ExpandDims();
    vector<int32_t> axs = {param->axis()};
    shapeConst = hiai::op::Const(opName + "_shape_const");
    ge::TensorDesc fdesc(ge::Shape({1}), ge::FORMAT_NCHW,  ge::DT_INT32);
    ge::TensorPtr filter = std::make_shared<ge::Tensor>();
    filter->SetTensorDesc(fdesc);
    filter->SetData((uint8_t *)axs.data(), sizeof(int32_t));
    shapeConst.set_attr_value(filter);

    shared_ptr<hiai::op::ExpandDims> prob(new hiai::op::ExpandDims(opName));
    if (mNpuBackend->mSclipMap.find(inputIndex) == mNpuBackend->mSclipMap.end()) {
        (*prob).set_input_x(*xOp.get());
    } else {
        (*prob).set_input_x(xOp->GetOutput(mNpuBackend->mSclipMap[inputIndex]));
    }
    (*prob).set_input_axis(shapeConst);
    mNpuBackend->setOutputOps(mOp, {prob}, outputs);

    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUExpandDims>> __expand_dims_op(OpType_ExpandDims);

} // namespace MNN