//
//  NPUBroadCastTo.cpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUBroadCastTo.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUBroadCastTo::NPUBroadCastTo(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {
}

ErrorCode NPUBroadCastTo::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto opName = mOp->name()->str();
    bool isConst1 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    if (isConst1) {
        auto depth = inputs[1];
        mConst_s = hiai::op::Const(opName + "_s_const");
        vector<int64_t> dims;
        for (int32_t i = 0; i < depth->buffer().dimensions; i++) {
            dims.push_back(depth->buffer().dim[i].extent);
        }
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        ge::TensorDesc fdesc(ge::Shape(dims), ge::FORMAT_NCHW, ge::DT_INT32);
        filter->SetData((uint8_t *)depth->host<int32_t>(), depth->elementSize() * sizeof(int32_t));
        filter->SetTensorDesc(fdesc);
        mConst_s.set_attr_value(filter);
    }
    mNpuBackend->setNetworkInput(inputs, mOp);
    shared_ptr<hiai::op::BroadcastTo> broadCastTo(new hiai::op::BroadcastTo(opName));
    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops       = mNpuBackend->mGrapMap[inputIndex]; // x
    auto xOp        = iops.back().first;
    if (isConst1) {
        (*broadCastTo).set_input_x(*xOp.get()).set_input_shape(mConst_s);
        mNpuBackend->setOutputOps(mOp, {broadCastTo}, outputs);
        return NO_ERROR;
    }
    return NOT_SUPPORT;
}

NPUCreatorRegister<TypedCreator<NPUBroadCastTo>> __BroadCastTo_op(OpType_BroadcastTo);

} // namespace MNN