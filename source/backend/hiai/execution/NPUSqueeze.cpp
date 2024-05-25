//
//  NPUSqueeze.cpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUSqueeze.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUSqueeze::NPUSqueeze(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {
}

ErrorCode NPUSqueeze::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();
    auto param = mOp->main_as_SqueezeParam();
    auto axis = param->squeezeDims();
    vector<int64_t> ax;
    if (axis != nullptr) {
        for (int32_t i = 0; i < axis->size(); i++) {
            ax.push_back(axis->Get(i));
        }
    } else {
        ax = {0};
    }
    auto xOp = mNpuBackend->getInputOps(mOp);
    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops       = mNpuBackend->mGrapMap[inputIndex]; // x
    xOp        = iops.back().first;
    if (mOp->type() == OpType_Squeeze) {
        shared_ptr<hiai::op::Squeeze> squeeze(new hiai::op::Squeeze(opName));
        if (mNpuBackend->mSclipMap.find(inputIndex) == mNpuBackend->mSclipMap.end()) {
            (*squeeze).set_input_x(*xOp.get());
        } else {
            (*squeeze).set_input_x(xOp->GetOutput(mNpuBackend->mSclipMap[inputIndex]));
        }
        (*squeeze).set_attr_axis(ax);
        mNpuBackend->setOutputOps(mOp, {squeeze}, outputs);
    } else {
        shapeConst = hiai::op::Const(opName + "_axis_const");
        if (ax.size() > 1) {
            std::cout<<"unsqueeze axis only one element const, not "<< ax.size() << std::endl;
            return NOT_SUPPORT;
        }
        vector<int32_t> axs = {static_cast<int32_t>(ax[0])};
        {
            ge::TensorDesc fdesc(ge::Shape({1}), ge::FORMAT_NCHW,  ge::DT_INT32);
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t *)axs.data(), sizeof(int32_t));
            shapeConst.set_attr_value(filter);
        }
        shared_ptr<hiai::op::ExpandDims> prob(new hiai::op::ExpandDims(opName));
        if (mNpuBackend->mSclipMap.find(inputIndex) == mNpuBackend->mSclipMap.end()) {
            (*prob).set_input_x(*xOp.get());
        } else {
            (*prob).set_input_x(xOp->GetOutput(mNpuBackend->mSclipMap[inputIndex]));
        }
        (*prob).set_input_axis(shapeConst);
        mNpuBackend->setOutputOps(mOp, {prob}, outputs);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUSqueeze>> __squeeze_op(OpType_Squeeze);
NPUCreatorRegister<TypedCreator<NPUSqueeze>> __unsqueeze_op(OpType_Unsqueeze);

} // namespace MNN
