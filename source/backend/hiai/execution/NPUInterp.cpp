//
//  NPUInterp.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUInterp.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUInterp::NPUInterp(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {}

ErrorCode NPUInterp::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();
    auto param = mOp->main_as_Interp();
    auto xOp = mNpuBackend->getInputOps(mOp);
    auto resizeType = param->resizeType();
    MNN_ASSERT(resizeType <= 3);
    if (resizeType > 3) {
        MNN_ERROR("npu Interp not support type: %d", resizeType);
        return NOT_SUPPORT;
    }
    vector<int32_t> hw = {outputs[0]->height(),outputs[0]->width()};
    mConstShape = hiai::op::Const(opName + "_w_const");
    {
        ge::TensorDesc fdesc(ge::Shape({2}), ge::FORMAT_NCHW, ge::DT_INT32); 
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)hw.data(), hw.size() * sizeof(int32_t));
        mConstShape.set_attr_value(filter);
    }

    if (resizeType == 1) {
        shared_ptr<hiai::op::ResizeNearestNeighbor> interp(new hiai::op::ResizeNearestNeighbor(opName));
        (*interp).set_input_x(*xOp)
                 .set_input_size(mConstShape)
                 .set_attr_align_corners(param->alignCorners());
        mNpuBackend->setOutputOps(mOp, {interp}, outputs);
    } else if (resizeType == 2) {
        shared_ptr<hiai::op::ResizeBilinear> interp(new hiai::op::ResizeBilinear(opName));
        (*interp).set_input_x(*xOp)
                 .set_input_size(mConstShape)
                 .set_attr_align_corners(param->alignCorners());
        mNpuBackend->setOutputOps(mOp, {interp}, outputs);
    } else if (resizeType == 3) {
        shared_ptr<hiai::op::ResizeBilinear> interp(new hiai::op::ResizeBilinear(opName));
        (*interp).set_input_x(*xOp)
                 .set_input_size(mConstShape)
                 .set_attr_align_corners(param->alignCorners());
        mNpuBackend->setOutputOps(mOp, {interp}, outputs);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUInterp>> __interp_op(OpType_Interp);

} // namespace MNN