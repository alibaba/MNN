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
    auto shapeFormt = tensorShapeFormat(outputs[0]);
    std::vector<int32_t> shapeDims (shapeFormt.begin(), shapeFormt.end());
    shapeConst = hiai::op::Const(opName + "_shape_const");
    {
        ge::TensorDesc fdesc(ge::Shape({static_cast<int64_t>(shapeDims.size())}), 
            ge::FORMAT_NCHW,  ge::DT_INT32);
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)shapeDims.data(), shapeDims.size() * sizeof(int32_t));

        shapeConst.set_attr_value(filter);
    }

    shared_ptr<hiai::op::Reshape> prob(new hiai::op::Reshape(opName));

    auto xOp = mNpuBackend->getInputOps(mOp);

    (*prob).set_input_x(*xOp.get()).set_input_shape(shapeConst);
    
    mNpuBackend->setOutputOps(mOp, {prob}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUSqueeze>> __squeeze_op(OpType_Squeeze);
NPUCreatorRegister<TypedCreator<NPUSqueeze>> __unsqueeze_op(OpType_Unsqueeze);

} // namespace MNN
