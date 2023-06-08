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
    auto shapeFormat = tensorShapeFormat(outputs[0]);
    std::vector<int32_t> shapeDims(shapeFormat.begin(), shapeFormat.end()); 
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

    auto output = outputs[0];

    (*prob).set_input_x(*xOp.get()).set_input_shape(shapeConst);

    mNpuBackend->setOutputOps(mOp, {prob}, outputs);

    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUExpandDims>> __expand_dims_op(OpType_ExpandDims);

} // namespace MNN