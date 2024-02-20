//
//  NPUTranspose.cpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUTranspose.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUTranspose::NPUTranspose(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) 
{
    const Tensor* perm  = inputs[1];
    for (int i = 0; i < perm->elementSize(); i++) {
        permutation.push_back(perm->host<int32_t>()[i]);
    }
}

static bool isPermNoChange(std::vector<int64_t>& perm)
{
    if((perm[0] == 0) && (perm[1] == 1) &&
       (perm[2] == 2) && (perm[3] == 3)) {
        return true;
    }
    return false;
}

ErrorCode NPUTranspose::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    
    auto opName = mOp->name()->str();
    auto xOp = mNpuBackend->getInputOps(mOp);

    std::vector<int32_t> shapeDims = outputs[0]->shape(); 
    shapeConst = hiai::op::Const(opName + "_shape_const");
    {
        ge::TensorDesc fdesc(ge::Shape({static_cast<int64_t>(shapeDims.size())}), 
            ge::FORMAT_NCHW,  ge::DT_INT32);
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)shapeDims.data(), shapeDims.size() * sizeof(int32_t));
        shapeConst.set_attr_value(filter);
    }

    MNN_ASSERT((permutation.size() == 4));

    if(isPermNoChange(permutation)) {
        shared_ptr<hiai::op::Reshape> reshape(new hiai::op::Reshape(opName));
        (*reshape).set_input_x(*xOp).set_input_shape(shapeConst);
        mNpuBackend->setOutputOps(mOp, {reshape}, outputs);
    } else {
        shared_ptr<hiai::op::Permute> permute(new hiai::op::Permute(opName));
        (*permute)
            .set_input_x(*xOp.get())
            .set_attr_order(permutation);
        mNpuBackend->setOutputOps(mOp, {permute}, outputs);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUTranspose>> __transpose_op(OpType_Transpose);

} // namespace MNN