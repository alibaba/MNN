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
    for (int i = 0; i < perm->buffer().dim[0].extent; i++) {
        permutation.push_back(axisFormat(inputs[0], perm->host<int32_t>()[i]));
    }
    auto dimSize = inputs[0]->buffer().dimensions;
    if(dimSize == 3) {
        permutation.insert(permutation.begin(),0);
    } else if (dimSize == 2) {
        permutation.insert(permutation.begin(),0);
        permutation.push_back(3);
    } else if (dimSize == 1) {
        permutation.insert(permutation.begin(),0);
        permutation.push_back(2);
        permutation.push_back(3);
    }
    if(TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC)
    {
        std::vector<int64_t> tmp = permutation;
        permutation = {tmp[0],tmp[3],tmp[1],tmp[2]};
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
    auto shapeDims = tensorShapeFormat(outputs[0]);

    MNN_ASSERT((permutation.size()==4));

    if(isPermNoChange(permutation)) {
        shared_ptr<ge::op::Reshape> reshape(new ge::op::Reshape(opName));
        (*reshape).set_input_tensor(*xOp).set_attr_shape(ge::AttrValue::LIST_INT(shapeDims));
        mNpuBackend->setOutputOps(mOp, {reshape}, outputs);
    } else {
        shared_ptr<ge::op::Permute> permute(new ge::op::Permute(opName));
        (*permute)
            .set_input_x(*xOp.get())
            .set_attr_order(permutation);
        mNpuBackend->setOutputOps(mOp, {permute}, outputs);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUTranspose>> __transpose_op(OpType_Transpose);

} // namespace MNN