//
//  NPUSlice.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUSlice.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUSlice::NPUSlice(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {}

ErrorCode NPUSlice::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();
    shared_ptr<hiai::op::SplitD> slice(new hiai::op::SplitD(opName));

    auto param = mOp->main_as_Slice();
    auto axis = param->axis();
    if (axis < 0) {
        axis = axis + inputs[0]->dimensions();
    }

    if(TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NHWC){
        axis = mNCHW[axis];
    }else{
        axis = mNHWC[axis];
    }

    auto xOp = mNpuBackend->getInputOps(mOp);

    (*slice)
        .set_input_x(*xOp.get())
        .set_attr_split_dim(axis)
        .set_attr_num_split(outputs.size())
        .create_dynamic_output_y(outputs.size());

    mNpuBackend->setOutputOps(mOp, {slice}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUSlice>> __slice_op(OpType_Slice);

} // namespace MNN