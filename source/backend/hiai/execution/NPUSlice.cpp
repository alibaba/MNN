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
    int64_t slice_num = 0;
    if (param->slicePoints() != nullptr) {
        if (param->slicePoints()->size() < outputs.size()) {
            slice_num = static_cast<int64_t>(outputs.size());
        } else if (param->slicePoints()->size() == 1) {
            slice_num = static_cast<int64_t>(param->slicePoints()->Get(0));
        } else {
            slice_num = static_cast<int64_t>(param->slicePoints()->size());
        }
    } else {
        slice_num = static_cast<int64_t>(outputs.size());
    }
    auto xOp = mNpuBackend->getInputOps(mOp);

    (*slice)
        .set_input_x(*xOp.get())
        .set_attr_split_dim(axis)
        .set_attr_num_split(slice_num)
        .create_dynamic_output_y(slice_num);

    mNpuBackend->setOutputOps(mOp, {slice}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUSlice>> __slice_op(OpType_Slice);

} // namespace MNN