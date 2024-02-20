//
//  NPUCrop.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUCrop.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUCrop::NPUCrop(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {
}

ErrorCode NPUCrop::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();
    shared_ptr<hiai::op::Crop> crop(new hiai::op::Crop(opName));
    auto xOp = mNpuBackend->getInputOps(mOp);

    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops       = mNpuBackend->mGrapMap[inputIndex];
    xOp        = iops.back().first;
    auto inputIndex1 = mOp->inputIndexes()->data()[1];
    auto iops1       = mNpuBackend->mGrapMap[inputIndex1];
    auto xOp1        = iops1.back().first;

    auto param = mOp->main_as_Crop();
    int32_t axis = param->axis();
    auto offsetTmp = param->offset();
    vector<int64_t> offset;
    for (int32_t i = 0; i < offsetTmp->size(); i++) {
        offset.push_back(offsetTmp->Get(i));
    } 
    (*crop).set_input_x(*xOp.get())
           .set_input_size(*xOp1.get())
           .set_attr_axis(axis)
           .set_attr_offsets(ge::AttrValue::LIST_INT(offset));
    mNpuBackend->setOutputOps(mOp, {crop}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUCrop>> __CropD_op(OpType_Crop);

} // namespace MNN
