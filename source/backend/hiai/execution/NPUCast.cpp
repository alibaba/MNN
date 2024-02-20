//
//  NPUCast.cpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUCast.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUCast::NPUCast(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {
}

ErrorCode NPUCast::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();
    shared_ptr<hiai::op::CastT> castTOp(new hiai::op::CastT(opName));
    auto xOp = mNpuBackend->getInputOps(mOp);
    auto castPara = mOp->main_as_CastParam();
    DataType dstT = castPara->dstT();

    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops = mNpuBackend->mGrapMap[inputIndex]; // x
    xOp = iops.back().first;
    if (mNpuBackend->mSclipMap.find(inputIndex) != mNpuBackend->mSclipMap.end()) {
        (*castTOp).set_input_x(xOp->GetOutput(mNpuBackend->mSclipMap[inputIndex]));
    } else {
        (*castTOp).set_input_x(*xOp.get());
    }
    (*castTOp).set_attr_dst_dtype(mapDataType(dstT));
    mNpuBackend->setOutputOps(mOp, {castTOp}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUCast>> __cast_op(OpType_Cast);

} // namespace MNN