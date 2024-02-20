//
//  NPULRN.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPULRN.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPULRN::NPULRN(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {}

ErrorCode NPULRN::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();
    shared_ptr<hiai::op::LRN> lrn(new hiai::op::LRN(opName));
    auto param = mOp->main_as_LRN();
    int32_t depth_radius = param->localSize();
    float bias = param->bias();
    float alpha = param->alpha();
    float beta = param->beta();
    int32_t normRegion = param->regionType();
    string normRegionName = "ACROSS_CHANNELS";
    if (normRegion == 1) {
        normRegionName = "WITHIN_CHANNEL";
    }
    auto xOp = mNpuBackend->getInputOps(mOp);
    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops       = mNpuBackend->mGrapMap[inputIndex]; // x
    xOp        = iops.back().first;
    (*lrn)
        .set_input_x(*xOp.get())
        .set_attr_depth_radius(depth_radius)
        .set_attr_bias(bias)
        .set_attr_alpha(alpha)
        .set_attr_beta(beta)
        .set_attr_norm_region(normRegionName);

    mNpuBackend->setOutputOps(mOp, {lrn}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPULRN>> __LRN_op(OpType_LRN);

} // namespace MNN