//
//  NPUTile.cpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUTile.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUTile::NPUTile(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {
}

ErrorCode NPUTile::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    bool isConst2 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    auto input = inputs[1];
    auto opName = mOp->name()->str();
    if (isConst2) {
        mConst_m = hiai::op::Const(opName + "_mul_const");
        vector<int64_t> dims;
        for (int32_t i = 0; i< input->buffer().dimensions; i++) {
            dims.push_back(static_cast<int64_t>(input->buffer().dim[i].extent));
        }
        ge::TensorDesc fdesc(ge::Shape(dims), ge::FORMAT_NCHW,  ge::DT_INT32);
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)input->host<int32_t>(), input->elementSize() * sizeof(int32_t));
        mConst_m.set_attr_value(filter);
    }
    mNpuBackend->setNetworkInput(inputs, mOp);
    shared_ptr<hiai::op::Tile> tile(new hiai::op::Tile(opName));
    auto xOp = mNpuBackend->getInputOps(mOp);
    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops = mNpuBackend->mGrapMap[inputIndex]; // x
    xOp = iops.back().first;
    if (isConst2) {
        (*tile).set_input_x(*xOp.get()).set_input_multiples(mConst_m);
        mNpuBackend->setOutputOps(mOp, {tile}, outputs);
        return NO_ERROR;
    }
    return NOT_SUPPORT;
}

NPUCreatorRegister<TypedCreator<NPUTile>> __Tile_op(OpType_Tile);

} // namespace MNN