//
//  NPUPooling3D.cpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUPooling3D.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUPooling3D::NPUPooling3D(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs,  const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {
}

ErrorCode NPUPooling3D::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();
    shared_ptr<hiai::op::PoolingD> poolingD(new hiai::op::PoolingD(opName));

    auto poolParam = mOp->main_as_Pool3D();

    //  0:NOTSET, 6:SAME 5:VALID. defaul default value is 0:NOTSET
    auto pad_mode = 0;
    int data_mode = 0;
    
    if (PoolPadType_VALID == poolParam->padType()) {
        pad_mode = 5;
        data_mode = 1;
    } else if (PoolPadType_SAME == poolParam->padType()) {
        pad_mode = 6;
        data_mode = 1;
    }

    bool ceilMode = 0;
    // 0:max Pooling3D   1:avg Pooling3D  2:L2 Pooling3D
    auto mode = 0; // TODO
    if (PoolType_MAXPOOL == poolParam->type()) {
        mode = 0;
    } else if (PoolType_AVEPOOL == poolParam->type()) {
        mode = 1;
    }
    int64_t kernelH = 1;
    int64_t kernelW = 1;
    if(poolParam->isGlobal() == true) {
        kernelH = inputs[0]->height();
        kernelW = inputs[0]->width();
    }

    auto xOp = mNpuBackend->getInputOps(mOp);

    int64_t strideWidth  = 1;
    int64_t strideHeight = 1;
    vector<int64_t> pads;
    if (poolParam->pads() != nullptr) {
        int32_t size = poolParam->pads()->size() / 2;
        for (int32_t i = 0; i < size; i++) {
            pads.push_back(static_cast<int64_t>(poolParam->pads()->data()[i]));
            pads.push_back(static_cast<int64_t>(poolParam->pads()->data()[i + size]));
        }
    } else {
        pads.push_back(0);
        pads.push_back(0);
        pads.push_back(0);
        pads.push_back(0);
    }
    (*poolingD)
        .set_input_x(*xOp.get())
        .set_attr_data_mode(data_mode) // data_mode, DOMI_CAFFE_DATA_MODE =0, TENSORFLOW_DATA_MODE = 1.  TODO
        .set_attr_pad_mode(pad_mode)
        .set_attr_ceil_mode(0) // Pooling3D ceil mode, 0: DOMI_Pooling3D_CEIL, 1:DOMI_Pooling3D_FLOOR
        .set_attr_mode(mode)
        .set_attr_pad(pads) // 上下左右
        .set_attr_window(ge::AttrValue::LIST_INT({kernelH, kernelW}))
        .set_attr_stride(ge::AttrValue::LIST_INT({strideHeight, strideWidth}))
        .set_attr_global_pooling(true);

    mNpuBackend->setOutputOps(mOp, {poolingD}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUPooling3D>> __Pooling3D_op(OpType_Pooling3D);

} // namespace MNN