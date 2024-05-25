//
//  NPUPooling.cpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUPooling.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUPooling::NPUPooling(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs,  const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {
}

ErrorCode NPUPooling::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();
    shared_ptr<hiai::op::PoolingD> poolingD(new hiai::op::PoolingD(opName));

    auto poolParam = mOp->main_as_Pool();

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

    // 0:max pooling   1:avg pooling  2:L2 pooling
    bool ceilMode = poolParam->ceilModel();
    auto mode = 0; // TODO
    if (PoolType_MAXPOOL == poolParam->type()) {
        mode = 0;
    } else if (PoolType_AVEPOOL == poolParam->type()) {
        mode = 1;
    }
    int64_t kernelH = poolParam->kernelY();
    int64_t kernelW = poolParam->kernelX();
    if(poolParam->isGlobal() == true) {
        kernelH = inputs[0]->height();
        kernelW = inputs[0]->width();
    }

    auto xOp = mNpuBackend->getInputOps(mOp);

    int64_t strideWidth  = std::max(poolParam->strideX(), 1);
    int64_t strideHeight = std::max(poolParam->strideY(), 1);
    vector<int64_t> pads;
    if (poolParam->pads() != nullptr) {
        int32_t size = poolParam->pads()->size() / 2;
        for (int32_t i = 0; i < size; i++) {
            pads.push_back(static_cast<int64_t>(poolParam->pads()->data()[i]));
            pads.push_back(static_cast<int64_t>(poolParam->pads()->data()[i + size]));
        }
    } else {
        pads.push_back(static_cast<int64_t>(poolParam->padY()));
        pads.push_back(static_cast<int64_t>(poolParam->padY()));
        pads.push_back(static_cast<int64_t>(poolParam->padX()));
        pads.push_back(static_cast<int64_t>(poolParam->padX()));
    }
    if (poolParam->isGlobal() == true && 
        kernelH%2 == 0 && kernelW%2==0 && kernelH*kernelW >65535) {
        shared_ptr<hiai::op::PoolingD> pooling2X2(new hiai::op::PoolingD(opName+"_2x2"));
        (*pooling2X2)
            .set_input_x(*xOp.get()).set_attr_data_mode(0) 
            .set_attr_pad_mode(0).set_attr_ceil_mode(ceilMode) 
            .set_attr_mode(mode)
            .set_attr_pad(ge::AttrValue::LIST_INT({0, 0, 0, 0})) // 上下左右
            .set_attr_window(ge::AttrValue::LIST_INT({2, 2}))
            .set_attr_stride(ge::AttrValue::LIST_INT({2, 2}))
            .set_attr_global_pooling(false);
        (*poolingD)
            .set_input_x(*pooling2X2.get())
            .set_attr_data_mode(data_mode) // data_mode, DOMI_CAFFE_DATA_MODE =0, TENSORFLOW_DATA_MODE = 1.  TODO
            .set_attr_pad_mode(pad_mode)
            .set_attr_ceil_mode(ceilMode) // pooling ceil mode, 0: DOMI_POOLING_CEIL, 1:DOMI_POOLING_FLOOR
            .set_attr_mode(mode)
            .set_attr_pad(pads) // 上下左右
            .set_attr_window(ge::AttrValue::LIST_INT({kernelH/2, kernelW/2}))
            .set_attr_stride(ge::AttrValue::LIST_INT({strideHeight, strideWidth}))
            .set_attr_global_pooling(poolParam->isGlobal());
            mNpuBackend->setOutputOps(mOp, {pooling2X2,poolingD}, outputs);
    } else {
        (*poolingD)
            .set_input_x(*xOp.get())
            .set_attr_data_mode(data_mode) // data_mode, DOMI_CAFFE_DATA_MODE =0, TENSORFLOW_DATA_MODE = 1.  TODO
            .set_attr_pad_mode(pad_mode)
            .set_attr_ceil_mode(ceilMode) // pooling ceil mode, 0: DOMI_POOLING_CEIL, 1:DOMI_POOLING_FLOOR
            .set_attr_mode(mode)
            .set_attr_pad(pads) // 上下左右
            .set_attr_window(ge::AttrValue::LIST_INT({kernelH, kernelW}))
            .set_attr_stride(ge::AttrValue::LIST_INT({strideHeight, strideWidth}))
            .set_attr_global_pooling(poolParam->isGlobal());

        mNpuBackend->setOutputOps(mOp, {poolingD}, outputs);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUPooling>> __pooling_op(OpType_Pooling);

} // namespace MNN