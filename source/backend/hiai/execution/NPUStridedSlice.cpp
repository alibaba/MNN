//
//  NPUStridedSlice.cpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUStridedSlice.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUStridedSlice::NPUStridedSlice(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) 
{
    isConst1 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    isConst2 = TensorUtils::getDescribe(inputs[2])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    isConst3 = false;
    isConst4 = false;
    auto opName = mOp->name()->str();
    Tensor *begin   = inputs[1];
    Tensor *end     = inputs[2];
    if(isConst1) {
        mConst_b = hiai::op::Const(opName + "_b_const");
        ge::TensorDesc fdesc(ge::Shape({begin->elementSize()}), ge::DT_INT32); 
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t*)begin->host<int32_t>(), begin->elementSize()*sizeof(int32_t));
        mConst_b.set_attr_value(filter);
    }
    if(isConst2) {
        mConst_e = hiai::op::Const(opName + "_e_const");
        ge::TensorDesc fdesc(ge::Shape({end->elementSize()}),  ge::DT_INT32); 
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)end->host<int32_t>(), end->elementSize()*sizeof(int32_t));
        mConst_e.set_attr_value(filter);
    }
    auto parameter = mOp->main_as_StridedSliceParam();
    beginMask = convertMask(begin, parameter->beginMask(),1); 
    endMask = convertMask(begin, parameter->endMask(),1);
    ellipsisMask = parameter->ellipsisMask(); //框架未使用
    newAxisMask = parameter->newAxisMask(); 
    shrinkAxisMask = convertMask(begin, parameter->shrinkAxisMask());
}

ErrorCode NPUStridedSlice::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();
    Tensor *axis = nullptr;
    Tensor *strides = nullptr;
    if (inputs.size() > 3) {
        axis = inputs[3];
        isConst3 = TensorUtils::getDescribe(inputs[3])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    }
    if (inputs.size() > 4) {
        strides = inputs[4];
        isConst4 = TensorUtils::getDescribe(inputs[4])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    }
    if (isConst3) {
        vector<int32_t> axisdims;
        vector<int32_t> axisdims1;
        for (int32_t i = 0; i < axis->elementSize(); i++) {
            axisdims.push_back(i);
            if (count(axisdims1.begin(), axisdims1.end(), axis->host<int32_t>()[i]) == 0) {
                axisdims1.push_back(axis->host<int32_t>()[i]);
            }
        }
        mConst_a = hiai::op::Const(opName + "_a_const");
        ge::TensorDesc fdesc(ge::Shape({axis->elementSize()}), ge::DT_INT32); 
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        if (axisdims1.size() != axisdims.size() || (axisdims.size() == 1 && axisdims1[0] == 1)) {
            filter->SetData((uint8_t*)axisdims.data(), axis->elementSize()*sizeof(int32_t));
        } else {
            filter->SetData((uint8_t*)axisdims1.data(), axis->elementSize()*sizeof(int32_t));
        }
        mConst_a.set_attr_value(filter);
    }
    if (isConst4) {
        mConst_s = hiai::op::Const(opName + "_s_const");
        ge::TensorDesc fdesc(ge::Shape({strides->elementSize()}), ge::DT_INT32); 
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t*)strides->host<int32_t>(), strides->elementSize()*sizeof(int32_t));
        mConst_s.set_attr_value(filter);
    } else {
        vector<int32_t> axisdims;
        for (int32_t i = 0; i < axis->elementSize(); i++) {
            axisdims.push_back(1);
        }
        mConst_s = hiai::op::Const(opName + "_s_const");
        ge::TensorDesc fdesc(ge::Shape({axis->elementSize()}), ge::DT_INT32); 
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t*)axisdims.data(), axis->elementSize()*sizeof(int32_t));
        mConst_s.set_attr_value(filter);
    }
    shared_ptr<hiai::op::StridedSliceV2> stride_slice(new hiai::op::StridedSliceV2(opName));

    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops       = mNpuBackend->mGrapMap[inputIndex]; // x
    auto xOp        = iops.back().first;
    (*stride_slice)
        .set_input_x(*xOp.get())
        .set_input_begin(mConst_b)
        .set_input_end(mConst_e);         
    if (isConst3) {
        (*stride_slice).set_input_axes(mConst_a);
    }
    (*stride_slice).set_input_strides(mConst_s);
    mNpuBackend->setOutputOps(mOp, {stride_slice}, outputs);

    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUStridedSlice>> __stride_slice_op(OpType_StridedSlice);

} // namespace MNN
