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
    bool isConst1 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    bool isConst2 = TensorUtils::getDescribe(inputs[2])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    bool isConst3 = TensorUtils::getDescribe(inputs[3])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    auto opName = mOp->name()->str();
    Tensor *begin   = inputs[1];
    Tensor *end     = inputs[2];
    Tensor *strided = inputs[3];

    if(isConst1 == true) {
        auto beginShape = convertShapeConstValue(begin, 0);
        mConst_b = ge::op::Const(opName + "_b_const");
        {
            ge::TensorDesc fdesc(ge::Shape({4}), ge::DT_INT32); 
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t *)&beginShape[0], 4*sizeof(int32_t));
            mConst_b.set_attr_value(filter);
        }
    }

    if(isConst2 == true) {
        auto endShape = convertShapeConstValue(end, 0);
        mConst_e = ge::op::Const(opName + "_e_const");
        {
            ge::TensorDesc fdesc(ge::Shape({4}),  ge::DT_INT32); 
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t *)&endShape[0], 4*sizeof(int32_t));
            mConst_e.set_attr_value(filter);
        }
    }

    if(isConst3 == true) {
        auto stridedShape = convertShapeConstValue(strided);
        mConst_s = ge::op::Const(opName + "_s_const");
        {
            ge::TensorDesc fdesc(ge::Shape({4}), ge::DT_INT32); 
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t *)&stridedShape[0], 4*sizeof(int32_t));
            mConst_s.set_attr_value(filter);
        }
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
    auto param  = mOp->main_as_Axis();

    shared_ptr<ge::op::StridedSlice> stride_slice(new ge::op::StridedSlice(opName));

    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops       = mNpuBackend->mGrapMap[inputIndex]; // x
    auto xOp        = iops.back().first;

    auto parameter = mOp->main_as_StridedSliceParam();

    (*stride_slice)
        .set_input_x(*xOp.get())
        .set_input_begin(mConst_b)
        .set_input_end(mConst_e)
        .set_input_strides(mConst_s)
        .set_attr_begin_mask(beginMask)
        .set_attr_end_mask(endMask)
        .set_attr_ellipsis_mask(ellipsisMask)
        .set_attr_new_axis_mask(newAxisMask)
        .set_attr_shrink_axis_mask(shrinkAxisMask);

    mNpuBackend->setOutputOps(mOp, {stride_slice}, outputs);

    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUStridedSlice>> __stride_slice_op(OpType_StridedSlice);

} // namespace MNN
