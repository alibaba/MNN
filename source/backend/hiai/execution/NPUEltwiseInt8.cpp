//
//  NPUEltwiseInt8.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUEltwiseInt8.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUEltwiseInt8::NPUEltwiseInt8(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {
}

ErrorCode NPUEltwiseInt8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();
    auto param    = mOp->main_as_EltwiseInt8();

    // 
    auto inputIndex0 = mOp->inputIndexes()->data()[0];
    auto iops0       = mNpuBackend->mGrapMap[inputIndex0]; // x
    auto xOp0        = iops0.back().first;

    
    auto inputIndex1 = mOp->inputIndexes()->data()[1];
    auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; // x
    auto xOp1        = iops1.back().first;

    mConst_scale0 = hiai::op::Const(opName + "_scale0_const");
    {
        int size = param->inputQuan0()->tensorScale()->size();
        auto inScalePtr = param->inputQuan0()->tensorScale()->data();
        auto outScalePtr = param->outputQuan()->tensorScale()->data();

        vector<float> scaleData;
        for (size_t i = 0; i < size; i++){
            scaleData.push_back(outScalePtr[i]*inScalePtr[i]);
        }
        
        ge::TensorDesc fdesc(ge::Shape({1, size, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT); 
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)scaleData.data(), scaleData.size() * sizeof(float));
        mConst_scale0.set_attr_value(filter);
    }

    mConst_scale1 = hiai::op::Const(opName + "_scale1_const");
    {
        int size = param->inputQuan1()->tensorScale()->size();
        auto inScalePtr = param->inputQuan1()->tensorScale()->data();
        auto outScalePtr = param->outputQuan()->tensorScale()->data();

        vector<float> scaleData;
        for (size_t i = 0; i < size; i++){
            scaleData.push_back(outScalePtr[i]*inScalePtr[i]);
        }
        
        ge::TensorDesc fdesc(ge::Shape({1, size, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT); 
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)scaleData.data(), scaleData.size() * sizeof(float));
        mConst_scale1.set_attr_value(filter);
    }

    mConstMin0 = hiai::op::Const(opName + "_clip_min0");
    {
        float minData = -127;
        ge::TensorDesc fdesc(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr constTensor = std::make_shared<ge::Tensor>();
        constTensor->SetTensorDesc(fdesc);
        constTensor->SetData((uint8_t *)(&minData), sizeof(float));
        mConstMin0.set_attr_value(constTensor);
    }

    mConstMax0 = hiai::op::Const(opName + "_clip_max0");
    {
        float maxData = 127;
        ge::TensorDesc fdesc(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr constTensor = std::make_shared<ge::Tensor>();
        constTensor->SetTensorDesc(fdesc);
        constTensor->SetData((uint8_t *)(&maxData), sizeof(float));
        mConstMax0.set_attr_value(constTensor);
    }

    shared_ptr<hiai::op::ClipByValue> clip0(new hiai::op::ClipByValue(opName + "_clip0"));

    (*clip0).set_input_x(*xOp0.get()).set_input_clip_value_min(mConstMin0).set_input_clip_value_max(mConstMax0);
    
    shared_ptr<hiai::op::Scale> scale0(new hiai::op::Scale(opName + "_scale0"));
    (*scale0).set_input_x(*clip0.get()).set_input_scale(mConst_scale0);

    mConstMin1 = hiai::op::Const(opName + "_clip_min1");
    {
        float minData = -127;
        ge::TensorDesc fdesc(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr constTensor = std::make_shared<ge::Tensor>();
        constTensor->SetTensorDesc(fdesc);
        constTensor->SetData((uint8_t *)(&minData), sizeof(float));
        mConstMin1.set_attr_value(constTensor);
    }

    mConstMax1 = hiai::op::Const(opName + "_clip_max1");
    {
        float maxData = 127;
        ge::TensorDesc fdesc(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr constTensor = std::make_shared<ge::Tensor>();
        constTensor->SetTensorDesc(fdesc);
        constTensor->SetData((uint8_t *)(&maxData), sizeof(float));
        mConstMax1.set_attr_value(constTensor);
    }

    shared_ptr<hiai::op::ClipByValue> clip1(new hiai::op::ClipByValue(opName + "_clip1"));

    (*clip1).set_input_x(*xOp1.get()).set_input_clip_value_min(mConstMin1).set_input_clip_value_max(mConstMax1);

    shared_ptr<hiai::op::Scale> scale1(new hiai::op::Scale(opName + "_scale1"));
    (*scale1).set_input_x(*clip1.get()).set_input_scale(mConst_scale1);

    shared_ptr<hiai::op::Eltwise> eltwise(new hiai::op::Eltwise(opName));
    int type = 1;
    (*eltwise)
        .create_dynamic_input_x(2)
        .set_dynamic_input_x(1, *scale0.get())
        .set_dynamic_input_x(2, *scale1.get())
        .set_attr_N(2)
        .set_attr_coeff(ge::AttrValue::LIST_FLOAT({1, 1}))
        .set_attr_mode(type); // mode  : Either 0 (product), 1 (sum), or 2 (max). Defaults to 1 (sum).

    mConstMin = hiai::op::Const(opName + "_clip_min");
    {
        float minData = -127;
        ge::TensorDesc fdesc(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr constTensor = std::make_shared<ge::Tensor>();
        constTensor->SetTensorDesc(fdesc);
        constTensor->SetData((uint8_t *)(&minData), sizeof(float));
        mConstMin.set_attr_value(constTensor);
    }

    mConstMax = hiai::op::Const(opName + "_clip_max");
    {
        float maxData = 127;
        ge::TensorDesc fdesc(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr constTensor = std::make_shared<ge::Tensor>();
        constTensor->SetTensorDesc(fdesc);
        constTensor->SetData((uint8_t *)(&maxData), sizeof(float));
        mConstMax.set_attr_value(constTensor);
    }

    shared_ptr<hiai::op::ClipByValue> clip(new hiai::op::ClipByValue(opName + "_clip"));

    (*clip)
        .set_input_x(*eltwise)
        .set_input_clip_value_min(mConstMin)
        .set_input_clip_value_max(mConstMax);

    mNpuBackend->setOutputOps(mOp, {scale0, scale1, clip0, clip1, eltwise, clip}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUEltwiseInt8>> __elewise_int8_op(OpType_EltwiseInt8);

} // namespace MNN