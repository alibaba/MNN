//
//  NPUFloatToInt8.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUFloatToInt8.hpp"
#include "NPUBackend.hpp"
#include "../3rdParty/include/graph/op/all_ops.h"
#include <core/TensorUtils.hpp>

using namespace std;

namespace MNN {

NPUFloatToInt8::NPUFloatToInt8(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs)
    : MNN::NPUCommonExecution(b,op) {}

ErrorCode NPUFloatToInt8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto scale         = mOp->main_as_QuantizedFloatParam()->tensorScale();

    auto opName = mOp->name()->str();
    shared_ptr<hiai::op::Scale> floatToInt8(new hiai::op::Scale(opName));
    auto xOp = mNpuBackend->getInputOps(mOp);

    // om input filter const op
    mConst_fliter = hiai::op::Const(opName + "_filter_const");
    {
        ge::TensorDesc fdesc(ge::Shape({1, scale->size(), 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT); // in o h w ?
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)scale->data(), scale->size() * sizeof(float));
        mConst_fliter.set_attr_value(filter);
    }

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
    shared_ptr<hiai::op::ClipByValue> clip(new hiai::op::ClipByValue(opName + "clip"));
    
    (*floatToInt8)
        .set_input_x(*xOp)
        .set_input_scale(mConst_fliter);

    (*clip)
        .set_input_x(*floatToInt8)
        .set_input_clip_value_min(mConstMin)
        .set_input_clip_value_max(mConstMax);

    mNpuBackend->setOutputOps(mOp, {floatToInt8, clip}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUFloatToInt8>> __float_to_int8_op(OpType_FloatToInt8);

} // namespace MNN
