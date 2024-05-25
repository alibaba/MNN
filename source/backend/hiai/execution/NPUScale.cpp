//
//  NPUScale.cpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUScale.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUScale::NPUScale(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {}

ErrorCode NPUScale::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();

    auto param     = mOp->main_as_Scale();
    auto scaleData = param->scaleData();
    auto biasData  = param->biasData();

    shared_ptr<hiai::op::Scale> scale(new hiai::op::Scale(opName + "_scale"));

    auto xOp = mNpuBackend->getInputOps(mOp);

    // om input filter const op
    mConst_fliter = hiai::op::Const(opName + "_filter_const");
    {
        ge::TensorDesc fdesc(ge::Shape({1, scaleData->size(), 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT); // in o h w ?
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)scaleData->data(), scaleData->size() * sizeof(float));
        mConst_fliter.set_attr_value(filter);
    }
    // om input bias const op
    mConst_bias = hiai::op::Const(opName + "_bias_const");
    {
        ge::TensorDesc fdesc(ge::Shape({1, biasData->size(), 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)biasData->data(), biasData->size() * sizeof(float));
        mConst_bias.set_attr_value(filter);
    }
    if (inputs[0]->buffer().dimensions == 2) {
        vector<int32_t> shape;
        for (int32_t i = 0; i < inputs[0]->buffer().dimensions; i++) {
            shape.push_back(inputs[0]->buffer().dim[i].extent);
        }
        for (int32_t i = inputs[0]->buffer().dimensions; i < 4; i++) {
            shape.push_back(1);
        }
        shapeConst = hiai::op::Const(opName + "_shape_const");
        {
            ge::TensorDesc fdesc(ge::Shape({static_cast<int64_t>(shape.size())}), ge::FORMAT_NCHW, ge::DT_INT32); // in o h w ?
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t *)shape.data(), shape.size() * sizeof(int32_t));
            shapeConst.set_attr_value(filter);
        }
        shared_ptr<hiai::op::Reshape> reshape(new hiai::op::Reshape(opName + "_reshape"));
        (*reshape).set_input_x(*xOp.get()).set_input_shape(shapeConst);
        (*scale).set_input_x(*reshape.get()).set_input_scale(mConst_fliter).set_input_bias(mConst_bias);
        mNpuBackend->setOutputOps(mOp, {reshape, scale}, outputs);
    } else {
        (*scale).set_input_x(*xOp.get()).set_input_scale(mConst_fliter).set_input_bias(mConst_bias);
        mNpuBackend->setOutputOps(mOp, {scale}, outputs);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUScale>> __scale_op(OpType_Scale);

} // namespace MNN