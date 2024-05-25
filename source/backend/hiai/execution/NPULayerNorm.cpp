//
//  NPULayerNorm.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPULayerNorm.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPULayerNorm::NPULayerNorm(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {}

ErrorCode NPULayerNorm::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();
    auto param = mOp->main_as_LayerNorm();
    auto xOp = mNpuBackend->getInputOps(mOp);
    shared_ptr<hiai::op::LayerNorm> layerNorm(new hiai::op::LayerNorm(opName));
    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops = mNpuBackend->mGrapMap[inputIndex]; // x
    xOp = iops.back().first;

    constw = hiai::op::Const(opName + "_w_const");
    constb = hiai::op::Const(opName + "_b_const");
    if (param->gamma() == nullptr && param->beta() == nullptr) {
        auto shape = inputs[0]->shape();
        int32_t size = shape[shape.size()-1];
        vector<float> data(size, 1);
        vector<float> data1(size, 0);
        vector<int64_t> shape1{static_cast<int64_t>(size)};
        ge::TensorDesc fdesc(ge::Shape(shape1), ge::FORMAT_NCHW, ge::DT_FLOAT); 
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)data.data(), size * sizeof(float));
        constw.set_attr_value(filter);

        ge::TensorDesc fdesc1(ge::Shape(shape1), ge::FORMAT_NCHW, ge::DT_FLOAT); 
        ge::TensorPtr filter1 = std::make_shared<ge::Tensor>();
        filter1->SetTensorDesc(fdesc1);
        filter1->SetData((uint8_t *)data1.data(), size * sizeof(float));
        constb.set_attr_value(filter1);
    } else {
        uint32_t size = param->gamma()->size();
        vector<int64_t> shape1{size};
        ge::TensorDesc fdesc(ge::Shape(shape1), ge::FORMAT_NCHW, ge::DT_FLOAT); 
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)param->gamma()->Data(), size * sizeof(float));
        constw.set_attr_value(filter);

        size = param->beta()->size();
        vector<int64_t> shape2{size};
        ge::TensorDesc fdesc1(ge::Shape(shape2), ge::FORMAT_NCHW, ge::DT_FLOAT); 
        ge::TensorPtr filter1 = std::make_shared<ge::Tensor>();
        filter1->SetTensorDesc(fdesc1);
        filter1->SetData((uint8_t *)param->beta()->Data(), size * sizeof(float));
        constb.set_attr_value(filter1);
    }
    float eps = param->epsilon();
    (*layerNorm).set_input_x(*xOp.get())
                .set_input_gamma(constw)
                .set_input_beta(constb)
                .set_attr_epsilon(eps);
    mNpuBackend->setOutputOps(mOp, {layerNorm}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPULayerNorm>> __LayerNorm_op(OpType_LayerNorm);

} // namespace MNN