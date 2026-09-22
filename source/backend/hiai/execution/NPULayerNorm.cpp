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

    // HiAI NPUCL rejects the native LayerNorm operator for some rank-2 affine
    // shapes (the production graph contains both 256 and 8192 elements).
    // Decompose those mathematically equivalent 64-aligned cases into NPU
    // operators instead of allowing HiAI to partition them to CPUCL.
    const int normalizedSize = param->gamma() == nullptr ? 0 : param->gamma()->size();
    const bool normalizeLastAxis = param->axis() != nullptr && param->axis()->size() == 1 &&
        (param->axis()->Get(0) == -1 || param->axis()->Get(0) == 1);
    if (mNpuBackend->isExplicitHiAISession() &&
        normalizedSize > 0 && normalizedSize % 64 == 0 &&
        inputs[0]->dimensions() == 2 &&
        inputs[0]->length(0) == 1 && inputs[0]->length(1) == normalizedSize &&
        inputs[0]->elementSize() == normalizedSize &&
        outputs[0]->elementSize() == normalizedSize && normalizeLastAxis &&
        param->group() == 1 && !param->useRMSNorm() &&
        param->beta() != nullptr && static_cast<int>(param->beta()->size()) == normalizedSize) {
        constw = hiai::op::Const(opName + "_decomposed_gamma");
        constb = hiai::op::Const(opName + "_decomposed_beta");
        {
            ge::TensorDesc desc(ge::Shape({1, 1, 64, normalizedSize / 64}),
                                ge::FORMAT_NCHW, ge::DT_FLOAT);
            ge::TensorPtr gamma = std::make_shared<ge::Tensor>();
            gamma->SetTensorDesc(desc);
            gamma->SetData(reinterpret_cast<const uint8_t*>(param->gamma()->Data()),
                           normalizedSize * sizeof(float));
            constw.set_attr_value(gamma);

            ge::TensorPtr beta = std::make_shared<ge::Tensor>();
            beta->SetTensorDesc(desc);
            beta->SetData(reinterpret_cast<const uint8_t*>(param->beta()->Data()),
                          normalizedSize * sizeof(float));
            constb.set_attr_value(beta);
        }

        constReshapeShape = hiai::op::Const(opName + "_decomposed_shape");
        {
            const std::vector<int32_t> shape = {1, 1, 64, normalizedSize / 64};
            ge::TensorDesc desc(ge::Shape({4}), ge::FORMAT_NCHW, ge::DT_INT32);
            ge::TensorPtr tensor = std::make_shared<ge::Tensor>();
            tensor->SetTensorDesc(desc);
            tensor->SetData(reinterpret_cast<const uint8_t*>(shape.data()),
                            shape.size() * sizeof(int32_t));
            constReshapeShape.set_attr_value(tensor);
        }

        constOutputShape = hiai::op::Const(opName + "_decomposed_output_shape");
        {
            const std::vector<int32_t> shape = {1, normalizedSize, 1, 1};
            ge::TensorDesc desc(ge::Shape({4}), ge::FORMAT_NCHW, ge::DT_INT32);
            ge::TensorPtr tensor = std::make_shared<ge::Tensor>();
            tensor->SetTensorDesc(desc);
            tensor->SetData(reinterpret_cast<const uint8_t*>(shape.data()),
                            shape.size() * sizeof(int32_t));
            constOutputShape.set_attr_value(tensor);
        }

        constNormGamma = hiai::op::Const(opName + "_instance_gamma");
        constNormBeta = hiai::op::Const(opName + "_instance_beta");
        {
            ge::TensorDesc desc(ge::Shape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
            const float one = 1.0f;
            const float zero = 0.0f;
            ge::TensorPtr gamma = std::make_shared<ge::Tensor>();
            gamma->SetTensorDesc(desc);
            gamma->SetData(reinterpret_cast<const uint8_t*>(&one), sizeof(float));
            constNormGamma.set_attr_value(gamma);
            ge::TensorPtr beta = std::make_shared<ge::Tensor>();
            beta->SetTensorDesc(desc);
            beta->SetData(reinterpret_cast<const uint8_t*>(&zero), sizeof(float));
            constNormBeta.set_attr_value(beta);
        }

        auto reshapedInput = std::make_shared<hiai::op::Reshape>(opName + "_decomposed_input_reshape");
        reshapedInput->set_input_x(*xOp).set_input_shape(constReshapeShape);
        auto normalized = std::make_shared<hiai::op::InstanceNorm>(opName + "_instance_norm");
        normalized->set_input_x(*reshapedInput)
                  .set_input_gamma(constNormGamma)
                  .set_input_beta(constNormBeta)
                  .set_attr_data_format("NCHW")
                  .set_attr_epsilon(eps);
        auto scaled = std::make_shared<hiai::op::Mul>(opName + "_decomposed_scaled");
        scaled->set_input_x1(*normalized).set_input_x2(constw);
        auto shifted = std::make_shared<hiai::op::Add>(opName + "_decomposed_shifted");
        shifted->set_input_x1(*scaled).set_input_x2(constb);
        auto restoredOutput = std::make_shared<hiai::op::Reshape>(
            opName + "_decomposed_output_reshape");
        restoredOutput->set_input_x(*shifted).set_input_shape(constOutputShape);

        mNpuBackend->setOutputOps(mOp,
            {reshapedInput, normalized, scaled, shifted, restoredOutput}, outputs);
        return NO_ERROR;
    }

    (*layerNorm).set_input_x(*xOp.get())
                .set_input_gamma(constw)
                .set_input_beta(constb)
                .set_attr_epsilon(eps);
    mNpuBackend->setOutputOps(mOp, {layerNorm}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPULayerNorm>> __LayerNorm_op(OpType_LayerNorm);

} // namespace MNN
