#include <cmath>
#include "backend/NeuropilotBackend.hpp"
#include "ConvolutionTfliteConverter.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/ConvolutionCommon.hpp"
#include "flatbuffers/flexbuffers.h"
namespace MNN {
template<typename T> void _tranposeWeight(T* dstWeight, const T* originWeight, int group, int kernelSize, int ic, int oc) {
    for (int oz=0; oz<oc/group; ++oz) {
        for (int k=0; k<kernelSize; ++k) {
            for (int iz=0; iz<ic; ++iz) {
                dstWeight[oz * kernelSize * ic + k * ic + iz] = originWeight[oz * kernelSize * ic + k + iz * kernelSize];
            }
        }
    }
}

struct ConvConstTensors {
    std::shared_ptr<Tensor> weightTensor;
    std::shared_ptr<Tensor> biasTensor;
    int bits = 0;
};
static ConvConstTensors _getConstTensor(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, ConvertTflite* root) {
    ConvConstTensors constRes;
    if (inputs.size() > 1) {
        return constRes;
    }
    auto conv2d = op->main_as_Convolution2D();
    auto common = op->main_as_Convolution2D()->common();
    int oc = common->outputCount();
    int group = common->group();
    int ic = common->inputCount();
    if (0 == ic) {
        ic = inputs[0]->channel();
    }
    int kernelX = common->kernelX();
    int kernelY = common->kernelY();

    std::shared_ptr<Tensor> weightTensor, biasTensor;
    bool useQuant = TensorUtils::getDescribe(outputs[0])->applyQuant;
    const float* originWeight = nullptr;
    const float* originBias   = nullptr;
    int originWeightSize   = 0;
    int originBiasSize     = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    if (nullptr != conv2d->quanParameter()) {
        bool forceFloat = false;
        if (conv2d->quanParameter()->index() != nullptr) {
            // The weight is storage as float sparse, but the backend don't support sparse compute, expand it
            forceFloat = true;
        }
        quanCommon = ConvolutionCommon::load(op, nullptr, forceFloat, useQuant);
        if (nullptr == quanCommon) {
            MNN_ERROR("Memory not Enough, can't extract IDST Convolution: %s \n", op->name()->c_str());
            return constRes;
        }
        // Back to float
        originWeight     = quanCommon->weightFloat.get();
        originWeightSize = quanCommon->weightFloat.size();
    } else if (nullptr == conv2d->weight() || nullptr == conv2d->bias()) {
        MNN_ERROR("%s has no weight or bias. The model may be benchmark model, please revert the weight/bias firstly\n", op->name()->c_str());
    }
    if (nullptr == originWeight && nullptr != op->main_as_Convolution2D()->weight()) {
        originWeight     = op->main_as_Convolution2D()->weight()->data();
        originWeightSize = op->main_as_Convolution2D()->weight()->size();
    }
    if (nullptr == originBias && op->main_as_Convolution2D()->bias()) {
        originBias     = op->main_as_Convolution2D()->bias()->data();
        originBiasSize = op->main_as_Convolution2D()->bias()->size();
    }

    if (useQuant) {
        std::unique_ptr<tflite::QuantizationParametersT> parameters(new tflite::QuantizationParametersT);
        biasTensor.reset(Tensor::create<int32_t>({oc}));
        TensorUtils::getDescribe(biasTensor.get())->usage = Tensor::InsideDescribe::CONSTANT;
        std::unique_ptr<tflite::QuantizationParametersT> biasParameters(new tflite::QuantizationParametersT);
        int originBits = quanCommon->originBits;
        if (conv2d->symmetricQuan() != nullptr) {
            originBits = conv2d->symmetricQuan()->nbits();
        }
        // Remove too small weight quant scale
        auto removeSmallQuantScale = [](int8_t* weight, float* scale, bool async, int alphaSize, int weightSize) {
            if (async) {
                int kernelCount = alphaSize / 2;
                int kernelSize = weightSize / kernelCount;
                for (int z=0; z<kernelCount; ++z) {
                    auto alpha = scale[2 * z + 1];
                    auto weightZ = weight + z * kernelSize;
                    if (fabsf(alpha) <= 0.0000000001f) {
                        scale[2 * z + 1] = 1.0f;
                        ::memset(weightZ, 0, kernelSize);
                    }
                }
            } else {
                int kernelCount = alphaSize;
                int kernelSize = weightSize / kernelCount;
                for (int z=0; z<kernelCount; ++z) {
                    auto alpha = scale[z];
                    auto weightZ = weight + z * kernelSize;
                    if (fabsf(alpha) <= 0.0000000001f) {
                        scale[z] = 1.0f;
                        ::memset(weightZ, 0, kernelSize);
                    }
                }
            }
        };
        constRes.bits = quanCommon->originBits;

        if (!quanCommon->canUseInt4) {
            weightTensor.reset(Tensor::create<int8_t>({oc/group, kernelY, kernelX, ic}));
            auto dstWeight = weightTensor->host<int8_t>();
            auto originWeight = quanCommon->weight.get();
            TensorUtils::getDescribe(weightTensor.get())->usage = Tensor::InsideDescribe::CONSTANT;
            removeSmallQuantScale(originWeight, quanCommon->alpha.get(), quanCommon->asymmetric, quanCommon->alphaSize, quanCommon->weight.size());
            _tranposeWeight(dstWeight, originWeight, group, kernelX * kernelY, ic, oc);
            if (conv2d->symmetricQuan() != nullptr && conv2d->symmetricQuan()->nbits() <= 4) {
                constRes.bits = conv2d->symmetricQuan()->nbits();
            }
        } else {
            weightTensor.reset(Tensor::create<int8_t>({oc/group, kernelY, kernelX, ic}));
            auto dstWeight = weightTensor->host<int8_t>();
            auto originWeight = quanCommon->weight.get();
            TensorUtils::getDescribe(weightTensor.get())->usage = Tensor::InsideDescribe::CONSTANT;
            int weightSize = oc/group * ic * kernelX * kernelY;
            std::vector<int8_t> tmpWeightStorage;
            auto tmpWeight = dstWeight;
            if (1 < kernelX * kernelY) {
                tmpWeightStorage.resize(weightSize);
                tmpWeight = tmpWeightStorage.data();
            }
            for (int index=0; index<weightSize; ++index) {
                uint8_t w_ = originWeight[index / 2];
                int truew = index % 2 ? (w_ & 0x0f) : (w_ >> 4);
                tmpWeight[index] = (truew - 8);
            }
            removeSmallQuantScale(tmpWeight, quanCommon->alpha.get(), quanCommon->asymmetric, quanCommon->alphaSize, quanCommon->weight.size());

//            static bool gFirst = false;
//            if (!gFirst) {
//                gFirst = true;
//                printf("%s: %d - %d - %d - %d\n", op->name()->c_str(), tmpWeight[0], tmpWeight[1], tmpWeight[2], tmpWeight[3]);
//            }
            if (tmpWeight != dstWeight) {
                _tranposeWeight(dstWeight, tmpWeight, group, kernelX * kernelY, ic, oc);
            }
        }
        if (constRes.bits <= 4) {
            root->pBackend->setPackTensor(weightTensor.get(), 4);
        }
        float maxWeight = (1 << (originBits - 1)) - 1;
        float minWeight = -(1 << (originBits - 1)); 
        if (quanCommon->asymmetric) {
            auto alpha = quanCommon->alpha.get();
            for (int i=0; i<oc; ++i) {
                auto scale = alpha[2*i+1];
                auto bias = alpha[2*i];
                if (fabsf(scale) >= 0.000000001f) {
                    parameters->scale.emplace_back(scale);
                    parameters->zero_point.emplace_back((int)(-bias/scale));
                    parameters->max.emplace_back(maxWeight * scale + bias);
                    parameters->min.emplace_back(minWeight * scale + bias);
                } else {
                    parameters->scale.emplace_back(0.000000001f);
                    parameters->zero_point.emplace_back(0);
                    parameters->max.emplace_back(bias);
                    parameters->min.emplace_back(bias);
                }
            }
        } else {
            for (int i=0; i<oc; ++i) {
                auto scale = quanCommon->alpha.get()[i];
                if (fabsf(scale) < 0.000000001f) {
                    scale = 0.00000001f;
                }
                parameters->scale.emplace_back(scale);
                parameters->max.emplace_back(maxWeight * quanCommon->alpha.get()[i]);
                parameters->min.emplace_back(minWeight * quanCommon->alpha.get()[i]);
            }
            parameters->zero_point = std::vector<int64_t>(oc, 0);
        }
        if (oc == group && group > 1) {
            parameters->quantized_dimension = 3;
        } else {
            parameters->quantized_dimension = 0;
        }
        auto dstBias = biasTensor->host<int32_t>();
        for (int i=0; i<oc; ++i) {
            float weightScale = parameters->scale[i];
            float inputScale = TensorUtils::getDescribe(inputs[0])->quantAttr->scale;
            float biasScale = inputScale * weightScale;
            float bias = originBias[i];
            if (biasScale > 0.0f) {
                dstBias[i] = bias / (biasScale);
                biasParameters->scale.emplace_back(biasScale);
            } else if (biasScale < 0.0f) {
                dstBias[i] = bias / (-biasScale);
                biasParameters->scale.emplace_back(-biasScale);
            } else {
                dstBias[i] = 0;
                biasParameters->scale.emplace_back(1.0f);
            }
            biasParameters->max.emplace_back(bias);
            biasParameters->min.emplace_back(bias);
            biasParameters->zero_point.emplace_back(0);
        }
        root->pBackend->prepareTensorQuantInfo(weightTensor.get(), std::move(parameters));
        root->pBackend->prepareTensorQuantInfo(biasTensor.get(), std::move(biasParameters));
    } else {
        weightTensor.reset(Tensor::create<float>({oc/group, kernelY, kernelX, ic}));
        TensorUtils::getDescribe(weightTensor.get())->usage = Tensor::InsideDescribe::CONSTANT;
        auto dstWeight = weightTensor->host<float>();
        _tranposeWeight(dstWeight, originWeight, group, kernelX * kernelY, ic, oc);
        biasTensor.reset(Tensor::create<float>({oc}));
        TensorUtils::getDescribe(biasTensor.get())->usage = Tensor::InsideDescribe::CONSTANT;
        ::memcpy(biasTensor->host<void>(), originBias, originBiasSize * sizeof(float));
    }
    constRes.weightTensor = weightTensor;
    constRes.biasTensor = biasTensor;
    root->pBackend->setTensorName(weightTensor.get(), op->name()->str() + ".weight");
    root->pBackend->setTensorName(biasTensor.get(), op->name()->str() + ".bias");
    return constRes;
}
static ConvertTflite::CommandBuffer _makeFullConnect(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, ConvertTflite* root, ConvConstTensors constRes) {
    auto conv2d = op->main_as_Convolution2D();
    auto common = conv2d->common();
    ConvertTflite::CommandBuffer res;
    Tensor* fcInput = nullptr;
    Tensor* fcOutput = nullptr;
    int planeSize = inputs[0]->length(0);
    for (int i=2; i<inputs[0]->dimensions(); ++i) {
        planeSize *= inputs[0]->length(i);
    }
    {
        ConvertTflite::Command cmd;
        std::vector<int> reshapeSize = {1, planeSize, inputs[0]->channel()};
        auto reshapeTensor = ConvertTflite::getIntArrayTensor(reshapeSize);
        cmd.op.reset(new tflite::OperatorT());
        cmd.op->opcode_index = root->getOpIndex(tflite::BuiltinOperator_RESHAPE);
        std::shared_ptr<Tensor> reshapeOutput(Tensor::createDevice<float>(reshapeSize));
        TensorUtils::getDescribe(reshapeOutput.get())->applyQuant = TensorUtils::getDescribe(inputs[0])->applyQuant;
        TensorUtils::getDescribe(reshapeOutput.get())->quantAttr = TensorUtils::getDescribe(inputs[0])->quantAttr;
        cmd.outputs = {reshapeOutput.get()};
        cmd.inputs = {inputs[0], reshapeTensor.get()};
        fcInput = reshapeOutput.get();

        res.extraConst.emplace_back(reshapeTensor);
        res.extraConst.emplace_back(reshapeOutput);
        res.commands.emplace_back(std::move(cmd));
    }
    {
        // FC
        ConvertTflite::Command cmd;
        cmd.op.reset(new tflite::OperatorT());
        cmd.op->opcode_index = root->getCustomOpIndex("MTKEXT_FULLY_CONNECTED");
        flexbuffers::Builder builder;
        auto start = builder.StartMap();
        builder.Int("fused_activation_function", 0);
        builder.Bool("keep_num_dims", true);
        builder.EndMap(start);
        builder.Finish();
        int oc = outputs[0]->channel();
        int ic = inputs[0]->channel();
        cmd.op->custom_options = builder.GetBuffer();
        std::shared_ptr<Tensor> fCOutputR(Tensor::createDevice<float>({1, planeSize, outputs[0]->channel()}));
        fcOutput = fCOutputR.get();
        // Set weight to oc, ic
        constRes.weightTensor->buffer().dimensions = 2;
        constRes.weightTensor->setLength(0, oc);
        constRes.weightTensor->setLength(1, ic);
        TensorUtils::setLinearLayout(constRes.weightTensor.get());
        cmd.inputs = {fcInput, constRes.weightTensor.get(), constRes.biasTensor.get()};
        cmd.outputs = {fcOutput};
        TensorUtils::getDescribe(fcOutput)->applyQuant = TensorUtils::getDescribe(outputs[0])->applyQuant;
        TensorUtils::getDescribe(fcOutput)->quantAttr = TensorUtils::getDescribe(outputs[0])->quantAttr;

        res.commands.emplace_back(std::move(cmd));
        res.extraConst.emplace_back(fCOutputR);
        res.extraConst.emplace_back(constRes.weightTensor);
        res.extraConst.emplace_back(constRes.biasTensor);
    }
    {
        ConvertTflite::Command cmd;
        std::vector<int> reshapeSize = ConvertTflite::getShapeOfTensor(outputs[0]);
        auto reshapeTensor = ConvertTflite::getIntArrayTensor(reshapeSize);
        cmd.op.reset(new tflite::OperatorT());
        cmd.op->opcode_index = root->getOpIndex(tflite::BuiltinOperator_RESHAPE);
        std::shared_ptr<Tensor> reshapeOutput(Tensor::createDevice<float>(reshapeSize));
        cmd.outputs = {outputs[0]};
        cmd.inputs = {fcOutput, reshapeTensor.get()};
        res.commands.emplace_back(std::move(cmd));
        res.extraConst.emplace_back(reshapeTensor);
    }
    return res;
}

ConvertTflite::CommandBuffer ConvolutionTfliteConverter::onExecute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, ConvertTflite* root) {
    auto conv2d = op->main_as_Convolution2D();
    auto common = conv2d->common();
    bool useQuant = TensorUtils::getDescribe(outputs[0])->applyQuant;
    auto constTensors = _getConstTensor(op, inputs, outputs, root);
    if (op->type() == OpType_Convolution) {
        if (1 == common->kernelX() && 1 == common->kernelY() && 1 == common->strideX() && 1 == common->strideY() && inputs[0]->width() == outputs[0]->width() && inputs[0]->height() == outputs[0]->height() && constTensors.bits == 4) {
            // Linear to Convolution
            return _makeFullConnect(op, inputs, outputs, root, constTensors);
        }
    }
    ConvertTflite::CommandBuffer res;
    res.op = op;
    ConvertTflite::Command cmd;
    cmd.op.reset(new tflite::OperatorT());
    switch (op->type()) {
        case OpType_Convolution:
            cmd.op->opcode_index = root->getOpIndex(tflite::BuiltinOperator_CONV_2D);
            break;
        case OpType_ConvolutionDepthwise:
            cmd.op->opcode_index = root->getOpIndex(tflite::BuiltinOperator_DEPTHWISE_CONV_2D);
            break;
        default:
            break;
    }
    cmd.outputs = outputs;
    int oc = common->outputCount();
    int group = common->group();
    int ic = common->inputCount();
    if (0 == ic) {
        ic = inputs[0]->channel();
    }
    int kernelX = common->kernelX();
    int kernelY = common->kernelY();
    if (op->type() == OpType_Convolution) {
        cmd.op->builtin_options.type = tflite::BuiltinOptions_Conv2DOptions;
        cmd.op->builtin_options.value = new tflite::Conv2DOptionsT;
        auto dstCommon = cmd.op->builtin_options.AsConv2DOptions();
        if (useQuant) {
            dstCommon->quantized_bias_type = tflite::TensorType_INT32;
        }
        dstCommon->dilation_h_factor = common->dilateY();
        dstCommon->dilation_w_factor = common->dilateX();
        dstCommon->stride_h = common->strideY();
        dstCommon->stride_w = common->strideX();
        // TODO: Fix padding error
        switch (common->padMode()) {
            case PadMode_VALID:
                dstCommon->padding = tflite::Padding_VALID;
                break;
            case PadMode_SAME:
                dstCommon->padding = tflite::Padding_SAME;
                break;
            default:
                dstCommon->padding = tflite::Padding_SAME;
                break;
        }
        if (common->relu()) {
            dstCommon->fused_activation_function = tflite::ActivationFunctionType_RELU;
        }
        if (common->relu6()) {
            dstCommon->fused_activation_function = tflite::ActivationFunctionType_RELU6;
        }
    } else {
        cmd.op->builtin_options.type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
        cmd.op->builtin_options.value = new tflite::DepthwiseConv2DOptionsT;
        auto dstCommon = cmd.op->builtin_options.AsDepthwiseConv2DOptions();
        dstCommon->dilation_h_factor = common->dilateY();
        dstCommon->dilation_w_factor = common->dilateX();
        dstCommon->stride_h = common->strideY();
        dstCommon->stride_w = common->strideX();
        dstCommon->depth_multiplier = 1;
        // TODO: Fix padding error
        switch (common->padMode()) {
            case PadMode_VALID:
                dstCommon->padding = tflite::Padding_VALID;
                break;
            case PadMode_SAME:
                dstCommon->padding = tflite::Padding_SAME;
                break;
            default:
                dstCommon->padding = tflite::Padding_SAME;
                break;
        }
        if (common->relu()) {
            dstCommon->fused_activation_function = tflite::ActivationFunctionType_RELU;
        }
        if (common->relu6()) {
            dstCommon->fused_activation_function = tflite::ActivationFunctionType_RELU6;
        }
    }

    if (1 == inputs.size()) {
        cmd.inputs = {inputs[0], constTensors.weightTensor.get(), constTensors.biasTensor.get()};
        res.extraConst.emplace_back(constTensors.weightTensor);
        res.extraConst.emplace_back(constTensors.biasTensor);
    }
    res.commands.emplace_back(std::move(cmd));
    return res;
}
};

