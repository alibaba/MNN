//
//  NNAPIConvolution.cpp
//  MNN
//
//  Created by MNN on 2022/09/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NNAPIConvolution.hpp"

namespace MNN {


NNAPIConvolution::NNAPIConvolution(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NNAPICommonExecution(b, op) {
    isDepthwise = mOp->type() == OpType_ConvolutionDepthwise;
    isDeconv = mOp->type() == OpType_Deconvolution;
}
template<typename T>
static void NCHW2NHWC(const T* source, T* dest, int b, int c, int area, bool isdeconv) {
    if (isdeconv) { // (n, c, h, w) -> (c, h, w, n)
        for (int ci = 0; ci < c; ++ci) {
            auto dstStride0 = dest + area * b * ci;
            auto srcStride0 = source + area * ci;
            for (int i = 0; i < area; ++i) {
                auto dstStride1 = dstStride0 + b * i;
                auto srcStride1 = srcStride0 + i;
                for (int bi = 0; bi < b; ++bi) {
                    dstStride1[bi] = srcStride1[bi * area * c];
                }
            }
        }
    } else { // (n, c, h, w) -> (n, h, w, c)
        int sourceBatchsize = c * area;
        int destBatchSize   = sourceBatchsize;
        for (int bi = 0; bi < b; ++bi) {
            auto srcBatch = source + bi * sourceBatchsize;
            auto dstBatch = dest + bi * destBatchSize;
            for (int i = 0; i < area; ++i) {
                auto srcArea = srcBatch + i;
                auto dstArea = dstBatch + i * c;
                for (int ci = 0; ci < c; ++ci) {
                    dstArea[ci] = srcArea[ci * area];
                }
            }
        }
    }
}
ErrorCode NNAPIConvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    bool isQuantInt8 = TensorUtils::getDescribe(inputs[0])->quantAttr.get();
    auto conv2D     = mOp->main_as_Convolution2D();
    auto common     = conv2D->common();
    int kernelX     = common->kernelX();
    int kernelY     = common->kernelY();
    int strideX     = common->strideX();
    int strideY     = common->strideY();
    int dilateX     = common->dilateX();
    int dilateY     = common->dilateY();
    int group       = common->group();
    uint32_t outputCount = common->outputCount();
    auto padMod     = common->padMode();
    bool relu       = common->relu();
    bool relu6      = common->relu6();
    int top, left, bottom, right;
    if (nullptr != common->pads()) {
        MNN_ASSERT(common->pads()->size() >= 4);
        top = common->pads()->Get(0);
        left = common->pads()->Get(1);
        bottom = common->pads()->Get(2);
        right = common->pads()->Get(3);
    } else {
        top = common->padY();
        left = common->padX();
        bottom = common->padY();
        right = common->padX();
    }
    if (padMod == PadMode_SAME) {
        int inputY = (outputs[0]->height() - 1) * strideY + (kernelY - 1) * dilateY + 1;
        int inputX = (outputs[0]->width() - 1) * strideX + (kernelX - 1) * dilateX + 1;
        int padY = std::max(inputY - inputs[0]->height(), 0);
        int padX = std::max(inputX - inputs[0]->width(), 0);
        top = bottom = padY / 2;
        left = right = padX / 2;
        top += padY % 2;
        left += padX % 2;
    }
    // NNAPI inputs:
    // conv2d: [input, weight, bias, pad_left, pad_right, pad_top, pad_bottom, stride_w, stride_h, fusecode, NCHW/NHWC, dilate_w, dilate_h]
    // depthwise_conv2d: [input, weight, bias, pad_left, pad_right, pad_top, pad_bottom, stride_w, stride_h, multiplier, fusecode, NCHW/NHWC, dilate_w, dilate_h]
    auto inputIdxs = getTensorIdxs(inputs);
    // inputs not contain weight and bias, read from param
    if (inputs.size() < 3) {
        const void *weightPtr, *biasPtr;
        int weightSize, biasSize;
        if (isQuantInt8) {
            weightPtr = conv2D->quanParameter()->buffer()->data();
            weightSize = conv2D->quanParameter()->buffer()->size();
        } else if (nullptr != conv2D->quanParameter()) {
            quanCommon = ConvolutionCommon::load(conv2D, backend(), true);
            if (nullptr == quanCommon) {
                MNN_ERROR("Memory not Enough, can't extract IDST Convolution: %s \n", mOp->name()->c_str());
            }
            if (quanCommon->weightFloat.get() == nullptr) {
                MNN_PRINT("quanCommon->weightFloat.get() == nullptr \n");
            }
            // Back to float
            weightPtr  = quanCommon->weightFloat.get();
            weightSize = quanCommon->weightFloat.size();
        } else {
            weightPtr  = conv2D->weight()->data();
            weightSize = conv2D->weight()->size();
        }
        biasSize = conv2D->bias()->size();
        biasPtr  = conv2D->bias()->data();
        uint32_t inputCount = weightSize / (kernelX * kernelY * outputCount);
        uint32_t n  = outputCount;
        uint32_t c  = inputCount;
        uint32_t h = kernelY;
        uint32_t w = kernelX;
        if (isDepthwise) {
            n = 1;
            c = outputCount;
        }
        std::vector<uint32_t> weightDims {n, h, w, c};
        std::vector<uint32_t> biasDims {outputCount};
        if (isQuantInt8) {
            outputs[0]->buffer().type = halide_type_of<int8_t>();
            quantWeight.reset(new int8_t[weightSize]);
            quantBias.reset(new int32_t[biasSize]);
            // [outputCount, inputChannel, h, w] -> [outputCount, h, w, inputChannel]
            NCHW2NHWC<int8_t>(reinterpret_cast<const int8_t*>(weightPtr), quantWeight.get(), n, c, h * w, isDeconv);
            // bias to int32
            auto alpha = conv2D->quanParameter()->alpha()->data();
            auto scaleIn = conv2D->quanParameter()->scaleIn();
            auto scaleOut = conv2D->quanParameter()->scaleOut();
            auto scaleAlpha  = scaleIn / scaleOut;
            for (int i = 0; i < outputCount; i++) {
                quantBias[i] = static_cast<int32_t>(((float*)biasPtr)[i] / (scaleIn * alpha[i]));
            }
            weightPtr = quantWeight.get();
            biasPtr = quantBias.get();
            inputIdxs.push_back(buildConstant(weightPtr, weightSize, ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL, weightDims, alpha, 0));
            inputIdxs.push_back(buildConstant(biasPtr, biasSize * sizeof(int), ANEURALNETWORKS_TENSOR_INT32, biasDims));
        } else {
            // std::unique_ptr<float[]> nhwcWeight(new float[weightSize]);
            nhwcWeight.reset(new float[weightSize]);
            // [outputCount, inputChannel, h, w] -> [outputCount, h, w, inputChannel]
            NCHW2NHWC<float>(reinterpret_cast<const float*>(weightPtr), nhwcWeight.get(), n, c, h * w, isDeconv);
            inputIdxs.push_back(buildConstant(nhwcWeight.get(), weightSize * sizeof(float), ANEURALNETWORKS_TENSOR_FLOAT32, weightDims));
            inputIdxs.push_back(buildConstant(biasPtr, biasSize * sizeof(float), ANEURALNETWORKS_TENSOR_FLOAT32, biasDims));
        }
    }
    // pad
    inputIdxs.push_back(buildScalar(left));
    inputIdxs.push_back(buildScalar(right));
    inputIdxs.push_back(buildScalar(top));
    inputIdxs.push_back(buildScalar(bottom));
    // stride
    inputIdxs.push_back(buildScalar(strideX));
    inputIdxs.push_back(buildScalar(strideY));
    if (isDepthwise) {
        int multiplier = outputCount / group;
        inputIdxs.push_back(buildScalar(multiplier));
    }
    // fusecode
    FuseCode code = ANEURALNETWORKS_FUSED_NONE;
    if (relu) code = ANEURALNETWORKS_FUSED_RELU;
    if (relu6) code = ANEURALNETWORKS_FUSED_RELU6;
    inputIdxs.push_back(buildScalar(code));
    // NCHW/NHWC
    inputIdxs.push_back(buildScalar(mNCHW));
    // dilate
    if (dilateX > 1 || dilateY > 1) {
        inputIdxs.push_back(buildScalar(dilateX));
        inputIdxs.push_back(buildScalar(dilateY));
    }
    auto op = ANEURALNETWORKS_CONV_2D;
    if (mOp->type() == OpType_ConvolutionDepthwise) {
        op = ANEURALNETWORKS_DEPTHWISE_CONV_2D;
    } else if (mOp->type() == OpType_Deconvolution){
        op = ANEURALNETWORKS_TRANSPOSE_CONV_2D;
    }
    return buildOperation(op, inputIdxs, getTensorIdxs(outputs));
}

REGISTER_NNAPI_OP_CREATOR(NNAPIConvolution, OpType_Convolution)
REGISTER_NNAPI_OP_CREATOR(NNAPIConvolution, OpType_ConvolutionDepthwise)
REGISTER_NNAPI_OP_CREATOR(NNAPIConvolution, OpType_Deconvolution)
} // namespace MNN
