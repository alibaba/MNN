//
//  ConvolutionTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <MNN/AutoTime.hpp>
#include <cstdio>
#include <vector>
#include "CaffeOp_generated.h"
#include "MNN/MNNDefine.h"
#include "MNN/MNNForwardType.h"
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "CommonOpCreator.hpp"
#include "core/Session.hpp"
#include "core/TensorUtils.hpp"
#include "core/MemoryFormater.h"
#include "core/CommonCompute.hpp"

#define TEST_RANDOM_SEED 100

using namespace MNN;
using namespace MNN::Express;
static void reference_conv2d(const std::vector<float>& input, const std::vector<float>& weight,
                             const std::vector<float>& bias, std::vector<float>& output, std::vector<float>& outputDataSeparateBias, int batch, int ic, int oc,
                             int ih, int iw, PadMode mode, int pad_h, int pad_w, int kh, int kw, int stride,
                             int dilation, int group, ConvertFP32 functor) {
    int oh, ow;
    if (mode == PadMode_SAME) {
        oh    = (ih + stride - 1) / stride; // oh = ceil(ih / stride)
        ow    = (iw + stride - 1) / stride; // ow = ceil(iw / stride)
        pad_h = ((oh - 1) * stride + (kh - 1) * dilation + 1 - ih) / 2;
        pad_w = ((ow - 1) * stride + (kw - 1) * dilation + 1 - iw) / 2;
    } else {
        if (mode == PadMode_VALID) {
            pad_h = pad_w = 0;
        }
        oh = (ih + 2 * pad_h - (kh - 1) * dilation - 1) / stride + 1;
        ow = (iw + 2 * pad_w - (kw - 1) * dilation - 1) / stride + 1;
    }

    MNN_ASSERT(oc % group == 0 && ic % group == 0);
    if (oh <= 0 || ow <= 0) {
        output.clear();
        return;
    }
    output.resize(batch * oh * ow * oc);
    /*
      In CPUConvolutionDepthwise, bias function 'MNNAxByClampBroadcastUnit' is called separately with MNNConvRunForLineDepthwise,
      this would affect the precision when using bf16 or fp16.
      winograd convolution also did this.
      we keep the two result for checking.
    */
    outputDataSeparateBias.resize(batch * oh * ow * oc);

    int ocGroup = oc / group, icGroup = ic / group;
    for (int b = 0; b < batch; ++b) {
        for (int oz = 0; oz < oc; ++oz) {
            int gId = oz / ocGroup;
            for (int oy = 0; oy < oh; ++oy) {
                for (int ox = 0; ox < ow; ++ox) {
                    float sum = 0;
                    auto destOffset = ((b * oc + oz) * oh + oy) * ow + ox;
                    for (int sz = gId * icGroup; sz < (gId + 1) * icGroup; ++sz) {
                        for (int ky = 0; ky < kh; ++ky) {
                            for (int kx = 0; kx < kw; ++kx) {
                                int ix = ox * stride + kx * dilation - pad_w, iy = oy * stride + ky * dilation - pad_h;
                                float xValue = 0.0f;
                                if (ix >= 0 && ix < iw && iy >= 0 && iy < ih) {
                                    xValue = input[(((b * ic + sz) * ih + iy) * iw + ix)];
                                }
                                float convertX = functor(xValue);
                                float convertW = functor(weight[(((gId * ocGroup + oz % ocGroup) * icGroup + sz % icGroup) * kh + ky) * kw + kx]);
                                sum += convertX * convertW;
                            }
                        }
                    }

                    output[destOffset] = functor(sum + functor(bias[oz]));
                    outputDataSeparateBias[destOffset] = functor(functor(sum) + functor(bias[oz]));
                }
            }
        }
    }
}

VARP _Conv(VARP weight, VARP bias, VARP x, PaddingMode pad = VALID, INTS stride = {1, 1}, INTS dilate = {1, 1},
           int group = 1, INTS pads = {0, 0}, MNN::SparseAlgo sparseAlgo = MNN::SparseAlgo_RANDOM, int sparseBlockOC = 1, bool sparse = false) {
std::unique_ptr<OpT> convOp(new OpT);
convOp->type = OpType_Convolution;
auto shape   = weight -> getInfo();
if (NHWC == shape->order) {
    weight = _Transpose(weight, {0, 3, 1, 2});
    shape  = weight->getInfo();
    }
    auto channel    = std::vector<int>{shape->dim[0], shape->dim[1]};
    auto kernelSize = std::vector<int>{shape->dim[3], shape->dim[2]};
    if (1 == channel[1] && channel[0] == group) {
        convOp->type = OpType_ConvolutionDepthwise;
        channel[1] = group;
    }
    convOp->main.type  = OpParameter_Convolution2D;
    convOp->main.value = new Convolution2DT;
    auto conv2D        = convOp->main.AsConvolution2D();
    conv2D->common.reset(new Convolution2DCommonT);
    if (pads.size() == 2) {
        conv2D->common->padX        = pads[0];
        conv2D->common->padY        = pads[1];
    } else {
        conv2D->common->pads = std::move(pads);
    }
    conv2D->common->padMode     = _convertPadMode(pad);
    conv2D->common->strideX     = stride[0];
    conv2D->common->strideY     = stride[1];
    conv2D->common->group       = group;
    conv2D->common->outputCount = channel[0];
    conv2D->common->inputCount  = channel[1];
    conv2D->common->dilateX     = dilate[0];
    conv2D->common->dilateY     = dilate[1];
    conv2D->common->kernelX     = kernelSize[0];
    conv2D->common->kernelY     = kernelSize[1];
    if (sparse) {
        size_t weightNNZElement, weightBlockNumber = 0;
        int weightSize = weight->getInfo()->size;
        int biasSize = bias->getInfo()->size;
        CommonCompute::statisticWeightSparsity(weightNNZElement, weightBlockNumber, weight->readMap<float>(), biasSize, weightSize / biasSize, sparseBlockOC);

        std::unique_ptr<MNN::AttributeT> arg1(new MNN::AttributeT);
        arg1->key = "sparseBlockOC";
        arg1->i = sparseBlockOC;

        std::unique_ptr<MNN::AttributeT> arg2(new MNN::AttributeT);;
        arg2->key = "sparseBlockKernel";
        arg2->i = 1;

        std::unique_ptr<MNN::AttributeT> arg3(new MNN::AttributeT);;
        arg3->key = "NNZElement";
        arg3->i = weightNNZElement;

        std::unique_ptr<MNN::AttributeT> arg4(new MNN::AttributeT);;
        arg4->key = "blockNumber";
        arg4->i = weightBlockNumber;

        flatbuffers::FlatBufferBuilder builder;
        std::vector<flatbuffers::Offset<MNN::Attribute>> argsVector;
        auto sparseArg1 = MNN::CreateAttribute(builder, arg1.get());
        auto sparseArg2 = MNN::CreateAttribute(builder, arg2.get());
        auto sparseArg3 = MNN::CreateAttribute(builder, arg3.get());
        auto sparseArg4 = MNN::CreateAttribute(builder, arg4.get());

        argsVector.emplace_back(sparseArg1);
        argsVector.emplace_back(sparseArg2);
        argsVector.emplace_back(sparseArg3);
        argsVector.emplace_back(sparseArg4);

        auto sparseArgs = builder.CreateVectorOfSortedTables<MNN::Attribute>(&argsVector);
        auto sparseCom = MNN::CreateSparseCommon(builder, sparseAlgo, sparseArgs);
        builder.Finish(sparseCom);
        auto sparseComPtr = flatbuffers::GetRoot<MNN::SparseCommon>(builder.GetBufferPointer())->UnPack();

        conv2D->sparseParameter.reset(sparseComPtr);
    }
    if (nullptr == bias) {
        return (Variable::create(Expr::create(convOp.get(), {x, weight})));
    }
    return (Variable::create(Expr::create(convOp.get(), {x, weight, bias})));
}
VARP _Conv(std::vector<float>&& weight, std::vector<float>&& bias, VARP x, INTS channel, INTS kernelSize,
           PaddingMode pad = VALID, INTS stride = {1, 1}, INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0},
           bool relu = false, bool relu6 = false, MNN::SparseAlgo sparseAlgo = MNN::SparseAlgo_RANDOM, int sparseBlockOC = 1, bool sparese = false, float threshold = 0.0f) {
    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type = OpType_Convolution;
    if (channel[0] == channel[1] && channel[0] == group) {
        convOp->type = OpType_ConvolutionDepthwise;
    }
    convOp->main.type  = OpParameter_Convolution2D;
    convOp->main.value = new Convolution2DT;
    auto conv2D        = convOp->main.AsConvolution2D();
    conv2D->common.reset(new Convolution2DCommonT);
    conv2D->common->padMode     = _convertPadMode(pad);
    if (pads.size() == 2) {
        conv2D->common->padX        = pads[0];
        conv2D->common->padY        = pads[1];
    } else {
        conv2D->common->pads = std::move(pads);
    }
    conv2D->common->strideX     = stride[0];
    conv2D->common->strideY     = stride[1];
    conv2D->common->group       = group;
    conv2D->common->outputCount = channel[1];
    conv2D->common->inputCount  = channel[0];
    conv2D->common->dilateX     = dilate[0];
    conv2D->common->dilateY     = dilate[1];
    conv2D->common->kernelX     = kernelSize[0];
    conv2D->common->kernelY     = kernelSize[1];
    conv2D->common->relu6 = relu6;
    conv2D->common->relu = relu;
    conv2D->common->threshold = threshold;
    MNN_ASSERT(weight.size() == channel[1] * (channel[0] / group) * kernelSize[0] * kernelSize[1]);
    conv2D->weight = std::move(weight);
    MNN_ASSERT(bias.size() == channel[1]);
    conv2D->bias = std::move(bias);
    if (sparese) {
        size_t weightNNZElement, weightBlockNumber = 0;
        CommonCompute::statisticWeightSparsity(weightNNZElement, weightBlockNumber, conv2D->weight.data(), conv2D->bias.size(), conv2D->weight.size() / conv2D->bias.size(), sparseBlockOC);

        std::unique_ptr<MNN::AttributeT> arg1(new MNN::AttributeT);
        arg1->key = "sparseBlockOC";
        arg1->i = sparseBlockOC;

        std::unique_ptr<MNN::AttributeT> arg2(new MNN::AttributeT);;
        arg2->key = "sparseBlockKernel";
        arg2->i = 1;

        std::unique_ptr<MNN::AttributeT> arg3(new MNN::AttributeT);;
        arg3->key = "NNZElement";
        arg3->i = weightNNZElement;

        std::unique_ptr<MNN::AttributeT> arg4(new MNN::AttributeT);;
        arg4->key = "blockNumber";
        arg4->i = weightBlockNumber;

        flatbuffers::FlatBufferBuilder builder;
        std::vector<flatbuffers::Offset<MNN::Attribute>> argsVector;
        auto sparseArg1 = MNN::CreateAttribute(builder, arg1.get());
        auto sparseArg2 = MNN::CreateAttribute(builder, arg2.get());
        auto sparseArg3 = MNN::CreateAttribute(builder, arg3.get());
        auto sparseArg4 = MNN::CreateAttribute(builder, arg4.get());

        argsVector.emplace_back(sparseArg1);
        argsVector.emplace_back(sparseArg2);
        argsVector.emplace_back(sparseArg3);
        argsVector.emplace_back(sparseArg4);

        auto sparseArgs = builder.CreateVectorOfSortedTables<MNN::Attribute>(&argsVector);
        auto sparseCom = MNN::CreateSparseCommon(builder, sparseAlgo, sparseArgs);
        builder.Finish(sparseCom);
        auto sparseComPtr = flatbuffers::GetRoot<MNN::SparseCommon>(builder.GetBufferPointer())->UnPack();

        conv2D->sparseParameter.reset(sparseComPtr);
        CommonCompute::compressFloatWeightToSparse(convOp.get());
    }
    return (Variable::create(Expr::create(convOp.get(), {x})));
}

VARP _Conv(float weight, float bias, VARP x, INTS channel, INTS kernelSize, PaddingMode pad = VALID,
           INTS stride = {1, 1}, INTS dilate = {1, 1}, int group = 1, MNN::SparseAlgo sparseAlgo = MNN::SparseAlgo_RANDOM, int sparseBlockOC = 1, bool sparse = false) {
    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type = OpType_Convolution;
    if (channel[0] == channel[1] && channel[0] == group) {
        convOp->type = OpType_ConvolutionDepthwise;
    }
    convOp->main.type  = OpParameter_Convolution2D;
    convOp->main.value = new Convolution2DT;
    auto conv2D        = convOp->main.AsConvolution2D();
    conv2D->common.reset(new Convolution2DCommonT);
    conv2D->common->padMode     = _convertPadMode(pad);
    conv2D->common->strideX     = stride[0];
    conv2D->common->strideY     = stride[1];
    conv2D->common->group       = group;
    conv2D->common->outputCount = channel[1];
    conv2D->common->inputCount  = channel[0];
    conv2D->common->dilateX     = dilate[0];
    conv2D->common->dilateY     = dilate[1];
    conv2D->common->kernelX     = kernelSize[0];
    conv2D->common->kernelY     = kernelSize[1];
    conv2D->weight.resize(channel[1] * (channel[0] / group) * kernelSize[0] * kernelSize[1]);
    std::fill(conv2D->weight.begin(), conv2D->weight.end(), weight);
    conv2D->bias.resize(channel[1]);
    std::fill(conv2D->bias.begin(), conv2D->bias.end(), bias);
    if (sparse) {
        size_t weightNNZElement, weightBlockNumber = 0;
        CommonCompute::statisticWeightSparsity(weightNNZElement, weightBlockNumber, conv2D->weight.data(), conv2D->bias.size(), conv2D->weight.size() / conv2D->bias.size(), sparseBlockOC);

        std::unique_ptr<MNN::AttributeT> arg1(new MNN::AttributeT);
        arg1->key = "sparseBlockOC";
        arg1->i = sparseBlockOC;

        std::unique_ptr<MNN::AttributeT> arg2(new MNN::AttributeT);;
        arg2->key = "sparseBlockKernel";
        arg2->i = 1;

        std::unique_ptr<MNN::AttributeT> arg3(new MNN::AttributeT);;
        arg3->key = "NNZElement";
        arg3->i = weightNNZElement;

        std::unique_ptr<MNN::AttributeT> arg4(new MNN::AttributeT);;
        arg4->key = "blockNumber";
        arg4->i = weightBlockNumber;

        flatbuffers::FlatBufferBuilder builder;
        std::vector<flatbuffers::Offset<MNN::Attribute>> argsVector;
        auto sparseArg1 = MNN::CreateAttribute(builder, arg1.get());
        auto sparseArg2 = MNN::CreateAttribute(builder, arg2.get());
        auto sparseArg3 = MNN::CreateAttribute(builder, arg3.get());
        auto sparseArg4 = MNN::CreateAttribute(builder, arg4.get());

        argsVector.emplace_back(sparseArg1);
        argsVector.emplace_back(sparseArg2);
        argsVector.emplace_back(sparseArg3);
        argsVector.emplace_back(sparseArg4);

        auto sparseArgs = builder.CreateVectorOfSortedTables<MNN::Attribute>(&argsVector);
        auto sparseCom = MNN::CreateSparseCommon(builder, sparseAlgo, sparseArgs);
        builder.Finish(sparseCom);
        auto sparseComPtr = flatbuffers::GetRoot<MNN::SparseCommon>(builder.GetBufferPointer())->UnPack();

        conv2D->sparseParameter.reset(sparseComPtr);
    }
    return (Variable::create(Expr::create(convOp.get(), {x})));
}

class ConvolutionCommonTest : public MNNTestCase {
protected:
    bool mSparse = false;
    bool mBenchSpeed = false;
public:
    virtual ~ConvolutionCommonTest() = default;
    virtual bool run (int precision) {
        return true;
    }

public:
    virtual void generateWeight(std::vector<float>& weightData, int ic, int oc, int kh, int kw, int dilation, int group, int sparseBlockOC) {
        for (int i = 0; i < group * (oc / group) * (ic / group) * kw * kh; i++) {
            auto data      = ((((i / kw)% 1317) * ((i / kh) % 1317)) % 1317 + i / ic + i / oc + (((oc - i) % 1317) * ic) % 1317 + i * ((oc - i) % 1317)) % 1317;
            auto floatData      = (float)(data % 255) / 255.0f / 1000.0f;
            weightData.push_back(floatData);
        }

    }
    ConvolutionCommonTest& speed() {
        mBenchSpeed = true;
        return *this;
    }
    bool test(MNNForwardType type, const std::string& device_name, const std::string& test_op_name, int batch,
                     int ic, int oc, int ih, int iw, PadMode mode, int pad_h, int pad_w, int kh, int kw, int stride,
              int dilation, int group, int precision, MNN::SparseAlgo sparseAlgo = MNN::SparseAlgo_RANDOM, int sparseBlockOC = 1, bool debug = false, bool testRelu = false) {
        using namespace MNN::Express;
        std::map<PadMode, Express::PaddingMode> padMap = {
            {PadMode_CAFFE, CAFFE}, {PadMode_VALID, VALID}, {PadMode_SAME, SAME}};
        std::vector<float> weightData, biasData;

        generateWeight(weightData, ic, oc, kh, kw, dilation, group, sparseBlockOC);

        for (int i = 0; i < oc; i++) {
            auto data      = (((i / kw) % 1317) * ((i / kh) % 1317) + i / ic + i / oc + (oc - i) * ic + i * (oc - i)) % 1317;
            auto floatData = (float)(data % 255) / 255.0f;
            data           = data * data;
            biasData.push_back(floatData);
            // biasData.push_back(0.0f);
        }

        std::vector<float> inputData, outputData, outputDataSeparateBias;
        for (int i = 0; i < ih * iw * ic * batch; ++i) {
            auto data      = ((i / kw) % 1317) * ((i / kh) % 1317) + ((i / ic)% 1317) * ((i / oc) % 1317) + ((oc - i) % 1317) * ic + (i % 1317) * ((oc - i) % 1317);
            data = data % 1317;
            data           = (data * data) % 1317;
            auto floatData = (float)(data % 255) / 255.0f;
            inputData.push_back(floatData);
        }
        reference_conv2d(inputData, weightData, biasData, outputData, outputDataSeparateBias, batch, ic, oc, ih, iw, mode, pad_h, pad_w, kh, kw,
                         stride, dilation, group, FP32Converter[precision]);
        if (outputData.size() == 0) {
            return true;
        }

        auto input = _Input({batch, ic, ih, iw}, NCHW, halide_type_of<float>());
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
         // Multi Conv
        // if (group == 1 || (group == ic && ic == oc)) {
        //     VARP weightVar;
        //     if (group == 1) {
        //         weightVar = _Const(weightData.data(), {oc, ic, kh, kw}, NCHW, halide_type_of<float>());
        //     } else {
        //         weightVar = _Const(weightData.data(), {oc, ic / group, kh, kw}, NCHW, halide_type_of<float>());
        //     }
        //     auto biasVar = _Const(biasData.data(), {oc}, NCHW, halide_type_of<float>());
        //     auto out     = _Conv(weightVar, biasVar, _Convert(input, NC4HW4), padMap[mode], {stride, stride}, {dilation, dilation}, group,
        //                      {pad_w, pad_h}, sparseAlgo, sparseBlockOC, mSparse);
        //     out = _Convert(out, NCHW);
        //     auto outputPtr = out->readMap<float>();
        //     if (!checkVectorByRelativeError<float>(outputPtr, outputData.data(), outputData.size(), 0.05)) {
        //         MNN_PRINT("multi expect:\t real:\n");
        //         for (int i = 0; i < outputData.size(); ++i)
        //         {
        //             MNN_PRINT("%f\t, %f\n", outputData[i], outputPtr[i]);
        //         }
        //         MNN_ERROR("%s(%s) multi test failed!\n", test_op_name.c_str(), device_name.c_str());
        //         return false;
        //     }
        // }
        // Single Conv
        std::vector<std::pair<bool, bool>> activations = {
            {false, false},
        };

        if (testRelu) {
            activations = {
                {false, false},
                {true, false},
                {false, true}
            };
        }
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 100; // winograd error in 16-bits is relatively large
        for (auto activation : activations) {
            auto newWeight = weightData;
            auto newBias = biasData;
            auto toutputData = outputData;
            auto toutputBias = outputDataSeparateBias;
            float maxV = -10000.0f;
            float minV = 10000.0f;
            if (activation.first) {
                for (auto& t : toutputData) {
                    maxV = ALIMAX(maxV, t);
                    minV = ALIMIN(minV, t);
                    t = ALIMAX(0.0f, t);
                }
                for (auto& t : toutputBias) {
                    maxV = ALIMAX(maxV, t);
                    minV = ALIMIN(minV, t);
                    t = ALIMAX(0.0f, t);
                }
            }
            if (activation.second) {
                for (auto& t : toutputData) {
                    t = ALIMAX(0.0f, t);
                    t = ALIMIN(6.0f, t);
                }
                for (auto& t : toutputBias) {
                    t = ALIMAX(0.0f, t);
                    t = ALIMIN(6.0f, t);
                }
            }
            auto output = _Conv(std::move(newWeight), std::move(newBias), input, {ic, oc}, {kw, kh}, padMap[mode],
                                {stride, stride}, {dilation, dilation}, group, {pad_w, pad_h}, activation.first, activation.second, sparseAlgo, sparseBlockOC, mSparse);

            // difference below 0.5% relative error is considered correct.
            auto outputPtr = output->readMap<float>();

            // when using low precision, im2col or strassen convolution error rate to reference value is about 1e-4, winograd has larger error rate.

           if (!checkVectorByRelativeError<float>(outputPtr, toutputData.data(), toutputBias.data(), toutputData.size(), 0.001 * errorScale)) {
               MNN_PRINT("precision:%d, expect:\t expect2:\t real:\t\n", precision);
               for (int i = 0; i < toutputData.size(); ++i)
               {
                   MNN_PRINT("%f\t, %f\t, %f\n", toutputData[i],toutputBias[i], outputPtr[i]);
               }
               MNN_ERROR("%s(%s) test failed!\n", test_op_name.c_str(), device_name.c_str());
               return false;
           }
        }
        return true;
    }
};

class SparseConvolutionCommonTest : public ConvolutionCommonTest {

public:
    SparseConvolutionCommonTest() {
        mSparse = true;
    }
    virtual void generateWeight(std::vector<float>& weightData, int ic, int oc, int kh, int kw, int dilation, int group, int sparseBlockOC) {
        assert(sparseBlockOC);
        int ocEven = (group * (oc / group) / sparseBlockOC) * sparseBlockOC;
        int reduceDimLength = (ic / group) * kw * kh;
        weightData.resize(group * (oc / group) * reduceDimLength);
        size_t ioc = 0;
        size_t index = 0;
        for (; ioc < ocEven; ioc += sparseBlockOC) {
            for (size_t i = 0; i < reduceDimLength; i++) {
                index = ioc * reduceDimLength + i;
                bool isZero = index % 4 != 0;
                for (int iblock = 0; iblock < sparseBlockOC; iblock++) {
                    if(isZero) {
                        weightData[index] = 0;
                    } else {
                        auto data      = (index / kw) * (index / kh) + index / ic + index / oc + (oc - index) * ic + index * (oc - index);
                        weightData[index] = (float)(data % 255) / 255.0f / 1000.0f;
                    }
                    index += reduceDimLength;
                }
            }
        }
        for (; ioc < oc; ioc++) {
            for (size_t i = 0; i < reduceDimLength; i++) {
                index = ioc * reduceDimLength + i;
                bool isZero = index % 4 != 0;
                if(isZero) {
                    weightData[index] = 0;
                } else {
                    auto data      = (index / kw) * (index / kh) + index / ic + index / oc + (oc - index) * ic + index * (oc - index);
                    weightData[index] = (float)(data % 255) / 255.0f;
                }
            }
        }
        return;
    }
};

class ConvolutionInt8CommonTest : public ConvolutionCommonTest {
public:
    virtual ~ConvolutionInt8CommonTest() = default;
    virtual bool run (int precision) {
        return true;
    }

public:
    virtual void generateWeight(std::vector<float>& weightData, int ic, int oc, int kh, int kw, int dilation, int group, int sparseBlockOC) {
        auto numbers = group * (oc / group) * (ic / group) * kw * kh;
        weightData.resize(numbers);
        float rate = 1.0f / numbers;
        for (int ri = 0; ri < numbers; ri++) {
            int data = ri - numbers / 2;
            auto floatData = (float)(data) * rate;
            weightData[ri] = floatData;
        }
    }
    ConvolutionInt8CommonTest& speed() {
        mBenchSpeed = true;
        return *this;
    }

    bool testUnit(MNNForwardType type, const std::string& device_name, const std::string& test_op_name, int batch,
                     int ic, int oc, int ih, int iw, PadMode mode, int pad_h, int pad_w, int kh, int kw, int stride,
                  int dilation, int group, int precision, MNN::SparseAlgo sparseAlgo = MNN::SparseAlgo_RANDOM, int sparseBlockOC = 1, bool debug = false, int nbit = 4, bool async = false) {
        using namespace MNN::Express;
        std::map<PadMode, Express::PaddingMode> padMap = {
            {PadMode_CAFFE, CAFFE}, {PadMode_VALID, VALID}, {PadMode_SAME, SAME}};
        std::vector<float> weightData, biasData;

        generateWeight(weightData, ic, oc, kh, kw, dilation, group, sparseBlockOC);

        for (int i = 0; i < oc; i++) {
            auto data      = (((i / kw) % 1317) * ((i / kh) % 1317) + i / ic + i / oc + (oc - i) * ic + i * (oc - i)) % 1317;
            auto floatData = (float)(data % 255) / 255.0f;
            biasData.push_back(floatData);
        }

        std::vector<float> inputData, outputData, outputDataSeparateBias;
        float rate = 1.0f;
        if (ih * iw * ic * batch > 10000) {
            // Avoid exceed fp16 limit
            rate = 0.01f;
        }
        for (int i = 0; i < ih * iw * ic * batch; ++i) {
            auto data      = ((i / kw) % 1317) * ((i / kh) % 1317) + ((i / ic)% 1317) * ((i / oc) % 1317) + ((oc - i) % 1317) * ic + (i % 1317) * ((oc - i) % 1317);
            data = data % 1317;
            data           = (data * data) % 1317;
            auto floatData = (float)(data % 255) / 255.0f * rate;
            inputData.push_back(floatData);
        }
        float fac = 1.23;
        int res = 10;
        float tail = 0.2;
        float threshold = (float)(1 << (nbit - 1)) - 1.0f;
        float clampMin = -threshold;
        if (async) {
            clampMin = -threshold - 1;
        }
        int kernel_size = ic * kw * kh;
        std::vector<int8_t> quantWeight(oc*ic*kw*kh);
        std::vector<float> wScale;
        if (async) {

            wScale.resize(2 * oc);
            for (int k = 0; k < oc; ++k) {
                int beginIndex = k * kernel_size;
                auto minMax = findMinMax(weightData.data() + beginIndex, kernel_size);
                auto minValue = minMax.first;
                wScale[2*k] = minMax.first;
                auto absMax = minMax.second - minMax.first;
                wScale[2*k+1] = 0;
                
                float quantscale = 1.0f;
                if (absMax >= 0.000001f) {
                    wScale[2 * k + 1] = absMax / (threshold - clampMin);
                    quantscale = 1.0f / wScale[2*k+1];
                    
                }
                float* ptr = weightData.data() + beginIndex;
                for (int i = 0; i < kernel_size; ++i) {
                    int8_t quantValue = int8_t(std::round((ptr[i] - minValue) * quantscale + clampMin));
                    float floatValue = ((float)quantValue - clampMin) * wScale[2*k+1] + minValue;
                    quantWeight[k * kernel_size + i] = quantValue;
                    ptr[i] = floatValue;
                }
            }
        } else {
            wScale.resize(oc);
            for (int k = 0; k < oc; ++k) {
                int beginIndex = k * kernel_size;
                auto absMax = findAbsMax(weightData.data() + beginIndex, kernel_size);
                wScale[k] = absMax / threshold;

                float* ptr = weightData.data() + beginIndex;
                for (int i = 0; i < kernel_size; ++i) {
                    int8_t quantVal = (int8_t)(fmax(fmin(round(ptr[i] / wScale[k]), threshold), clampMin));
                    quantWeight[k * kernel_size + i] = quantVal;
                    ptr[i] = (float)quantVal * wScale[k];
                }
            }
        }
        reference_conv2d(inputData, weightData, biasData, outputData, outputDataSeparateBias, batch, ic, oc, ih, iw, mode, pad_h, pad_w, kh, kw, stride, dilation, group, FP32Converter[precision]);
        if (outputData.size() == 0) {
            return true;
        }

        auto input = _Input({batch, ic, ih, iw}, NCHW, halide_type_of<float>());
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        // Single Conv
        auto weightLength = weightData.size();
        float errorScale = 1.0f;
        if (nbit == 4 && weightLength > 10000) {
            errorScale = 50.0f;
        }
        int memory = MNNTestSuite::get()->pStaus.memory;
        if (precision > MNN::BackendConfig::Precision_High || memory > MNN::BackendConfig::Memory_High) {
            errorScale = 100.0f;
        }
        std::vector<std::pair<bool, bool>> activations = {
            {false, false},
            {true, false},
            {false, true}
        };
        for (auto& activation : activations) {
            auto output     = _HybridConv(weightData, biasData, wScale, input,
                                          {ic, oc}, {kw, kh}, padMap[mode],  {stride, stride}, {dilation, dilation}, group, {pad_w, pad_h}, activation.first, activation.second, nbit, async);
            auto toutputData = outputData;
            float maxV = -10000.0f;
            float minV = 10000.0f;
            if (activation.first) {
                for (auto& t : toutputData) {
                    maxV = ALIMAX(maxV, t);
                    minV = ALIMIN(minV, t);
                    t = ALIMAX(0.0f, t);
                }
//                MNN_PRINT("Max: %f -> Min:%f\n", maxV, minV);
            }
            if (activation.second) {
                for (auto& t : toutputData) {
                    t = ALIMAX(0.0f, t);
                    t = ALIMIN(6.0f, t);
                }
            }

            // difference below 0.5% relative error is considered correct.
            output = _Convert(output, NCHW);
            auto outputPtr = output->readMap<float>();
            // when using low precision, im2col or strassen convolution error rate to reference value is about 1e-4, winograd has larger error rate.

            if (!checkVectorByRelativeError<float>(outputPtr, toutputData.data(), toutputData.data(), toutputData.size(), 0.001 * errorScale)) {
                MNN_PRINT("precision:%d, memory:%d\n", precision, memory);
                MNN_PRINT("expect:\t real:\t\n");
                for (int i = 0; i < toutputData.size(); ++i)
                {
                    MNN_PRINT("%f, %f\n", toutputData[i], outputPtr[i]);
                }
                MNN_PRINT("output shape: n=%d c=%d h=%d w=%d\n", output->getInfo()->dim[0], output->getInfo()->dim[1], output->getInfo()->dim[2], output->getInfo()->dim[3]);
                MNN_ERROR("%s(%s) test failed for %d bits, async=%d , relu: %d, relu6: %d!\n", test_op_name.c_str(), device_name.c_str(), nbit, async, activation.first, activation.second);
                return false;
            }
        }
        return true;
    }
    bool test(MNNForwardType type, const std::string& device_name, const std::string& test_op_name, int batch,
                     int ic, int oc, int ih, int iw, PadMode mode, int pad_h, int pad_w, int kh, int kw, int stride,
              int dilation, int group, int precision, MNN::SparseAlgo sparseAlgo = MNN::SparseAlgo_RANDOM, int sparseBlockOC = 1, bool debug = false) {
        auto res = testUnit(type, device_name, test_op_name, batch, ic, oc, ih, iw, mode, pad_h, pad_w, kh, kw, stride, dilation, group, precision, sparseAlgo, sparseBlockOC, debug, 8, true);
        if (!res) {
            FUNC_PRINT(1);
            return res;
        }
        res = res && testUnit(type, device_name, test_op_name, batch, ic, oc, ih, iw, mode, pad_h, pad_w, kh, kw, stride, dilation, group, precision, sparseAlgo, sparseBlockOC, debug, 8, false);
        if (!res) {
            FUNC_PRINT(1);
            return res;
        }
        res = res && testUnit(type, device_name, test_op_name, batch, ic, oc, ih, iw, mode, pad_h, pad_w, kh, kw, stride, dilation, group, precision, sparseAlgo, sparseBlockOC, debug, 4, true);
        if (!res) {
            FUNC_PRINT(1);
            return res;
        }
        res = res && testUnit(type, device_name, test_op_name, batch, ic, oc, ih, iw, mode, pad_h, pad_w, kh, kw, stride, dilation, group, precision, sparseAlgo, sparseBlockOC, debug, 4, false);
        if (!res) {
            FUNC_PRINT(1);
            return res;
        }
        return res;
    }
};

template <typename ConvolutionType>
class ConvolutionSpeedTest : public ConvolutionType {
public:
    virtual ~ConvolutionSpeedTest() = default;


protected:
    static bool test(MNNForwardType type, const std::string& device_name, int precision, MNN::SparseAlgo sparseAlgo, int MaxBlock) {
        int padW = 1, padH = 1, iw = 28, ih = 28, ic = 128, oc = 128;
        std::vector<std::vector<int>> kernels = {
            {1, 1}, {3, 3}, {5, 5}, {7, 1}, {1, 7} // {w, h}
        };
        std::vector<std::string> titles = {"3x3", "5x5", "1x7", "7x1"};
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = ConvolutionType().speed().test(type, device_name, "Conv2D Speed",
                            1, ic, oc, ih, iw, PadMode_CAFFE, padH, padW, kernels[i][1], kernels[i][0], 1, 1, 1, precision);
            if (!res) {
                MNN_ERROR("Error for test kernel %s for ConvolutionSpeedTest\n", titles[i].c_str());
                return false;
            }
        }
        return true;
    }
};


template <typename ConvolutionType>
class ConvolutionTest : public ConvolutionType {
public:
    virtual ~ConvolutionTest() = default;

protected:
    static bool test(MNNForwardType type, const std::string& device_name, int precision, MNN::SparseAlgo sparseAlgo, std::vector<int> blocks, bool checkSpectial = false) {
        int ocStep = 1;
        int icStep = 1;
        int isStep = 3;
        std::vector<int> ocSize = {
            1, 4, 3, 10, 17
        };
        std::vector<int> icSize = {
            1, 4, 3, 8, 11
        };
        std::vector<int> isSize = {
            1, 7, 9
        };

        for (int b = 1; b <= 2; b++) {
            for (auto oc : ocSize) {
                for (auto ic : icSize) {
                    for (auto is : isSize) {
                        for (int kw = 1; kw <= 3 && kw <= is; kw+=2) {
                            for (int kh = 1; kh <= 3 && kh <= is; kh+=3) {
                                for (int d = 1; d <= 2; d++) {
                                    if (d > is || d * (kw - 1) + 1 > is || d * (kh - 1) + 1 > is)
                                        continue;

                                    for (int s = 1; s <= 2; s++) {
                                        for (auto block : blocks) {
                                            for (int p = 0; p <= 1; p++) {
                                                bool succ =  ConvolutionType().test(type, device_name, "Conv2D", b, ic, oc, is, is, PadMode_CAFFE, p, p, kh, kw, s, d, 1, precision, sparseAlgo, block, false);
                                                if (!succ) {
                                                    MNN_ERROR(
                                                        "Error for conv b=%d, oc=%d, ic=%d, ih=%d, "
                                                        "iw=%d,kw=%d,kh=%d,d=%d,s=%d,p=%d, block=%d\n",
                                                        b, oc, ic, is, is, kw, kh, d, s, p, block);
                                                    return false;
                                                }
                                            }


                                            {
                                                bool succ =
                                                    ConvolutionType().test(type, device_name, "Conv2D", b, ic, oc, is,
                                                                                is, PadMode_VALID, 0, 0, kh, kw, s, d, 1, precision, sparseAlgo, block, false);
                                                if (!succ) {
                                                    MNN_ERROR(
                                                        "Error for conv b=%d, oc=%d, ic=%d, is=%d, is=%d, kw=%d,kh=%d,d=%d,s=%d, block=%d, "
                                                        "valid pad\n",
                                                        b, oc, ic, is, is, kw, kh, d, s, block);
                                                    return false;
                                                }
                                            }

                                            {
                                                bool succ =
                                                    ConvolutionType().test(type, device_name, "Conv2D", b, ic, oc, is,
                                                                                is, PadMode_SAME, 0, 0, kh, kw, s, d, 1, precision, sparseAlgo, block, false);
                                                if (!succ) {
                                                    MNN_ERROR(
                                                        "Error for conv b=%d, oc=%d, ic=%d, is=%d, is=%d, kw=%d, kh=%d, d=%d, s=%d, block=%d, "
                                                        "same pad\n",
                                                        b, oc, ic, is, is, kw, kh, d, s, block);
                                                    return false;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (!checkSpectial) {
            return true;
        }
        // Check Long convolution
         bool succ =
            ConvolutionType().test(type, device_name, "Conv2D", 1, 256, 256, 24, 24, PadMode_SAME, 0, 0, 3, 3, 1, 1, 1, precision, sparseAlgo, 4, false);
        if (!succ) {
            MNN_ERROR("Error for long conv\n");
            return false;
        }
        // // uncovered and easily wrong case.
        succ =
            ConvolutionType().test(type, device_name, "Conv2D", 1, 3, 16, 256, 256, PadMode_CAFFE, 1, 1, 3, 3, 1, 1, 1, precision, sparseAlgo, 4, false);
        if (!succ) {
            MNN_ERROR("Error in pick up case 1.\n");
            return false;
        }
        succ =
            ConvolutionType().test(type, device_name, "Conv2D", 1, 1, 8, 28, 28, PadMode_CAFFE, 2, 2, 5, 5, 1, 1, 1, precision, sparseAlgo, 1, false);
        if (!succ) {
            MNN_ERROR("Error in pick up case 2.\n");
            return false;
        }

        succ =
            ConvolutionType().test(type, device_name, "Conv2D", 1, 1, 8, 14, 14, PadMode_CAFFE, 2, 2, 5, 5, 1, 1, 1, precision, sparseAlgo, 1, false);
        if (!succ) {
            MNN_ERROR("Error in pick up case 3.\n");
            return false;
        }

        return true;
    }

};

using DenseConvolutionTest = ConvolutionTest<ConvolutionCommonTest>;
class ConvolutionTestOnCPU : public DenseConvolutionTest {
public:
    ~ConvolutionTestOnCPU() = default;
    virtual bool run(int precision) {
        return DenseConvolutionTest::test(MNN_FORWARD_CPU, "CPU", precision, MNN::SparseAlgo_RANDOM, {1}, true);
    }
};

using DenseConvolutionInt8Test = ConvolutionTest<ConvolutionInt8CommonTest>;
class ConvolutionInt8Test : public DenseConvolutionInt8Test {
public:
    ~ConvolutionInt8Test() = default;
    virtual bool run(int precision) {
        return DenseConvolutionInt8Test::test(MNN_FORWARD_OPENCL, "OpenCL", precision, MNN::SparseAlgo_RANDOM, {1});
    }
};

using DenseConvolutionSpeedTest = ConvolutionSpeedTest<ConvolutionCommonTest>;
class ConvolutionSpeedTestOnCPU : public DenseConvolutionSpeedTest {
public:
    ~ConvolutionSpeedTestOnCPU() = default;
    virtual bool run(int precision) {
        return DenseConvolutionSpeedTest::test(MNN_FORWARD_CPU, "CPU", precision, MNN::SparseAlgo_RANDOM, 1);
    }
};


using SparseConvolutionTest = ConvolutionTest<SparseConvolutionCommonTest>;
class SparseConvolutionTestOnCPU : public SparseConvolutionTest {
public:
    ~SparseConvolutionTestOnCPU() = default;
    virtual bool run(int precision) {
        std::vector<int> blocks = {1, 4, 8};
        return SparseConvolutionTest::test(MNN_FORWARD_CPU, "CPU", precision, MNN::SparseAlgo_SIMD_OC, blocks);
    }
};

class DepthwiseConvolutionTest : public ConvolutionCommonTest {
public:
    virtual ~DepthwiseConvolutionTest() = default;

protected:
    virtual bool run(int precision) {
        return test(MNN_FORWARD_CPU, "CPU", precision);
    }

    static bool test(MNNForwardType type, const std::string& device_name, int precision) {
        srand(TEST_RANDOM_SEED);
        // correct unit test
        for (int b = 1; b <= 2; b++) {
            for (int oc = 4; oc <= 16; oc *= 2) {
                for (int ic = oc; ic <= oc; ic++) {
                    for (int isw = 1; isw <= 8; isw+=2) {
                        for (int ish = 1; ish <= 8; ish*=2) {
                            for (int kw = 1; kw <= 4; kw++) {
                                for (int kh = 1; kh <= 4; kh++) {
                                    for (int d = 1; d <= 2; d++) {
                                        for (int s = 1; s <= 2; s++) {
                                            for (int p = 0; p <= std::min(kw, kh); p++) {
                                                // depthwise <==> group == outputChannel
                                                bool succ = ConvolutionCommonTest().test(
                                                                                         type, device_name, "DepthwiseConv2D", b, ic, oc, ish, isw, PadMode_CAFFE,
                                                                                         p, p, kh, kw, s, d, oc, precision, MNN::SparseAlgo_RANDOM, 1, false, true);
                                                if (!succ) {
                                                    MNN_ERROR(
                                                              "Error for dw oc=%d, ic=%d, ih=%d, iw = %d, kw=%d,kh=%d,d=%d,s=%d,p=%d\n", oc,
                                                              ic, ish, isw, kw, kh, d, s, p);
#ifdef DEBUG
                                                    //Rerun test for easy to test
                                                    ConvolutionCommonTest().test(
                                                                                             type, device_name, "DepthwiseConv2D", b, ic, oc, ish, isw, PadMode_CAFFE,
                                                                                             p, p, kh, kw, s, d, oc, precision);
#endif
                                                    return false;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // memory leak unit test
        int b = 1, oc = 4, ic = oc, group = oc, is = 2, p = 1, kh = 3, kw = 3, s = 2, d = 1;
        return ConvolutionCommonTest().test(type, device_name, "DepthwiseConv2D", b, ic, oc, is, is,
                                           PadMode_CAFFE, p, p, kh, kw, s, d, group, precision);
    }
};

class GroupConvolutionTest : public ConvolutionCommonTest {
public:
    virtual ~GroupConvolutionTest() = default;

protected:
    static bool test(MNNForwardType type, const std::string& device_name, int precision) {
        srand(TEST_RANDOM_SEED);
        bool succ = ConvolutionCommonTest().test(
            type, device_name, "GroupConv2D", 2, 8, 16, 1, 1, PadMode_CAFFE,
            0, 0, 1, 1, 1, 1, 2, precision, MNN::SparseAlgo_RANDOM, 1, false);
        return succ;
        for (int b = 1; b <= 2; b++) {
            for (int g = 2; g <= 4; g *= 2) {
                for (int oc = g * 4; oc <= 4 * g * 4; oc += g * 4) {
                    for (int ic = g * 4; ic <= 4 * g * 4; ic += g * 4) {
                        for (int is = 1; is <= 8; is *= 2) {
                            for (int kw = 1; kw <= 3 && kw <= is; kw++) {
                                for (int kh = 1; kh <= 3 && kh <= is; kh++) {
                                    for (int d = 1; d <= 2; d++) {
                                        if (d > std::min(kw, kh) || d * (std::max(kw, kh) - 1) + 1 > is)
                                            continue;
                                        for (int s = 1; s <= 2; s++) {
                                            for (int p = 0; p <= 1; p++) {
                                                bool debug = false;
                                                bool succ = ConvolutionCommonTest().test(
                                                    type, device_name, "GroupConv2D", b, ic, oc, is, is, PadMode_CAFFE,
                                                    p, p, kh, kw, s, d, g, precision, MNN::SparseAlgo_RANDOM, 1, debug);
                                                if (!succ) {
                                                    MNN_PRINT("convolution group b=%d, oc=%d, ic=%d, is=%d,kw=%d,kh=%d,d=%d,s=%d,g=%d,p=%d\n", b, oc,
                                                    ic, is, kw, kh, d, s, g, p);
                                                    return false;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return true;
    }
};

class GroupConvolutionTestOnCPU : public GroupConvolutionTest {
public:
    virtual ~GroupConvolutionTestOnCPU() = default;
    virtual bool run(int precision) {
        return GroupConvolutionTest::test(MNN_FORWARD_CPU, "CPU", precision);
    }
};

class Conv1x1OpenCLTest : public ConvolutionCommonTest {
public:
    virtual bool run(int precision) {
        // 专门测试1x1卷积的参数组合
        return test(MNN_FORWARD_OPENCL, "OpenCL", "Conv1x1", 
                   1,    // batch = 1
                   4096,   // ic = 64 (能被4整除)
                   4096,  // oc = 128 (能被4整除)  
                   1,   // ih = 56
                   1,   // iw = 56
                   PadMode_VALID, // 填充模式
                   0, 0, // pad_h, pad_w = 0
                   1, 1, // kh, kw = 1 (关键：1x1卷积)
                   1,    // stride = 1
                   1,    // dilation = 1
                   1,    // group = 1
                   precision);
    }
};

class Conv1x1SparseOpenCLTest : public ConvolutionCommonTest{
public:

    bool test(MNNForwardType type, const std::string& device_name, const std::string& test_op_name, int batch,
                        int ic, int oc, int ih, int iw, PadMode mode, int pad_h, int pad_w, int kh, int kw, int stride,
                        int dilation, int group, int precision, MNN::SparseAlgo sparseAlgo = MNN::SparseAlgo_RANDOM, int sparseBlockOC = 1, bool debug = false, bool testRelu = false){
        using namespace MNN::Express;
        std::map<PadMode, Express::PaddingMode> padMap = {
            {PadMode_CAFFE, CAFFE}, {PadMode_VALID, VALID}, {PadMode_SAME, SAME}};
        std::vector<float> weightData, biasData;

        generateWeight(weightData, ic, oc, kh, kw, dilation, group, sparseBlockOC);

        for (int i = 0; i < oc; i++) {
            auto data      = (((i / kw) % 1317) * ((i / kh) % 1317) + i / ic + i / oc + (oc - i) * ic + i * (oc - i)) % 1317;
            auto floatData = (float)(data % 255) / 255.0f;
            data           = data * data;
            biasData.push_back(floatData);
            //biasData.push_back(0.0f);
        }

        std::vector<float> referenceData, inputData, outputData, outputDataSeparateBias;
        for (int i = 0; i < ih * iw * ic * batch; ++i) {
            float floatData = 0.0f;
            if (i >= ih * iw * ic * batch / 2)
            {
                floatData = (float)i;
            }
            referenceData.push_back(floatData);
            inputData.push_back(float(i));
        }
        //打乱数据
        std::vector<int> indices(referenceData.size());
        for (int i = 0; i < indices.size(); ++i) indices[i] = i;

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        std::vector<float> referenceDataShuffled, inputDataShuffled;
        for (int i = 0; i < indices.size(); ++i) {
            referenceDataShuffled.push_back(referenceData[indices[i]]);
            inputDataShuffled.push_back(inputData[indices[i]]);
        }
        referenceData = referenceDataShuffled;
        inputData = inputDataShuffled;
        float threshold = ih * iw * ic * batch / 2;
        reference_conv2d(referenceData, weightData, biasData, outputData, outputDataSeparateBias, batch, ic, oc, ih, iw, mode, pad_h, pad_w, kh, kw,
                         stride, dilation, group, FP32Converter[precision]);
        if (outputData.size() == 0) {
            return true;
        }
        auto input = _Input({batch, ic, ih, iw}, NCHW, halide_type_of<float>());
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));

        std::vector<std::pair<bool, bool>> activations = {
            {false, false},
        };

        if (testRelu) {
            activations = {
                {false, false},
                {true, false},
                {false, true}
            };
        }
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 100; // winograd error in 16-bits is relatively large
        for (auto activation : activations) {
            auto newWeight = weightData;
            auto newBias = biasData;
            auto toutputData = outputData;
            auto toutputBias = outputDataSeparateBias;
            float maxV = -10000.0f;
            float minV = 10000.0f;
            if (activation.first) {
                for (auto& t : toutputData) {
                    maxV = ALIMAX(maxV, t);
                    minV = ALIMIN(minV, t);
                    t = ALIMAX(0.0f, t);
                }
                for (auto& t : toutputBias) {
                    maxV = ALIMAX(maxV, t);
                    minV = ALIMIN(minV, t);
                    t = ALIMAX(0.0f, t);
                }
            }
            if (activation.second) {
                for (auto& t : toutputData) {
                    t = ALIMAX(0.0f, t);
                    t = ALIMIN(6.0f, t);
                }
                for (auto& t : toutputBias) {
                    t = ALIMAX(0.0f, t);
                    t = ALIMIN(6.0f, t);
                }
            }
            
            auto output = _Conv(std::move(newWeight), std::move(newBias), input, {ic, oc}, {kw, kh}, padMap[mode],
                                {stride, stride}, {dilation, dilation}, group, {pad_w, pad_h}, activation.first, activation.second, sparseAlgo, sparseBlockOC, mSparse, threshold);

            // difference below 0.5% relative error is considered correct.
            auto outputPtr = output->readMap<float>();

            // when using low precision, im2col or strassen convolution error rate to reference value is about 1e-4, winograd has larger error rate.

            if (!checkVectorByRelativeError<float>(outputPtr, toutputData.data(), toutputBias.data(), toutputData.size(), 0.001 * errorScale)) {
                MNN_PRINT("precision:%d, expect:\t expect2:\t real:\t\n", precision);
                for (int i = 0; i < toutputData.size(); ++i)
                {
                    MNN_PRINT("%f\t, %f\t, %f\n", toutputData[i],toutputBias[i], outputPtr[i]);
                }
                MNN_ERROR("%s(%s) test failed!\n", test_op_name.c_str(), device_name.c_str());
                return false;
            }
        }
        return true;
    }
    virtual bool run(int precision) {
        // 专门测试1x1卷积的参数组合
        return test(MNN_FORWARD_OPENCL, "OpenCL", "Conv1x1", 
                   1,    // batch = 1
                   4096,   // ic = 64 (能被4整除)
                   4096,  // oc = 128 (能被4整除)  
                   1,   // ih = 56
                   1,   // iw = 56
                   PadMode_VALID, // 填充模式
                   0, 0, // pad_h, pad_w = 0
                   1, 1, // kh, kw = 1 (关键：1x1卷积)
                   1,    // stride = 1
                   1,    // dilation = 1
                   1,    // group = 1
                   precision);
    }
};

class ConvolutionInt84GEMVSparseTest : public ConvolutionCommonTest {
public:
    virtual ~ConvolutionInt84GEMVSparseTest() = default;
    virtual bool run(int precision) {
        return testUnit(MNN_FORWARD_OPENCL, "OpenCL", "ConvInt81x1", 
                        1, 
                        4096, 
                        4096 , 
                        1, 
                        1, 
                        MNN::PadMode_VALID, 
                        0, 0, 
                        1, 1, 
                        1, 
                        1, 
                        1,
                        precision,
                        MNN::SparseAlgo_RANDOM,
                        1,
                        false,
                        4,
                        false);
    }
private:

std::tuple<std::vector<float>, std::vector<float>, float, float> generateHalfStructuredSparseData(
    int ih, int iw, int ic, int batch, float rate = 1.0f) {
    
    std::vector<float> inputData;
    int totalElements = ih * iw * ic * batch;
    int groupSize = 4;
    int numGroups = (totalElements + groupSize - 1) / groupSize;  // 向上取整

    // 设置阈值范围
    float sparseThreshold = 0.1f * rate;  // 稀疏值的上限
    float denseMinValue = 0.2f * rate;    // 密集值的下限

    inputData.reserve(totalElements);

    // 预定义几种稀疏模式：4个位置中选2个为稀疏
    std::vector<std::vector<int>> sparsePatterns = {
        {0, 1},  // 前两个为稀疏
        {0, 2},  // 第1和第3个为稀疏  
        {0, 3},  // 第1和第4个为稀疏
        {1, 2},  // 第2和第3个为稀疏
        {1, 3},  // 第2和第4个为稀疏
        {2, 3}   // 后两个为稀疏
    };

    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_int_distribution<int> patternDist(0, sparsePatterns.size() - 1);

    // 为每个组生成数据
    for (int groupIdx = 0; groupIdx < numGroups; ++groupIdx) {
        // 随机选择一种稀疏模式
        auto selectedPattern = sparsePatterns[patternDist(g)];
        
        // 标记哪些位置是稀疏的
        std::vector<bool> isSparse(groupSize, false);
        for (int sparsePos : selectedPattern) {
            isSparse[sparsePos] = true;
        }
        
        // 为这个组生成4个值
        for (int elemInGroup = 0; elemInGroup < groupSize; ++elemInGroup) {
            int globalIdx = groupIdx * groupSize + elemInGroup;
            if (globalIdx >= totalElements) break;
            
            float value;
            if (isSparse[elemInGroup]) {
                // 稀疏位置：生成小于阈值的值
                auto seed = globalIdx * 97 + elemInGroup * 193;
                auto data = ((seed * 1103515245) + 12345) & 0x7fffffff;
                data = data % 1000;
                value = (float)(data % 500) / 5000.0f * rate;  // 范围：[0, sparseThreshold)
            } else {
                // 密集位置：生成大于阈值的值
                auto seed = globalIdx * 73 + elemInGroup * 149;
                auto data = ((seed * 1103515245) + 12345) & 0x7fffffff;
                data = data % 1000;
                value = denseMinValue + (float)(data % 800) / 1000.0f * rate;  // 范围：[denseMinValue, rate]
            }
            
            inputData.push_back(value);
        }
    }

    // 计算实际的稀疏阈值
    float actualThreshold = (sparseThreshold + denseMinValue) / 2.0f;

    // ====== 调试：打印前几组的inputData ======
    printf("\n=== DEBUG: Half-Structured Sparse InputData ===\n");
    printf("Rate: %.6f, SparseThreshold: %.6f, DenseMinValue: %.6f, ActualThreshold: %.6f\n", 
           rate, sparseThreshold, denseMinValue, actualThreshold);
    
    int maxGroupsToPrint = std::min(20, numGroups);
    for (int groupIdx = 0; groupIdx < maxGroupsToPrint; ++groupIdx) {
        printf("Group %2d: ", groupIdx);
        
        int sparseCount = 0;
        int denseCount = 0;
        
        for (int elemInGroup = 0; elemInGroup < groupSize; ++elemInGroup) {
            int globalIdx = groupIdx * groupSize + elemInGroup;
            if (globalIdx < inputData.size()) {
                printf("%.6f ", inputData[globalIdx]);
                if (inputData[globalIdx] < actualThreshold) {
                    sparseCount++;
                } else {
                    denseCount++;
                }
            }
        }
        
        printf("| Sparse: %d, Dense: %d %s\n", 
               sparseCount, denseCount, 
               (sparseCount == 2 && denseCount == 2) ? "✓" : "✗ ERROR!");
    }

    // 生成参考数据（小于阈值的置零）
    std::vector<float> referenceData;
    referenceData.reserve(inputData.size());

    for (int i = 0; i < inputData.size(); ++i) {
        if (inputData[i] < actualThreshold) {
            referenceData.push_back(0.0f);
        } else {
            referenceData.push_back(inputData[i]);
        }
    }

    // ====== 调试：验证参考数据的结构 ======
    printf("\n=== DEBUG: Half-Structured ReferenceData Verification ===\n");
    
    int totalZeros = 0;
    int totalNonZeros = 0;
    int correctGroups = 0;
    int incorrectGroups = 0;

    for (int groupIdx = 0; groupIdx < maxGroupsToPrint; ++groupIdx) {
        printf("Group %2d: ", groupIdx);
        
        int groupZeros = 0;
        int groupNonZeros = 0;
        
        for (int elemInGroup = 0; elemInGroup < groupSize; ++elemInGroup) {
            int globalIdx = groupIdx * groupSize + elemInGroup;
            if (globalIdx < referenceData.size()) {
                printf("%.6f ", referenceData[globalIdx]);
                if (referenceData[globalIdx] == 0.0f) {
                    groupZeros++;
                    totalZeros++;
                } else {
                    groupNonZeros++;
                    totalNonZeros++;
                }
            }
        }
        
        bool isCorrect = (groupZeros == 2 && groupNonZeros == 2);
        if (isCorrect) {
            correctGroups++;
        } else {
            incorrectGroups++;
        }
        
        printf("| Zeros: %d, NonZeros: %d %s\n", 
               groupZeros, groupNonZeros, isCorrect ? "✓" : "✗ ERROR!");
    }

    // 验证整体统计
    for (int i = maxGroupsToPrint * groupSize; i < referenceData.size(); i += groupSize) {
        int groupZeros = 0;
        for (int j = 0; j < groupSize && (i + j) < referenceData.size(); ++j) {
            if (referenceData[i + j] == 0.0f) {
                groupZeros++;
                totalZeros++;
            } else {
                totalNonZeros++;
            }
        }
        if (groupZeros == 2) {
            correctGroups++;
        } else {
            incorrectGroups++;
        }
    }

    float sparsityRatio = (float)totalZeros / referenceData.size();

    printf("\n=== Half-Structured Sparse Summary ===\n");
    printf("- Total elements: %d\n", (int)inputData.size());
    printf("- Total groups (4-channel): %d\n", numGroups);
    printf("- Correct groups (2 zeros + 2 non-zeros): %d\n", correctGroups);
    printf("- Incorrect groups: %d\n", incorrectGroups);
    printf("- Total zeros: %d\n", totalZeros);
    printf("- Total non-zeros: %d\n", totalNonZeros);
    printf("- Sparsity ratio: %.2f%% (should be ~50%%)\n", sparsityRatio * 100.0f);
    printf("- Threshold used: %.6f\n", actualThreshold);
    printf("- Success rate: %.2f%%\n", (float)correctGroups / numGroups * 100.0f);
    printf("==========================================\n\n");
    
    return std::make_tuple(inputData, referenceData, sparsityRatio, actualThreshold);
}
    // 生成随机稀疏数据（中位数阈值）
std::tuple<std::vector<float>, std::vector<float>, float, float> generateRandomSparseData(
    int ih, int iw, int ic, int batch, float rate = 1.0f) {
    
    std::vector<float> inputData;
    int totalElements = ih * iw * ic * batch;
    
    // 生成随机数据
    for (int i = 0; i < totalElements; ++i) {
        auto data = (i * 73) ^ ((i * 149) >> 3) ^ ((i * 251) << 2);  
        data = ((data * 1103515245) + 12345) & 0x7fffffff;  
        data = data % 1317;
        data = (data * data) % 1317;
        auto floatData = (float)(data % 255) / 255.0f * rate;
        inputData.push_back(floatData);
    }

    // 计算中位数作为阈值
    std::vector<float> sortedData = inputData;  
    std::sort(sortedData.begin(), sortedData.end());  

    float sparseThreshold;
    int totalSize = sortedData.size();
    if (totalSize % 2 == 0) {
        sparseThreshold = (sortedData[totalSize/2 - 1] + sortedData[totalSize/2]) / 2.0f;
    } else {
        sparseThreshold = sortedData[totalSize/2];
    }

    // 生成参考数据（小于阈值的置零）
    std::vector<float> referenceData;
    referenceData.reserve(inputData.size());
    for (int i = 0; i < inputData.size(); ++i) {
        if (inputData[i] < sparseThreshold) {
            referenceData.push_back(0.0f);  
        } else {
            referenceData.push_back(inputData[i]);  
        }
    }

    // 计算稀疏率
    int zeroCount = 0;
    for (const auto& val : referenceData) {
        if (val == 0.0f) zeroCount++;
    }
    float sparsityRatio = (float)zeroCount / referenceData.size();
    
    printf("Random Sparse Info:\n");
    printf("- Total elements: %d\n", totalElements);
    printf("- Sparsity ratio: %.2f%%\n", sparsityRatio * 100.0f);
    printf("- Threshold used: %.6f\n", sparseThreshold);
    
    return std::make_tuple(inputData, referenceData, sparsityRatio, sparseThreshold);
}

// 生成结构化稀疏数据（4-channel组级稀疏，30%稀疏率）
std::tuple<std::vector<float>, std::vector<float>, float, float> generateStructuredSparseData(
    int ih, int iw, int ic, int batch, float rate = 1.0f, float targetSparsityRatio = 0.8f) {
    
    std::vector<float> inputData;
    int totalElements = ih * iw * ic * batch;
    int groupSize = 4;
    int numGroups = (totalElements + groupSize - 1) / groupSize;  // 向上取整
    int sparseGroups = (int)(numGroups * targetSparsityRatio);
    int denseGroups = numGroups - sparseGroups;

    // 设置阈值范围
    float sparseThreshold = 0.1f * rate;  // 稀疏组的阈值
    float denseMinValue = 0.2f * rate;    // 密集组的最小值

    inputData.reserve(totalElements);

    // 生成组的标记：0表示稀疏组，1表示密集组
    std::vector<int> groupLabels;
    groupLabels.reserve(numGroups);

    // 先添加稀疏组标记
    for (int i = 0; i < sparseGroups; ++i) {
        groupLabels.push_back(0);
    }
    // 再添加密集组标记
    for (int i = 0; i < denseGroups; ++i) {
        groupLabels.push_back(1);
    }

    // 随机打乱组的分布
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(groupLabels.begin(), groupLabels.end(), g);

    // 为每个组生成数据
    for (int groupIdx = 0; groupIdx < numGroups; ++groupIdx) {
        bool isDenseGroup = (groupLabels[groupIdx] == 1);
        
        // 为这个组生成4个值
        for (int elemInGroup = 0; elemInGroup < groupSize; ++elemInGroup) {
            int globalIdx = groupIdx * groupSize + elemInGroup;
            if (globalIdx >= totalElements) break;  // 处理最后一组可能不足4个元素的情况
            
            float value;
            if (isDenseGroup) {
                // 密集组：生成大于阈值的值
                auto seed = globalIdx * 73 + elemInGroup * 149;
                auto data = ((seed * 1103515245) + 12345) & 0x7fffffff;
                data = data % 1000;
                value = denseMinValue + (float)(data % 800) / 1000.0f * rate;  // 范围：[denseMinValue, rate]
            } else {
                // 稀疏组：生成小于阈值的值
                auto seed = globalIdx * 97 + elemInGroup * 193;
                auto data = ((seed * 1103515245) + 12345) & 0x7fffffff;
                data = data % 1000;
                value = (float)(data % 500) / 5000.0f * rate;  // 范围：[0, sparseThreshold)
            }
            
            inputData.push_back(value);
        }
    }

    // 计算实际的稀疏阈值（应该在稀疏值和密集值之间）
    float actualThreshold = (sparseThreshold + denseMinValue) / 2.0f;

    // ====== 添加调试代码：打印前几组的inputData ======
    printf("\n=== DEBUG: InputData Structure Analysis ===\n");
    printf("Rate: %.6f, SparseThreshold: %.6f, DenseMinValue: %.6f, ActualThreshold: %.6f\n", 
           rate, sparseThreshold, denseMinValue, actualThreshold);
    
    int maxGroupsToPrint = std::min(20, numGroups);  // 打印前20组
    for (int groupIdx = 0; groupIdx < maxGroupsToPrint; ++groupIdx) {
        printf("Group %2d [%s]: ", groupIdx, (groupLabels[groupIdx] == 1) ? "Dense" : "Sparse");
        
        for (int elemInGroup = 0; elemInGroup < groupSize; ++elemInGroup) {
            int globalIdx = groupIdx * groupSize + elemInGroup;
            if (globalIdx < inputData.size()) {
                printf("%.6f ", inputData[globalIdx]);
            }
        }
        
        // 检查这个组是否真的符合预期
        bool shouldBeDense = (groupLabels[groupIdx] == 1);
        bool actuallyAllAboveThreshold = true;
        bool actuallyAllBelowThreshold = true;
        
        for (int elemInGroup = 0; elemInGroup < groupSize; ++elemInGroup) {
            int globalIdx = groupIdx * groupSize + elemInGroup;
            if (globalIdx < inputData.size()) {
                if (inputData[globalIdx] < actualThreshold) {
                    actuallyAllAboveThreshold = false;
                } else {
                    actuallyAllBelowThreshold = false;
                }
            }
        }
        
        printf("| Expected: %s, Actual: %s\n", 
               shouldBeDense ? "Dense" : "Sparse",
               actuallyAllAboveThreshold ? "Dense" : (actuallyAllBelowThreshold ? "Sparse" : "MIXED!"));
    }

    // 生成参考数据（用于验证结构化稀疏性）
    std::vector<float> referenceData;
    referenceData.reserve(inputData.size());

    for (int i = 0; i < inputData.size(); i += groupSize) {
        // 检查这个组的第一个元素来决定整个组的处理方式
        bool isGroupSparse = (inputData[i] < actualThreshold);
        
        // 整个组使用相同的处理方式
        for (int j = 0; j < groupSize && (i + j) < inputData.size(); ++j) {
            if (isGroupSparse) {
                referenceData.push_back(0.0f);  // 稀疏组全部置零
            } else {
                referenceData.push_back(inputData[i + j]);  // 密集组保持原值
            }
        }
    }

    // ====== 添加调试代码：打印前几组的referenceData ======
    printf("\n=== DEBUG: ReferenceData Structure Analysis ===\n");
    for (int groupIdx = 0; groupIdx < maxGroupsToPrint; ++groupIdx) {
        printf("Group %2d: ", groupIdx);
        
        bool isGroupAllZero = true;
        bool isGroupAllNonZero = true;
        
        for (int elemInGroup = 0; elemInGroup < groupSize; ++elemInGroup) {
            int globalIdx = groupIdx * groupSize + elemInGroup;
            if (globalIdx < referenceData.size()) {
                printf("%.6f ", referenceData[globalIdx]);
                if (referenceData[globalIdx] == 0.0f) {
                    isGroupAllNonZero = false;
                } else {
                    isGroupAllZero = false;
                }
            }
        }
        
        printf("| %s\n", isGroupAllZero ? "SPARSE" : (isGroupAllNonZero ? "DENSE" : "MIXED!"));
    }

    // 验证结构化稀疏性
    int zeroCount = 0;
    int structuredGroups = 0;
    bool isStructuredSparse = true;
    int mixedGroups = 0;

    for (int i = 0; i < referenceData.size(); i += groupSize) {
        bool groupAllZero = true;
        bool groupAllNonZero = true;
        
        for (int j = 0; j < groupSize && (i + j) < referenceData.size(); ++j) {
            if (referenceData[i + j] == 0.0f) {
                zeroCount++;
                groupAllNonZero = false;
            } else {
                groupAllZero = false;
            }
        }
        
        // 检查是否满足结构化稀疏要求（全零或全非零）
        if (!(groupAllZero || groupAllNonZero)) {
            isStructuredSparse = false;
            mixedGroups++;
            if (mixedGroups <= 5) {  // 只打印前5个混合组
                printf("WARNING: Group starting at index %d is not structured sparse!\n", i);
            }
        }
        
        if (groupAllZero) {
            structuredGroups++;
        }
    }

    float sparsityRatio = (float)zeroCount / referenceData.size();
    float structuredSparsityRatio = (float)structuredGroups / ((referenceData.size() + groupSize - 1) / groupSize);

    printf("\n=== Structured Sparse Summary ===\n");
    printf("- Total elements: %d\n", (int)inputData.size());
    printf("- Total groups (4-channel): %d\n", (int)((inputData.size() + 3) / 4));
    printf("- Sparse groups: %d\n", structuredGroups);
    printf("- Mixed groups (ERROR): %d\n", mixedGroups);
    printf("- Element sparsity ratio: %.2f%%\n", sparsityRatio * 100.0f);
    printf("- Group sparsity ratio: %.2f%%\n", structuredSparsityRatio * 100.0f);
    printf("- Is structured sparse: %s\n", isStructuredSparse ? "Yes" : "No");
    printf("- Threshold used: %.6f\n", actualThreshold);
    printf("=====================================\n\n");
    
    return std::make_tuple(inputData, referenceData, sparsityRatio, actualThreshold);
}


public:
    bool testUnit(MNNForwardType type, const std::string& device_name, const std::string& test_op_name, int batch,
                     int ic, int oc, int ih, int iw, PadMode mode, int pad_h, int pad_w, int kh, int kw, int stride,
                  int dilation, int group, int precision, MNN::SparseAlgo sparseAlgo = MNN::SparseAlgo_RANDOM, int sparseBlockOC = 1, bool debug = false, int nbit = 8, bool async = false) {
        using namespace MNN::Express;
        std::map<PadMode, Express::PaddingMode> padMap = {
            {PadMode_CAFFE, CAFFE}, {PadMode_VALID, VALID}, {PadMode_SAME, SAME}};
        std::vector<float> weightData, biasData;

        generateWeight(weightData, ic, oc, kh, kw, dilation, group, sparseBlockOC);

        for (int i = 0; i < oc; i++) {
            auto data      = (((i / kw) % 1317) * ((i / kh) % 1317) + i / ic + i / oc + (oc - i) * ic + i * (oc - i)) % 1317;
            auto floatData = (float)(data % 255) / 255.0f;
            biasData.push_back(floatData);
            //biasData.push_back(0.0f);
        }

        std::vector<float> inputData, outputData, outputDataSeparateBias;
        float rate = 1.0f;
        if (ih * iw * ic * batch > 10000) {
            // Avoid exceed fp16 limit
            rate = 0.001f;
        }
        bool useStructuredSparse = true;  // 设置为true使用结构化稀疏，false使用随机稀疏

        std::vector<float> referenceData;
        float sparsityRatio, sparseThreshold;

        if (useStructuredSparse) {
            // 使用结构化稀疏数据（30%稀疏率）
            std::tie(inputData, referenceData, sparsityRatio, sparseThreshold) = 
                generateHalfStructuredSparseData(ih, iw, ic, batch, rate);
        } else {
            // 使用随机稀疏数据（中位数阈值）
            std::tie(inputData, referenceData, sparsityRatio, sparseThreshold) = 
                generateRandomSparseData(ih, iw, ic, batch, rate);
        }

        // print referenceData
        

        float fac = 1.23;
        int res = 10;
        float tail = 0.2;
        float threshold = (float)(1 << (nbit - 1)) - 1.0f;
        float clampMin = -threshold;
        if (async) {
            clampMin = -threshold - 1;
        }
        int kernel_size = ic * kw * kh;
        std::vector<int8_t> quantWeight(oc*ic*kw*kh);
        std::vector<float> wScale;
        if (nbit == 4) {
            // 缩小权重范围，提高量化精度
            for (auto& w : weightData) {
                w *= 0.5f;  // 将权重范围从[-1,1]缩小到[-0.5,0.5]
            }
        }
        
        if (async) {

            wScale.resize(2 * oc);
            for (int k = 0; k < oc; ++k) {
                int beginIndex = k * kernel_size;
                auto minMax = findMinMax(weightData.data() + beginIndex, kernel_size);
                auto minValue = minMax.first;
                wScale[2*k] = minMax.first;
                auto absMax = minMax.second - minMax.first;
                wScale[2*k+1] = 0;
                
                float quantscale = 1.0f;
                if (absMax >= 0.000001f) {
                    wScale[2 * k + 1] = absMax / (threshold - clampMin);
                    quantscale = 1.0f / wScale[2*k+1];
                    
                }
                float* ptr = weightData.data() + beginIndex;
                for (int i = 0; i < kernel_size; ++i) {
                    int8_t quantValue = int8_t(std::round((ptr[i] - minValue) * quantscale + clampMin));
                    float floatValue = ((float)quantValue - clampMin) * wScale[2*k+1] + minValue;
                    quantWeight[k * kernel_size + i] = quantValue;
                    ptr[i] = floatValue;
                }
            }
        } else {
            wScale.resize(oc);
            for (int k = 0; k < oc; ++k) {
                int beginIndex = k * kernel_size;
                auto absMax = findAbsMax(weightData.data() + beginIndex, kernel_size);
                wScale[k] = absMax / threshold;

                float* ptr = weightData.data() + beginIndex;
                for (int i = 0; i < kernel_size; ++i) {
                    int8_t quantVal = (int8_t)(fmax(fmin(round(ptr[i] / wScale[k]), threshold), clampMin));
                    quantWeight[k * kernel_size + i] = quantVal;
                    ptr[i] = (float)quantVal * wScale[k];
                }
            }
        }
        reference_conv2d(referenceData, weightData, biasData, outputData, outputDataSeparateBias, batch, ic, oc, ih, iw, mode, pad_h, pad_w, kh, kw, stride, dilation, group, FP32Converter[precision]);
        if (outputData.size() == 0) {
            return true;
        }

        auto input = _Input({batch, ic, ih, iw}, NCHW, halide_type_of<float>());
        ::memcpy(input->writeMap<float>(), referenceData.data(), inputData.size() * sizeof(float));
        // Single Conv
        //for (int i = 0; i < inputData.size(); ++i){
        //    printf("%f ", input->readMap<float>()[i]);
        //}

        auto weightLength = weightData.size();
        float errorScale = 1.0f;
        if (nbit == 4 && weightLength > 10000) {
            errorScale = 500.0f;
        }
        int memory = MNNTestSuite::get()->pStaus.memory;
        if (precision > MNN::BackendConfig::Precision_High || memory > MNN::BackendConfig::Memory_High) {
            errorScale = 100.0f;
        }
        std::vector<std::pair<bool, bool>> activations = {
            {false, false}
        };
        for (auto& activation : activations) {
            auto output     = _HybridConv(weightData, biasData, wScale, input,
                                            {ic, oc}, {kw, kh}, padMap[mode],  {stride, stride}, {dilation, dilation}, group, {pad_w, pad_h}, activation.first, activation.second, nbit, async, sparseThreshold);
            auto toutputData = outputData;
            float maxV = -10000.0f;
            float minV = 10000.0f;
            if (activation.first) {
                for (auto& t : toutputData) {
                    maxV = ALIMAX(maxV, t);
                    minV = ALIMIN(minV, t);
                    t = ALIMAX(0.0f, t);
                }
    //                MNN_PRINT("Max: %f -> Min:%f\n", maxV, minV);
            }
            if (activation.second) {
                for (auto& t : toutputData) {
                    t = ALIMAX(0.0f, t);
                    t = ALIMIN(6.0f, t);
                }
            }

            // difference below 0.5% relative error is considered correct.
            output = _Convert(output, NCHW);
            auto outputPtr = output->readMap<float>();
            // when using low precision, im2col or strassen convolution error rate to reference value is about 1e-4, winograd has larger error rate.

            //if (!checkVectorByRelativeError<float>(outputPtr, toutputData.data(), toutputData.data(), toutputData.size(), 0.001 * errorScale)) {
            //    MNN_PRINT("precision:%d, memory:%d\n", precision, memory);
            //    MNN_PRINT("expect:\t real:\t\n");
            //    for (int i = 0; i < toutputData.size(); ++i)
            //    {
            //        MNN_PRINT("%f, %f\n", toutputData[i], outputPtr[i]);
            //    }
            //    MNN_PRINT("output shape: n=%d c=%d h=%d w=%d\n", output->getInfo()->dim[0], output->getInfo()->dim[1], output->getInfo()->dim[2], output->getInfo()->dim[3]);
            //    MNN_ERROR("%s(%s) test failed for %d bits, async=%d , relu: %d, relu6: %d!\n", test_op_name.c_str(), device_name.c_str(), nbit, async, activation.first, activation.second);
            //    return false;
            //}
        }
        return true;
    }
    /*
    // 批量测试不同配置
    bool testGEMVBatch(MNNForwardType type, const std::string& device_name) {
        // 针对GEMV优化的测试用例
        std::vector<std::tuple<int,int,int,int,int>> test_cases = {
            // batch, ic, oc, ih, iw
            {1, 32, 64, 56, 56},     // 小尺寸
            {1, 64, 128, 28, 28},    // 中等尺寸  
            {1, 128, 256, 14, 14},   // 大尺寸
            {1, 256, 512, 7, 7},     // 高通道数
            {2, 64, 128, 32, 32},    // 多batch
        };
        
        for (auto& test_case : test_cases) {
            int batch, ic, oc, ih, iw;
            std::tie(batch, ic, oc, ih, iw) = test_case;
            
            // 测试8bit同步和异步
            if (!testGEMV(type, device_name, batch, ic, oc, ih, iw, 8, false)) {
                MNN_ERROR("GEMV test failed: 8bit sync, batch=%d, ic=%d, oc=%d\n", batch, ic, oc);
                return false;
            }
            
            if (!testGEMV(type, device_name, batch, ic, oc, ih, iw, 8, true)) {
                MNN_ERROR("GEMV test failed: 8bit async, batch=%d, ic=%d, oc=%d\n", batch, ic, oc);
                return false;
            }
            
            // 可选：测试4bit
            // if (!testGEMV(type, device_name, batch, ic, oc, ih, iw, 4, false)) {
            //     MNN_ERROR("GEMV test failed: 4bit sync, batch=%d, ic=%d, oc=%d\n", batch, ic, oc);
            //     return false;
            // }
        }
        
        return true;
    }
    */
};

    

MNNTestSuiteRegister(ConvolutionTestOnCPU, "op/convolution/conv2d");
MNNTestSuiteRegister(ConvolutionInt8Test, "op/convolution/weighti8i4conv2d");
MNNTestSuiteRegister(ConvolutionSpeedTestOnCPU, "speed/convolution/conv2d");
MNNTestSuiteRegister(SparseConvolutionTestOnCPU, "op/convolution/sparse_conv2d");
MNNTestSuiteRegister(DepthwiseConvolutionTest, "op/convolution/depthwise_conv");
MNNTestSuiteRegister(GroupConvolutionTestOnCPU, "op/convolution/conv_group");
MNNTestSuiteRegister(Conv1x1OpenCLTest, "op/convolution/conv1x1_opencl");
MNNTestSuiteRegister(Conv1x1SparseOpenCLTest, "op/convolution/conv1x1_sparse_opencl");
MNNTestSuiteRegister(ConvolutionInt84GEMVSparseTest, "op/convolution/conv1x1i8i4_sparse_opencl");