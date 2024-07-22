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
#include <vector>
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
           bool relu = false, bool relu6 = false, MNN::SparseAlgo sparseAlgo = MNN::SparseAlgo_RANDOM, int sparseBlockOC = 1, bool sparese = false) {
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
                     int dilation, int group, int precision, MNN::SparseAlgo sparseAlgo = MNN::SparseAlgo_RANDOM, int sparseBlockOC = 1, bool debug = false) {
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

        if (debug) {
           std::vector<float> printCache(inputData.size());
           for (int i = 0; i < inputData.size(); ++i) {
               printCache[i] = FP32Converter[precision](inputData[i]);
           }
           MNN_PRINT("input:");
           formatMatrix(printCache.data(), {batch, ic, ih, iw});
           printCache.resize(weightData.size());
           for (int i = 0; i < weightData.size(); ++i) {
               printCache[i] = FP32Converter[precision](weightData[i]);
           }
           MNN_PRINT("weight:");
           formatMatrix(printCache.data(), {oc, ic, kh, kw});
           printCache.resize(biasData.size());
           for (int i = 0; i < biasData.size(); ++i) {
               printCache[i] = FP32Converter[precision](biasData[i]);
           }
           MNN_PRINT("bias:");
           formatMatrix(printCache.data(), {oc});

        }

        reference_conv2d(inputData, weightData, biasData, outputData, outputDataSeparateBias, batch, ic, oc, ih, iw, mode, pad_h, pad_w, kh, kw,
                         stride, dilation, group, FP32Converter[precision]);
        if (outputData.size() == 0) {
            return true;
        }

        auto input = _Input({batch, ic, ih, iw}, NCHW, halide_type_of<float>());
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
         // Multi Conv
         if (group == 1 || (group == ic && ic == oc)) {
             VARP weightVar;
             if (group == 1) {
                 weightVar = _Const(weightData.data(), {oc, ic, kh, kw}, NCHW, halide_type_of<float>());
             } else {
                 weightVar = _Const(weightData.data(), {oc, ic / group, kh, kw}, NCHW, halide_type_of<float>());
             }
             auto biasVar = _Const(biasData.data(), {oc}, NCHW, halide_type_of<float>());
             auto out     = _Conv(weightVar, biasVar, _Convert(input, NC4HW4), padMap[mode], {stride, stride}, {dilation, dilation}, group,
                              {pad_w, pad_h}, sparseAlgo, sparseBlockOC, mSparse);
             out = _Convert(out, NCHW);
             auto outputPtr = out->readMap<float>();
             if (!checkVectorByRelativeError<float>(outputPtr, outputData.data(), outputData.size(), 0.05)) {
                 MNN_PRINT("multi expect:\t real:\n");
                 for (int i = 0; i < outputData.size(); ++i)
                 {
                     MNN_PRINT("%f\t, %f\n", outputData[i], outputPtr[i]);
                 }
                 MNN_ERROR("%s(%s) multi test failed!\n", test_op_name.c_str(), device_name.c_str());
                 return false;
             }
         }
        // Single Conv
        auto output = _Conv(std::move(weightData), std::move(biasData), input, {ic, oc}, {kw, kh}, padMap[mode],
                            {stride, stride}, {dilation, dilation}, group, {pad_w, pad_h}, false, false, sparseAlgo, sparseBlockOC, mSparse);

        // difference below 0.5% relative error is considered correct.
        auto outputPtr = output->readMap<float>();

        if (debug) {
            MNN_PRINT("\ndata NCHW shape:");
            printDims(input->getInfo()->dim);
            MNN_PRINT("\nweight OIHW shape:");
            printDims({oc, ic, kh, kw});
            MNN_PRINT("\noutput NCHW shape:");
            printDims(output->getInfo()->dim);
            MNN_PRINT("\nexpected output:");
            formatMatrix(outputData.data(), output->getInfo()->dim);
            MNN_PRINT("\nexpected output 2:");
            formatMatrix(outputDataSeparateBias.data(), output->getInfo()->dim);
            MNN_PRINT("\nreal output:");
            formatMatrix(outputPtr, output->getInfo()->dim);
        }
        // when using low precision, im2col or strassen convolution error rate to reference value is about 1e-4, winograd has larger error rate.

        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 100; // winograd error in 16-bits is relatively large
        if (!checkVectorByRelativeError<float>(outputPtr, outputData.data(), outputDataSeparateBias.data(), outputData.size(), 0.001 * errorScale)) {
            MNN_PRINT("precision:%d, expect:\t expect2:\t real:\t\n", precision);
            for (int i = 0; i < outputData.size(); ++i)
            {
                MNN_PRINT("%f\t, %f\t, %f\n", outputData[i],outputDataSeparateBias[i], outputPtr[i]);
            }
            MNN_ERROR("%s(%s) test failed!\n", test_op_name.c_str(), device_name.c_str());
            return false;
        }


        if (mBenchSpeed) {
            int oh = output->getInfo()->dim[2], ow = output->getInfo()->dim[3];
            input.fix(VARP::INPUT);
            MNN::Timer _t;
            const int LOOP = 20;
            for (int i = 0; i < LOOP; ++i) {
                input->writeMap<float>();
                output->readMap<float>();
            }
            auto time = (float)_t.durationInUs() / 1000.0f;
            MNN_PRINT("kernel=(%dx%d) input=(1x%dx%dx%d) output=(1x%dx%dx%d) stride=(%dx%d), avg time = %f\n",
                      kh, kw, ic, ih, iw, oc, oh, ow, stride, stride, 1.0 * time / LOOP);
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
            weightData[ri] = data;
        }
    }
    ConvolutionInt8CommonTest& speed() {
        mBenchSpeed = true;
        return *this;
    }

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
            data           = data * data;
            biasData.push_back(floatData);
            // biasData.push_back(0.0f);
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
                wScale[2*k+1] = absMax / (threshold - clampMin);
                float scale = 0.0f;
                if (absMax >= 0.000001f) {
                    scale = 1.0f / wScale[2*k+1];
                }
                float* ptr = weightData.data() + beginIndex;
                for (int i = 0; i < kernel_size; ++i) {
                    int8_t quantValue = int8_t(std::round((ptr[i] - minValue) * scale + clampMin));
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
        reference_conv2d(inputData, weightData, biasData, outputData, outputDataSeparateBias, batch, ic, oc, ih, iw, mode, pad_h, pad_w, kh, kw,
                         stride, dilation, group, FP32Converter[precision]);
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
        if (precision > MNN::BackendConfig::Precision_High) {
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
            auto outputPtr = output->readMap<float>();
            // when using low precision, im2col or strassen convolution error rate to reference value is about 1e-4, winograd has larger error rate.

            if (!checkVectorByRelativeError<float>(outputPtr, toutputData.data(), toutputData.data(), toutputData.size(), 0.001 * errorScale)) {
                MNN_PRINT("precision:%d, expect:\t expect2:\t real:\t\n", precision);
                for (int i = 0; i < toutputData.size(); ++i)
                {
                    MNN_PRINT("%f\t, %f\n", toutputData[i], outputPtr[i]);
                }
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
            1, 3, 10, 17
        };
        std::vector<int> icSize = {
            1, 3, 10, 17
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
                                                bool succ =
                                                    ConvolutionType().test(type, device_name, "Conv2D", b, ic, oc, is,
                                                                                is, PadMode_CAFFE, p, p, kh, kw, s, d, 1, precision, sparseAlgo, block, false);
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
        return DenseConvolutionInt8Test::test(MNN_FORWARD_CPU, "CPU", precision, MNN::SparseAlgo_RANDOM, {1});
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
                    for (int isw = 1; isw <= 8; ++isw) {
                        for (int ish = 1; ish <= 8; ++ish) {
                            for (int kw = 1; kw <= 4; kw++) {
                                for (int kh = 1; kh <= 4; kh++) {
                                    for (int d = 1; d <= 2; d++) {
                                        for (int s = 1; s <= 2; s++) {
                                            for (int p = 0; p <= std::min(kw, kh); p++) {
                                                // depthwise <==> group == outputChannel
                                                bool succ = ConvolutionCommonTest().test(
                                                                                         type, device_name, "DepthwiseConv2D", b, ic, oc, ish, isw, PadMode_CAFFE,
                                                                                         p, p, kh, kw, s, d, oc, precision);
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

MNNTestSuiteRegister(ConvolutionTestOnCPU, "op/convolution/conv2d");
MNNTestSuiteRegister(ConvolutionInt8Test, "op/convolution/weighti8i4conv2d");
MNNTestSuiteRegister(ConvolutionSpeedTestOnCPU, "speed/convolution/conv2d");
MNNTestSuiteRegister(SparseConvolutionTestOnCPU, "op/convolution/sparse_conv2d");
MNNTestSuiteRegister(DepthwiseConvolutionTest, "op/convolution/depthwise_conv");
MNNTestSuiteRegister(GroupConvolutionTestOnCPU, "op/convolution/conv_group");
