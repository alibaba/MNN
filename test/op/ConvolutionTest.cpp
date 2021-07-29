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
#include <vector>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "TestUtils.h"
#include "core/Session.hpp"
#include "core/TensorUtils.hpp"
#include "core/MemoryFormater.h"
#include "core/OpCommonUtils.hpp"

#define TEST_RANDOM_SEED 100

using namespace MNN;
using namespace MNN::Express;
static void reference_conv2d(const std::vector<float>& input, const std::vector<float>& weight,
                             const std::vector<float>& bias, std::vector<float>& output, int batch, int ic, int oc,
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
    output.resize(batch * oh * ow * oc);
    int ocGroup = oc / group, icGroup = ic / group;
    for (int b = 0; b < batch; ++b) {
        for (int oz = 0; oz < oc; ++oz) {
            int gId = oz / ocGroup;
            for (int oy = 0; oy < oh; ++oy) {
                for (int ox = 0; ox < ow; ++ox) {
                    float summber = bias[oz];
                    auto destOffset = ((b * oc + oz) * oh + oy) * ow + ox;
                    for (int sz = gId * icGroup; sz < (gId + 1) * icGroup; ++sz) {
                        for (int ky = 0; ky < kh; ++ky) {
                            for (int kx = 0; kx < kw; ++kx) {
                                int ix = ox * stride + kx * dilation - pad_w, iy = oy * stride + ky * dilation - pad_h;
                                float xValue = 0.0f;
                                if (ix >= 0 && ix < iw && iy >= 0 && iy < ih) {
                                    xValue = input[(((b * ic + sz) * ih + iy) * iw + ix)];
                                }
                                summber += xValue * weight[(((gId * ocGroup + oz % ocGroup) * icGroup + sz % icGroup) * kh + ky) * kw + kx];
                            }
                        }
                    }
                    output[destOffset] = summber;
                }
            }
        }
    }
}

static PadMode _convertPadMode(PaddingMode mode) {
    switch (mode) {
        case CAFFE:
            return PadMode_CAFFE;
        case VALID:
            return PadMode_VALID;
        case SAME:
            return PadMode_SAME;
        default:
            break;
    }
    return PadMode_CAFFE;
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
        OpCommonUtils::statisticWeightSparsity(weightNNZElement, weightBlockNumber, weight->readMap<float>(), biasSize, weightSize / biasSize, sparseBlockOC);
        
        MNN::AttributeT* arg1(new MNN::AttributeT);
        arg1->key = "sparseBlockOC";
        arg1->i = sparseBlockOC;

        MNN::AttributeT* arg2(new MNN::AttributeT);
        arg2->key = "sparseBlockKernel";
        arg2->i = 1;

        MNN::AttributeT* arg3(new MNN::AttributeT);
        arg3->key = "NNZElement";
        arg3->i = weightNNZElement;

        MNN::AttributeT* arg4(new MNN::AttributeT);
        arg4->key = "blockNumber";
        arg4->i = weightBlockNumber;

        flatbuffers::FlatBufferBuilder builder;
        std::vector<flatbuffers::Offset<MNN::Attribute>> argsVector;
        auto sparseArg1 = MNN::CreateAttribute(builder, arg1);
        auto sparseArg2 = MNN::CreateAttribute(builder, arg2);
        auto sparseArg3 = MNN::CreateAttribute(builder, arg3);
        auto sparseArg4 = MNN::CreateAttribute(builder, arg4);

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
    if (sparese) {
        size_t weightNNZElement, weightBlockNumber = 0;
        OpCommonUtils::statisticWeightSparsity(weightNNZElement, weightBlockNumber, weight.data(), bias.size(), weight.size() / bias.size(), sparseBlockOC);
        
        MNN::AttributeT* arg1(new MNN::AttributeT);
        arg1->key = "sparseBlockOC";
        arg1->i = sparseBlockOC;

        MNN::AttributeT* arg2(new MNN::AttributeT);
        arg2->key = "sparseBlockKernel";
        arg2->i = 1;

        MNN::AttributeT* arg3(new MNN::AttributeT);
        arg3->key = "NNZElement";
        arg3->i = weightNNZElement;

        MNN::AttributeT* arg4(new MNN::AttributeT);
        arg4->key = "blockNumber";
        arg4->i = weightBlockNumber;

        flatbuffers::FlatBufferBuilder builder;
        std::vector<flatbuffers::Offset<MNN::Attribute>> argsVector;
        auto sparseArg1 = MNN::CreateAttribute(builder, arg1);
        auto sparseArg2 = MNN::CreateAttribute(builder, arg2);
        auto sparseArg3 = MNN::CreateAttribute(builder, arg3);
        auto sparseArg4 = MNN::CreateAttribute(builder, arg4);

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
    MNN_ASSERT(weight.size() == channel[1] * (channel[0] / group) * kernelSize[0] * kernelSize[1]);
    conv2D->weight = std::move(weight);
    MNN_ASSERT(bias.size() == channel[1]);
    conv2D->bias = std::move(bias);
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
        OpCommonUtils::statisticWeightSparsity(weightNNZElement, weightBlockNumber, conv2D->weight.data(), conv2D->bias.size(), conv2D->weight.size() / conv2D->bias.size(), sparseBlockOC);
        
        MNN::AttributeT* arg1(new MNN::AttributeT);
        arg1->key = "sparseBlockOC";
        arg1->i = sparseBlockOC;

        MNN::AttributeT* arg2(new MNN::AttributeT);
        arg2->key = "sparseBlockKernel";
        arg2->i = 1;

        MNN::AttributeT* arg3(new MNN::AttributeT);
        arg3->key = "NNZElement";
        arg3->i = weightNNZElement;

        MNN::AttributeT* arg4(new MNN::AttributeT);
        arg4->key = "blockNumber";
        arg4->i = weightBlockNumber;

        flatbuffers::FlatBufferBuilder builder;
        std::vector<flatbuffers::Offset<MNN::Attribute>> argsVector;
        auto sparseArg1 = MNN::CreateAttribute(builder, arg1);
        auto sparseArg2 = MNN::CreateAttribute(builder, arg2);
        auto sparseArg3 = MNN::CreateAttribute(builder, arg3);
        auto sparseArg4 = MNN::CreateAttribute(builder, arg4);

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
public:
    virtual ~ConvolutionCommonTest() = default;
    virtual bool run (int precision) {
        return true;
    }

public:
    virtual void generateWeight(std::vector<float>& weightData, int ic, int oc, int kh, int kw, int dilation, int group, int sparseBlockOC) {
        for (int i = 0; i < group * (oc / group) * (ic / group) * kw * kh; i++) {
            auto data      = ((((i / kw)% 1317) * ((i / kh) % 1317)) % 1317 + i / ic + i / oc + (((oc - i) % 1317) * ic) % 1317 + i * ((oc - i) % 1317)) % 1317;
            auto floatData      = (float)(data % 255) / 255.0f;
            weightData.push_back(floatData);
        }

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

        std::vector<float> inputData, outputData;
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
           for (int i = 0; i < inputData.size(); ++i) {
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

        reference_conv2d(inputData, weightData, biasData, outputData, batch, ic, oc, ih, iw, mode, pad_h, pad_w, kh, kw,
                         stride, dilation, group, FP32Converter[precision]);

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
            auto out     = _Conv(weightVar, biasVar, input, padMap[mode], {stride, stride}, {dilation, dilation}, group,
                             {pad_w, pad_h}, sparseAlgo, sparseBlockOC, mSparse);
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
            MNN_PRINT("\nreal output:");
            formatMatrix(outputPtr, output->getInfo()->dim);
        }
        if (!checkVectorByRelativeError<float>(outputPtr, outputData.data(), outputData.size(), 0.05)) {
            MNN_PRINT("expect:\t real:\t\n");
            for (int i = 0; i < outputData.size(); ++i)
            {
                MNN_PRINT("%f\t, %f\n", outputData[i], outputPtr[i]);
            }
            MNN_ERROR("%s(%s) test failed!\n", test_op_name.c_str(), device_name.c_str());
            return false;
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
                        weightData[index] = (float)(data % 255) / 255.0f;
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

template <typename ConvolutionType>
class ConvolutionTest : public ConvolutionType {
public:
    virtual ~ConvolutionTest() = default;


protected:
    static bool test(MNNForwardType type, const std::string& device_name, int precision, MNN::SparseAlgo sparseAlgo, int MaxBlock) {
        for (int b = 1; b <= 2; b++) {
            for (int oc = 1; oc <= 16; oc *= 2) {
                for (int ic = 1; ic <= 8; ic *= 2) {
                    for (int is = 1; is <= 8; is *= 2) {
                        for (int kw = 1; kw <= 3 && kw <= is; kw++) {
                            for (int kh = 1; kh <= 3 && kh <= is; kh++) {
                                for (int d = 1; d <= 2; d++) {
                                    if (d > std::min(kw, kh) || d * (std::max(kw, kh) - 1) + 1 > is)
                                        continue;

                                    for (int s = 1; s <= 2; s++) {
                                        for (int block = 1; block <= MaxBlock; block *= 4) {
                                            for (int p = 0; p <= 1; p++) {

                                                bool succ =
                                                    ConvolutionType().test(type, device_name, "Conv2D", b, ic, oc, is,
                                                                                is, PadMode_CAFFE, p, p, kh, kw, s, d, 1, precision, sparseAlgo, block, false);
                                                if (!succ) {
                                                    MNN_ERROR(
                                                        "Error for conv b=%d, oc=%d, ic=%d, "
                                                        "is=%d,kw=%d,kh=%d,d=%d,s=%d,p=%d, block=%d\n",
                                                        b, oc, ic, is, kw, kh, d, s, p, block);
                                                    return false;
                                                }
                                            }

                                            {
                                                bool succ =
                                                    ConvolutionType().test(type, device_name, "Conv2D", b, ic, oc, is,
                                                                                is, PadMode_VALID, 0, 0, kh, kw, s, d, 1, precision, sparseAlgo, block, false);
                                                if (!succ) {
                                                    MNN_ERROR(
                                                        "Error for conv b=%d, oc=%d, ic=%d, is=%d,kw=%d,kh=%d,d=%d,s=%d, block=%d, "
                                                        "valid pad\n",
                                                        b, oc, ic, is, kw, kh, d, s, block);
                                                    return false;
                                                }
                                            }
                                            {
                                                bool succ =
                                                    ConvolutionType().test(type, device_name, "Conv2D", b, ic, oc, is,
                                                                                is, PadMode_SAME, 0, 0, kh, kw, s, d, 1, precision, sparseAlgo, block, false);
                                                if (!succ) {
                                                    MNN_ERROR(
                                                        "Error for conv b=%d, oc=%d, ic=%d, is=%d,kw=%d,kh=%d,d=%d,s=%d, block=%d, "
                                                        "same pad\n",
                                                        b, oc, ic, is, kw, kh, d, s, block);
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
        // TODO: Fix bug and use it
        // Check Long convolution
//        bool succ =
//            ConvolutionType().test(type, device_name, "Conv2D", 1, 3072, 16, 128, 1, PadMode_SAME, 0, 0, 1, 1, 1, 1, 1, precision, sparseAlgo, 4, false);
//        if (!succ) {
//            MNN_ERROR("Error for long conv\n");
//            return false;
//        }
        return true;
    }

};

using DenseConvolutionTest = ConvolutionTest<ConvolutionCommonTest>;
class ConvolutionTestOnCPU : public DenseConvolutionTest {
public:
    ~ConvolutionTestOnCPU() = default;
    virtual bool run(int precision) {
        return DenseConvolutionTest::test(MNN_FORWARD_CPU, "CPU", precision, MNN::SparseAlgo_RANDOM, 1);
    }
};

using SparseConvolutionTest = ConvolutionTest<SparseConvolutionCommonTest>;
class SparseConvolutionTestOnCPU : public SparseConvolutionTest {
public:
    ~SparseConvolutionTestOnCPU() = default;
    virtual bool run(int precision) {
        return SparseConvolutionTest::test(MNN_FORWARD_CPU, "CPU", precision, MNN::SparseAlgo_SIMD_OC, 4);
    }
};

class DepthwiseConvolutionTest : public ConvolutionCommonTest {
public:
    virtual ~DepthwiseConvolutionTest() = default;

protected:
    static bool test(MNNForwardType type, const std::string& device_name, int precision) {
        srand(TEST_RANDOM_SEED);
        // correct unit test
        for (int b = 1; b <= 2; b++) {
            for (int oc = 4; oc <= 16; oc *= 2) {
                for (int ic = oc; ic <= oc; ic++) {
                    for (int is = 1; is <= 8; is *= 2) {
                        for (int kw = 1; kw <= 3 && kw <= is; kw++) {
                            for (int kh = 1; kh <= 3 && kh <= is; kh++) {
                                for (int d = 1; d <= 2; d++) {
                                    if (d > std::min(kw, kh) || d * (std::max(kw, kh) - 1) + 1 > is)
                                        continue;
                                    for (int s = 1; s <= 2; s++) {
                                        for (int p = 0; p <= 1; p++) {
                                            // depthwise <==> group == outputChannel
                                            bool succ = ConvolutionCommonTest().test(
                                                type, device_name, "DepthwiseConv2D", b, ic, oc, is, is, PadMode_CAFFE,
                                                p, p, kh, kw, s, d, oc, precision);
                                            if (!succ) {
                                                MNN_ERROR(
                                                    "Error for dw oc=%d, ic=%d, is=%d,kw=%d,kh=%d,d=%d,s=%d,p=%d\n", oc,
                                                    ic, is, kw, kh, d, s, p);
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
        // memory leak unit test
        int b = 1, oc = 4, ic = oc, group = oc, is = 2, p = 1, kh = 3, kw = 3, s = 2, d = 1;
        return ConvolutionCommonTest().test(type, device_name, "DepthwiseConv2D", b, ic, oc, is, is,
                                           PadMode_CAFFE, p, p, kh, kw, s, d, group, precision);
    }
};

class DepthwiseConvolutionTestOnCPU : public DepthwiseConvolutionTest {
public:
    ~DepthwiseConvolutionTestOnCPU() = default;
    virtual bool run(int precision) {
        return DepthwiseConvolutionTest::test(MNN_FORWARD_CPU, "CPU", precision);
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
MNNTestSuiteRegister(SparseConvolutionTestOnCPU, "op/convolution/sparse_conv2d");
MNNTestSuiteRegister(DepthwiseConvolutionTestOnCPU, "op/convolution/depthwise_conv");
MNNTestSuiteRegister(GroupConvolutionTestOnCPU, "op/convolution/conv_group");
