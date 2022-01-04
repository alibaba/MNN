//
//  ConvInt8Test.cpp
//  MNNTests
//
//  Created by MNN on b'2020/02/19'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include <random>
#include <MNN/expr/ExprCreator.hpp>
#include "MNN_generated.h"
#include "MNNTestSuite.h"
#include "common/CommonCompute.hpp"
#include "common/MemoryFormater.h"
#include "common/WinogradInt8Helper.hpp"
#include <MNN/AutoTime.hpp>

using namespace MNN::Express;
using namespace MNN;
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

inline int8_t int32ToInt8(int data, int bias, float scale) {
    float value = roundf((float)(data + bias) * scale);
    value       = std::max(value, -127.0f);
    value       = std::min(value, 127.0f);
    return static_cast<int8_t>(value);
}

VARP _Conv(std::vector<int8_t>&& weight, std::vector<int>&& bias, std::vector<float>&& scale, VARP x, INTS channel,
           INTS kernelSize, PaddingMode pad, INTS stride, INTS dilate, int group, INTS pads, bool relu,
           int8_t inputZeroPoint, int8_t outputZeroPoint, int8_t minValue, int8_t maxValue, bool accumulateToInt16,
           MNN::SparseAlgo sparseAlgo, int sparseBlockOC) {
    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type = OpType_ConvInt8;
    if (channel[0] == channel[1] && channel[0] == group) {
        convOp->type = OpType_DepthwiseConvInt8;
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
    conv2D->common->relu = relu;
    MNN_ASSERT(weight.size() == channel[1] * (channel[0] / group) * kernelSize[0] * kernelSize[1]);
    conv2D->symmetricQuan.reset(new QuantizedFloatParamT);

    if (sparseAlgo == MNN::SparseAlgo_RANDOM || sparseAlgo == MNN::SparseAlgo_SIMD_OC) {
        size_t weightNNZElement, weightBlockNumber = 0;
        CommonCompute::statisticWeightSparsity(weightNNZElement, weightBlockNumber, weight.data(), bias.size(), weight.size() / bias.size(), sparseBlockOC);
        std::unique_ptr<MNN::AttributeT> arg1(new MNN::AttributeT);
        arg1->key = "sparseBlockOC";
        arg1->i = sparseBlockOC;

        std::unique_ptr<MNN::AttributeT> arg2(new MNN::AttributeT);
        arg2->key = "sparseBlockKernel";
        arg2->i = 1;

        std::unique_ptr<MNN::AttributeT> arg3(new MNN::AttributeT);
        arg3->key = "NNZElement";
        arg3->i = weightNNZElement;

        std::unique_ptr<MNN::AttributeT> arg4(new MNN::AttributeT);
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

    if (bias.size() == 0) {
        bias.resize(channel[1]);
        std::fill(bias.begin(), bias.end(), 0);
    }
    conv2D->symmetricQuan->bias = std::move(bias);
    conv2D->symmetricQuan->scale = std::move(scale);
    conv2D->symmetricQuan->zeroPoint = std::move(inputZeroPoint);
    conv2D->symmetricQuan->outputZeroPoint = std::move(outputZeroPoint);
    MNN_ASSERT(maxValue > minValue);
    conv2D->symmetricQuan->clampMin = minValue;
    conv2D->symmetricQuan->clampMax = maxValue;
    conv2D->symmetricQuan->weight = std::move(weight);

    if (accumulateToInt16) {
        conv2D->symmetricQuan->method = MNN::QuantizeAlgo::QuantizeAlgo_OVERFLOW_AWARE;
    }


    return (Variable::create(Expr::create(convOp.get(), {x})));
}

// y = Conv(x, w), x and y is C4 ordered format, weight is [oc, ic, kh, kw] raw format.
// weight: [group, ocGroup, icGroup, kh, kw]
static std::vector<int8_t> naiveConvInt8(const int8_t* x, const int8_t* weight, const int* bias, const float* scale,
                                           int ow, int oh, int iw, int ih, int ic, int oc, int kw, int kh, int padX, int padY, int group, int padValue = 0,
                                           int strideX = 1, int strideY = 1, int dilateX = 1, int dilateY = 1, int batch = 1) {
    int ocGroup = oc / group, icGroup = ic / group;
    std::vector<int8_t> yCorrect(batch * oc * oh * ow, 0);
    for (int b = 0; b < batch; ++b) {
        for (int oz = 0; oz < oc; ++oz) {
            int gId = oz / ocGroup;
            for (int oy = 0; oy < oh; ++oy) {
                for (int ox = 0; ox < ow; ++ox) {
                    int32_t yInt32 = 0;
                    auto destOffset = ((b * oc + oz) * oh + oy) * ow + ox;
                    for (int sz = gId * icGroup; sz < (gId + 1) * icGroup; ++sz) {
                        for (int ky = 0; ky < kh; ++ky) {
                            for (int kx = 0; kx < kw; ++kx) {
                                int ix = ox * strideX + kx * dilateX - padX, iy = oy * strideY + ky * dilateY - padY;
                                int8_t xValue = padValue;
                                if (ix >= 0 && ix < iw && iy >= 0 && iy < ih) {
                                    xValue = x[(((b * ic + sz) * ih + iy) * iw + ix)];
                                }
                                yInt32 += xValue * weight[(((gId * ocGroup + oz % ocGroup) * icGroup + sz % icGroup) * kh + ky) * kw + kx];
                            }
                        }
                    }
                    yCorrect[destOffset] = int32ToInt8(yInt32, bias[oz], scale[oz]);
                }
            }
        }
    }
    return yCorrect;
}

class ConvInt8TestCommon : public MNNTestCase {
protected:
    virtual void generateWeight(std::vector<int8_t>& weight, int ic, int oc, int kh, int kw, int group, int xMax, int xMin, int sparseBlockOC) {
        for (int i = 0; i < oc/group; ++i) {
            for (int j = 0; j < ic; ++j) {
                auto weightCurrent = weight.data() + (i * ic + j) * kw * kh;
                for (int k = 0; k < kw * kh; ++k) {
                    weightCurrent[k] = ((i * i + j * j + k * k) % (xMax - xMin + 1)) + xMin; // w in [xMin, xMax]
                }
            }
        }
    }

    bool testKernel(INTS inputShape, INTS kernel, INTS channel, INTS pad, INTS strides, INTS dilate, int nbit = 8,
                    bool overflow = false, int group = 1, int batch = 1, MNN::SparseAlgo sparseAlgo = MNN::SparseAlgo_RANDOM,
                    int sparseBlockOC = 1, bool debug = false) {

        std::vector<int> bias(channel[1]);
        std::vector<float> scale(channel[1]);
        std::vector<int8_t> weight(channel[1] * channel[0] / group * kernel[0] * kernel[1]);
        int iw = inputShape[0], ih = inputShape[1];
        VARP x     = _Input({batch, channel[0], ih, iw}, NCHW, halide_type_of<int8_t>());
        auto xInfo = x->getInfo();
        auto xPtr  = x->writeMap<int8_t>();
        int8_t xMin = -(1<<(nbit-1))+1, xMax = (1<<(nbit-1))-1;
        for (int i = 0; i < xInfo->size; ++i) {
            xPtr[i] = (i % (xMax - xMin + 1)) + xMin; // x in [xMin, xMax]
        }

        for (int i = 0; i < bias.size(); ++i) {
        // bias[i] = 0;
        // scale[i] = 1;
        bias[i]  = (10000 + i * i * 10 - i * i * i) % 12580;
        scale[i] = ((127 - i) * i % 128) / 20000.0f;
        }

        generateWeight(weight, channel[0], channel[1], kernel[1], kernel[0], group, xMax, xMin, sparseBlockOC);

        if (debug) {
            MNN_PRINT("\nxPtr data :\n");
            formatMatrix(xPtr, {batch, channel[0], ih, iw});
            MNN_PRINT("\nweight data:\n");
            formatMatrix(weight.data(), {channel[1], channel[0], kernel[0], kernel[1]});
            MNN_PRINT("\nscale data:\n");
            formatMatrix(scale.data(), {static_cast<int>(scale.size())});
            MNN_PRINT("\nbias data:\n");
            formatMatrix(bias.data(), {static_cast<int>(bias.size())});
        }

        auto saveWeight = weight;
        auto saveBias = bias;
        auto saveScale = scale;

        VARP y;
        auto xC4 = _Convert(x, NC4HW4);
        // For sse we use uint8 instead of int8, use FloatToInt8 to hidden detail
        xC4 = _FloatToInt8(_Cast<float>(xC4), _Scalar<float>(1.0f), -127, 127);
        if (overflow) {
            y     = _Conv(std::vector<int8_t>(weight), std::vector<int>(bias), std::vector<float>(scale), xC4,
                               channel, kernel, PaddingMode::CAFFE, strides, dilate, group, pad, false, 0, 0, -127, 127, true, sparseAlgo, sparseBlockOC);
        } else {
            y     = _Conv(std::vector<int8_t>(weight), std::vector<int>(bias), std::vector<float>(scale), xC4,
                               channel, kernel, PaddingMode::CAFFE, strides, dilate, group, pad, false, 0, 0, -127, 127, false, sparseAlgo, sparseBlockOC);
        }
        y = _Int8ToFloat(y, _Scalar<float>(1.0f));
        y = _Cast<int8_t>(y);
        y = _Convert(y, NCHW);
        auto yInfo = y->getInfo();
        auto ow = yInfo->dim[3], oh = yInfo->dim[2];
        auto targetValues = naiveConvInt8(xPtr, saveWeight.data(), saveBias.data(), saveScale.data(),
                                            ow, oh, iw, ih, channel[0], channel[1], kernel[0], kernel[1], pad[0], pad[1], group, 0, strides[0], strides[1], dilate[0], dilate[1], batch);
        auto yPtr  = y->readMap<int8_t>();
        if (debug) {
            MNN_PRINT("\ndebug expected output nchw");
            formatMatrix(targetValues.data(), {yInfo->dim[0], yInfo->dim[1]/4, yInfo->dim[2], yInfo->dim[3], 4});
            MNN_PRINT("\nreal output:");
            formatMatrix(yPtr, {yInfo->dim[0], yInfo->dim[1]/4, yInfo->dim[2], yInfo->dim[3], 4});
        }

        for (int i = 0; i < targetValues.size(); ++i) {
            int8_t targetValue = targetValues[i], computeResult = yPtr[i];
            // Because of round implement in ARM / X86 / PC may cause 1 / 0 / -1 diff, don't care about this error
            auto error = (int32_t)targetValue - (int32_t)computeResult;
            if (error * error > 1) {
                MNN_PRINT("%d x %d, ConvInt8 result %d Error: %d -> %d\n", ow, oh, i, targetValue, computeResult);
                MNN_PRINT("\nexpected output:");
                formatMatrix(targetValues.data(), {yInfo->dim[0], yInfo->dim[1]/4, yInfo->dim[2], yInfo->dim[3], 4});
                MNN_PRINT("\nreal output:");
                formatMatrix(yPtr, {yInfo->dim[0], yInfo->dim[1]/4, yInfo->dim[2], yInfo->dim[3], 4});

                return false;
            }
        }
        return true;
    }
};

class ConvInt8Im2colGemmTest : public ConvInt8TestCommon {
public:

    virtual bool run(int precision) {
        INTS strides = {1, 1}, dilate = {1, 1}, pad = {3, 4}, inputShape = {34, 23}; // {w, h}
        INTS channel = {64, 64}; // {ci, co}
        std::vector<std::vector<int>> kernels = {
            {4, 2}, {1, 5}, {7, 1}
        };
        std::vector<std::string> titles = {"4x2", "1x5", "7x1"};
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 8, false, 1, 2, MNN::SparseAlgo_RANDOM, 1, false);
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 215, 204 (im2col + gemm)\n", titles[i].c_str());
                return false;
            }
        }
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 3, true, 1, 3, MNN::SparseAlgo_RANDOM, 1, false);
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 215, 204 (im2col + gemm + overflow aware)\n", titles[i].c_str());
                return false;
            }
        }
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 8, false, 1, 5, MNN::SparseAlgo_RANDOM, 1, false);
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 215, 201 (im2col + gemm)\n", titles[i].c_str());
                return false;
            }
        }
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 3, true, 1, 2, MNN::SparseAlgo_RANDOM, 1, false);
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 215, 201 (im2col + gemm + overflow aware)\n", titles[i].c_str());
                return false;
            }
        }
        return true;
    }
};

class SparseConvInt8Im2colGemmTest : public ConvInt8TestCommon {
public:

    virtual void generateWeight(std::vector<int8_t>& weight, int ic, int oc, int kh, int kw, int group, int xMax, int xMin, int sparseBlockOC) {

        assert(sparseBlockOC);
        int ocEven = (group * (oc / group) / sparseBlockOC) * sparseBlockOC;
        int reduceDimLength = (ic / group) * kw * kh;
        weight.resize(group * (oc / group) * reduceDimLength);
        size_t ioc = 0;
        size_t index = 0;
        for (; ioc < ocEven; ioc += sparseBlockOC) {
            for (size_t i = 0; i < reduceDimLength; i++) {
                index = ioc * reduceDimLength + i;
                bool isZero = index % 4 != 0;
                for (int iblock = 0; iblock < sparseBlockOC; iblock++) {
                    if(isZero) {
                        weight[index] = 0;
                    } else {
                        auto data      = (index / kw) * (index / kh) + index / ic + index / oc + (oc - index) * ic + index * (oc - index);
                        weight[index] = (data % (xMax - xMin + 1)) + xMin;
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
                    weight[index] = 0;
                } else {
                    auto data      = (index / kw) * (index / kh) + index / ic + index / oc + (oc - index) * ic + index * (oc - index);
                    weight[index] = (data % (xMax - xMin + 1)) + xMin;
                }
            }
        }
        return;
    }
    virtual bool run(int precision) {

        std::vector<std::pair<MNN::SparseAlgo, int>> SparseList = {{SparseAlgo_RANDOM, 1}, {MNN::SparseAlgo_SIMD_OC, 4}};

        for (int is = 0; is < SparseList.size(); ++is)
        {
            // INTS strides = {1, 1}, dilate = {1, 1}, pad = {3, 4}, inputShape = {215, 204}; // {w, h}
            // INTS channel = {64, 64}; // {ci, co}
            // std::vector<std::vector<int>> kernels = {
            //     {4, 2}, {1, 5}, {7, 1}
            // };

            INTS strides = {1, 1}, dilate = {1, 1}, pad = {0, 0}, inputShape = {6, 6}; // {w, h}
            INTS channel = {8, 8}; // {ci, co}
            std::vector<std::vector<int>> kernels = {
                {3, 3}, {1, 5}, {7, 1}
            };

            std::vector<std::string> titles = {"4x2", "1x5", "7x1"};
            for (int i = 0; i < kernels.size(); ++i) {
                auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 8, false, 1, 2, SparseList[is].first, SparseList[is].second, false);
                if (!res) {
                    MNN_ERROR("Error for test kernel %s for convint8 215, 204 (im2col + gemm)\n", titles[i].c_str());
                    return false;
                }
            }
            for (int i = 0; i < kernels.size(); ++i) {
                auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 3, true, 1, 3, SparseList[is].first, SparseList[is].second, false);
                if (!res) {
                    MNN_ERROR("Error for test kernel %s for convint8 215, 204 (im2col + gemm + overflow aware)\n", titles[i].c_str());
                    return false;
                }
            }
            inputShape = {215, 201};
            for (int i = 0; i < kernels.size(); ++i) {
                auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 8, false, 1, 5, SparseList[is].first, SparseList[is].second, false);
                if (!res) {
                    MNN_ERROR("Error for test kernel %s for convint8 215, 201 (im2col + gemm)\n", titles[i].c_str());
                    return false;
                }
            }
            for (int i = 0; i < kernels.size(); ++i) {
                auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 3, true, 1, 2, SparseList[is].first, SparseList[is].second, false);
                if (!res) {
                    MNN_ERROR("Error for test kernel %s for convint8 215, 201 (im2col + gemm + overflow aware)\n", titles[i].c_str());
                    return false;
                }
            }
        }
        return true;
    }
};

class ConvInt8WinogradTestCommon : public MNNTestCase {
public:
    static bool testKernel(INTS inputShape, INTS kernel, INTS channel, INTS pad, bool speed, std::string title) {
        auto createWinogradConv = [](std::vector<int8_t>& weight, std::vector<int>& bias, std::vector<float>& scale,
                                     std::vector<int>& attrs, VARP x, INTS kernel, INTS channel, INTS pad,
                                     int8_t inputZeroPoint, int8_t outputZeroPoint) -> VARP {
            std::unique_ptr<OpT> convOp(new OpT);
            convOp->type = OpType_ConvInt8;
            convOp->main.type  = OpParameter_Convolution2D;
            convOp->main.value = new Convolution2DT;
            auto conv2D        = convOp->main.AsConvolution2D();
            conv2D->common.reset(new Convolution2DCommonT);
            conv2D->common->padMode = PadMode_CAFFE;
            conv2D->common->padX = pad[0];
            conv2D->common->padY = pad[1];
            conv2D->common->inputCount  = channel[0];
            conv2D->common->outputCount = channel[1];
            conv2D->common->kernelX = kernel[0];
            conv2D->common->kernelY = kernel[1];
            conv2D->common->relu = false;
            conv2D->symmetricQuan.reset(new QuantizedFloatParamT);
            conv2D->symmetricQuan->weight = weight;
            conv2D->symmetricQuan->bias = bias;
            conv2D->symmetricQuan->scale = scale;
            conv2D->symmetricQuan->zeroPoint = inputZeroPoint;
            conv2D->symmetricQuan->outputZeroPoint = outputZeroPoint;
            conv2D->symmetricQuan->clampMin = -127;
            conv2D->symmetricQuan->clampMax = 127;
            conv2D->symmetricQuan->winogradAttr = attrs;
            return (Variable::create(Expr::create(convOp.get(), {x})));
        };
        int ic = channel[0], oc = channel[1], iw = inputShape[0], ih = inputShape[1], kx = kernel[0], ky = kernel[1];
        int8_t inputZeroPoint = 0, outputZeroPoint = 0, xMin = -31, xMax = 31;
        std::vector<int> bias(oc, 0);
        std::vector<float> scale(oc, 1.0f);
        std::vector<int8_t> weight(oc * ic * ky * kx);
        VARP x = _Input({1, ic, ih, iw}, NCHW, halide_type_of<int8_t>());
        auto xInfo = x->getInfo();
        auto xPtr  = x->writeMap<int8_t>();
        for (int i = 0; i < xInfo->size; ++i) {
            xPtr[i] = (i % (xMax - xMin + 1)) + xMin; // x in [xMin, xMax]
        }
        for (int oz = 0; oz < oc; ++oz) {
            for (int sz = 0; sz < ic; ++sz) {
                for (int k = 0; k < ky * kx; ++k) {
                    auto w = (oz * oz + sz * sz + k * k) % 15 - 7;
                    weight[(oz * ic + sz) * ky * kx + k] = w * 4; // ww = 4*w, w in [-7, 7]
                }
            }
        }
        std::vector<float> weightFloat(weight.size()), transWeightFloat;
        std::vector<int> attrs;
        std::transform(weight.begin(), weight.end(), weightFloat.begin(), [](int8_t val) -> float { return val; });
        WinogradInt8Helper::transformWeight(weightFloat, transWeightFloat, attrs, oc, ic, ky, kx);
        std::vector<int8_t> transWeight(transWeightFloat.size());
        std::transform(transWeightFloat.begin(), transWeightFloat.end(), transWeight.begin(), [=](float val) -> int8_t { return val; });
        x = _Convert(x, NC4HW4);
        // For sse we use uint8 instead of int8, use FloatToInt8 to hidden detail
        x = _FloatToInt8(_Cast<float>(x), _Scalar<float>(1.0f), -127, 127);
        VARP y = createWinogradConv(transWeight, bias, scale, attrs, x, kernel, channel, pad, inputZeroPoint, outputZeroPoint);
        VARP yr = _Cast<int8_t>(_Int8ToFloat(y, _Scalar<float>(1.0f)));
        yr = _Convert(yr, NCHW);
        auto yInfo = yr->getInfo();
        auto yPtr  = yr->readMap<int8_t>();
        auto ow = yInfo->dim[3], oh = yInfo->dim[2];
        auto targetValues = naiveConvInt8(xPtr, weight.data(), bias.data(), scale.data(),
                                            ow, oh, iw, ih, channel[0], channel[1], kernel[0], kernel[1], pad[0], pad[1], 1, 0, 1, 1, 1, 1);
        for (int i = 0; i < targetValues.size(); ++i) {
            int8_t targetValue = targetValues[i], computeResult = yPtr[i];
            if (targetValue != computeResult) {
                MNN_PRINT("ConvInt8 Winograd %s %d x %d, ConvInt8 result %d Error: %d -> %d\n", title.c_str(), ow, oh, i, targetValue, computeResult);
                return false;
            }
        }
        if (speed) {
            x.fix(VARP::INPUT);
            MNN::Timer _t;
            const int LOOP = 20;
            for (int i = 0; i < LOOP; ++i) {
                x->writeMap<float>();
                y->readMap<float>();
            }
            auto time = (float)_t.durationInUs() / 1000.0f;
            MNN_PRINT("ConvInt8 Winograd %s input=(1x%dx%dx%d) output=(1x%dx%dx%d) avg time = %f\n",
                      title.c_str(), ic, ih, iw, oc, oh, ow, 1.0 * time / LOOP);
        }
        return true;
    }
};

class ConvInt8WinogradTest : public ConvInt8WinogradTestCommon {
    virtual bool run(int precision) {
        INTS pad = {1, 1}, inputShape = {128, 128}; // {w, h}
        INTS channel = {32, 32}; // {ci, co}

        std::vector<std::vector<int>> kernels = {
            {3, 3}, {3, 2}, {2, 3}, {2, 2}, {4, 4}, {7, 1}, {1, 7} // {w, h}
        };
        std::vector<std::string> titles = {
            "3x3", "2x3", "3x2", "2x2", "4x4", "1x7", "7x1"
        };
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, false, titles[i]);
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 (winograd)\n", titles[i].c_str());
                return false;
            }
        }
        return true;
    }
};

class ConvSpeedInt8WinogradTest : public ConvInt8WinogradTestCommon {
    public:
    virtual bool run(int precision) {
        INTS pad = {1, 1}, inputShape = {28, 28}; // {w, h}
        INTS channel = {128, 128};
        std::vector<INTS> kernels = {
            {3, 3}, {5, 5}, {7, 1}, {1, 7} // {w, h}
        };
        
        std::vector<std::string> titles = {"3x3", "5x5", "1x7", "7x1"};
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, true, titles[i]);
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 (winograd)\n", titles[i].c_str());
                return false;
            }
        }
        return true;
    }
};

class DepthwiseConvInt8Test : public ConvInt8TestCommon {
public:
    virtual bool run(int precision) {
        INTS strides = {1, 1}, dilate = {1, 1}, pad = {0, 0}, inputShape = {21, 13}; // {w, h}
        int channel = 64;
        std::vector<std::vector<int>> kernels = {
            {3, 3}
        };
        std::vector<std::string> titles = {
            "3x3"
        };
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], {channel, channel}, pad, strides, dilate, 8, false, channel, 4, MNN::SparseAlgo_RANDOM, 1, false);
            if (!res) {
                FUNC_PRINT(1);
                return false;
            }
        }
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], {channel, channel}, pad, strides, dilate, 3, true, channel, 1, MNN::SparseAlgo_RANDOM, 1, false);
            if (!res) {
                FUNC_PRINT(1);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ConvInt8Im2colGemmTest, "op/ConvInt8/im2col_gemm");
#if defined(__arm__) || defined(__aarch64__) // arm32 or arm64
MNNTestSuiteRegister(SparseConvInt8Im2colGemmTest, "op/ConvInt8/im2col_spmm");
#endif
//MNNTestSuiteRegister(ConvInt8WinogradTest, "op/ConvInt8/winograd");
MNNTestSuiteRegister(ConvSpeedInt8WinogradTest, "speed/ConvInt8/winograd");
MNNTestSuiteRegister(DepthwiseConvInt8Test, "op/ConvInt8/depthwise");
