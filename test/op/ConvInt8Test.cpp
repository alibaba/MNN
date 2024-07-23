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
#include "TestUtils.h"
#include "core/CommonCompute.hpp"
#include "core/MemoryFormater.h"
#include "core/WinogradInt8Attr.hpp"
#include "math/WingoradGenerater.hpp"
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
    float value = 0.f;
    value = roundf((float)(data + bias) * scale);

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
        arg3->i = static_cast<int32_t>(weightNNZElement);

        std::unique_ptr<MNN::AttributeT> arg4(new MNN::AttributeT);
        arg4->key = "blockNumber";
        arg4->i = static_cast<int32_t>(weightBlockNumber);

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
                    int sparseBlockOC = 1, bool debug = false, bool speed = false) {

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
        bool testDepthwise = false;
        if (channel[0] == channel[1] && channel[0] == group) {
            testDepthwise = true;
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
                MNN_PRINT("ic=%d, oc=%d, ow=%d, oh=%d, ConvInt8 result No.%d Error: right=%d, error=%d\n", channel[0], channel[1], ow, oh, i, targetValue, computeResult);
#ifdef DEBUG
                x->writeMap<int8_t>();
                auto ptr = y->readMap<int8_t>();
                FUNC_PRINT_ALL(ptr, p);
#endif
                return false;
            }
        }
        if (speed) {
            x.fix(VARP::INPUT);
            // warm up, do onResize first for shapeDirty
            x->writeMap<float>();
            y->readMap<float>();

            MNN::Timer _t;
            const int LOOP = 100;
            for (int i = 0; i < LOOP; ++i) {
                x->writeMap<float>();
                y->readMap<float>();
            }
            auto time = (float)_t.durationInUs() / 1000.0f;
            MNN_PRINT("DepthwiseConvInt8 Speed: input = (1x%dx%dx%d), kernel=(%dx%dx%d), avg time=%f\n",
                      channel[0], ih, iw, channel[0], kernel[0], kernel[1], time);
        }
        return true;
    }
};

class ConvInt8Im2colGemmTest : public ConvInt8TestCommon {
public:

    virtual bool run(int precision) {
        std::vector<std::vector<int>> kernels = {
            {4, 2}, {1, 5}, {7, 1}
        };
        int iw = 14; int ih = 11;
        std::vector<std::string> titles = {"4x2", "1x5", "7x1"};
        for (int sx=1; sx<2; ++sx) {
            for (int sy=1; sy<2; ++sy) {
                for (int dx=1; dx<2; ++dx) {
                    for (int dy=1; dy<2; ++dy) {
                        for (int px=2; px<4; ++px) {
                            for (int py=3; py<4; ++py) {
                                for (int ic=1; ic<=64; ic*=8) {
                                    for (int oc=1; oc<=64; oc*=8) {
                                        INTS strides = {sx, sy}, dilate = {dx, dy}, pad = {px, py}, inputShape = {iw, ih};
                                        INTS channel = {ic, oc};
                                        for (int i = 0; i < kernels.size(); ++i) {
                                            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 8, false, 1, 2, MNN::SparseAlgo_RANDOM, 1, false);
                                            if (!res) {
                                                MNN_ERROR("Error for test kernel %s for convint8 215, 204 (im2col + gemm)\n", titles[i].c_str());
                                                MNN_ERROR("overflow=false, bit=8, batch=2, Conv info: sx=%d, sy=%d, dx=%d, dy=%d, px=%d, py=%d, ic=%d, oc=%d\n", sx, sy, dx, dy, px, py, ic, oc);
                                                return false;
                                            }
                                        }
                                        for (int i = 0; i < kernels.size(); ++i) {
                                            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 3, true, 1, 3, MNN::SparseAlgo_RANDOM, 1, false);
                                            if (!res) {
                                                MNN_ERROR("Error for test kernel %s for convint8 215, 204 (im2col + gemm + overflow aware)\n", titles[i].c_str());
                                                MNN_ERROR("overflow=true,bit=3, batch=3, Conv info: sx=%d, sy=%d, dx=%d, dy=%d, px=%d, py=%d, ic=%d, oc=%d\n", sx, sy, dx, dy, px, py, ic, oc);
                                                return false;
                                            }
                                        }
                                        for (int i = 0; i < kernels.size(); ++i) {
                                            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 8, false, 1, 5, MNN::SparseAlgo_RANDOM, 1, false);
                                            if (!res) {
                                                MNN_ERROR("Error for test kernel %s for convint8 215, 201 (im2col + gemm)\n", titles[i].c_str());
                                                MNN_ERROR("overflow=false,bit=8, batch=5, Conv info: sx=%d, sy=%d, dx=%d, dy=%d, px=%d, py=%d, ic=%d, oc=%d\n", sx, sy, dx, dy, px, py, ic, oc);
                                                return false;
                                            }
                                        }
                                        for (int i = 0; i < kernels.size(); ++i) {
                                            auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 3, true, 1, 2, MNN::SparseAlgo_RANDOM, 1, false);
                                            if (!res) {
                                                MNN_ERROR("Error for test kernel %s for convint8 215, 201 (im2col + gemm + overflow aware)\n", titles[i].c_str());
                                                MNN_ERROR("overflow=true,bit=3, batch=2, Conv info: sx=%d, sy=%d, dx=%d, dy=%d, px=%d, py=%d, ic=%d, oc=%d\n", sx, sy, dx, dy, px, py, ic, oc);
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
                    MNN_ERROR("Error for test kernel %s for convint8 (im2col + gemm + overflow aware)\n", titles[i].c_str());
                    return false;
                }
            }
            inputShape = {123, 65};
            for (int i = 0; i < kernels.size(); ++i) {
                auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 8, false, 1, 5, SparseList[is].first, SparseList[is].second, false);
                if (!res) {
                    MNN_ERROR("Error for test kernel %s for convint8 (im2col + gemm)\n", titles[i].c_str());
                    return false;
                }
            }
            for (int i = 0; i < kernels.size(); ++i) {
                auto res = testKernel(inputShape, kernels[i], channel, pad, strides, dilate, 3, true, 1, 2, SparseList[is].first, SparseList[is].second, false);
                if (!res) {
                    MNN_ERROR("Error for test kernel %s for convint8 (im2col + gemm + overflow aware)\n", titles[i].c_str());
                    return false;
                }
            }
        }
        return true;
    }
};

class ConvInt8WinogradTestCommon : public MNNTestCase {
public:
    static VARP referenceWinograd(const VARP xInt, const std::vector<int8_t>& weight, const std::vector<float>& wScale, const std::vector<float>& bias, INTS kernel, INTS channel, INTS pads, const WinogradInt8Attr::Attr& attr, float xScale, float yScale, int8_t xZeroPoint, int8_t yZeroPoint, bool relu) {
        auto clamp = [](VARP x) {return _Maximum(_Minimum(x, _Scalar<float>(127)), _Scalar<float>(-127));};

        //auto round = [](VARP x) { return _Round(x); };
        auto roundWithEps = [](VARP x) { return _Round(x + _Sign(x) * _Scalar<float>(1e-6)); };

        auto inDims = xInt->getInfo()->dim;
        int batch = inDims[0], inH = inDims[2], inW = inDims[3];
        int outChannel = channel[1], inChannel = channel[0], kernelH = kernel[1], kernelW = kernel[0];
        int padW = pads[0], padH = pads[1];
        int outH = inH + 2 * padH - kernelH + 1, outW = inW + 2 * padW - kernelW + 1;
        int unitH = attr.unitY, unitW = attr.unitX, unitNumH = UP_DIV(outH, unitH), unitNumW = UP_DIV(outW, unitW);
        int alphaH = unitH + kernelH - 1, alphaW = unitW + kernelW - 1;

        int needH = unitNumH * unitH + kernelH - 1, needW = unitNumW * unitW + kernelW - 1;
        int paddings[] = {0, 0, 0, 0, padH, needH - inH - padH, padW, needW - inW - padW};

        auto xx = _Int8ToFloat(xInt, _Scalar<float>(xScale), xZeroPoint);
        xx = _Convert(xx, NCHW);
        xx = _Pad(xx, _Const(paddings, {8}, NCHW, halide_type_of<int32_t>()));
        // [ic * alphaH * alphaW, N * h_unit_num * w_unit_num]
        xx = _Im2Col(xx, {alphaW, alphaH}, {1, 1}, {0, 0}, {unitW, unitH});
        // [N * h_unit_num * w_unit_num, ic, alphaH, alphaW]
        xx = _Transpose(_Reshape(xx, {inChannel, alphaH, alphaW, -1}), {3, 0, 1, 2});
        Math::WinogradGenerater genH(unitH, kernelH, 1, true), genW(unitW, kernelW, 1, true);
        auto srcTransH = _Const(genH.B()->host<void>(), {alphaH, alphaH}, NCHW);
        auto srcTransW = _Const(genW.B()->host<void>(), {alphaW, alphaW}, NCHW);
        xx = _MatMul(_MatMul(_Transpose(srcTransH, {1, 0}), xx), srcTransW);
        // [alphaH * alphaW, ic, N * h_unit_num * w_unit_num]
        xx = _Reshape(_Transpose(xx, {2, 3, 1, 0}), {alphaH * alphaW, inChannel, -1});

        // simulate input asym quant
        auto xxScale = _Const(attr.inputScales.data(), {alphaH * alphaW, 1, 1}, NCHW);
        auto xxZeroPoint = _Cast<float>(_Const(attr.inputZeroPoints.data(), {alphaH * alphaW, 1, 1}, NCHW, halide_type_of<int>()));
        xx = (clamp(_Round(xx / xxScale + xxZeroPoint)) - xxZeroPoint) * xxScale;

        auto w = _Const(weight.data(), {outChannel, inChannel, kernelH, kernelW}, NCHW, halide_type_of<int8_t>());
        w = _Cast<float>(w) * _Const(wScale.data(), {outChannel, 1, 1, 1}, NCHW);
        auto wTransH = _Const(genH.G()->host<void>(), {alphaH, kernelH}, NCHW);
        auto wTransW = _Const(genW.G()->host<void>(), {alphaW, kernelW}, NCHW);
        // [oc, ic, alphaH, alphaW]
        auto ww = _MatMul(_MatMul(wTransH, w), _Transpose(wTransW, {1, 0}));
        // [alphaH * alphaW, oc, ic]
        ww = _Transpose(_Reshape(ww, {outChannel, inChannel, -1}), {2, 0, 1});
        // simulate weight quant
        auto wwScale = _Const(attr.weightScales.data(), {alphaH * alphaW, outChannel, 1}, NCHW);
        ww = clamp(roundWithEps(ww / wwScale));
        ww = ww * wwScale;

        // [alphaH * alphaW, oc, N * h_unit_num * w_unit_num]
        auto yy = _MatMul(ww, xx);
        // [oc, N * h_unit_num * w_unit_num, alphaH, alphaW]
        yy = _Reshape(_Transpose(yy, {1, 2, 0}), {outChannel, -1, alphaH, alphaW});
        auto dstTransH = _Const(genH.A()->host<void>(), {alphaH, unitH}, NCHW);
        auto dstTransW = _Const(genW.A()->host<void>(), {alphaW, unitW}, NCHW);
        // [oc, N * h_unit_num * w_unit_num, unitH, unitW]
        yy = _MatMul(_MatMul(_Transpose(dstTransH, {1, 0}), yy), dstTransW);
        // [N, oc, h_unit_num * unitH, w_unit_num * unitW]
        yy = _Reshape(_Transpose(_Reshape(yy, {outChannel, batch, unitNumH, unitNumW, unitH, unitW}), {1, 0, 2, 4, 3, 5}), {batch, outChannel, unitNumH * unitH, unitNumW * unitW});
        int sliceStartData[] = {0, 0, 0, 0}, sliceEndData[] = {-1, -1, outH, outW};
        yy = _Slice(yy, _Const(sliceStartData, {4}, NCHW), _Const(sliceEndData, {4}, NCHW));
        // TODO: add operator!= to VARP
        if (!bias.empty()) {
            yy = yy + _Const(bias.data(), {1, outChannel, 1, 1}, NCHW);
        }
        if (relu) {
            yy = _Maximum(yy, _Scalar<float>(0));
        }
        yy = _Convert(yy, NC4HW4);
        yy = _FloatToInt8(yy, _Scalar<float>(1.0 / yScale), -127, 127, yZeroPoint);
        return yy;
    }
    static bool testKernel(INTS inputShape, INTS kernel, INTS channel, INTS pads, INTS alphas, bool speed, std::string title, bool relu = true) {
        int ic = channel[0], oc = channel[1], iw = inputShape[0], ih = inputShape[1], kx = kernel[0], ky = kernel[1], alpha2 = alphas[0] * alphas[1];
        for (int batchSize = 1; batchSize <= 3; ++batchSize) {
            VARP x = _Input({batchSize, ic, ih, iw}, NCHW);
            auto xPtr  = x->writeMap<float>();
            float xMin = std::numeric_limits<float>::max(), xMax = std::numeric_limits<float>::lowest();
            for (int i = 0; i < x->getInfo()->size; ++i) {
                xPtr[i] = i % 128; // x in [0, 127], same as relu output, test asym quant
                xMin = std::min(xMin, xPtr[i]);
                xMax = std::max(xMax, xPtr[i]);
            }
            float xScale = (xMax - xMin) / (2.0 * 127), yScale = 0.5;
            int8_t xZeroPoint = roundf((0 - xMin) / xScale - 127), yZeroPoint = 1;

            int wMin = -3, wMax = 3;
            std::vector<float> wScale(oc), bias(oc);
            std::vector<int8_t> weight(oc * ic * ky * kx);
            for (int oz = 0; oz < oc; ++oz) {
                wScale[oz] = (oz % 11) * 0.1 + 0.5; // wScale in [0.5, 1.5]
                bias[oz] = (oz % 5) * 0.5 - 1; // bias in [-1, 1]
                for (int sz = 0; sz < ic; ++sz) {
                    for (int k = 0; k < ky * kx; ++k) {
                        weight[(oz * ic + sz) * ky * kx + k] = ((oz * ic + sz) * ky * kx + k) % (wMax - wMin + 1) + wMin;
                        //weight[(oz * ic + sz) * ky * kx + k] = (oz * oz + sz * sz + k * k) % (wMax - wMin + 1) + wMin; // w in [wMin, wMax]
                    }
                }
            }

            x = _Convert(x, NC4HW4);
            // For sse we use uint8 instead of int8, use FloatToInt8 to hidden detail
            x = _FloatToInt8(x, _Scalar<float>(1.0 / xScale), -127, 127, xZeroPoint);

            WinogradInt8Attr attrs;
            std::vector<float> transInputScales(alpha2, 0.9), transWeightScales(alpha2 * oc, 1.1);
            std::vector<int> transInputZeroPoint(alpha2, 1);
            attrs.add(0, 0, ky, kx, alphas[1] - ky + 1, alphas[0] - kx + 1, transInputScales, transWeightScales, transInputZeroPoint);
            auto yTarget = referenceWinograd(x, weight, wScale, bias, kernel, channel, pads, attrs.attrs[0], xScale, yScale, xZeroPoint, yZeroPoint, relu);
            auto y = _Conv(std::move(weight), std::move(bias), std::move(wScale), x, channel,
                           kernel, CAFFE, {1, 1}, {1, 1}, 1, pads, relu, xScale, yScale, xZeroPoint, yZeroPoint,
                           -127, 127, 127, false);
            y = attrs.turnToWinogradConv(y);

            yTarget = _Convert(_Cast<int>(_Int8ToFloat(yTarget, _Scalar<float>(1.0))), NCHW);
            y = _Convert(_Cast<int>(_Int8ToFloat(y, _Scalar<float>(1.0))), NCHW);
            auto yTargetInfo = yTarget->getInfo(), yInfo = y->getInfo();
            if (yTargetInfo == nullptr || yInfo == nullptr || yTargetInfo->size != yInfo->size) {
                MNN_ERROR("[ConvInt8WinogradTestCommon] getInfo not match\n");
                return false;
            }
            auto yTargetPtr = yTarget->readMap<int>(), yPtr = y->readMap<int>();
            if (yTargetPtr == nullptr || yPtr == nullptr) {
                MNN_ERROR("[ConvInt8WinogradTestCommon] result is nullptr\n");
                return false;
            }
            if (!checkVector<int>(yPtr, yTargetPtr, yInfo->size, 1)) {
                MNN_ERROR("[ConvInt8WinogradTestCommon] result error for batchSize = %d, oc=%d, oh=%d, ow=%d\n", batchSize, yInfo->dim[1], yInfo->dim[2], yInfo->dim[3]);
                return false;
            }
            if (speed) {
                x.fix(VARP::INPUT);
                // warm up, do onResize first for shapeDirty
                x->writeMap<float>();
                y->readMap<float>();

                MNN::Timer _t;
                const int LOOP = 20;
                for (int i = 0; i < LOOP; ++i) {
                    x->writeMap<float>();
                    y->readMap<float>();
                }
                auto time = (float)_t.durationInUs() / 1000.0f;
                MNN_PRINT("ConvInt8 Winograd %s input=(1x%dx%dx%d) kernel=(%dx%dx%dx%d) avg time = %.2f\n",
                          title.c_str(), ic, ih, iw, oc, ic, ky, kx, 1.0 * time / LOOP);
            }
        }
        return true;
    }
};

class ConvInt8WinogradTest : public ConvInt8WinogradTestCommon {
    virtual bool run(int precision) {
        INTS pad = {1, 1}, inputShape = {47, 39}; // {w, h}
        INTS channel = {32, 32}; // {ci, co}

        std::vector<std::vector<int>> kernels = {
            {3, 3}//, {3, 2}, {2, 3}, {2, 2}//, {4, 4}, {7, 1}, {1, 7} // {w, h}
        };
        std::vector<std::string> titles = {
            "3x3", "2x3", "3x2", "2x2", "4x4", "1x7", "7x1"
        };
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, {4, 4}, false, titles[i] + ",alpha=4");
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 (winograd)\n", titles[i].c_str());
                return false;
            }
            /*res = testKernel(inputShape, kernels[i], channel, pad, {6, 6}, false, titles[i] + ",alpha=6");
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 (winograd)\n", titles[i].c_str());
                return false;
            }*/
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
            {3, 3}//, {5, 5}, {7, 1}, {1, 7} // {w, h}
        };

        std::vector<std::string> titles = {"3x3", "5x5", "1x7", "7x1"};
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], channel, pad, {4, 4}, true, titles[i] + ",alpha=4");
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 (winograd)\n", titles[i].c_str());
                return false;
            }
            res = testKernel(inputShape, kernels[i], channel, pad, {6, 6}, true, titles[i] + ",alpha=6");
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
        printf("Test strides=1\n");
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
        printf("strides=2\n");
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], {channel, channel}, pad, {2, 2}, dilate, 8, true, channel, 1, MNN::SparseAlgo_RANDOM, 1, false);
            if (!res) {
                FUNC_PRINT(1);
                return false;
            }
        }
        return true;
    }
};

class DepthwiseConvSpeedInt8Test : public ConvInt8TestCommon {
public:
    virtual bool run(int precision) {
        INTS strides = {1, 1}, dilate = {1, 1}, pad = {0, 0}, inputShape = {112, 144}; // {w, h}
        int channel = 16;
        std::vector<std::vector<int>> kernels = {
            {3, 3}
        };
        std::vector<std::string> titles = {
            "3x3"
        };
        printf("Depthwise Speed Test Strides=1.\n");
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], {channel, channel}, pad, strides, dilate, 8, false, channel, 4, MNN::SparseAlgo_RANDOM, 1, false, true);
            if (!res) {
                FUNC_PRINT(1);
                return false;
            }
        }
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], {channel, channel}, pad, strides, dilate, 3, true, channel, 1, MNN::SparseAlgo_RANDOM, 1, false, true);
            if (!res) {
                FUNC_PRINT(1);
                return false;
            }
        }
        printf("Depthwise Speed Test Strides=2\n");
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernel(inputShape, kernels[i], {channel, channel}, pad, {2, 2}, dilate, 8, true, channel, 1, MNN::SparseAlgo_RANDOM, 1, false, true);
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
MNNTestSuiteRegister(ConvInt8WinogradTest, "op/ConvInt8/winograd");
MNNTestSuiteRegister(ConvSpeedInt8WinogradTest, "speed/ConvInt8/winograd");
MNNTestSuiteRegister(DepthwiseConvInt8Test, "op/ConvInt8/depthwise");
MNNTestSuiteRegister(DepthwiseConvSpeedInt8Test, "speed/ConvInt8/depthwise");
