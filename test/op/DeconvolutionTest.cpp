//
//  DeconvolutionTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <string>
#include <sstream>
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "core/IDSTEncoder.hpp"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

static void reference_deconv2d(const float* input, const std::vector<float>& weight,
                             const std::vector<float>& bias, std::vector<float>& output, int batch, int ic, int oc,
                             int ih, int iw, int pad_h, int pad_w, int kh, int kw, int stridew, int strideh,
                             int dilation) {
    int oh, ow;
    ow  = (iw - 1) * stridew + dilation * (kw - 1) + 1 - pad_w * 2;
    oh = (ih - 1) * strideh + dilation * (kh - 1) + 1 - pad_h * 2;

    if (oh <= 0 || ow <= 0) {
        output.clear();
        return;
    }
    output.resize(batch * oh * ow * oc);
    for (int b = 0; b < batch; ++b) {
        for (int oz = 0; oz < oc; ++oz) {
            auto outputPtr = output.data() + b * oh * ow * oc + oz * ow * oh;
            for (int s=0; s<oh*ow; ++s) {
                outputPtr[s] = bias[oz];
            }
        }
    }

    for (int b = 0; b < batch; ++b) {
        for (int oz = 0; oz < oc; ++oz) {
            // Revert input, output
            for (int iy = 0; iy < ih; ++iy) {
                for (int ix = 0; ix < iw; ++ix) {
                    auto destOffset = (b * oc + oz) * oh * ow;
                    for (int sz = 0; sz < ic; ++sz) {
                        auto srcOffset = ((b * ic + sz) * ih + iy)*iw+ix;
                        auto xValue = input[srcOffset];
                        auto wOffset = sz * oc * kh * kw + oz * kh * kw;
                        for (int ky = 0; ky < kh; ++ky) {
                            for (int kx = 0; kx < kw; ++kx) {
                                int ox = ix * stridew + kx * dilation - pad_w;
                                int oy = iy * strideh + ky * dilation - pad_h;
                                if (ox >= 0 && ox < ow && oy >= 0 && oy < oh) {
                                    auto w = weight[wOffset+ky*kw+kx];
                                    output[destOffset + oy * ow + ox] += xValue * w;
                                }
                            }
                        }
                    }
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

VARP _Deconv(std::vector<int8_t>&& weight, std::vector<float>&& bias, std::vector<float>&& scale, VARP x, INTS channel, INTS kernelSize,
           PaddingMode pad, INTS stride, INTS dilate, int group, INTS pads, bool relu, bool relu6, int8_t inputZeroPoint, int8_t outputZeroPoint,
             int8_t maxValue, int8_t minValue) {
    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type = OpType_Deconvolution;
    if (channel[0] == channel[1] && channel[0] == group) {
        convOp->type = OpType_DeconvolutionDepthwise;
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
    conv2D->quanParameter = IDSTEncoder::encode(nullptr, scale, channel[1], channel[0] * kernelSize[0] * kernelSize[1], false, weight.data(), -128);
    conv2D->symmetricQuan.reset(new QuantizedFloatParamT);
    conv2D->bias = std::move(bias);
    return (Variable::create(Expr::create(convOp.get(), {x})));
}

class DeconvolutionCommonTest : public MNNTestCase {
public:
    virtual ~DeconvolutionCommonTest() = default;

protected:
    static bool test(const std::string& test_op_name,
                    int batch, int ic, int oc, int ih, int iw, int pad_h, int pad_w, int kh,
                    int kw, int stride, int dilation, int group, int precision) {
        int ow  = (iw - 1) * stride + dilation * (kw - 1) + 1 - pad_w * 2;
        int oh = (ih - 1) * stride + dilation * (kh - 1) + 1 - pad_h * 2;
        if (ow <=0 || oh <= 0) {
            return true;
        }
        auto input = _Input({batch, ic, ih, iw}, NCHW, halide_type_of<float>());
        auto inputPtr = input->writeMap<float>();
        {
            int size = input->getInfo()->size;
            for (int i=0; i<size; ++i) {
                inputPtr[i] = (float)((i+1) % 10) / 100.0f;
            }
        }
        std::vector<float> weightData(ic*oc*kh*kw);
        for (int i=0; i<weightData.size(); ++i) {
            weightData[i] = (float)(10-(i%10)) / 10.0f;
        }
        std::vector<float> biasData(oc);
        for (int i=0; i<oc; ++i) {
            biasData[i] = (float)(5-(i%10)) / 10.0f;
        }
        std::vector<float> rightOutData;
        reference_deconv2d(inputPtr, weightData, biasData, rightOutData, batch, ic, oc, ih, iw, pad_h, pad_w, kh, kw, stride, stride, dilation);
        input = _Convert(input, NC4HW4);
        auto output = _Deconv(std::move(weightData), std::move(biasData), input, {ic, oc}, {kw, kh}, VALID,
                              {stride, stride}, {dilation, dilation}, group, {pad_w, pad_h}, false, false);
        output = _Convert(output, NCHW);
        if (rightOutData.size() != output->getInfo()->size) {
            FUNC_PRINT(1);
            return false;
        }
        

        // difference below 0.5% relative error is considered correct.
        auto outputPtr = output->readMap<float>();
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 20;
        if (!checkVectorByRelativeError<float>(outputPtr, rightOutData.data(), rightOutData.size(), 0.005 * errorScale)) {
            MNN_ERROR("%s test failed!\n", test_op_name.c_str());
            return false;
        }
        return true;
    }
    static bool test(const std::string& test_op_name,
                    vector<float>& inputData, vector<float>& weightData, vector<float>& biasData, vector<float>& rightOutData,
                    int batch, int ic, int oc, int ih, int iw, PadMode mode, int pad_h, int pad_w, int kh,
                    int kw, int stride, int dilation, int group, int precision) {
        std::map<PadMode, Express::PaddingMode> padMap = {
            {PadMode_CAFFE, CAFFE}, {PadMode_VALID, VALID}, {PadMode_SAME, SAME}};
        auto input = _Input({batch, ic, ih, iw}, NCHW, halide_type_of<float>());
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        auto output = _Deconv(std::move(weightData), std::move(biasData), input, {ic, oc}, {kw, kh}, padMap[mode],
                              {stride, stride}, {dilation, dilation}, group, {pad_w, pad_h}, false, false);

        // difference below 0.5% relative error is considered correct.
        auto outputPtr = output->readMap<float>();
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 20;
        if (!checkVectorByRelativeError<float>(outputPtr, rightOutData.data(), rightOutData.size(), 0.005 * errorScale)) {
            MNN_ERROR("%s test failed!\n", test_op_name.c_str());
            return false;
        }
        return true;
    }
};

class DeconvolutionCommonTestInt8 : public MNNTestCase {
public:
    virtual ~DeconvolutionCommonTestInt8() = default;

protected:
    static bool test(const std::string& device_name, const std::string& test_op_name,
                    vector<float>& inputData, vector<int8_t>& weightData, vector<float>& biasData, vector<float>& rightOutData,
                    int batch, int ic, int oc, int ih, int iw, PadMode mode, int pad_h, int pad_w, int kh,
                    int kw, int stride, int dilation, int group, int precision, vector<float>& scale, vector<float>& zeroPoints, vector<float>& quantScales) {
        std::map<PadMode, Express::PaddingMode> padMap = {
            {PadMode_CAFFE, CAFFE}, {PadMode_VALID, VALID}, {PadMode_SAME, SAME}};
        auto input = _Input({batch, ic, ih, iw}, NCHW, halide_type_of<float>());
        input->writeScaleMap(quantScales[0], zeroPoints[0]);
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        auto xC4 = _Convert(input, NC4HW4);
        auto output = _Deconv(std::move(weightData), std::move(biasData), std::move(scale), xC4, {ic, oc}, {kw, kh}, padMap[mode], {stride, stride}, {dilation, dilation}, group, {pad_w, pad_h}, false, false, (int8_t)zeroPoints[0], (int8_t)zeroPoints[1], 127, -127);
        output->writeScaleMap(quantScales[1], zeroPoints[1]);
        auto y = _Convert(output, NCHW);
        // difference below 0.5% relative error is considered correct.
        auto outputPtr = y->readMap<float>();
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 20;
        if (!checkVectorByRelativeError<float>(outputPtr, rightOutData.data(), rightOutData.size(), 0.005 * errorScale)) {
            MNN_ERROR("%s(%s) test failed: batch=%d, oc=%d, oh=%d, ow=%d!\n", test_op_name.c_str(), device_name.c_str(), y->getInfo()->dim[0], y->getInfo()->dim[1], y->getInfo()->dim[2], y->getInfo()->dim[3]);
            return false;
        }
        return true;
    }
};
class DeconvolutionFullTest : public DeconvolutionCommonTest {
public:
    virtual ~DeconvolutionFullTest() = default;
    virtual bool run(int precision) {
        if (MNN_FORWARD_OPENCL == getCurrentType()) {
            MNN_ERROR("Currently opencl run deconvolution has error, skip it\n");
            return true;
        }
        int ocStep = 1;
        int icStep = 1;
        int isStep = 3;
        std::vector<int> ocSize = {
            1, 3, 10, 17
        };
        std::vector<int> icSize = {
            1, 4, 3, 8, 11
        };
        std::vector<int> isSize = {
            1, 7, 9, 13
        };

        for (int batch = 1; batch <= 2; batch++) {
            for (auto oc : ocSize) {
                for (auto ic : icSize) {
                    for (auto is : isSize) {
                        int ih = is;
                        int iw = is;
                        for (int kw = 1; kw <= 7 && kw <= is; kw+=2) {
                            for (int kh = 1; kh <= 7 && kh <= is; kh+=3) {
                                for (int d = 1; d <= 2; d++) {
                                    for (int s = 1; s <= 2; s++) {
                                        int stride = s;
                                        for (int p = 0; p <= 1; p++) {
                                            std::ostringstream name;
                                            int pad_w = p;
                                            int pad_h = p;
                                            name << "Deconvolution: " << batch <<","<< oc <<","<<ic<<","<<is<<"["<< kw <<","<<kh <<"][s:" << s << ",p:"
                                            <<p << ",d:"<<d<<"]";
                                            bool succ = DeconvolutionCommonTest::test(name.str(),batch, ic, oc, ih, iw, pad_h, pad_w, kh, kw, stride, d, 1, precision);
                                            if (!succ) {
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

class DeconvolutionTest : public DeconvolutionCommonTest {
public:
    virtual ~DeconvolutionTest() = default;
    virtual bool run(int precision) {
        MNN_PRINT("beigin testcase 0\n");

        {
            std::vector<float> data_a = {// channel 0
                                         1.0, 2.0, 4.0, 5.0,
                                         // channel 1
                                         1.1, 2.1, 4.1, 5.1,
                                         // channel 2
                                         1.2, 2.2, 4.2, 5.2};

            std::vector<float> weight = {//IOHW
                // input channel0
                // output channel0
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                // output channel1
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,

                // input channel1
                // output channel0
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                // output channel1
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,

                // input channel2
                // output channel0
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                // output channel1
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            };
            std::vector<float> bias   = {0.0, 0.0};
            std::vector<float> data_c = {3.3,  3.3,  9.6,  6.3,  6.3,  3.3,  3.3,  9.6,  6.3,  6.3,  15.6, 15.6, 37.2,
                                         21.6, 21.6, 12.3, 12.3, 27.6, 15.3, 15.3, 12.3, 12.3, 27.6, 15.3, 15.3,

                                         6.6,  6.6,  19.2, 12.6, 12.6, 6.6,  6.6,  19.2, 12.6, 12.6, 31.2, 31.2, 74.4,
                                         43.2, 43.2, 24.6, 24.6, 55.2, 30.6, 30.6, 24.6, 24.6, 55.2, 30.6, 30.6};

            int ic = 3, oc = 2;
            int kw = 3, kh = 3, ih = 2, iw = 2;
            int stride = 2, dilation = 1;
            int group = 1, batch = 1;
            int pad_w = 0, pad_h = 0;

            bool succ = DeconvolutionCommonTest::test("DeconvolutionTest0", data_a, weight, bias, data_c,
                                                      batch, ic, oc, ih, iw, PadMode_VALID, pad_h, pad_w, kh, kw,
                                                      stride, dilation, group, precision);
            if (!succ) {
                return false;
            }
        }

        MNN_PRINT("beigin testcase 1\n");
        {
            std::vector<float> data_a = {// channel 0
                                         1.0, 2.0, 4.0, 5.0,
                                         // channel 1
                                         1.1, 2.1, 4.1, 5.1,
                                         // channel 2
                                         1.2, 2.2, 4.2, 5.2};

            std::vector<float> weight = {//IOHW
                // input channel0
                // output channel0
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                // output channel1
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,

                // input channel1
                // output channel0
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                // output channel1
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,

                // input channel2
                // output channel0
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                // output channel1
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            };
            std::vector<float> bias   = {1.0, 2.0};
            std::vector<float> data_c = {
                4.3, 10.6, 10.6, 7.3,  16.6, 38.2, 38.2, 22.6, 16.6, 38.2, 38.2, 22.6, 13.3, 28.6, 28.6, 16.3,

                8.6, 21.2, 21.2, 14.6, 33.2, 76.4, 76.4, 45.2, 33.2, 76.4, 76.4, 45.2, 26.6, 57.2, 57.2, 32.6,
            };
            int ic = 3, oc = 2;
            int kw = 4, kh = 4, ih = 2, iw = 2;
            int stride = 2, dilation = 1;
            int group = 1, batch = 1;
            int pad_w = 1, pad_h = 1;

            bool succ = DeconvolutionCommonTest::test("Deconv", data_a, weight, bias, data_c,
                                                      batch, ic, oc, ih, iw, PadMode_VALID, pad_h, pad_w, kh, kw,
                                                      stride, dilation, group, precision);
            if (!succ) {
                return false;
            }
        }

        MNN_PRINT("beigin testcase 2\n");
        {
            std::vector<float> data_a = {// channel 0
                                         1.0, 2.0, 4.0, 5.0,
                                         // channel 1
                                         1.1, 2.1, 4.1, 5.1,
                                         // channel 2
                                         1.2, 2.2, 4.2, 5.2};

            std::vector<float> weight = {//IOHW
                // input channel0
                // output channel0
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                // output channel1
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,

                // input channel1
                // output channel0
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                // output channel1
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,

                // input channel2
                // output channel0
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                // output channel1
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            };
            std::vector<float> bias   = {0.0, 0.0};
            std::vector<float> data_c = {3.3,  3.3,  9.6,  6.3,  3.3,  3.3,  9.6,  6.3, 15.6, 15.6, 37.2,
                                         21.6, 12.3, 12.3, 27.6, 15.3,

                                         6.6,  6.6,  19.2, 12.6, 6.6,  6.6,  19.2, 12.6, 31.2, 31.2, 74.4,
                                         43.2, 24.6, 24.6, 55.2, 30.6};
            int ic = 3, oc = 2;
            int kw = 3, kh = 3, ih = 2, iw = 2;
            int stride = 2, dilation = 1;
            int group = 1, batch = 1;
            int pad_w = 0, pad_h = 0;

            bool succ = DeconvolutionCommonTest::test("Deconv", data_a, weight, bias, data_c,
                                                      batch, ic, oc, ih, iw, PadMode_SAME, pad_h, pad_w, kh, kw,
                                                      stride, dilation, group, precision);
            if (!succ) {
                return false;
            }
        }

        return true;
    }
};

class DeconvolutionInt8Test : public DeconvolutionCommonTestInt8 {
public:
    virtual ~DeconvolutionInt8Test() = default;
    virtual bool run(int precision) {
        MNN_PRINT("begin testcase 0\n");

        {
            std::vector<float> data_a = {// channel 0
                                         1.0, 2.0, 4.0, 5.0,
                                         // channel 1
                                         1.1, 2.1, 4.1, 5.1,
                                         // channel 2
                                         1.2, 2.2, 4.2, 5.2};

            std::vector<int8_t> weight = {//IOHW
                // input channel0
                // output channel0
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                // output channel1
                2, 2, 2, 2, 2, 2, 2, 2, 2,

                // input channel1
                // output channel0
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                // output channel1
                2, 2, 2, 2, 2, 2, 2, 2, 2,

                // input channel2
                // output channel0
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                // output channel1
                2, 2, 2, 2, 2, 2, 2, 2, 2,
            };
            std::vector<float> bias = {0, 0};
            std::vector<float> data_c = {3.3,  3.3,  9.6,  6.3,  6.3,  3.3,  3.3,  9.6,  6.3,  6.3,  15.6, 15.6, 37.2,
                                         21.6, 21.6, 12.3, 12.3, 27.6, 15.3, 15.3, 12.3, 12.3, 27.6, 15.3, 15.3,

                                         6.6,  6.6,  19.2, 12.6, 12.6, 6.6,  6.6,  19.2, 12.6, 12.6, 31.2, 31.2, 74.4,
                                         43.2, 43.2, 24.6, 24.6, 55.2, 30.6, 30.6, 24.6, 24.6, 55.2, 30.6, 30.6};
            
            std::vector<float> scale = {1., 1.};
            std::vector<float> zeroPoints = {0, 0};
            std::vector<float> quantScales = {0.0416, 0.58582677};

            int ic = 3, oc = 2;
            int kw = 3, kh = 3, ih = 2, iw = 2;
            int stride = 2, dilation = 1;
            int group = 1, batch = 1;
            int pad_w = 0, pad_h = 0;

            bool succ = DeconvolutionCommonTestInt8::test("CPU", "DeconvolutionTest0", data_a, weight, bias, data_c,
                                                      batch, ic, oc, ih, iw, PadMode_VALID, pad_h, pad_w, kh, kw,
                                                      stride, dilation, group, precision, scale, zeroPoints, quantScales);
            if (!succ) {
                return false;
            }
        }
        
        MNN_PRINT("begin testcase 1\n");
        {
            std::vector<float> data_a = {// channel 0
                                         1.0, 2.0, 4.0, 5.0,
                                         // channel 1
                                         1.1, 2.1, 4.1, 5.1,
                                         // channel 2
                                         1.2, 2.2, 4.2, 5.2};

            std::vector<int8_t> weight = {//IOHW
                // input channel0
                // output channel0
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                // output channel1
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,

                // input channel1
                // output channel0
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                // output channel1
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,

                // input channel2
                // output channel0
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                // output channel1
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            };
            std::vector<float> bias   = {1, 2};
            std::vector<float> data_c = {
                4.3, 10.6, 10.6, 7.3,  16.6, 38.2, 38.2, 22.6, 16.6, 38.2, 38.2, 22.6, 13.3, 28.6, 28.6, 16.3,

                8.6, 21.2, 21.2, 14.6, 33.2, 76.4, 76.4, 45.2, 33.2, 76.4, 76.4, 45.2, 26.6, 57.2, 57.2, 32.6,
            };
            int ic = 3, oc = 2;
            int kw = 4, kh = 4, ih = 2, iw = 2;
            int stride = 2, dilation = 1;
            int group = 1, batch = 1;
            int pad_w = 1, pad_h = 1;
            
            std::vector<float> scale = {1., 1.};
            std::vector<float> zeroPoints = {0, 0};
            std::vector<float> quantScales = {0.0416, 0.6112};

            bool succ = DeconvolutionCommonTestInt8::test("CPU", "Deconv", data_a, weight, bias, data_c,
                                                      batch, ic, oc, ih, iw, PadMode_VALID, pad_h, pad_w, kh, kw,
                                                      stride, dilation, group, precision, scale, zeroPoints, quantScales);
            if (!succ) {
                return false;
            }
        }

        MNN_PRINT("begin testcase 2\n");
        {
            std::vector<float> data_a = {// channel 0
                                         1.0, 2.0, 4.0, 5.0,
                                         // channel 1
                                         1.1, 2.1, 4.1, 5.1,
                                         // channel 2
                                         1.2, 2.2, 4.2, 5.2};

            std::vector<int8_t> weight = {//IOHW
                // input channel0
                // output channel0
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                // output channel1
                2, 2, 2, 2, 2, 2, 2, 2, 2,

                // input channel1
                // output channel0
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                // output channel1
                2, 2, 2, 2, 2, 2, 2, 2, 2,

                // input channel2
                // output channel0
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                // output channel1
                2, 2, 2, 2, 2, 2, 2, 2, 2,
            };
            std::vector<float> bias   = {0, 0};
            std::vector<float> data_c = {3.3,  3.3,  9.6,  6.3,  3.3,  3.3,  9.6,  6.3, 15.6, 15.6, 37.2,
                                         21.6, 12.3, 12.3, 27.6, 15.3,

                                         6.6,  6.6,  19.2, 12.6, 6.6,  6.6,  19.2, 12.6, 31.2, 31.2, 74.4,
                                         43.2, 24.6, 24.6, 55.2, 30.6};
            int ic = 3, oc = 2;
            int kw = 3, kh = 3, ih = 2, iw = 2;
            int stride = 2, dilation = 1;
            int group = 1, batch = 1;
            int pad_w = 0, pad_h = 0;

            std::vector<float> scale = {1., 1.};
            std::vector<float> zeroPoints = {0, 0};
            std::vector<float> quantScales = {0.0416, 0.6112};

            bool succ = DeconvolutionCommonTestInt8::test("CPU", "Deconv", data_a, weight, bias, data_c,
                                                      batch, ic, oc, ih, iw, PadMode_SAME, pad_h, pad_w, kh, kw,
                                                      stride, dilation, group, precision, scale, zeroPoints, quantScales);
            if (!succ) {
                return false;
            }
        }
        MNN_PRINT("begin testcase 3\n");
        {
            std::vector<float> data_a = {// channel 0
                                         1.0, 2.0, 4.0, 5.0,
                                         // channel 1
                                         1.1, 2.1, 4.1, 5.1,
                                         // channel 2
                                         1.2, 2.2, 4.2, 5.2};

            std::vector<int8_t> weight = {//IOHW
                // input channel0

                1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2,
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2,
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2,
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2,
                1, 1, 1, 1, 1, 1, 1, 1, 1,

                // input channel1

                1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2,
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2,
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2,
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2,
                1, 1, 1, 1, 1, 1, 1, 1, 1,

                // input channel2

                1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2,
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2,
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2,
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2,
                1, 1, 1, 1, 1, 1, 1, 1, 1,
            };
            std::vector<float> bias(9, 0);
            std::vector<float> data_c = {3.3,  3.3,  9.6,  6.3,  3.3,  3.3,  9.6,  6.3, 15.6, 15.6, 37.2,
                                         21.6, 12.3, 12.3, 27.6, 15.3,

                                         6.6,  6.6,  19.2, 12.6, 6.6,  6.6,  19.2, 12.6, 31.2, 31.2, 74.4,
                                         43.2, 24.6, 24.6, 55.2, 30.6};
            int ic = 3, oc = 9;
            int kw = 3, kh = 3, ih = 2, iw = 2;
            int stride = 2, dilation = 1;
            int group = 1, batch = 1;
            int pad_w = 0, pad_h = 0;

            std::vector<float> scale = {1., 1.};
            std::vector<float> zeroPoints = {0, 0};
            std::vector<float> quantScales = {0.0416, 0.6112};

            bool succ = DeconvolutionCommonTestInt8::test("CPU", "Deconv", data_a, weight, bias, data_c,
                                                      batch, ic, oc, ih, iw, PadMode_SAME, pad_h, pad_w, kh, kw,
                                                      stride, dilation, group, precision, scale, zeroPoints, quantScales);
            if (!succ) {
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(DeconvolutionFullTest, "op/Deconvolutionfull");
MNNTestSuiteRegister(DeconvolutionTest, "op/Deconvolution");
MNNTestSuiteRegister(DeconvolutionInt8Test, "op/DeconvolutionInt8");


