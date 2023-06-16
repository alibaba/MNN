//
//  ConvSpeedInt8Test.cpp
//  MNNTests
//
//  Created by MNN on 2019/010/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include "core/Session.hpp"
#include <thread>
#include "MNN_generated.h"
using namespace MNN::Express;
using namespace MNN;
inline int8_t int32ToInt8(int data, int bias, float scale) {
    float value = roundf((float)(data + bias) * scale);
    value       = std::max(value, -127.0f);
    value       = std::min(value, 127.0f);
    return static_cast<int8_t>(value);
}

// y = Conv(x, w), x and y is C4 ordered format, weight is [oc, ic, kh, kw] raw format.
static std::vector<int8_t> naiveConvInt8C4(const int8_t* x, const int8_t* weight, const int* bias, const float* scale,
                                           int ow, int oh, int iw, int ih, int ic, int oc, int kw, int kh, int padX, int padY, int padValue = 0,
                                           int strideX = 1, int strideY = 1, int dilateX = 1, int dilateY = 1, int batch = 1) {
    int ic4 = (ic + 3) / 4, oc4 = (oc + 3) / 4;
    std::vector<int8_t> yCorrect(batch * oc4 * oh * ow * 4, 0);
    for (int b = 0; b < batch; ++b) {
        for (int oz = 0; oz < oc; ++oz) {
            int ozC4 = oz / 4, ozRemain = oz % 4;
            for (int oy = 0; oy < oh; ++oy) {
                for (int ox = 0; ox < ow; ++ox) {
                    int32_t yInt32 = 0;
                    for (int sz = 0; sz < ic; ++sz) {
                        int szC4 = sz / 4, szRemain = sz % 4;
                        for (int ky = 0; ky < kh; ++ky) {
                            for (int kx = 0; kx < kw; ++kx) {
                                int ix = ox * strideX + kx * dilateX - padX, iy = oy * strideY + ky * dilateY - padY;
                                int8_t xValue = padValue;
                                if (ix >= 0 && ix < iw && iy >= 0 && iy < ih) {
                                    xValue = x[(((b * ic4 + szC4) * ih + iy) * iw + ix) * 4 + szRemain];
                                }
                                yInt32 += xValue * weight[((oz * ic + sz) * kh + ky) * kw + kx];
                            }
                        }
                    }
                    yCorrect[(((b * oc4 + ozC4) * oh + oy) * ow + ox) * 4 + ozRemain] = int32ToInt8(yInt32, bias[oz], scale[oz]);
                }
            }
        }
    }
    return yCorrect;
}

class ConvSpeedInt8TestCommon : public MNNTestCase {
protected:
    static bool testKernelV2(std::string title, INTS inputShape, INTS kernel, INTS channel, INTS pad, INTS strides, INTS dilate, int nbit = 8) {
        int iw = inputShape[0], ih = inputShape[1], kw = kernel[0], kh = kernel[1], ic = channel[0], oc = channel[1];
        std::vector<int> bias(channel[1]);
        std::vector<float> scale(channel[1]);
        std::vector<int8_t> weight(oc * ic * kw * kh);
        VARP x = _Input({1, ic, ih, iw}, NC4HW4, halide_type_of<int8_t>());
        auto xInfo = x->getInfo();
        int8_t xMin = -(1<<(nbit-1))+1, xMax = (1<<(nbit-1))-1;
        auto y     = _Conv(std::vector<int8_t>(weight), std::vector<int>(bias), std::vector<float>(scale), x,
                           channel, kernel, PaddingMode::CAFFE, strides, dilate, 1, pad, false, 0, 0, -127, 127, false);
        if (nbit != 8) {
            std::unique_ptr<MNN::OpT> op(y->expr().first->get()->UnPack());
            op->main.AsConvolution2D()->symmetricQuan->nbits = nbit;
            y = Variable::create(Expr::create(op.get(), {x}));
            op.reset();
        }
        auto yInfo = y->getInfo();
        auto ow = yInfo->dim[3], oh = yInfo->dim[2];
        std::unique_ptr<NetT> net(new NetT);
        Variable::save({y}, net.get());
        y = nullptr;
        x = nullptr;
        flatbuffers::FlatBufferBuilder builder;
        auto len = MNN::Net::Pack(builder, net.get());
        builder.Finish(len);
        net.reset();
        std::vector<std::thread> threads;
        std::vector<std::shared_ptr<Interpreter>> inters;
        ScheduleConfig config;
        config.numThread = 1;
        std::vector<MNN::Session*> sessions;
        for (int i = 0; i < 4; ++i) {
            std::shared_ptr<Interpreter> interMNN(Interpreter::createFromBuffer(builder.GetBufferPointer(), builder.GetSize()));
            auto session = interMNN->createSession(config);
            sessions.emplace_back(session);
            inters.emplace_back(interMNN);
        }
        auto f = [&] (int index) {
            {
                MNN::Timer _t;
                const int LOOP = 20;
                for (int i = 0; i < LOOP; ++i) {
                    inters[index]->runSession(sessions[index]);
                }
                auto time = (float)_t.durationInUs() / 1000.0f;
                MNN_PRINT("%s kernel=(%dx%d) input=(1x%dx%dx%d) output=(1x%dx%dx%d) stride=(%dx%d), avg time = %f\n",
                          title.c_str(), kh, kw, ic, ih, iw, oc, oh, ow, strides[1], strides[0], 1.0 * time / LOOP);
            }
        };
        MNN_PRINT("Run 4 instance\n");
        for (int i = 0; i < 4; ++i) {
            int index = i;
            threads.emplace_back(std::thread([&, index]() {
                f(index);
            }));
        }
        for (auto& t : threads) {
            t.join();
        }
        MNN_PRINT("Run 1 instance\n");
        f(0);
        return true;
    }
    static bool testKernel(std::string title, INTS inputShape, INTS kernel, INTS channel, INTS pad, INTS strides, INTS dilate, int nbit = 8) {
        int iw = inputShape[0], ih = inputShape[1], kw = kernel[0], kh = kernel[1], ic = channel[0], oc = channel[1];
        std::vector<int> bias(channel[1]);
        std::vector<float> scale(channel[1]);
        std::vector<int8_t> weight(oc * ic * kw * kh);
        VARP x = _Input({1, ic, ih, iw}, NC4HW4, halide_type_of<int8_t>());
        auto xInfo = x->getInfo();
        auto xPtr = x->writeMap<int8_t>();
        int8_t xMin = -(1<<(nbit-1))+1, xMax = (1<<(nbit-1))-1;
        for (int i=0; i<xInfo->size; ++i) {
            xPtr[i] = (i % (xMax - xMin + 1)) + xMin; // x in [xMin, xMax]
        }
        for (int i = 0; i < oc; ++i) {
            bias[i] = (10000 + i*i*10 - i*i*i) % 12580;
            scale[i] = fabs(((127-i)*i % 128) / 20000.0f);
            for (int j = 0; j < ic; ++j) {
                auto weightCurrent = weight.data() + (i * ic + j) * kw * kh;
                for (int k = 0; k < kw * kh; ++k) {
                    weightCurrent[k] = ((i * i + j * j + k * k) % (xMax - xMin + 1)) + xMin; // w in [xMin, xMax]
                }
            }
        }
        x = _FloatToInt8(_Cast<float>(x), _Scalar<float>(1.0f), -127, 127);
        //x.fix(MNN::Express::VARP::CONSTANT);
        auto y     = _Conv(std::vector<int8_t>(weight), std::vector<int>(bias), std::vector<float>(scale), x,
                           channel, kernel, PaddingMode::CAFFE, strides, dilate, 1, pad, false, 0, 0, -127, 127, false);
        if (nbit != 8) {
            std::unique_ptr<MNN::OpT> op(y->expr().first->get()->UnPack());
            op->main.AsConvolution2D()->symmetricQuan->nbits = nbit;
            y = Variable::create(Expr::create(op.get(), {x}));
            op.reset();
        }
        auto yr = _Int8ToFloat(y, _Scalar<float>(1.0f));
        yr = _Cast<int8_t>(yr);
        auto yInfo = y->getInfo();
        auto yPtr  = yr->readMap<int8_t>();
        auto ow = yInfo->dim[3], oh = yInfo->dim[2];
        auto targetValues = naiveConvInt8C4(xPtr, weight.data(), bias.data(), scale.data(),
                                            ow, oh, iw, ih, channel[0], channel[1], kernel[0], kernel[1], pad[0], pad[1]);
        for (int i = 0; i < targetValues.size(); ++i) {
            int8_t targetValue = targetValues[i], computeResult = yPtr[i];
            if (targetValue != computeResult) {
                MNN_PRINT("ConvInt8 result Error: %d -> %d\n", targetValue, computeResult);
                break;
            }
        }
        {
            x.fix(VARP::INPUT);
            MNN::Timer _t;
            const int LOOP = 20;
            for (int i = 0; i < LOOP; ++i) {
                x->writeMap<float>();
                y->readMap<float>();
            }
            auto time = (float)_t.durationInUs() / 1000.0f;
            MNN_PRINT("%s kernel=(%dx%d) input=(1x%dx%dx%d) output=(1x%dx%dx%d) stride=(%dx%d), avg time = %f\n",
                      title.c_str(), kh, kw, ic, ih, iw, oc, oh, ow, strides[1], strides[0], 1.0 * time / LOOP);
        }

        return true;
    }
};

class ConvSpeedInt8Test : public ConvSpeedInt8TestCommon {
public:
    virtual bool run(int precision) {
        INTS strides = {1, 1}, dilate = {1, 1}, pad = {1, 1}, inputShape = {28, 28}; // {w, h}
        INTS channel = {128, 128}; // {ci, co}
        std::vector<std::vector<int>> kernels = {
            {1, 1}, {3, 3}, {5, 5}, {7, 1}, {1, 7} // {w, h}
        };
        std::vector<std::string> titles = {"3x3", "5x5", "1x7", "7x1"};
        std::vector<int> weightBits = {8, 7};
        for (auto& bits : weightBits) {
            MNN_PRINT("Bits=%d\n", bits);
            inputShape = {28, 28};
            for (int i = 0; i < kernels.size(); ++i) {
                auto res = testKernel("ConvInt8 (im2col + gemm)", inputShape, kernels[i], channel, pad, strides, dilate, bits);
                if (!res) {
                    MNN_ERROR("Error for test kernel %s for convint8 (im2col + gemm)\n", titles[i].c_str());
                    return false;
                }
            }
            inputShape = {129, 412};
            for (int i = 0; i < 1; ++i) {
                auto res = testKernel("ConvInt8 (im2col + gemm)", inputShape, kernels[i], channel, pad, strides, dilate, bits);
                if (!res) {
                    MNN_ERROR("Error for test kernel %s for convint8 129,412 (im2col + gemm)\n", titles[i].c_str());
                    return false;
                }
            }
        }
        return true;
    }
};

class ConvSpeedInt8MultiInstanceTest : public ConvSpeedInt8TestCommon {
    public:
    virtual bool run(int precision) {
        INTS strides = {1, 1}, dilate = {1, 1}, pad = {3, 4}, inputShape = {215, 204}; // {w, h}
        INTS channel = {32, 56}; // {ci, co}
        std::vector<std::vector<int>> kernels = {
            {3, 3}
        };
        std::vector<std::string> titles = {"3x3"};
        for (int i = 0; i < kernels.size(); ++i) {
            auto res = testKernelV2("ConvInt8 (im2col + gemm)", inputShape, kernels[i], channel, pad, strides, dilate);
            if (!res) {
                MNN_ERROR("Error for test kernel %s for convint8 (im2col + gemm)\n", titles[i].c_str());
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ConvSpeedInt8Test, "speed/ConvInt8/im2col_gemm");
MNNTestSuiteRegister(ConvSpeedInt8MultiInstanceTest, "speed/ConvInt8/multi_instance");
