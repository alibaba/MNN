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
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"
using namespace MNN::Express;
class DeconvolutionTest : public MNNTestCase {
public:
    virtual ~DeconvolutionTest() = default;
    virtual bool run() {
        MNN_PRINT("beigin testcase 0\n");
        {
            auto input = _Input({1, 3, 2, 2}, NCHW, halide_type_of<float>());

            std::vector<float> data_a = {// channel 0
                                         1.0, 2.0, 4.0, 5.0,
                                         // channel 1
                                         1.1, 2.1, 4.1, 5.1,
                                         // channel 2
                                         1.2, 2.2, 4.2, 5.2};

            std::vector<float> weight = {
                // output channel0
                // input channel0
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                // input channel1
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                // input channel2
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,

                // output channel1
                // input channel0
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                // input channel1
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                // input channel2
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
            };
            std::vector<float> bias   = {0.0, 0.0};
            std::vector<float> data_c = {3.3,  3.3,  9.6,  6.3,  6.3,  3.3,  3.3,  9.6,  6.3,  6.3,  15.6, 15.6, 37.2,
                                         21.6, 21.6, 12.3, 12.3, 27.6, 15.3, 15.3, 12.3, 12.3, 27.6, 15.3, 15.3,

                                         6.6,  6.6,  19.2, 12.6, 12.6, 6.6,  6.6,  19.2, 12.6, 12.6, 31.2, 31.2, 74.4,
                                         43.2, 43.2, 24.6, 24.6, 55.2, 30.6, 30.6, 24.6, 24.6, 55.2, 30.6, 30.6};
            int ic = 3, oc = 2;
            int kw = 3, kh = 3;
            int stride = 2, dilation = 1;
            int group = 1;
            int pad_w = 0, pad_h = 0;

            auto output = _Deconv(std::move(weight), std::move(bias), input, {ic, oc}, {kw, kh}, VALID,
                                  {stride, stride}, {dilation, dilation}, group, {pad_w, pad_h}, false, false);

            ::memcpy(input->writeMap<float>(), data_a.data(), data_a.size() * sizeof(float));

            if (!checkVectorByRelativeError<float>(output->readMap<float>(), data_c.data(), data_c.size(), 0.005)) {
                MNN_ERROR("DeconvolutionTest0 test failed!\n");
                return false;
            }
        }

        MNN_PRINT("beigin testcase 1\n");
        {
            auto input = _Input({1, 3, 2, 2}, NCHW, halide_type_of<float>());

            std::vector<float> data_a = {// channel 0
                                         1.0, 2.0, 4.0, 5.0,
                                         // channel 1
                                         1.1, 2.1, 4.1, 5.1,
                                         // channel 2
                                         1.2, 2.2, 4.2, 5.2};

            std::vector<float> weight = {
                // output channel0
                // input channel0
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                // input channel1
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                // input channel2
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,

                // output channel1
                // input channel0
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                // input channel1
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                // input channel2
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
            };
            std::vector<float> bias   = {1.0, 2.0};
            std::vector<float> data_c = {
                4.3, 10.6, 10.6, 7.3,  16.6, 38.2, 38.2, 22.6, 16.6, 38.2, 38.2, 22.6, 13.3, 28.6, 28.6, 16.3,

                8.6, 21.2, 21.2, 14.6, 33.2, 76.4, 76.4, 45.2, 33.2, 76.4, 76.4, 45.2, 26.6, 57.2, 57.2, 32.6,
            };
            int ic = 3, oc = 2;
            int kw = 4, kh = 4;
            int stride = 2, dilation = 1;
            int group = 1;
            int pad_w = 1, pad_h = 1;

            auto output = _Deconv(std::move(weight), std::move(bias), input, {ic, oc}, {kw, kh}, VALID,
                                  {stride, stride}, {dilation, dilation}, group, {pad_w, pad_h}, false, false);

            ::memcpy(input->writeMap<float>(), data_a.data(), data_a.size() * sizeof(float));

            if (!checkVectorByRelativeError<float>(output->readMap<float>(), data_c.data(), data_c.size(), 0.005)) {
                MNN_ERROR("DeconvolutionTest1 test failed!\n");
                return false;
            }
        }

        MNN_PRINT("beigin testcase 2\n");
        {
            auto input = _Input({1, 3, 2, 2}, NCHW, halide_type_of<float>());

            std::vector<float> data_a = {// channel 0
                                         1.0, 2.0, 4.0, 5.0,
                                         // channel 1
                                         1.1, 2.1, 4.1, 5.1,
                                         // channel 2
                                         1.2, 2.2, 4.2, 5.2};

            std::vector<float> weight = {
                // output channel0
                // input channel0
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                // input channel1
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                // input channel2
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,

                // output channel1
                // input channel0
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                // input channel1
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                // input channel2
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
            };
            std::vector<float> bias   = {0.0, 0.0};
            std::vector<float> data_c = {3.3,  3.3,  9.6,  6.3,  3.3,  3.3,  9.6,  6.3, 15.6, 15.6, 37.2,
                                         21.6, 12.3, 12.3, 27.6, 15.3,

                                         6.6,  6.6,  19.2, 12.6, 6.6,  6.6,  19.2, 12.6, 31.2, 31.2, 74.4,
                                         43.2, 24.6, 24.6, 55.2, 30.6};
            int ic = 3, oc = 2;
            int kw = 3, kh = 3;
            int stride = 2, dilation = 1;
            int group = 1;
            int pad_w = 0, pad_h = 0;

            auto output = _Deconv(std::move(weight), std::move(bias), input, {ic, oc}, {kw, kh}, SAME,
                                  {stride, stride}, {dilation, dilation}, group, {pad_w, pad_h}, false, false);

            ::memcpy(input->writeMap<float>(), data_a.data(), data_a.size() * sizeof(float));

            if (!checkVectorByRelativeError<float>(output->readMap<float>(), data_c.data(), data_c.size(), 0.005)) {
                MNN_ERROR("DeconvolutionTest2 test failed!\n");
                return false;
            }
        }

        return true;
    }
};
MNNTestSuiteRegister(DeconvolutionTest, "op/Deconvolution");
