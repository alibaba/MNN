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
#include "CommonOpCreator.hpp"
using namespace MNN::Express;
using namespace MNN;

class HybridConvSpeedTestCommon : public MNNTestCase {
protected:
    static bool testKernel(std::string title, INTS inputShape, INTS kernel, INTS channel, INTS pad, INTS strides, INTS dilate, int batch = 1, int nbit = 8, int precision = 1, bool testSpeed = false, int blocksize = 0) {
        float fac = 0.23;
        int res = 10;
        float tail = 0.05;
        int ic = channel[0], oc = channel[1];
        int iw = inputShape[0], ih = inputShape[1];
        std::vector<float> bias(oc), biastest(oc), biasdup(oc);
        int area = kernel[0] * kernel[1];
        int blocknum = 1;
        if (0 == blocksize || ic % blocksize != 0) {
            blocksize = ic;
            blocknum = 1;
        } else {
            blocknum = ic / blocksize;
        }
        
        std::vector<float> weightFp32(oc * ic * area);
        std::vector<float> wScale(2 * oc * blocknum);

        float threshold = (float)(1 << (nbit - 1)) - 1.0f;
        float clampMin = -threshold - 1;
        VARP x = _Input({batch, ic, ih, iw}, NCHW, halide_type_of<float>());
        auto xInfo = x->getInfo();
        auto xPtr = x->writeMap<float>();
        int8_t xMin = -(1<<(nbit-1)), xMax = (1<<(nbit-1))-1;
        for (int i=0; i<xInfo->size; ++i) {
            xPtr[i] = (i % (xMax - xMin + 1) - (xMax / 2)) * 0.017;
        }
        x = _Convert(x, NC4HW4);
        for (int i = 0; i < oc; ++i) {
            bias[i] = i % 10 + 0.005;
            for (int j = 0; j < ic; ++j) {
                for (int k = 0; k < area; k++) {
                    weightFp32[(i * ic + j) * area + k] = ((i * ic + j) * area + k) % res * fac + tail;
                }
            }
        }
        ::memcpy(biastest.data(), bias.data(), oc * sizeof(float));
        ::memcpy(biasdup.data(), bias.data(), oc * sizeof(float));
        int kernel_size = ic * area;
        auto newWeightFp32 = weightFp32;
        for (int k = 0; k < oc; ++k) {
            int beginIndex = k * kernel_size;
            for (int j = 0; j < blocknum; ++j) {
                auto index = k * blocknum + j;
                auto minmax = findMinMax(weightFp32.data() + k * ic * area + j * blocksize * area, blocksize * area);
                auto scale_ = (minmax.second - minmax.first) / (threshold - clampMin);
                wScale[2 * index] = minmax.first;
                wScale[2 * index + 1] = scale_;
                for (int u = 0; u < blocksize; ++u) {
                    for (int i = 0; i < area; ++i) {
                        int idx = k * ic * area + j * blocksize * area + u * area + i;
                        int q_weight = (weightFp32[idx] - minmax.first) * (threshold - clampMin) / (minmax.second - minmax.first) + clampMin;
                        newWeightFp32[idx] = (q_weight - xMin) * scale_ + minmax.first;
                    }
                }
            }
        }
        auto y     = _HybridConv(weightFp32, std::move(bias), std::move(wScale), x, channel, kernel, PaddingMode::CAFFE, strides, dilate, 1, pad, false, false, nbit, true);
        auto yfp32 = _Conv(std::move(newWeightFp32), std::move(biasdup), x, {ic, oc}, kernel, PaddingMode::CAFFE, strides, dilate, 1, pad);
        auto yInfo = y->getInfo();
        auto ow = yInfo->dim[3], oh = yInfo->dim[2];
#if defined (__aarch64__) && (precision == 2)
#define FLOAT_T __fp16
#else
#define FLOAT_T float
#endif
        y = _Convert(y, NCHW);
        yfp32 = _Convert(yfp32, NCHW);
        auto yPtr  = y->readMap<FLOAT_T>();
        auto tgPtr = yfp32->readMap<FLOAT_T>();
        auto elesize = yfp32->getInfo()->size;
        float limit = 0.1f;
        bool correct = true;
        float maxValue = 0.001f;
        for (int i = 0; i < elesize; ++i) {
            maxValue = fmaxf(maxValue, fabsf(tgPtr[i]));
        }

        for (int i = 0; i < elesize; ++i) {
            float targetValue = tgPtr[i], computeResult = yPtr[i];
            float diff = targetValue - computeResult;
            float ratio = fabsf(diff) / maxValue;
            if (ratio > limit) {
                MNN_PRINT("%d result Error ratio=%f: right=%f, error=%f\n", i, ratio, targetValue, computeResult);
                MNN_PRINT("conv info: input=(%dx%dx%dx%d) output=(%dx%dx%dx%d)\n", batch, ic, ih, iw, batch, oc, oh, ow);
                correct = false;
                break;
            }
        }
        if (testSpeed) {
            x.fix(VARP::INPUT);
            const int LOOP = 20;
            {
                x->writeMap<FLOAT_T>();
                y->readMap<FLOAT_T>();
            }
            MNN::Timer _t;
            for (int i = 0; i < LOOP; ++i) {
                x->writeMap<FLOAT_T>();
                y->readMap<FLOAT_T>();
            }
            auto time = (float)_t.durationInUs() / 1000.0f;
            MNN_PRINT("%s input=(%dx%dx%dx%d) output=(%dx%dx%dx%d) avg time = %f\n",
                      title.c_str(), batch, ic, ih, iw, batch, oc, oh, ow, 1.0 * time / LOOP);
        }
        return correct;
    }
};

class HybridConvSpeedInt8Test : public HybridConvSpeedTestCommon {
public:
    virtual bool run(int precision) {
        INTS strides = {1, 1}, dilate = {1, 1};
        int batch[] = {1, 512};
        std::vector<int> blocks = {0, 128};
        std::vector<std::vector<int>> channels = { {1536, 2048}, {2048, 2048}, {1536, 1536}};

        std::vector<std::vector<int>> kernels = {{1, 1}};
        std::vector<std::vector<int>> pads = {{0, 0}};
        std::vector<std::vector<int>> Shapes = {{1, 1}};
        std::vector<int> weightBits = {4, 8};
        int batchNum = sizeof(batch) / sizeof(int);
        bool correct = true;
        for (auto& bits : weightBits) {
            for (auto &channel: channels) {
                for (auto &kernel: kernels) {
                    for (auto &pad: pads) {
                        for (auto &inputShape: Shapes) {
                            for (auto block : blocks) {
                                MNN_PRINT("Test for %d bits, channel{%d,%d}, kernel={%d,%d}, pad={%d,%d}, block=%d\n", bits, channel[0], channel[1], kernel[0], kernel[1], pad[0], pad[1], block);
                                for (int n = 0; n < batchNum; ++n) {
                                    if (dilate[0] > inputShape[0] || dilate[0] * (kernel[0] - 1) + 1 > inputShape[0] || dilate[0] * (kernel[1] - 1) + 1 > inputShape[1])
                                        continue;
                                    auto res = testKernel("Low memory HybridConv test:", inputShape, kernel, channel, pad, strides, dilate, batch[n], bits, precision, true, block);
                                    if (!res) {
                                        MNN_ERROR("Error: low memory hybridConv when bits=%d, n=%d, ic=%d, oc=%d, block=%d, pad={%d,%d}, kernel={%d,%d}\n", bits, batch[n], channel[0], channel[1], block, pad[0], pad[1], kernel[0], kernel[1]);
                                        correct = false;
                                        return false;
                                    }
                                }
                            } //
                        }
                    }
                }
            }
        }
        return correct;
    }
};

class ConvInt8BlockQuantTest : public HybridConvSpeedTestCommon {
public:
    virtual bool run(int precision) {
        INTS strides = {1, 1}, dilate = {1, 1}, pad = {0, 0}, inputShape = {1, 17}; // {w, h}
        int batch[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
        std::vector<int> blocks = {0, 64, 32};
        std::vector<std::vector<int>> channels = {{320, 320}, {640, 200}, {128, 79}};

        std::vector<int> kernels = {1, 3};
        std::vector<int> weightBits = {4, 8};
        int batchNum = sizeof(batch) / sizeof(int);
        bool correct = true;
        for (auto& bits : weightBits) {
            for (auto &channel: channels) {
                for (auto block : blocks) {
                    for (int n = 0; n < batchNum; ++n) {
                        auto res = testKernel("Low memory HybridConv test:", inputShape, kernels, channel, pad, strides, dilate, batch[n], bits, precision, false, block);
                        if (!res) {
                            MNN_ERROR("Error: low memory hybridConv when bits=%d, n=%d, block=%d, ic=%d, oc=%d\n", bits, batch[n], block, channel[0], channel[1]);
                            correct = false;
                            return false;
                        }
                    }
                }
            }
        }
        return correct;
    }
};

class HybridConvInt8Test : public HybridConvSpeedTestCommon {
public:
    virtual bool run(int precision) {
        INTS strides = {1, 1}, dilate = {1, 1}, pad = {0, 0}; // {w, h}
        int batch[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 25, 26, 27, 28, 29, 30};
        std::vector<int> blocks = {0, 32, 128};
        std::vector<std::vector<int>> channels = {{3, 7}, {4, 18}, {5, 22}, {12, 16}, {8, 8}, {8, 9}, {8, 16}, {7, 20}, {9, 24}, {2048, 54}, {1, 10}, {20, 153}, {9, 18}, {64, 28}, {1496, 11}, {10, 9}};
        std::vector<std::vector<int>> inputShapes = {{1, 1}};
        std::vector<std::vector<int>> kernels = {{1, 1}};
        std::vector<int> weightBits = {4, 8};
        int batchNum = sizeof(batch) / sizeof(int);
        bool correct = true;
        for (auto kernel: kernels) {
            for (auto inputShape: inputShapes) {
                for (auto block : blocks) {
                    for (auto& bits : weightBits) {
                        for (auto &channel: channels) {
                            if (dilate[0] > inputShape[0] || dilate[0] * (kernel[0] - 1) + 1 > inputShape[0] || dilate[0] * (kernel[1] - 1) + 1 > inputShape[1])
                                continue;
                            if (block > 0 && channel[0] % block != 0)
                                continue;
                            for (int n = 0; n < batchNum; ++n) {
                                auto res = testKernel("Low memory HybridConv test:", inputShape, kernel, channel, pad, strides, dilate, batch[n], bits, precision, false, block);
                                if (!res) {
                                    MNN_ERROR("Error: low memory hybridConv when bits=%d, n=%d, ic=%d, oc=%d, block=%d\n", bits, batch[n], channel[0], channel[1], block);
                                    return false;
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

class DenseConvInt8Test : public HybridConvSpeedTestCommon {
public:
    virtual bool run(int precision) {
        std::vector< std::vector<int>> channels = {{4, 17}, {8, 256}, {5, 8}, {3, 17}, {7, 26}, {9, 26}, {1, 8}, {7, 9}, {256, 256}, {1024, 2048}};
        INTS strides = {1, 1}, dilate = {1, 3}, pad = {0, 3}, inputShape = {1, 11}; // {w, h}
        std::vector<int> batch = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 21, 22, 25, 28};
        std::vector<std::vector<int>> kernels = {{1, 1}, {1, 3}};
        std::vector<int> weightBits = {4, 8};
        std::vector<int> blocks = {0, 32};
        bool lowmemory = true;
        int n = 0;
        for (auto& bits : weightBits) {
            for (int n = 0; n < batch.size(); ++n) {
                for (int i = 0; i < channels.size(); ++i) {
                    for (auto kernel : kernels) {
                        for (auto block : blocks) {
                            if (block > 0 && channels[i][0] % block != 0) {
                                continue;
                            }
                            if (dilate[0] > inputShape[0] || dilate[0] * (kernel[0] - 1) + 1 > inputShape[0] || dilate[0] * (kernel[1] - 1) + 1 > inputShape[1])
                                continue;
                            auto res = testKernel("Low memory ConvInt8 with kernel test:", inputShape, kernel, channels[i], pad, strides, dilate, batch[n], bits, precision, false, block);
                            if (!res) {
                                MNN_ERROR("Error: low memory ConvInt8 with %dx%d kernel when bits=%d, n=%d, ic=%d, oc=%d, block=%d\n", kernel[0], kernel[1], bits, batch[n], channels[i][0], channels[i][1], block);
                                return false;
                            }
                        }
                    }
                }
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(DenseConvInt8Test, "op/lowMemory/DenseConv");
MNNTestSuiteRegister(HybridConvInt8Test, "op/lowMemory/HybridConv");
MNNTestSuiteRegister(HybridConvSpeedInt8Test, "speed/HybridConv");
MNNTestSuiteRegister(ConvInt8BlockQuantTest, "op/lowMemory/blockConv");
