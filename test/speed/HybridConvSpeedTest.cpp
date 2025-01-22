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
    static bool testKernel(std::string title, INTS inputShape, INTS kernel, INTS channel, INTS pad, INTS strides, INTS dilate, int batch = 1, int nbit = 8, int precision = 1, bool testSpeed = false, int block = 0) {
        float fac = 1.23;
        int res = 10;
        float tail = 0.2;
        int ic = channel[0], oc = channel[1];
        int iw = inputShape[0], ih = inputShape[1];
        std::vector<float> bias(oc), biastest(oc), biasdup(oc);
        int area = kernel[0] * kernel[1];
        if (0 == block || ic % block != 0 || area > 1) {
            block = area * ic;
        }
        int group = (area * ic) / block;
        std::vector<float> weightFp32(oc * ic * area);
        std::vector<float> wScale(oc * group);

        float threshold = (float)(1 << (nbit - 1)) - 1.0f;
        float clampMin = -threshold;
        VARP x = _Input({batch, ic, ih, iw}, NCHW, halide_type_of<float>());
        auto xInfo = x->getInfo();
        auto xPtr = x->writeMap<float>();
        int8_t xMin = -(1<<(nbit-1))+1, xMax = (1<<(nbit-1))-1;
        for (int i=0; i<xInfo->size; ++i) {
            xPtr[i] = (i % (xMax - xMin + 1) - (xMax / 2)) * 0.27;
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
            for (int v=0; v<group; ++v) {
                auto absMax = findAbsMax(weightFp32.data() + beginIndex + v*block, block);
                float scale = absMax / threshold;
                wScale[k*group+v] = scale;
                for (int u=0; u<block; ++u) {
                    auto index = beginIndex+v*block+u;
                    auto value = (int)round(weightFp32[index] / scale);
                    newWeightFp32[index] = value * scale;
                }
            }
        }
        auto y     = _HybridConv(weightFp32, std::move(bias), std::move(wScale), x,
                           channel, kernel, PaddingMode::CAFFE, strides, dilate, 1, pad, false, false, nbit, false);
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
                correct = false;
                break;
            }
        }
        if (testSpeed) {
            x.fix(VARP::INPUT);
            MNN::Timer _t;
            const int LOOP = 20;
            for (int i = 0; i < LOOP; ++i) {
                x->writeMap<FLOAT_T>();
                y->readMap<FLOAT_T>();
            }
            auto time = (float)_t.durationInUs() / 1000.0f;
            MNN_PRINT("%s input=(%dx%dx%dx%d) output=(%dx%dx%dx%d) avg time = %f\n",
                      title.c_str(), batch, ic, 1, 1, batch, oc, 1, 1, 1.0 * time / LOOP);
        }
        return correct;
    }
};

class HybridConvSpeedInt8Test : public HybridConvSpeedTestCommon {
public:
    virtual bool run(int precision) {
        INTS strides = {1, 1}, dilate = {1, 1}, pad = {0, 0}, inputShape = {1, 1}; // {w, h}
        INTS channel0 = {4096, 4096}; // {ic, co}
        INTS channel1 = {1496, 256};
        int batch[3] = {23, 13, 1};
        std::vector<int> blocks = {32, 128, 0};

        std::vector<int> kernels = {1, 1};
        std::vector<int> weightBits = {4, 8};
        bool lowmemory = true;
        int batchNum = sizeof(batch) / sizeof(int);
        bool correct = true;
        for (auto block : blocks) {
            for (auto& bits : weightBits) {
                MNN_PRINT("Test for %d bits, block=%d\n", bits, block);
                for (int n = 0; n < batchNum; ++n) {
                    auto res = testKernel("Low memory HybridConv test:", inputShape, kernels, channel0, pad, strides, dilate, batch[n], bits, precision, true, block);
                    if (!res) {
                        MNN_ERROR("Error: low memory hybridConv when n=%d, ic=%d, oc=%d\n", batch[n], channel0[0], channel0[1]);
                        correct = false;
                    }
                }
                for (int n = 0; n < batchNum; ++n) {
                    auto res = testKernel("Low memory HybridConv test:", inputShape, kernels, channel1, pad, strides, dilate, batch[n], bits, precision, true, block);
                    if (!res) {
                        MNN_ERROR("Error: low memory hybridConv when n=%d, ic=%d, oc=%d\n", batch[n], channel1[0], channel1[1]);
                        correct = false;
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
        
        INTS strides = {1, 1}, dilate = {1, 1}, pad = {0, 0}, inputShape = {1, 1}; // {w, h}
        int testBatchCount = 5;
        // std::vector<int> batch(testBatchCount);
        std::vector<int> batch = {1, 23, 149, 38, 29};
        std::vector<int> kernels = {1, 1};
        bool lowmemory = true;
        {
           std::vector< std::vector<int>> channels = {{7, 9}, {9, 9}, {2048, 54}, {1, 10}, {20, 153}, {9, 18}};
           for (int i = 0; i < channels.size(); ++i) {
               for (int n = 0; n < batch.size(); ++n) {
                   auto res = testKernel("Low memory HybridConv test:", inputShape, kernels, channels[i], pad, strides, dilate, batch[n], 8, precision);
                   res &= testKernel("Low memory HybridConv test:", inputShape, kernels, channels[i], pad, strides, dilate, batch[n], 4, precision);
                   if (!res) {
                       MNN_ERROR("Error: low memory hybridConv when bits=8, n=%d, ic=%d, oc=%d\n", batch[n], channels[i][0], channels[i][1]);
                       return false;
                   }
               }
           }
        }
        {
            std::vector< std::vector<int>> channels = {{12, 16}, {2048, 54}, {8, 8}, {8, 9}, {8, 16}};
            for (int i = 0; i < channels.size(); ++i) {
                for (int n = 0; n < batch.size(); ++n) {
                    auto res = testKernel("Low memory HybridConv test:", inputShape, kernels, channels[i], pad, strides, dilate, batch[n], 4, precision);
                    if (!res) {
                        MNN_ERROR("Error: low memory hybridConv when bits=4, n=%d, ic=%d, oc=%d\n", batch[n], channels[i][0], channels[i][1]);
                        return false;
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
        std::vector< std::vector<int>> channels = {{4, 256}, {512, 128}, {1, 8}, {7, 9}};
        INTS strides = {1, 1}, dilate = {1, 3}, pad = {0, 3}, inputShape = {1, 131}; // {w, h}
        std::vector<int> batch = {1, 13};
        std::vector<std::vector<int>> kernels = {{1, 1}, {1, 3}};
        std::vector<int> weightBits = {4, 8};
        bool lowmemory = true;
        int n = 0;
        for (auto& bits : weightBits) {
            for (int n = 0; n < batch.size(); ++n) {
                for (int i = 0; i < channels.size(); ++i) {
                    for (auto kernel : kernels) {
                        std::vector<int> blocks = {0};
                        if (kernel[0] == 1 && kernel[1] == 1) {
                            blocks = {0, 32};
                        }
                        for (auto block : blocks) {
                            auto res = testKernel("Low memory ConvInt8 with kernel test:", inputShape, kernel, channels[i], pad, strides, dilate, batch[n], bits, precision, false, block);
                            if (!res) {
                                MNN_ERROR("Error: low memory ConvInt8 with %dx%d kernel when bits=%d n=%d, ic=%d, oc=%d, block=%d\n", kernel[0], kernel[1], bits, batch[n], channels[i][0], channels[i][1], block);
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
