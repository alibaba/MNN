//
//  nnGradTest.cpp
//  MNN
//
//  Created by MNN on 2019/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "ADAM.hpp"
#include "DemoUnit.hpp"
#include "NN.hpp"
#include "SGD.hpp"
using namespace MNN::Express;
using namespace MNN::Train;
#include <random>
std::random_device gDevice;
class NNGrad : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        MNN_PRINT("Test grad for convolution, pool, concat\n");
        int ic         = 13;
        int oc         = 11;
        int kw         = 3;
        int kh         = 4;
        int iw         = 100;
        int ih         = 120;
        int weightSize = ic * oc * kw * kh;
        std::vector<float> targetVecs(weightSize);
        for (int i = 0; i < weightSize; ++i) {
            auto v        = ((float)(gDevice() % 2000) - 1000.0f) / 1000.0f;
            targetVecs[i] = v;
        }
        auto weightTarget = _Const(targetVecs.data(), {oc, ic, kh, kw}, NCHW);
        std::vector<float> targetVecsBias(oc);
        for (int i = 0; i < oc; ++i) {
            targetVecsBias[i] = ((float)(gDevice() % 2000) - 1000.0f) / 1000.0f;
        }
        auto biasTarget = _Const(targetVecsBias.data(), {oc}, NCHW);

        NN::ConvOption convOption;
        convOption.channel    = {ic, oc};
        convOption.kernelSize = {kw, kh};
        convOption.stride     = {2, 2};
        convOption.dilate     = {1, 2};
        auto convModule       = NN::Conv(convOption);

        std::shared_ptr<SGD> sgd(new SGD);
        sgd->setLearningRate(0.01f);
        sgd->append(convModule->parameters());
        std::vector<float> randomInputs(1 * ic * ih * iw);
        for (int i = 0; i < randomInputs.size(); ++i) {
            randomInputs[i] = ((float)(gDevice() % 2000) - 1000.0f) / 1000.0f;
        }

        for (int i = 0; i < 100; ++i) {
            auto input    = _Input({1, ic, ih, iw}, NCHW);
            auto inputPtr = input->writeMap<float>();
            ::memcpy(inputPtr, randomInputs.data(), randomInputs.size() * sizeof(float));

            auto targetValue  = _Conv(weightTarget, biasTarget, _Convert(input, NC4HW4), convOption.padMode,
                                     convOption.stride, convOption.dilate);
            auto predictValue = convModule->forward(input);

            auto targetValue1  = _MaxPool(targetValue, {2, 2}, {2, 2});
            auto targetValue2  = _AvePool(targetValue, {2, 2}, {2, 2});
            auto predictValue1 = _MaxPool(predictValue, {2, 2}, {2, 2});
            auto predictValue2 = _AvePool(predictValue, {2, 2}, {2, 2});
            targetValue        = _Concat({targetValue1, targetValue2}, 1);
            predictValue       = _Concat({predictValue1, predictValue2}, 1);
            targetValue        = _Convert(targetValue, NCHW);
            predictValue       = _Convert(predictValue, NCHW);
            auto loss          = _ReduceMean(_Square(_Subtract(targetValue, predictValue)), {});
            MNN_PRINT("Loss = %f\n", loss->readMap<float>()[0]);
            sgd->step(loss);
        }
        return 0;
    }
};
class NNGradV2 : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        MNN_PRINT("Test grad for concat, split, transpose\n");
        int ic         = 7;
        int oc         = 7;
        int kw         = 3;
        int kh         = 4;
        int iw         = 100;
        int ih         = 120;
        int weightSize = ic * oc * kw * kh;
        std::vector<float> targetVecs(weightSize);
        for (int i = 0; i < weightSize; ++i) {
            auto v        = ((float)(gDevice() % 2000) - 1000.0f) / 1000.0f;
            targetVecs[i] = v;
        }
        auto weightTarget = _Const(targetVecs.data(), {1, ic, kh, kw}, NCHW);
        std::vector<float> targetVecsBias(oc);
        for (int i = 0; i < oc; ++i) {
            targetVecsBias[i] = ((float)(gDevice() % 2000) - 1000.0f) / 1000.0f;
        }
        auto biasTarget = _Const(targetVecsBias.data(), {oc}, NCHW);

        NN::ConvOption convOption;
        convOption.channel    = {ic, oc};
        convOption.kernelSize = {kw, kh};
        convOption.stride     = {2, 2};
        convOption.dilate     = {1, 2};
        convOption.depthwise  = true;
        auto convModule       = NN::Conv(convOption);

        std::shared_ptr<SGD> sgd(new SGD);
        sgd->setLearningRate(0.1f);
        sgd->append(convModule->parameters());
        sgd->setWeightDecay(0.0f);
        sgd->setMomentum(0.0f);

        std::vector<float> randomInputs(1 * ic * ih * iw);
        for (int i = 0; i < randomInputs.size(); ++i) {
            randomInputs[i] = ((float)(gDevice() % 2000) - 1000.0f) / 1000.0f;
        }

        for (int i = 0; i < 100; ++i) {
            auto input    = _Input({1, ic, ih, iw}, NCHW);
            auto inputPtr = input->writeMap<float>();
            ::memcpy(inputPtr, randomInputs.data(), randomInputs.size() * sizeof(float));
            auto targetValue  = _Conv(weightTarget, biasTarget, _Convert(input, NC4HW4), convOption.padMode,
                                     convOption.stride, convOption.dilate, ic);
            auto predictValue = convModule->forward(input);

            auto targetValue1  = _MaxPool(targetValue, {2, 2}, {2, 2});
            auto targetValue2  = _AvePool(targetValue, {2, 2}, {2, 2});
            auto predictValue1 = _MaxPool(predictValue, {2, 2}, {2, 2});
            auto predictValue2 = _AvePool(predictValue, {2, 2}, {2, 2});
            targetValue        = _Concat({targetValue1, targetValue2}, 1);
            predictValue       = _Concat({predictValue1, predictValue2}, 1);

            auto slicetarget  = _Split(targetValue, {2}, 2);
            auto slicePredict = _Split(predictValue, {2}, 2);
            targetValue       = slicetarget[0];
            predictValue      = slicePredict[0];
            targetValue       = _Convert(targetValue, NCHW);
            targetValue       = _Transpose(targetValue, {1, 3, 2, 0});
            predictValue      = _Convert(predictValue, NCHW);
            predictValue      = _Transpose(predictValue, {1, 3, 2, 0});
            auto loss         = _ReduceMean(_Square(_Subtract(targetValue, predictValue)), {});
            MNN_PRINT("Loss = %f\n", loss->readMap<float>()[0]);
            sgd->step(loss);
        }
        return 0;
    }
};
class NNGradV3 : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        MNN_PRINT("Test grad for Deconvolution(+dw), Resize\n");
        int ic         = 13;
        int oc         = 11;
        int kw         = 3;
        int kh         = 4;
        int iw         = 100;
        int ih         = 120;
        int weightSize = ic * oc * kw * kh;
        std::vector<float> targetVecs(weightSize);
        for (int i = 0; i < weightSize; ++i) {
            auto v        = ((float)(gDevice() % 2000) - 1000.0f) / 1000.0f;
            targetVecs[i] = v;
        }
        auto weightTarget = _Const(targetVecs.data(), {ic, oc, kh, kw}, NCHW);
        std::vector<float> targetVecsBias(oc);
        for (int i = 0; i < oc; ++i) {
            targetVecsBias[i] = ((float)(gDevice() % 2000) - 1000.0f) / 1000.0f;
        }
        auto biasTarget = _Const(targetVecsBias.data(), {oc}, NCHW);

        NN::ConvOption convOption;
        convOption.channel    = {ic, oc};
        convOption.kernelSize = {kw, kh};
        convOption.stride     = {2, 2};
        convOption.dilate     = {1, 2};
        auto convModule       = NN::ConvTranspose(convOption);

        convOption.depthwise = true;
        convOption.channel   = {oc, oc};
        auto convModule2     = NN::ConvTranspose(convOption, false);
        VARP weightTarget2;
        {
            int weightSize = oc * kw * kh;
            std::vector<float> targetVecs(weightSize);
            for (int i = 0; i < weightSize; ++i) {
                auto v        = ((float)(gDevice() % 2000) - 1000.0f) / 1000.0f;
                targetVecs[i] = v;
            }
            weightTarget2 = _Const(targetVecs.data(), {1, oc, kh, kw}, NCHW);
        }

        std::shared_ptr<ADAM> sgd(new ADAM);
        sgd->setLearningRate(0.01f);
        sgd->append(convModule->parameters());
        std::vector<float> randomInputs(1 * ic * ih * iw);
        for (int i = 0; i < randomInputs.size(); ++i) {
            randomInputs[i] = ((float)(gDevice() % 2000) - 1000.0f) / 1000.0f;
        }

        for (int i = 0; i < 1000; ++i) {
            auto input    = _Input({1, ic, ih, iw}, NCHW);
            auto inputPtr = input->writeMap<float>();
            ::memcpy(inputPtr, randomInputs.data(), randomInputs.size() * sizeof(float));

            auto targetValue  = _Deconv(weightTarget, biasTarget, _Convert(input, NC4HW4), convOption.padMode,
                                       convOption.stride, convOption.dilate);
            auto predictValue = convModule->forward(input);
            targetValue       = _Deconv(weightTarget2, nullptr, targetValue, convOption.padMode, convOption.stride,
                                  convOption.dilate, oc);
            predictValue      = convModule2->forward(predictValue);

            auto targetValue1  = _MaxPool(targetValue, {2, 2}, {2, 2});
            auto targetValue2  = _AvePool(targetValue, {2, 2}, {2, 2});
            auto predictValue1 = _MaxPool(predictValue, {2, 2}, {2, 2});
            auto predictValue2 = _AvePool(predictValue, {2, 2}, {2, 2});
            targetValue        = _Concat({targetValue1, targetValue2}, 1);
            predictValue       = _Concat({predictValue1, predictValue2}, 1);
            targetValue        = _Resize(targetValue, 0.5f, 0.5f);
            predictValue       = _Resize(predictValue, 0.5f, 0.5f);

            targetValue  = _Convert(targetValue, NCHW);
            predictValue = _Convert(predictValue, NCHW);
            auto loss    = _ReduceMean(_Square(_Subtract(targetValue, predictValue)), {});
            MNN_PRINT("Loss = %f\n", loss->readMap<float>()[0]);
            sgd->step(loss);
        }
        return 0;
    }
};
class MatMulGradTest : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        MNN_PRINT("Test grad for MatMul, BatchMatMul\n");
        {
            int e          = 13;
            int l          = 11;
            int h          = 30;
            int weightSize = l * h;
            std::vector<float> targetVecs(weightSize);
            for (int i = 0; i < weightSize; ++i) {
                auto v        = ((float)(gDevice() % 2000) - 1000.0f) / 1000.0f;
                targetVecs[i] = v;
            }
            auto weightTarget = _Const(targetVecs.data(), {l, h}, NCHW);
            auto weightOrigin = _Const(0.0f, {l, h}, NCHW);
            std::shared_ptr<SGD> sgd(new SGD);
            sgd->setLearningRate(0.01f);
            sgd->append({weightOrigin});
            std::vector<float> randomInputs(e * l);
            for (int i = 0; i < randomInputs.size(); ++i) {
                randomInputs[i] = ((float)(gDevice() % 2000) - 1000.0f) / 1000.0f;
            }

            for (int i = 0; i < 1000; ++i) {
                auto input    = _Input({e, l}, NCHW);
                auto inputPtr = input->writeMap<float>();
                ::memcpy(inputPtr, randomInputs.data(), randomInputs.size() * sizeof(float));

                auto targetValue  = _MatMul(input, weightTarget);
                auto predictValue = _MatMul(input, weightOrigin);
                auto loss         = _ReduceMean(_Square(_Subtract(targetValue, predictValue)), {});
                if (i % 100 == 0) {
                    MNN_PRINT("Loss = %f\n", loss->readMap<float>()[0]);
                }
                sgd->step(loss);
            }
        }
        MNN_PRINT("Test for BatchMatMul\n");
        {
            int e          = 13;
            int l          = 11;
            int h          = 30;
            int b          = 5;
            int weightSize = b * l * h;
            std::vector<float> targetVecs(weightSize);
            for (int i = 0; i < weightSize; ++i) {
                auto v        = ((float)(gDevice() % 2000) - 1000.0f) / 1000.0f;
                targetVecs[i] = v;
            }
            auto weightTarget = _Const(targetVecs.data(), {b, l, h}, NCHW);
            auto weightOrigin = _Const(0.0f, {b, l, h}, NCHW);
            std::shared_ptr<ADAM> sgd(new ADAM);
            sgd->setLearningRate(0.01f);
            sgd->append({weightOrigin});
            std::vector<float> randomInputs(b * e * l);
            for (int i = 0; i < randomInputs.size(); ++i) {
                randomInputs[i] = ((float)(gDevice() % 2000) - 1000.0f) / 1000.0f;
            }

            for (int i = 0; i < 10000; ++i) {
                auto input    = _Input({b, e, l}, NCHW);
                auto inputPtr = input->writeMap<float>();
                ::memcpy(inputPtr, randomInputs.data(), randomInputs.size() * sizeof(float));

                auto targetValue  = _BatchMatMul(input, weightTarget);
                auto predictValue = _BatchMatMul(input, weightOrigin);
                auto loss         = _ReduceMean(_Square(_Subtract(targetValue, predictValue)), {});
                if (i % 1000 == 0) {
                    MNN_PRINT("Loss = %f\n", loss->readMap<float>()[0]);
                }
                sgd->step(loss);
            }
        }
        return 0;
    }
};

DemoUnitSetRegister(NNGrad, "NNGrad");
DemoUnitSetRegister(NNGradV2, "NNGradV2");
DemoUnitSetRegister(NNGradV3, "NNGradV3");
DemoUnitSetRegister(MatMulGradTest, "MatMulGradTest");
