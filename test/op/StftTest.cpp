//
//  StftTest.cpp
//  MNNTests
//
//  Created by MNN on 2024/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_BUILD_AUDIO
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class StftTest : public MNNTestCase {
public:
    virtual ~StftTest() = default;
    virtual bool run(int precision) {
        /*
        python:
            import torch
            freq = 5, sample_rate = 100, duration = 0.2
            t = torch.arange(0, duration, 1.0 / sample_rate)
            sine_wave = torch.sin(2 * torch.pi * freq * t)
            n_fft = 8, hop_length = 4, win_length = 8
            window = torch.hann_window(win_length)
            stft_result = torch.stft(sine_wave, n_fft=n_fft, hop_length=hop_length,
                                     win_length=win_length, window=window, center=False)
            magnitude = torch.abs(stft_result).transpose(1, 0)
        */
        auto signal = _Input({ 20 }, NCHW);
        auto window = _Input({  8 }, NCHW);
        signal->setName("signal");
        window->setName("window");
        const float signalData[] = {
            0.000, 0.309,  0.588,  0.809,  0.951,  1.000,  0.951,  0.809,  0.588,  0.309,
            0.000, -0.309, -0.588, -0.809, -0.951, -1.000, -0.951, -0.809, -0.588, -0.309
        };
        const float windowData[] = { 0.000, 0.146, 0.500, 0.854, 1.000, 0.854, 0.500, 0.146 };
        auto signalPtr           = signal->writeMap<float>();
        auto windowPtr           = window->writeMap<float>();
        memcpy(signalPtr, signalData, 20 * sizeof(float));
        memcpy(windowPtr, windowData,  8 * sizeof(float));
        auto output                  = _Stft(signal, window, 8, 4);
        const float expectedOutput[] = {
            3.428, 1.958, 0.203, 0.029, 0.013, 2.119, 1.501, 0.261, 0.041, 0.008,
            2.119, 1.501, 0.261, 0.041, 0.008, 3.428, 1.958, 0.203, 0.029, 0.013
        };
        auto gotOutput = output->readMap<float>();
        for (int i = 0; i < 20; ++i) {
            auto diff = ::fabsf(gotOutput[i] - expectedOutput[i]);
            if (diff > 0.01) {
                MNN_ERROR("StftTest test failed: %f - %f!\n", expectedOutput[i], gotOutput[i]);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(StftTest, "op/stft");
#endif // MNN_BUILD_AUDIO