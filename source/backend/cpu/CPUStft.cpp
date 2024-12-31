//
//  CPUStft.cpp
//  MNN
//
//  Created by MNN on 2024/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_BUILD_AUDIO
#ifndef M_PI
#define M_PI 3.141592654
#endif
#include <algorithm>
#include <cmath>
#include "backend/cpu/CPUStft.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include "core/Macro.h"
#include "compute/CommonOptFunction.h"

namespace MNN {
static void MNNDftAbs(const float* input, const float* window, float* output, float* buffer, int nfft) {
    for (int i = 0; i < nfft; ++i) {
        buffer[i] = input[i] * window[i];
    }
    for (int k = 0; k < nfft / 2 + 1; ++k) {
        float real_sum = 0.f, imag_sum = 0.f;
        for (int n = 0; n < nfft; ++n) {
            float angle = 2 * M_PI * k * n / nfft;
            real_sum += buffer[n] * cosf(angle);
            imag_sum -= buffer[n] * sinf(angle);
        }
        output[k] = sqrtf(real_sum * real_sum + imag_sum * imag_sum);
    }
}


CPUStft::CPUStft(Backend* backend, int nfft, int hop_length, bool abs)
    : Execution(backend), mNfft(nfft), mHopLength(hop_length), mAbs(abs) {
    // nothing to do
}

ErrorCode CPUStft::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto cpuBn = static_cast<CPUBackend*>(backend());
    mTmpFrames.buffer().dim[0].extent = cpuBn->threadNumber();
    mTmpFrames.buffer().dim[1].extent = mNfft;
    TensorUtils::getDescribe(&mTmpFrames)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
    mTmpFrames.buffer().dimensions    = 2;
    mTmpFrames.buffer().type          = inputs[0]->getType();
    backend()->onAcquireBuffer(&mTmpFrames, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mTmpFrames, Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUStft::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const float* sample = inputs[0]->host<float>();
    const float* window = inputs[1]->host<float>();
    float* buffer = mTmpFrames.host<float>();
    float* output = outputs[0]->host<float>();
    auto outputShape = outputs[0]->shape();
    int frames = outputShape[0];
    int col = outputShape[1];
    auto cpuBn = static_cast<CPUBackend*>(backend());
    int threadNum = cpuBn->threadNumber();
    // div frames to threadNum
    int threadNumber = std::min(threadNum, frames);
    int sizeDivide = frames / threadNumber;
    MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
        int number = sizeDivide;
        if (tId == threadNumber - 1) {
            number = frames - tId * sizeDivide;
        }
        for (int i = tId * sizeDivide; i < tId * sizeDivide + number; ++i) {
            MNNDftAbs(sample + i * mHopLength, window, output + i * col, buffer + tId * mNfft, mNfft);
        }
    };
    MNN_CONCURRENCY_END();

    return NO_ERROR;
}

class CPUStftCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        auto stft = op->main_as_StftParam();
        return new CPUStft(backend, stft->n_fft(), stft->hop_length(), stft->abs());
    }
};

REGISTER_CPU_OP_CREATOR_AUDIO(CPUStftCreator, OpType_Stft);
} // namespace MNN
#endif // MNN_BUILD_AUDIO
