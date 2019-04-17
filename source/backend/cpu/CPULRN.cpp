//
//  CPULRN.cpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPULRN.hpp"
#include <math.h>
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Concurrency.h"
#include "Macro.h"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

CPULRN::CPULRN(Backend *backend, int regionType, int localSize, float alpha, float beta)
    : Execution(backend), mRegionType(regionType), mLocalSize(localSize), mAlpha(alpha), mBeta(beta) {
    // nothing to do
}

ErrorCode CPULRN::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // input transform space
    auto &input = inputs[0]->buffer();
    memcpy(mInput.buffer().dim, input.dim, sizeof(halide_dimension_t) * input.dimensions);
    backend()->onAcquireBuffer(&mInput, Backend::DYNAMIC);

    // output transform space
    auto &output = outputs[0]->buffer();
    memcpy(mOutput.buffer().dim, output.dim, sizeof(halide_dimension_t) * output.dimensions);
    backend()->onAcquireBuffer(&mOutput, Backend::DYNAMIC);

    // square space
    memcpy(mSquare.buffer().dim, input.dim, sizeof(halide_dimension_t) * input.dimensions);
    if (mRegionType == 1 && mLocalSize > 1) {
        mSquare.buffer().dim[3].extent += mLocalSize - 1;
        mSquare.buffer().dim[2].extent += mLocalSize - 1;
    }
    backend()->onAcquireBuffer(&mSquare, Backend::DYNAMIC);

    // release temp buffer space
    backend()->onReleaseBuffer(&mInput, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mOutput, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mSquare, Backend::DYNAMIC);
    return NO_ERROR;
}

void CPULRN::executeAcrossChannels() {
    const auto size     = mInput.width() * mInput.height();
    const auto channels = mInput.channel();

    // calc pow
    MNN_CONCURRENCY_BEGIN(c, channels) {
        auto inChannel   = mInput.host<float>() + c * size;
        auto sqrtChannel = mSquare.host<float>() + c * size;
        int i            = 0;
#ifdef MNN_USE_NEON
        for (; i + 3 < size; i += 4) {
            float32x4_t v4 = vld1q_f32(inChannel + i);
            vst1q_f32(sqrtChannel + i, v4 * v4);
        }
#endif
        for (; i < size; i++) {
            float v        = inChannel[i];
            sqrtChannel[i] = v * v;
        }
    }
    MNN_CONCURRENCY_END()

    // clear output
    memset(mOutput.host<float>(), 0, size * channels * sizeof(float));
    auto outFactor = mAlpha / mLocalSize;

    // calc output
    MNN_CONCURRENCY_BEGIN(c, channels) {
        auto outChannel   = mOutput.host<float>() + (int)c * size;
        auto inChannel    = mInput.host<float>() + (int)c * size;
        auto startChanenl = std::max((int)c - mLocalSize / 2, 0);
        auto endChannel   = std::min((int)c + mLocalSize / 2, channels - 1);

        for (int lc = startChanenl; lc <= endChannel; lc++) {
            auto sqrtChannel = mSquare.host<float>() + lc * size;
            int i            = 0;
#ifdef MNN_USE_NEON
            for (; i + 3 < size; i += 4) {
                vst1q_f32(outChannel + i, vld1q_f32(outChannel + i) + vld1q_f32(sqrtChannel + i));
            }
#endif
            for (; i < size; i++) {
                outChannel[i] += sqrtChannel[i];
            }
        }

#pragma clang loop vectorize(enable)
        for (int i = 0; i < size; i++) {
            outChannel[i] = inChannel[i] * pow(1.f + outFactor * outChannel[i], -mBeta);
        }
    }
    MNN_CONCURRENCY_END()
}

void CPULRN::executeWithInChannels() {
    const auto width    = mInput.width();
    const auto height   = mInput.height();
    const auto channels = mInput.channel();
    const auto size     = width * height;

    // calc padding
    int padF     = mLocalSize / 2;
    int padWidth = width, padHeight = height;
    if (padF > 0) {
        padWidth += mLocalSize - 1; // front = mLocalSize / 2, behind = mLocalSize - front - 1
        padHeight += mLocalSize - 1;
    }
    int padSize = padWidth * padHeight;

    // calc pow
    MNN_CONCURRENCY_BEGIN(c, channels) {
        auto inPtr   = mInput.host<float>() + c * size;
        auto sqrtPtr = mSquare.host<float>() + c * padSize + padF * padWidth + padF;
        for (int h = 0; h < height; h++, inPtr += width, sqrtPtr += padWidth) {
            for (int w = 0; w < width; w++) {
                float v    = inPtr[w];
                sqrtPtr[w] = v * v;
            }
        }
    }
    MNN_CONCURRENCY_END()

    // norm window offsets
    auto area    = mLocalSize * mLocalSize;
    auto mapping = (int *)calloc(area, sizeof(int));
    {
        int inIndex   = 0;
        int sqrtIndex = 0;
        int gap       = padWidth - mLocalSize;
        for (int i = 0; i < mLocalSize; i++) {
            for (int j = 0; j < mLocalSize; j++) {
                mapping[inIndex++] = sqrtIndex++;
            }
            sqrtIndex += gap;
        }
    }

    // calc output
    auto outFactor = mAlpha / area;
    MNN_CONCURRENCY_BEGIN(c, channels) {
        auto inChannel   = mInput.host<float>() + c * size;
        auto outChannel  = mOutput.host<float>() + c * size;
        auto sqrtChannel = mSquare.host<float>() + c * padSize;

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                float sum = 0.f;
                for (int k = 0; k < area; k++) {
                    sum += sqrtChannel[mapping[k] + w];
                }
                outChannel[w] = inChannel[w] * pow(1.f + outFactor * sum, -mBeta);
            }
            inChannel += width;
            outChannel += width;
            sqrtChannel += padWidth;
        }
    }
    MNN_CONCURRENCY_END()

    free(mapping);
}

ErrorCode CPULRN::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto &input  = inputs[0];
    auto &output = outputs[0];

    // input transform
    MNNUnpackC4(mInput.host<float>(), input->host<float>(), input->width() * input->height(), input->channel());
    // clear square
    memset(mSquare.host<float>(), 0, mSquare.size());

    if (mRegionType == 0) {
        executeAcrossChannels();
    } else if (mRegionType == 1) {
        executeWithInChannels();
    } else {
        // not supported
    }

    // output transform
    MNNPackC4(output->host<float>(), mOutput.host<float>(), output->width() * output->height(), output->channel());
    return NO_ERROR;
}

class CPULRNCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto lrn = op->main_as_LRN();
        return new CPULRN(backend, lrn->regionType(), lrn->localSize(), lrn->alpha(), lrn->beta());
    }
};
REGISTER_CPU_OP_CREATOR(CPULRNCreator, OpType_LRN);

} // namespace MNN
