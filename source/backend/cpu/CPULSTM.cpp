//
//  CPULSTM.cpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPULSTM.hpp"
#include <math.h>
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Concurrency.h"
#include "Macro.h"
#include "TensorUtils.hpp"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

static inline float sigmoid(float x) {
    return 1. / (1. + expf(-x));
}

CPULSTM::CPULSTM(Backend *backend, const LSTM *LSTM) : Execution(backend), mLSTM(LSTM) {
    // nothing to do
}

CPULSTM::~CPULSTM() {
    if (mInit) {
        backend()->onReleaseBuffer(mWeightH.get(), Backend::STATIC);
        backend()->onReleaseBuffer(mWeightI.get(), Backend::STATIC);
        backend()->onReleaseBuffer(mBiasC.get(), Backend::STATIC);
    }
}

ErrorCode CPULSTM::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // input transform space
    auto &input = inputs[0];
    TensorUtils::copyShape(input, &mInput);
    mInput.buffer().dim[1].flags = 0;
    bool success                 = backend()->onAcquireBuffer(&mInput, Backend::DYNAMIC);

    // cont transform space
    if (inputs.size() > 1) {
        auto &cont = inputs[1];
        TensorUtils::copyShape(cont, &mCont);
        mCont.buffer().dim[1].flags = 0;
        success                     = success && backend()->onAcquireBuffer(&mCont, Backend::DYNAMIC);
    }

    // output transform space
    auto &output = outputs[0];
    TensorUtils::copyShape(output, &mOutput);
    mOutput.buffer().dim[1].flags = 0;
    success                       = success && backend()->onAcquireBuffer(&mOutput, Backend::DYNAMIC);

    // divide weight & bias if needed
    int iw = input->width(), ih = input->height(), inputSize = iw * ih;
    int ow = output->width(), oh = output->height(), outputSize = ow * oh;
    auto weightI   = mLSTM->weightI();
    auto weightH   = mLSTM->weightH();
    int weightSize = weightI->dims()->data()[0];

    // gate space
    mGates.buffer().dim[1].extent = input->channel() * mLSTM->outputCount();
    mGates.buffer().dim[0].extent = 4;
    mGates.buffer().dimensions    = 2;
    success                       = success && backend()->onAcquireBuffer(&mGates, Backend::DYNAMIC);

    // cell space
    mCell.buffer().dim[1].extent = mLSTM->outputCount();
    mCell.buffer().dim[0].extent = input->channel();
    mCell.buffer().dimensions    = 2;
    success                      = success && backend()->onAcquireBuffer(&mCell, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    // release temp buffer space
    backend()->onReleaseBuffer(&mInput, Backend::DYNAMIC);
    if (inputs.size() > 1) {
        backend()->onReleaseBuffer(&mCont, Backend::DYNAMIC);
    }
    backend()->onReleaseBuffer(&mOutput, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mGates, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mCell, Backend::DYNAMIC);

    if (mInit) {
        return NO_ERROR;
    }
    mInit       = true;
    auto devide = weightI && !weightH && weightSize == 4 * outputSize * (inputSize + outputSize + 2);
    mWeightI.reset(Tensor::createDevice<float>(std::vector<int>{inputSize * outputSize * 4}));
    mWeightH.reset(Tensor::createDevice<float>(std::vector<int>{outputSize * outputSize * 4}));
    mBiasC.reset(Tensor::createDevice<float>(std::vector<int>{outputSize * 4}));
    success = success && backend()->onAcquireBuffer(mWeightH.get(), Backend::STATIC);
    success = success && backend()->onAcquireBuffer(mWeightI.get(), Backend::STATIC);
    success = success && backend()->onAcquireBuffer(mBiasC.get(), Backend::STATIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    if (devide) {
        auto data = weightI->float32s()->data();
        {
            float *to = mWeightI->host<float>();
            int step  = inputSize * outputSize;
            memcpy(to, data, 2 * step * sizeof(float));
            to += 2 * step;
            data += 2 * step;                              // IF
            memcpy(to, data + step, step * sizeof(float)); // O
            memcpy(to + step, data, step * sizeof(float)); // G
            data += 2 * step;
        }
        {
            float *to = mWeightH->host<float>();
            int step  = outputSize * outputSize;
            memcpy(to, data, 2 * step * sizeof(float));
            to += 2 * step;
            data += 2 * step;                              // IF
            memcpy(to, data + step, step * sizeof(float)); // O
            memcpy(to + step, data, step * sizeof(float)); // G
            data += 2 * step;
        }
        {
            float *to = mBiasC->host<float>();
            int step  = outputSize;
            memcpy(to, data, 2 * step * sizeof(float));
            to += 2 * step;
            data += 2 * step;                              // IF
            memcpy(to, data + step, step * sizeof(float)); // O
            memcpy(to + step, data, step * sizeof(float)); // G
            // data += 2 * step;
        }
        return NO_ERROR;
    }
    ::memcpy(mWeightI->host<float>(), mLSTM->weightI()->float32s()->data(), mWeightI->size());
    ::memcpy(mBiasC->host<float>(), mLSTM->bias()->float32s()->data(), mBiasC->size());
    ::memcpy(mWeightH->host<float>(), mLSTM->weightH()->float32s()->data(), mWeightH->size());

    return NO_ERROR;
}

ErrorCode CPULSTM::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto &input           = inputs[0];
    const float *contData = nullptr;
    auto &output          = outputs[0];
    ::memset(mCell.host<float>(), 0, mCell.size());

    // tranform
    MNNUnpackC4(mInput.host<float>(), input->host<float>(), input->width() * input->height(), input->channel());
    if (inputs.size() > 1) {
        auto &cont = inputs[1];
        MNNUnpackC4(mCont.host<float>(), cont->host<float>(), cont->width() * cont->height(), cont->channel());
        contData = mCont.host<float>();
    }

    // calc weightI
    int iw = input->width(), ih = input->height(), inputSize = iw * ih;
    int ow = output->width(), oh = output->height(), outputSize = ow * oh;
    auto outputCount  = mCell.length(1);
    const auto xcStep = inputSize * outputCount;

    MNN_CONCURRENCY_BEGIN(ic, input->channel()) {
        auto inPtr     = mInput.host<const float>() + ic * inputSize;
        auto gatesPtr  = mGates.host<float>() + ic * outputCount * 4;
        auto weightXCI = mWeightI->host<const float>();

        for (int oc = 0; oc < outputCount; oc++, weightXCI += inputSize, gatesPtr += 4) {
            auto weightXCF = weightXCI + xcStep;
            auto weightXCO = weightXCF + xcStep;
            auto weightXCG = weightXCO + xcStep;

            float I = 0, F = 0, O = 0, G = 0;
            int i = 0;
#ifdef MNN_USE_NEON
#if !(defined(__ARM_FEATURE_FMA) && defined(__aarch64__))
#define vaddvq_f32(__v4) (__v4[0] + __v4[1] + __v4[2] + __v4[3]) // support A64 only
#endif
            float32x4_t I4 = vdupq_n_f32(0);
            float32x4_t F4 = vdupq_n_f32(0);
            float32x4_t O4 = vdupq_n_f32(0);
            float32x4_t G4 = vdupq_n_f32(0);
            for (; i + 3 < iw; i += 4) {
                const float32x4_t x4 = vld1q_f32(inPtr + i);
                I4 += vld1q_f32(weightXCI + i) * x4;
                F4 += vld1q_f32(weightXCF + i) * x4;
                O4 += vld1q_f32(weightXCO + i) * x4;
                G4 += vld1q_f32(weightXCG + i) * x4;
            }
            I += vaddvq_f32(I4);
            F += vaddvq_f32(F4);
            O += vaddvq_f32(O4);
            G += vaddvq_f32(G4);
#endif
            for (; i < iw; i++) {
                const float x = inPtr[i];
                I += weightXCI[i] * x;
                F += weightXCF[i] * x;
                O += weightXCO[i] * x;
                G += weightXCG[i] * x;
            }

            gatesPtr[0] = I;
            gatesPtr[1] = F;
            gatesPtr[2] = O;
            gatesPtr[3] = G;
        }
    }
    MNN_CONCURRENCY_END();

    // calc weightHC
    auto cellData     = mCell.host<float>();
    const auto hcStep = outputSize * outputCount;
    for (int ic = 0; ic < input->channel(); ic++) {
        // clip hidden by continuation indicator
        auto cont       = ic > 0 && (!contData || contData[ic]);
        auto gatesPtr   = mGates.host<const float>() + ic * outputCount * 4;
        auto weightHCI  = mWeightH->host<const float>();
        auto outChannel = mOutput.host<float>() + ic * outputSize;
        for (int oc = 0; oc < outputCount; oc++, gatesPtr += 4, weightHCI += outputSize) {
            float I = gatesPtr[0], F = gatesPtr[1], O = gatesPtr[2], G = gatesPtr[3];

            // hidden
            if (cont) {
                auto weightHCF = weightHCI + hcStep;
                auto weightHCO = weightHCF + hcStep;
                auto weightHCG = weightHCO + hcStep;
                auto hiddenPtr = mOutput.host<float>() + (ic - 1) * outputSize;

                int i = 0;
#ifdef MNN_USE_NEON
                float32x4_t Ix4 = vdupq_n_f32(0);
                float32x4_t Fx4 = vdupq_n_f32(0);
                float32x4_t Ox4 = vdupq_n_f32(0);
                float32x4_t Gx4 = vdupq_n_f32(0);
                for (; i + 3 < outputCount; i += 4) {
                    const float32x4_t hiddenData = vld1q_f32(hiddenPtr + i);
                    Ix4 += vld1q_f32(weightHCI + i) * hiddenData;
                    Fx4 += vld1q_f32(weightHCF + i) * hiddenData;
                    Ox4 += vld1q_f32(weightHCO + i) * hiddenData;
                    Gx4 += vld1q_f32(weightHCG + i) * hiddenData;
                }
                I += vaddvq_f32(Ix4);
                F += vaddvq_f32(Fx4);
                O += vaddvq_f32(Ox4);
                G += vaddvq_f32(Gx4);
#endif
                for (; i < outputCount; i++) {
                    const float hiddenData = hiddenPtr[i];
                    I += weightHCI[i] * hiddenData;
                    F += weightHCF[i] * hiddenData;
                    O += weightHCO[i] * hiddenData;
                    G += weightHCG[i] * hiddenData;
                }
            }

            // add bias
            auto biasPtr = mBiasC->host<float>() + oc;
            I            = sigmoid(*biasPtr + I);
            biasPtr      = biasPtr + outputCount;
            F            = cont ? sigmoid(*biasPtr + F) : 0.f;
            biasPtr      = biasPtr + outputCount;
            O            = sigmoid(*biasPtr + O);
            biasPtr      = biasPtr + outputCount;
            G            = tanhf(*biasPtr + G);

            auto newCell   = F * cellData[oc] + I * G;
            cellData[oc]   = newCell;
            auto H         = O * tanhf(newCell);
            outChannel[oc] = H;
        }
    }
    MNNPackC4(output->host<float>(), mOutput.host<float>(), output->width() * output->height(), output->channel());

    return NO_ERROR;
}

class CPULSTMCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPULSTM(backend, op->main_as_LSTM());
    }
};
REGISTER_CPU_OP_CREATOR(CPULSTMCreator, OpType_LSTM);

} // namespace MNN
