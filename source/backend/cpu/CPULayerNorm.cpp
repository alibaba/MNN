//
//  CPULayerNorm.cpp
//  MNN
//
//  Created by MNN on 2020/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cmath>
#include "backend/cpu/CPULayerNorm.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "CPUCast.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Execution.hpp"
#include "core/Concurrency.h"
#include "core/SimdHeader.h"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"

namespace MNN {

static void normPackedC4Float(float* normOutput, float* sumOutput, const float* input0, const float* input1,
                              const float* gamma, const float* beta, float epsilon, int batch, int channel,
                              bool rmsNorm, int tId, int threadNumber) {
    constexpr int pack = 4;
    constexpr int tokenTile = 4;
    const int channelBlocks = UP_DIV(channel, pack);
    const int tileCount = UP_DIV(batch, tokenTile);
#if defined(MNN_USE_NEON) && defined(__aarch64__)
    if (rmsNorm && input1 == nullptr && channel % pack == 0) {
        for (int tile = tId; tile < tileCount; tile += threadNumber) {
            const int tokenBase = tile * tokenTile;
            const int tokenCount = ALIMIN(tokenTile, batch - tokenBase);
            float32x4_t squareSums[tokenTile];
            for (int token = 0; token < tokenCount; ++token) {
                squareSums[token] = vdupq_n_f32(0.0f);
            }
            for (int block = 0; block < channelBlocks; ++block) {
                for (int token = 0; token < tokenCount; ++token) {
                    const int offset = (block * batch + tokenBase + token) * pack;
                    auto value = vld1q_f32(input0 + offset);
                    squareSums[token] = vmlaq_f32(squareSums[token], value, value);
                }
            }
            float invStds[tokenTile];
            for (int token = 0; token < tokenCount; ++token) {
                invStds[token] =
                    1.0f / std::sqrt(vaddvq_f32(squareSums[token]) / static_cast<float>(channel) + epsilon);
            }
            const bool affine = gamma != nullptr && beta != nullptr;
            for (int block = 0; block < channelBlocks; ++block) {
                float32x4_t gammaValue;
                float32x4_t betaValue;
                if (affine) {
                    gammaValue = vld1q_f32(gamma + block * pack);
                    betaValue = vld1q_f32(beta + block * pack);
                }
                for (int token = 0; token < tokenCount; ++token) {
                    const int offset = (block * batch + tokenBase + token) * pack;
                    auto value = vmulq_n_f32(vld1q_f32(input0 + offset), invStds[token]);
                    if (affine) {
                        value = vmlaq_f32(betaValue, value, gammaValue);
                    }
                    vst1q_f32(normOutput + offset, value);
                }
            }
        }
        return;
    }
#endif
    for (int tile = tId; tile < tileCount; tile += threadNumber) {
        const int tokenBase = tile * tokenTile;
        const int tokenCount = ALIMIN(tokenTile, batch - tokenBase);
        float sums[tokenTile] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (input1 != nullptr || !rmsNorm) {
            for (int block = 0; block < channelBlocks; ++block) {
                const int valid = ALIMIN(pack, channel - block * pack);
                for (int token = 0; token < tokenCount; ++token) {
                    const int offset = (block * batch + tokenBase + token) * pack;
                    for (int lane = 0; lane < valid; ++lane) {
                        float value = input0[offset + lane];
                        if (input1 != nullptr) {
                            value += input1[offset + lane];
                            sumOutput[offset + lane] = value;
                        }
                        if (!rmsNorm) {
                            sums[token] += value;
                        }
                    }
                    if (sumOutput != nullptr) {
                        for (int lane = valid; lane < pack; ++lane) {
                            sumOutput[offset + lane] = 0.0f;
                        }
                    }
                }
            }
        }

        float means[tokenTile] = {0.0f, 0.0f, 0.0f, 0.0f};
        float squareSums[tokenTile] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (!rmsNorm) {
            for (int token = 0; token < tokenCount; ++token) {
                means[token] = sums[token] / static_cast<float>(channel);
            }
        }
        const float* normInput = sumOutput != nullptr ? sumOutput : input0;
        for (int block = 0; block < channelBlocks; ++block) {
            const int valid = ALIMIN(pack, channel - block * pack);
            for (int token = 0; token < tokenCount; ++token) {
                const int offset = (block * batch + tokenBase + token) * pack;
                for (int lane = 0; lane < valid; ++lane) {
                    const float diff = normInput[offset + lane] - means[token];
                    squareSums[token] += diff * diff;
                }
            }
        }

        float invStds[tokenTile] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int token = 0; token < tokenCount; ++token) {
            invStds[token] = 1.0f / std::sqrt(squareSums[token] / static_cast<float>(channel) + epsilon);
        }
        for (int block = 0; block < channelBlocks; ++block) {
            const int channelBase = block * pack;
            const int valid = ALIMIN(pack, channel - channelBase);
            for (int token = 0; token < tokenCount; ++token) {
                const int offset = (block * batch + tokenBase + token) * pack;
                for (int lane = 0; lane < valid; ++lane) {
                    const int c = channelBase + lane;
                    float value = (normInput[offset + lane] - means[token]) * invStds[token];
                    if (gamma != nullptr && beta != nullptr) {
                        value = value * gamma[c] + beta[c];
                    }
                    normOutput[offset + lane] = value;
                }
                for (int lane = valid; lane < pack; ++lane) {
                    normOutput[offset + lane] = 0.0f;
                }
            }
        }
    }
}

CPULayerNorm::CPULayerNorm(std::shared_ptr<Resource> res, Backend* backend) : Execution(backend) {
    mResource = res;
}

std::shared_ptr<CPULayerNorm::Resource> CPULayerNorm::makeResource(const MNN::Op* op, Backend* backend) {
    const auto* layer_norm_param = op->main_as_LayerNorm();
    std::shared_ptr<CPULayerNorm::Resource> res(new Resource);
    res->mAxis = 0;
    if (nullptr != layer_norm_param->axis()) {
        res->mAxis = layer_norm_param->axis()->size();
    }
    res->mGroup = layer_norm_param->group();
    res->mEpsilon = layer_norm_param->epsilon();
    res->mRMSNorm = layer_norm_param->useRMSNorm();
    bool hasGammaBeta = (layer_norm_param->gamma() && layer_norm_param->beta());
    int gammasize = 0;
    if (hasGammaBeta) {
        MNN_ASSERT(layer_norm_param->gamma()->size() == layer_norm_param->beta()->size());
        gammasize = layer_norm_param->gamma()->size();
    }
    hasGammaBeta = hasGammaBeta || (layer_norm_param->external() && layer_norm_param->external()->size() > 1 &&
                                    layer_norm_param->external()->data()[1] > 0);
    if (hasGammaBeta && gammasize == 0) {
        gammasize = layer_norm_param->external()->data()[1] / sizeof(float);
    }
    if (hasGammaBeta) {
        res->mIniGammaBeta = true;
        // Use uint8_t to avoid lowp reduce float bytes
        res->mGamma.reset(Tensor::createDevice<uint8_t>({gammasize * 4}));
        res->mBeta.reset(Tensor::createDevice<uint8_t>({gammasize * 4}));
        auto status = backend->onAcquireBuffer(res->mGamma.get(), Backend::STATIC) &&
                      backend->onAcquireBuffer(res->mBeta.get(), Backend::STATIC);
        if (!status) {
            MNN_ERROR("Out of memory when gamma is acquired in CPULayerNorm.\n");
            return nullptr;
        }
        bool useCachedMmap = backend->getRuntime()->hint().useCachedMmap > 1;
        if (useCachedMmap) {
            return res;
        }

        const float* gamma_data = layer_norm_param->gamma()->data();
        memcpy(res->mGamma->host<float>(), gamma_data, gammasize * sizeof(float));
        const float* beta_data = layer_norm_param->beta()->data();
        memcpy(res->mBeta->host<float>(), beta_data, gammasize * sizeof(float));
    }
    return res;
}

ErrorCode CPULayerNorm::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const float* gamma = mResource->mIniGammaBeta ? mResource->mGamma->host<float>() : nullptr;
    const float* beta = mResource->mIniGammaBeta ? mResource->mBeta->host<float>() : nullptr;
    auto bn = static_cast<CPUBackend*>(backend());
    auto core = bn->functions();
    auto threadNumber = bn->threadNumber();
    threadNumber = ALIMIN(threadNumber, mOutterSize);
    auto int8core = bn->int8Functions();
    int bytes = core->bytes;
    auto inputQuan = TensorUtils::getDescribe(inputs[0])->quantAttr.get();
    auto outputQuan = TensorUtils::getDescribe(outputs[0])->quantAttr.get();

    if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) {
        bytes = 1;
    }

    if (mNeedUnpackC4 && (bytes == 2 || bytes == 4)) {
        const int batch = inputs[0]->length(0);
        const int channel = inputs[0]->length(1);
        auto inputPtr = inputs[0]->host<uint8_t>();
        auto outputPtr = outputs[0]->host<uint8_t>();
        const uint8_t* input1Ptr = nullptr;
        uint8_t* output1Ptr = nullptr;
        if (inputs.size() == 2 && outputs.size() == 2) {
            input1Ptr = inputs[1]->host<uint8_t>();
            output1Ptr = outputs[1]->host<uint8_t>();
        }
        if (batch == 1) {
            if (bytes == 4) {
                auto inputFloat = reinterpret_cast<const float*>(inputPtr);
                auto outputFloat = reinterpret_cast<float*>(outputPtr);
                if (input1Ptr != nullptr) {
                    auto input1Float = reinterpret_cast<const float*>(input1Ptr);
                    auto output1Float = reinterpret_cast<float*>(output1Ptr);
                    for (int c = 0; c < channel; ++c) {
                        outputFloat[c] = inputFloat[c] + input1Float[c];
                    }
                    MNNNorm(output1Float, outputFloat, gamma, beta, mResource->mEpsilon, channel, mResource->mRMSNorm);
                } else {
                    MNNNorm(outputFloat, inputFloat, gamma, beta, mResource->mEpsilon, channel, mResource->mRMSNorm);
                }
            } else {
                auto inputLowp = reinterpret_cast<const int16_t*>(inputPtr);
                auto outputLowp = reinterpret_cast<int16_t*>(outputPtr);
                auto tmpInput = reinterpret_cast<float*>(mTmpInputFloat.ptr());
                auto tmpOutput = reinterpret_cast<float*>(mTmpOutputFloat.ptr());
                core->MNNLowpToFp32(inputLowp, tmpInput, channel);
                if (input1Ptr != nullptr) {
                    auto input1Lowp = reinterpret_cast<const int16_t*>(input1Ptr);
                    auto output1Lowp = reinterpret_cast<int16_t*>(output1Ptr);
                    core->MNNLowpToFp32(input1Lowp, tmpOutput, channel);
                    for (int c = 0; c < channel; ++c) {
                        tmpInput[c] += tmpOutput[c];
                    }
                    core->MNNFp32ToLowp(tmpInput, outputLowp, channel);
                    MNNNorm(tmpOutput, tmpInput, gamma, beta, mResource->mEpsilon, channel, mResource->mRMSNorm);
                    core->MNNFp32ToLowp(tmpOutput, output1Lowp, channel);
                } else {
                    MNNNorm(tmpOutput, tmpInput, gamma, beta, mResource->mEpsilon, channel, mResource->mRMSNorm);
                    core->MNNFp32ToLowp(tmpOutput, outputLowp, channel);
                }
            }
            return NO_ERROR;
        }
        if (bytes == 4) {
            auto inputFloat = reinterpret_cast<const float*>(inputPtr);
            auto outputFloat = reinterpret_cast<float*>(outputPtr);
            auto input1Float = reinterpret_cast<const float*>(input1Ptr);
            auto output1Float = reinterpret_cast<float*>(output1Ptr);
            if (core->pack == 4) {
                constexpr int tokenTile = 4;
                const int packedThreadNumber = ALIMIN(threadNumber, UP_DIV(batch, tokenTile));
                MNN_CONCURRENCY_BEGIN(tId, packedThreadNumber) {
                    normPackedC4Float(output1Float != nullptr ? output1Float : outputFloat,
                                      output1Float != nullptr ? outputFloat : nullptr, inputFloat, input1Float, gamma,
                                      beta, mResource->mEpsilon, batch, channel, mResource->mRMSNorm, tId,
                                      packedThreadNumber);
                }
                MNN_CONCURRENCY_END();
            } else if (core->MNNNormPacked != nullptr) {
                if (input1Float != nullptr) {
                    const int elementSize = static_cast<CPUBackend*>(backend())->getTensorSize(inputs[0]);
                    core->MNNMatrixAdd(outputFloat, inputFloat, input1Float, elementSize / core->pack, 0, 0, 0, 1);
                    core->MNNNormPacked(output1Float, outputFloat, gamma, beta, mResource->mEpsilon, batch, channel,
                                        mResource->mRMSNorm);
                } else {
                    core->MNNNormPacked(outputFloat, inputFloat, gamma, beta, mResource->mEpsilon, batch, channel,
                                        mResource->mRMSNorm);
                }
            } else {
                return NOT_SUPPORT;
            }
            return NO_ERROR;
        }
        auto unpackedInput = mTmpUnpackedInput.ptr();
        auto unpackedOutput = mTmpUnpackedOutput.ptr();
        int unpackOffset[2] = {batch, channel};
        core->MNNUnpackCUnitTranspose(reinterpret_cast<float*>(unpackedInput), reinterpret_cast<const float*>(inputPtr),
                                      batch, channel, unpackOffset);
        if (input1Ptr != nullptr) {
            core->MNNUnpackCUnitTranspose(reinterpret_cast<float*>(unpackedOutput),
                                          reinterpret_cast<const float*>(input1Ptr), batch, channel, unpackOffset);
        }
        auto unpackedInputLowp = reinterpret_cast<int16_t*>(unpackedInput);
        auto unpackedOutputLowp = reinterpret_cast<int16_t*>(unpackedOutput);
        MNN_CONCURRENCY_BEGIN(ttId, threadNumber) {
            auto tmpInput = reinterpret_cast<float*>(mTmpInputFloat.ptr() + ttId * channel * sizeof(float));
            auto tmpOutput = reinterpret_cast<float*>(mTmpOutputFloat.ptr() + ttId * channel * sizeof(float));
            for (int n = ttId; n < batch; n += threadNumber) {
                auto inputRow = unpackedInputLowp + n * channel;
                auto outputRow = unpackedOutputLowp + n * channel;
                core->MNNLowpToFp32(inputRow, tmpInput, channel);
                if (input1Ptr != nullptr) {
                    core->MNNLowpToFp32(outputRow, tmpOutput, channel);
                    for (int c = 0; c < channel; ++c) {
                        tmpInput[c] += tmpOutput[c];
                    }
                    core->MNNFp32ToLowp(tmpInput, inputRow, channel);
                }
                MNNNorm(tmpOutput, tmpInput, gamma, beta, mResource->mEpsilon, channel, mResource->mRMSNorm);
                core->MNNFp32ToLowp(tmpOutput, outputRow, channel);
            }
        }
        MNN_CONCURRENCY_END();
        int packOffset[2] = {channel, batch};
        if (output1Ptr != nullptr) {
            core->MNNPackCUnitTranspose(reinterpret_cast<float*>(outputPtr),
                                        reinterpret_cast<const float*>(unpackedInput), batch, channel, packOffset);
            core->MNNPackCUnitTranspose(reinterpret_cast<float*>(output1Ptr),
                                        reinterpret_cast<const float*>(unpackedOutput), batch, channel, packOffset);
        } else {
            core->MNNPackCUnitTranspose(reinterpret_cast<float*>(outputPtr),
                                        reinterpret_cast<const float*>(unpackedOutput), batch, channel, packOffset);
        }
        return NO_ERROR;
    }

    auto input = inputs[0]->host<uint8_t>();
    auto output = outputs[0]->host<uint8_t>();
    MNN_CONCURRENCY_BEGIN(ttId, threadNumber) {
        for (int tId = ttId; tId < mOutterSize; tId += threadNumber) {
            const float* inner_input = (const float*)(input + tId * mInnerSize * bytes);
            float* inner_output = (float*)(output + tId * mInnerSize * bytes);
            if (bytes != 4) {
                auto tmpInput = (float*)(mTmpInputFloat.ptr() + ttId * mInnerSize * sizeof(float));
                auto tmpOutput = (float*)(mTmpOutputFloat.ptr() + ttId * mInnerSize * sizeof(float));
                if (bytes == 1) {
                    CPUCastCreator::cast(inner_input, tmpInput, CPUCastCreator::INT8_TO_FlOAT, mInnerSize,
                                         inputQuan->scale, inputQuan->zero, inputQuan->min, inputQuan->max, bn);
                } else {
                    core->MNNLowpToFp32((const int16_t*)inner_input, tmpInput, mInnerSize);
                }
                MNNNorm(tmpOutput, tmpInput, gamma, beta, mResource->mEpsilon, mInnerSize, mResource->mRMSNorm);
                if (bytes == 1) {
                    CPUCastCreator::cast(tmpOutput, inner_output, CPUCastCreator::FlOAT_TO_INT8, mInnerSize,
                                         outputQuan->scale, outputQuan->zero, outputQuan->min, outputQuan->max, bn);
                } else {
                    core->MNNFp32ToLowp(tmpOutput, (int16_t*)inner_output, mInnerSize);
                }
            } else {
                MNNNorm(inner_output, inner_input, gamma, beta, mResource->mEpsilon, mInnerSize, mResource->mRMSNorm);
            }
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

ErrorCode CPULayerNorm::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mOutterSize = 1;
    mInnerSize = 1;
    const auto layout = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
    mNeedUnpackC4 = (layout == MNN_DATA_FORMAT_NC4HW4);
    do {
        // Compute outter and inner
        int rank = inputs.at(0)->dimensions();
        if (mResource->mGroup > 1) {
            mOutterSize = inputs.at(0)->length(0) * mResource->mGroup;
            for (int i = 1; i < rank; i++) {
                mInnerSize *= inputs.at(0)->length(i);
            }
            mInnerSize /= mResource->mGroup;
            if (mResource->mIniGammaBeta) {
                MNN_ASSERT(mResource->mGamma->size() == mInnerSize * sizeof(float));
            }
            break;
        }
        for (int i = 0; i < rank - mResource->mAxis; ++i) {
            mOutterSize *= inputs.at(0)->length(i);
        }
        for (int i = rank - mResource->mAxis; i < rank; ++i) {
            mInnerSize *= inputs.at(0)->length(i);
        }
        if (mResource->mIniGammaBeta && !mNeedUnpackC4) {
            MNN_ASSERT(mResource->mGamma->size() == mInnerSize * sizeof(float));
        }
    } while (false);
    auto bn = static_cast<CPUBackend*>(backend());
    auto threadNumber = ALIMIN(bn->threadNumber(), mOutterSize);
    auto buf = bn->getBufferAllocator();

    if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1 ||
        bn->functions()->bytes != 4) {
        int tmpSize = mNeedUnpackC4 ? inputs[0]->length(1) : mInnerSize;
        int tmpThreadNumber = mNeedUnpackC4 ? bn->threadNumber() : threadNumber;
        mTmpInputFloat = buf->alloc(tmpThreadNumber * tmpSize * sizeof(float));
        mTmpOutputFloat = buf->alloc(tmpThreadNumber * tmpSize * sizeof(float));
        buf->free(mTmpInputFloat);
        buf->free(mTmpOutputFloat);
    }
    if (mNeedUnpackC4 && bn->functions()->bytes == 2) {
        const int elementCount = inputs[0]->length(0) * inputs[0]->length(1);
        mTmpUnpackedInput = buf->alloc(elementCount * sizeof(int16_t));
        mTmpUnpackedOutput = buf->alloc(elementCount * sizeof(int16_t));
        buf->free(mTmpUnpackedInput);
        buf->free(mTmpUnpackedOutput);
    }
    return NO_ERROR;
}

CPULayerNorm::~CPULayerNorm() {
    // Do nothing
}
bool CPULayerNorm::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    *dst = new CPULayerNorm(mResource, bn);
    return true;
}

class CPULayerNormCreator : public CPUBackend::Creator {
public:
    Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                        Backend* backend) const override {
        auto res = CPULayerNorm::makeResource(op, backend);
        if (nullptr == res.get()) {
            return nullptr;
        }
        return new CPULayerNorm(res, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPULayerNormCreator, OpType_LayerNorm);

} // namespace MNN
