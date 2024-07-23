//
//  ConvolutionCommon.cpp
//  MNN
//
//  Created by MNN on 2020/03/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvolutionCommon.hpp"
#include <math.h>
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/CPUBackend.hpp"
#include "half.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/IDSTDecoder.hpp"

namespace MNN {

std::shared_ptr<ConvolutionCommon::Int8Common> ConvolutionCommon::load(const Convolution2D *conv, Backend* backend, bool forceFloat, bool forceInt8) {
    auto quan = conv->quanParameter();
    auto result = std::make_shared<Int8Common>();
    result->quan = quan;
    size_t buffer_size = 0, alpha_size = 0;
    const int8_t* buffer_ptr = nullptr;
    const float* alpha_ptr = nullptr;
    if (quan->buffer()) {
        buffer_size = quan->buffer()->size();
        buffer_ptr = quan->buffer()->data();
    }
    if (quan->alpha()) {
        alpha_size = quan->alpha()->size();
        alpha_ptr = quan->alpha()->data();
    }
    if (quan->index() != nullptr) {
        if (forceFloat) {
            // Expand sparse to dense
            result->weightFloat.reset(quan->weightSize());
            if (nullptr == result->weightFloat.get()) {
                return nullptr;
            }
            ::memset(result->weightFloat.get(), 0, quan->weightSize() * sizeof(float));
            auto index = quan->index()->data();
            auto indexSize = quan->index()->size();
            if (nullptr == alpha_ptr || alpha_size != indexSize) {
                MNN_ERROR("The model is error, don't has alpha but has index\n");
                return nullptr;
            }
            for (uint32_t i=0; i<indexSize; ++i) {
                result->weightFloat.get()[index[i]] = alpha_ptr[i];
            }
        } // Otherwise needn't treat, just return result with quan info
        return result;
    }
    size_t weightLength = 0;
    int8_t *buffer        = nullptr;
    auto originBuffer     = (unsigned char *)buffer_ptr;

    if (1 == quan->type()) {
        buffer = IDSTDecoder::ReadQuanData_c(originBuffer, &weightLength, result.get(), quan->shapeInt32());
    }
    if (2 == quan->type()) {
        buffer = IDSTDecoder::ReadSparseQuanData_c(originBuffer, &weightLength, alpha_ptr, alpha_size, result.get(), quan->shapeInt32());
    }
    if (result->weightMap.size() > 0) {
        result->canUseInt4 = true;
        for (auto value : result->weightMap) {
            if (value < -8 || value > 7) {
                result->canUseInt4 = false;
            }
        }
    }
    // read fp16 data
    if (3 == quan->type()) {
        weightLength = buffer_size / sizeof(half_float::half);
        std::vector<int8_t> tempHalfWeight(buffer_size);
        ::memcpy(tempHalfWeight.data(), buffer_ptr, buffer_size);
        auto halfWeight = reinterpret_cast<half_float::half *>(tempHalfWeight.data());
        result->weightFloat.reset(weightLength);
        if (nullptr == result->weightFloat.get()) {
            MNN_PRINT("Alloc memory error for extract fp16 back to float\n");
            return nullptr;
        }
        std::transform(halfWeight, halfWeight + weightLength, result->weightFloat.get(),
                       [](half_float::half h) { return float(h); });
        return result;
    }

    // weight int8 only
    if (4 == quan->type()) {
        weightLength = buffer_size;
        result->weight.reset(weightLength);
        ::memcpy(result->weight.get(), buffer_ptr, weightLength);
    }

    if (result->weight.get() == nullptr) {
        if (nullptr == buffer) {
            MNN_PRINT("Alloc memory error for extract idst int8\n");
            return nullptr;
        }
        result->weight.set(buffer, weightLength);
    }
    result->alpha.reset(alpha_size);
    if (nullptr == result->alpha.get()) {
        MNN_PRINT("Alloc memory error for extract idst int8\n");
        return nullptr;
    }
    ::memcpy(result->alpha.get(), alpha_ptr, alpha_size * sizeof(float));
    {
        int outputCount = 0;
        bool oldType4 = (quan->type() == 4 && quan->aMin() == 0 && std::abs(quan->quantScale()) < 1e-6);
        if (quan->readType() != 0 || oldType4) {
            result->asymmetric = true;
            outputCount   = result->alpha.size() / 2;
        } else {
            result->asymmetric = false;
            outputCount   = result->alpha.size(); // backward compability with previous symmetric quantization
        }
        if (result->asymmetric) {
            // clampMin is minVal in asymmetric quant, clampMin = -(2^(bit))
            // and old version clampMin is -128
            float clampMin = quan->aMin() == 0 ? -128 : quan->aMin();
            for (int o = 0; o < outputCount; ++o) {
                result->alpha.get()[2 * o] = result->alpha.get()[2 * o] - clampMin * result->alpha.get()[2 * o + 1];
            }
        }
        if (!quan->has_scaleInt()) {
            float extraFactor = quan->quantScale();
            // for old type 4 models, their quan->quantScale is 0. which will introduce a bug here
            if (oldType4) {
                extraFactor = 1.0f;
            }
            for (int o=0; o<result->alpha.size(); ++o) {
                result->alpha.get()[o] *= extraFactor;
            }
        }
    }
    if (forceInt8) {
        return result;
    }
    if (!quan->has_scaleInt() || forceFloat) {
        // Back to float
        result->weightFloat.reset(weightLength);
        if (nullptr == result->weightFloat.get()) {
            MNN_PRINT("Alloc memory error for extract idst int8/ Back to float\n");
            return nullptr;
        }
        int outputCount = 0;
        if (result->asymmetric) {
            outputCount = result->alpha.size() / 2;
        } else {
            outputCount = result->alpha.size();
        }
        int partWeightSize = weightLength / outputCount;
        for (int o = 0; o < outputCount; ++o) {
            float min = 0.0f;
            float alpha = 0.0f;
            if (result->asymmetric) {
                min = result->alpha.get()[2*o];
                alpha = result->alpha.get()[2*o+1];
            } else {
                alpha = result->alpha.get()[o];
            }
            auto dstW   = result->weightFloat.get() + o * partWeightSize;
            auto srcW   = result->weight.get() + o * partWeightSize;
            for (int v=0; v < partWeightSize; ++v) {
                dstW[v] = (float)srcW[v] * alpha + min;
            }
        }
        result->weight.release();
        result->alpha.release();
    }
    return result;
}

void ConvolutionCommon::getConvParameters(std::shared_ptr<Int8Common> *quanCommon, Backend* backend, const MNN::Convolution2D *conv2d, const float** originWeight, int* originWeightSize) {
    *originWeight = nullptr;
    *originWeightSize = 0;
    if (nullptr != conv2d->quanParameter()) {
        bool forceFloat = conv2d->quanParameter()->index() != nullptr;
        *quanCommon = load(conv2d, backend, forceFloat);
        *originWeight     = (*quanCommon)->weightFloat.get();
        *originWeightSize = (*quanCommon)->weightFloat.size();
    }
    if (*originWeight == nullptr) {
        *originWeight = conv2d->weight()->data();
        *originWeightSize = conv2d->weight()->size();
    }
}

bool ConvolutionCommon::getConvInt8Parameters(const MNN::Convolution2D* conv2d, std::shared_ptr<Int8Common>& quanCommon, Backend* backend,
                                              const int8_t*& weight, int& weightSize, float*& scale, int32_t*& bias, int32_t*& weightQuantZeroPoint) {
    int outputCount = conv2d->common()->outputCount();
    weightSize = 0;
    auto core = static_cast<CPUBackend*>(backend)->functions();
    // fix xcode UndefinedBehaviorSanitizer
    if (conv2d->symmetricQuan() && conv2d->symmetricQuan()->weight() != nullptr) {
        weight = conv2d->symmetricQuan()->weight()->data();
        weightSize = conv2d->symmetricQuan()->weight()->size();
    }
    if (conv2d->quanParameter() && conv2d->quanParameter()->buffer()) { // int8 weight
        quanCommon = ConvolutionCommon::load(conv2d, backend, false, true);
        MNN_ASSERT(quanCommon != nullptr);
        weight = quanCommon->weight.get();
        weightSize = quanCommon->weight.size();
    }
    if (weight == nullptr) {
        MNN_ERROR("ConvolutionCommon::getConvInt8Parameters: No weight data!");
        return false;
    }
    bool weightAsy = false;
    if (quanCommon && quanCommon->asymmetric) {
        weightAsy = true;
    }
    if (conv2d->symmetricQuan() && conv2d->symmetricQuan()->bias() && conv2d->symmetricQuan()->scale()) {
        // Compability for old model
        MNN_ASSERT(conv2d->symmetricQuan()->bias()->size() == outputCount && conv2d->symmetricQuan()->scale()->size() == outputCount);
        ::memcpy(bias, conv2d->symmetricQuan()->bias()->data(), outputCount * sizeof(int32_t));
        ::memcpy(scale, conv2d->symmetricQuan()->scale()->data(), outputCount * sizeof(float));
        return true;
    }
    if (conv2d->bias()) {
        ::memcpy(bias, conv2d->bias()->data(), outputCount * sizeof(float));
    }
    if (conv2d->quanParameter() && conv2d->quanParameter()->alpha()) {
        auto alphaAndBeta = conv2d->quanParameter()->alpha()->data();
        int quantCount    = conv2d->quanParameter()->alpha()->size();
        if (false == weightAsy) { // symmetric quant
            if (core->bytes == 2) {
                core->MNNFp32ToLowp(quanCommon->alpha.get(), reinterpret_cast<int16_t*>(scale), quantCount);
            } else {
                ::memcpy(scale, conv2d->quanParameter()->alpha()->data(), quantCount * core->bytes);
            }
        } else if (true == weightAsy) { // asymmetric
            // int ocx2 = 2 * outputCount;
            int scaleSize = quantCount / 2;
            float clampMin = conv2d->quanParameter()->aMin() == 0 ? -128 : conv2d->quanParameter()->aMin();
            if (core->bytes == 2) {
                std::unique_ptr<int16_t[]> tmp(new int16_t[quantCount]);
                core->MNNFp32ToLowp(alphaAndBeta, tmp.get(), quantCount);
                for (int i = 0; i < scaleSize; ++i) {
                    weightQuantZeroPoint[i] = static_cast<int32_t>(roundf((-1) * tmp[2 * i] / tmp[2 * i + 1]) + clampMin);
                    reinterpret_cast<int16_t*>(scale)[i] = tmp[2 * i + 1];
                }
            } else {
                for (int i = 0; i < scaleSize; ++i) {
                    weightQuantZeroPoint[i] = static_cast<int32_t>(roundf((-1) * alphaAndBeta[2 * i] / alphaAndBeta[2 * i + 1])  + clampMin);
                    scale[i] = alphaAndBeta[2 * i + 1];
                }
            }
        }
        return true;
    }
    MNN_ERROR("ConvolutionCommon::getConvInt8Parameters: No bias & scale data!");
    return false;
}

std::pair<int, int> ConvolutionCommon::convolutionPad(const Tensor *input, const Tensor *output,
                                                      const Convolution2DCommon *mCommon) {
    if (mCommon->padMode() == PadMode_SAME) {
        int kernelWidthSize  = (mCommon->kernelX() - 1) * mCommon->dilateX() + 1;
        int kernelHeightSize = (mCommon->kernelY() - 1) * mCommon->dilateY() + 1;

        int padNeededWidth  = (output->width() - 1) * mCommon->strideX() + kernelWidthSize - input->width();
        int padNeededHeight = (output->height() - 1) * mCommon->strideY() + kernelHeightSize - input->height();
        auto mPadX          = padNeededWidth / 2;
        auto mPadY          = padNeededHeight / 2;
        return std::make_pair(mPadX, mPadY);
    }
    auto mPadX = mCommon->padX();
    auto mPadY = mCommon->padY();
    if (nullptr != mCommon->pads() && mCommon->pads()->size() >= 2) {
        mPadX = mCommon->pads()->data()[1];
        mPadY = mCommon->pads()->data()[0];
    }
    return std::make_pair(mPadX, mPadY);
}

std::tuple<int, int, int, int> ConvolutionCommon::convolutionPadFull(const Tensor* input, const Tensor* output,
                                                         const Convolution2DCommon* common) {
    auto pad = convolutionPad(input, output, common);
    int iw = input->width();
    int ih = input->height();
    int ow = output->width();
    int oh = output->height();

    int right = (ow - 1) * common->strideX() + (common->kernelX() - 1) * common->dilateX() - pad.first;
    int padRight = 0;
    if (right >= iw) {
        padRight = right - iw + 1;
    }
    int bottom = (oh - 1) * common->strideY() + (common->kernelY() - 1) * common->dilateY() - pad.second;
    int padBottom = 0;
    if (bottom >= ih) {
        padBottom = bottom - ih + 1;
    }
    return std::make_tuple(pad.first, pad.second, padRight, padBottom);
}

std::pair<int, int> ConvolutionCommon::convolutionTransposePad(const Tensor *input, const Tensor *output,
                                                               const Convolution2DCommon *mCommon) {
    if (mCommon->padMode() == PadMode_SAME) {
        const int outputWidth  = output->width();
        const int outputHeight = output->height();

        const int outputWidthPadded  = (input->width() - 1) * mCommon->strideX() + mCommon->kernelX();
        const int outputHeightPadded = (input->height() - 1) * mCommon->strideY() + mCommon->kernelY();

        const int padNeededWidth  = outputWidthPadded - outputWidth;
        const int padNeededHeight = outputHeightPadded - outputHeight;

        auto mPadX = padNeededWidth / 2;
        auto mPadY = padNeededHeight / 2;
        return std::make_pair(mPadX, mPadY);
    }
    auto mPadX = mCommon->padX();
    auto mPadY = mCommon->padY();
    if (nullptr != mCommon->pads() && mCommon->pads()->size() >= 2) {
        mPadY = mCommon->pads()->data()[0];
        mPadX = mCommon->pads()->data()[1];
    }
    return std::make_pair(mPadX, mPadY);
}

} // namespace MNN
