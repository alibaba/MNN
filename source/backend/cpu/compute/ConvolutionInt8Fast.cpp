//
//  ConvolutionInt8Fast.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvolutionInt8Fast.hpp"
#include "AutoStorage.h"
#include "CommonOptFunction.h"
#include "Concurrency.h"
#include "ConvOpt.h"
#include "Macro.h"

static const int gUnit  = 8;
static const int gUnit2 = 64;

namespace MNN {
ConvolutionInt8Fast::ConvolutionInt8Fast(const Convolution2DCommon* convOp, Backend* b,
                                         const ConvolutionIntFactory::Int8Common* common, const float* bias,
                                         size_t biasSize)
    : MNN::CPUConvolution(convOp, b) {
    mQuan = common->quan;
    MNN_ASSERT(nullptr != mQuan);
    mBias.reset((int)ALIGN_UP4(biasSize));
    mBias.clear();
    ::memcpy(mBias.get(), bias, biasSize * sizeof(float));

    mAlpha.reset((int)ALIGN_UP4(biasSize));
    mAlpha.clear();
    ::memcpy(mAlpha.get(), common->alpha.get(), biasSize * sizeof(float));

    auto weightLength = common->weight.size();
    mSrcCount         = (int)weightLength / mCommon->kernelX() / mCommon->kernelY() / biasSize;
    auto layer        = mCommon;
    int kx            = layer->kernelX();
    int ky            = layer->kernelY();

    int srcCount    = mSrcCount;
    int outputCount = (int)biasSize;
    int srcCountD8  = UP_DIV(srcCount, gUnit);
    int dstCountD8  = UP_DIV(outputCount, gUnit);

    int cur            = 0;
    auto dstWeightSize = srcCountD8 * dstCountD8 * gUnit2 * kx * ky * sizeof(int8_t);
    mWeight.reset((int)dstWeightSize / sizeof(int8_t));
    int8_t* reorderedWeight = mWeight.get();
    ::memset(reorderedWeight, 0, dstWeightSize);
    auto originWeight = common->weight.get();

    for (int dz = 0; dz < outputCount; ++dz) {
        int dzD8   = dz / gUnit;
        int my     = dz % gUnit;
        auto dstDz = reorderedWeight + dzD8 * srcCountD8 * kx * ky * gUnit2;

        for (int sz = 0; sz < srcCount; ++sz) {
            int szD8 = sz / gUnit;
            int mx   = sz % gUnit;

            auto dstSz = dstDz + szD8 * gUnit2;
            for (int i = 0; i < kx * ky; ++i) {
                auto index                     = i * srcCountD8 * gUnit2;
                dstSz[index + my * gUnit + mx] = originWeight[cur++];
            }
        }
    }
}

ErrorCode ConvolutionInt8Fast::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto layer  = mCommon;
    auto input  = inputs[0];
    auto output = outputs[0];
    int iw      = input->width();
    int ih      = input->height();
    int ic      = input->channel();
    int icD8    = UP_DIV(ic, gUnit);
    int iPlane  = iw * ih;

    int ow   = output->width();
    int oh   = output->height();
    int oc   = output->channel();
    int dcD8 = UP_DIV(oc, gUnit);
    int dcD4 = UP_DIV(oc, 4);

    CONV_SETUP_KERNELSIZE(gUnit);
    int weight_sy_step = gUnit * gUnit * kernel_width;
    int weight_z_step  = kernel_height * kernel_width * src_depth_quad * gUnit * gUnit;

    AutoStorage<int8_t> srcBuffer(icD8 * gUnit * iPlane);
    auto srcOrigin    = srcBuffer.get();
    auto postFunction = MNNScaleBias2FloatC4;
    if (layer->relu()) {
        postFunction = MNNScaleBias2FloatC4Relu;
    }
    if (layer->relu6()) {
        postFunction = MNNScaleBias2FloatC4Relu6;
    }

    for (int batchIndex = 0; batchIndex < input->batch(); ++batchIndex) {
        // AutoTime __t(__LINE__, __FILE__);
        int inputTotalSize = input->stride(0);
        AutoStorage<int8_t> srcCopyBuffer(inputTotalSize);
        auto srcCopy    = srcCopyBuffer.get();
        float quanScale = mQuan->quantScale();
        float quan[] = {
            quanScale,
            quanScale,
            quanScale,
            quanScale
        };
        MNNFloat2Int8(input->host<float>() + inputTotalSize * batchIndex, srcCopy, inputTotalSize / 4, quan,
                      mQuan->aMin(), mQuan->aMax());
        auto dstFloatOrigin = output->host<float>() + output->stride(0) * batchIndex;

        // Reorder, Depth First
        int icD4 = UP_DIV(ic, 4);
        for (int sz = 0; sz < icD4; ++sz) {
            auto dstZ = srcOrigin + sz * 4;
            auto srcZ = srcCopy + sz * iPlane * 4;
            for (int v = 0; v < iPlane; ++v) {
                auto dstV = dstZ + v * icD8 * gUnit;
                for (int j = 0; j < 4; ++j) {
                    dstV[j] = srcZ[4 * v + j];
                }
            }
        }

        // Compute
        MNN_CONCURRENCY_BEGIN(dz, dcD8) {
            AutoStorage<int16_t> dstTempBuffer(oh * ow * gUnit);
            auto dstTemp   = dstTempBuffer.get();
            auto weight_dz = mWeight.get() + dz * weight_z_step;

            for (int dy = 0; dy < oh; ++dy) {
                int srcStartY = dy * strideY - padY;
                auto dst_y    = dstTemp + ow * gUnit * dy;
                auto src_dy   = srcOrigin + srcStartY * iw * gUnit * icD8;
                int sfy       = ALIMAX(0, (UP_DIV(-srcStartY, dilateY)));
                int efy       = ALIMIN(kernel_height, UP_DIV(ih - srcStartY, dilateY));
                int yCount    = efy - sfy;

                for (int dx = 0; dx < ow; ++dx) {
                    int srcStartX     = dx * strideX - padX;
                    auto src_dx       = src_dy + gUnit * srcStartX * icD8;
                    auto dst_x        = dst_y + gUnit * dx;
                    int sfx           = ALIMAX(0, (UP_DIV(-srcStartX, dilateX)));
                    int efx           = ALIMIN(kernel_width, UP_DIV(iw - srcStartX, dilateX));
                    int xCount        = efx - sfx;
                    auto src_unit     = src_dx + icD8 * (sfx * dilateX_step + sfy * dilateY_step);
                    auto weight_start = weight_dz + icD8 * (gUnit2 * sfx + weight_sy_step * sfy);

                    MNNConvolutionInt8Run8x8(dst_x, src_unit, weight_start, icD8, xCount, yCount,
                                             icD8 * (dilateY_step - xCount * dilateX_step),
                                             icD8 * (dilateX_step - gUnit), icD8 * (weight_sy_step - xCount * gUnit2));
                }
            }

            auto dstZ   = dstFloatOrigin + oh * ow * gUnit * dz;
            auto alphaZ = mAlpha.get() + gUnit * dz;
            auto biasZ  = mBias.get() + gUnit * dz;

            postFunction(dstZ, dstTemp, alphaZ, biasZ, ow * oh);
            if (dz * 2 + 1 < dcD4) {
                postFunction(dstZ + ow * oh * 4, dstTemp + 4, alphaZ + 4, biasZ + 4, ow * oh);
            }
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

} // namespace MNN
