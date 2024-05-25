//
//  ConvolutionPackFreeWinograd.cpp
//  MNN
//
//  Created by MNN on 2022/01/20.
//  Copyright © 2018 - 2022, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvolutionPackFreeWinograd.hpp"
#include <math.h>
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "math/WingoradGenerater.hpp"
#include <MNN/AutoTime.hpp>
#include "core/MemoryFormater.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

constexpr int FUSE_THRESHHOLD_NUMERATOR = 10;
constexpr int FUSE_THRESHHOLD_DENOMINATOR = 10;

using namespace MNN::Math;

namespace MNN {
ConvolutionPackFreeWinograd::ConvolutionPackFreeWinograd(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output,
                                         Backend *b, const float *originWeight, size_t originWeightSize,
                                         const float *bias, size_t biasSize, WinogradConfig config)
    : MNN::ConvolutionWinogradImpl(convOp, b) {

    mResource.reset(new Resource);
    mResource->backend = b;
    mDestUnrollTransform.reset(new CoreFunctions::WinoUnrollDestTransFunc[CONVOLUTION_WINOGRAD_MAX_UNIT + 1],
        std::default_delete<CoreFunctions::WinoUnrollDestTransFunc[]>());

    if (!mResource->copyBiasAlign(bias, biasSize)) {
        MNN_ERROR("Not Enough Memory\n");
        mValid = false;
        return;
    }
    mConvPerfconfig = config;
    mOriginWeight = originWeight;
    updateWinogradBuffer(input, output);

}

ConvolutionPackFreeWinograd::~ConvolutionPackFreeWinograd() {
    // Do nothing
}
bool ConvolutionPackFreeWinograd::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvolutionPackFreeWinograd(mResource, op->main_as_Convolution2D()->common(), bn);
    dstExe->mA = mA;
    dstExe->mB = mB;
    dstExe->mTempBuffer.reset(Tensor::createDevice<uint8_t>(mTempBuffer->shape()));
    dstExe->mTransformMidBuffer.reset(Tensor::createDevice<uint8_t>(mTransformMidBuffer->shape()));
    dstExe->mGemmMidBuffer.reset(Tensor::createDevice<uint8_t>(mGemmMidBuffer->shape()));
    dstExe->mSourceTransformPack = mSourceTransformPack;
    dstExe->mSourceUnrollTransform = mSourceUnrollTransform;
    dstExe->mConvPerfconfig = mConvPerfconfig;
    dstExe->mDestUnrollTransform = mDestUnrollTransform;
    dstExe->mPostParameters = mPostParameters;
    *dst = dstExe;
    return true;
}

// #define PROFILE_DETAIL

ErrorCode ConvolutionPackFreeWinograd::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int pack = core->pack, bytes = core->bytes;

    auto input   = inputs[0];
    auto output  = outputs[0];
    auto dstUnit = mA->length(1);
    auto srcUnit = mA->length(0);
    int ePackMax, lPack, hPack;
    core->MNNGetMatMulPackMode(&ePackMax, &lPack, &hPack);
    int ePack = mConvPerfconfig.ePack;

    auto srcUnit2 = srcUnit * srcUnit;
    auto alphaXStride = srcUnit * ePack * pack;
    auto IC4alpha2Stride = srcUnit2 * ePack * pack;

    int ow   = output->width();
    int oh   = output->height();
    int iw   = input->width();
    int ih   = input->height();
    int oc = output->channel();
    int ic = input->channel();
    int ic_roundup = ROUND_UP(ic, lPack);
    int ic_4 = UP_DIV(input->channel(), pack);
    int dc_4 = UP_DIV(output->channel(), pack);
    int batch = input->batch();

    int padY = mPadY;
    int padX = mPadX;

    auto wUnit = UP_DIV(ow, dstUnit);
    auto hUnit = UP_DIV(oh, dstUnit);

    auto totalCount   = wUnit * hUnit * batch;
    int threadNumber = std::max(((CPUBackend *)backend())->threadNumber(), 1);
    int eRemain = totalCount % ePack;
    int tileCount = UP_DIV(totalCount, mConvPerfconfig.eTile);

    std::vector<size_t> parameters(7);
    parameters[0] = eRemain * bytes;
    parameters[1] = input->channel();
    parameters[2] = output->channel();
    parameters[3] = ePack * pack * bytes;
    parameters[4] = 0;
    parameters[5] = 0;
    parameters[6] = 0;

    std::vector<size_t> parametersRemain = parameters;
    parametersRemain[3]                  = eRemain * pack * bytes;

    std::vector<size_t> Tile2MatMulParameters = {
        static_cast<size_t>(ePack * ic_4 * pack * bytes),
        static_cast<size_t>(ic),
        0,
        0,
        static_cast<size_t>(ic_roundup * mConvPerfconfig.hPack * bytes),
        static_cast<size_t>(mConvPerfconfig.hPack * bytes),
        0};

    auto inputOrigin     = input->host<uint8_t>();
    auto outputOrigin    = output->host<uint8_t>();
    auto srcOrigin       = inputOrigin;
    auto dstOrigin       = outputOrigin;
    auto midBuffer0Bytes = srcUnit2 * pack * bytes;

    bool allow_x86_bf16_winograd = true;
#ifdef MNN_USE_SSE
    allow_x86_bf16_winograd = bytes != 2;
#endif

    using ElementType = float;
    auto weight    = mResource->mWeight->host<uint8_t>();
    auto bias      = mResource->mBias->host<uint8_t>();


        auto _srcOrigin = mTempBuffer->host<uint8_t>();
        auto gemmBuffer = (mGemmMidBuffer->host<uint8_t>());
        auto midBuffer0 = mTransformMidBuffer->host<uint8_t>();
        auto midBuffer1 = midBuffer0 + midBuffer0Bytes;

        auto parallelInnerSourceFunction = [&](int tId, int tIndex) {

            int eTile = mConvPerfconfig.eTile;
            int hPackDynamic = mConvPerfconfig.hPack;
            int ic_pack = ROUND_UP(ic, pack);
            int xIndex  = (int)tIndex * eTile;
            int xReamin = totalCount - xIndex;
            int eTileReal = xReamin > eTile ? eTile : xReamin;

            /*Source Transform Begin*/
            const int bTransStride = wUnit * hUnit;
            const int ib_stride = iw * ih;
            const int pack_stride = pack * bytes;

            const int ICUnitStep    = ic_4 * eTileReal * pack;
            const int sourceZStep = ib_stride * batch * pack_stride;
            const int IcBufferOffset = mTransformMidBuffer->stride(0);

            for (int tile_k_z = tId; tile_k_z < ic_4 * eTileReal; tile_k_z += threadNumber) {
                int z = tile_k_z / eTileReal;
                int eTileNumber = tile_k_z % eTileReal;
                int tile_k = eTileNumber + xIndex;
                int bIndex = tile_k / bTransStride;
                int hwIndex = tile_k % bTransStride;
                int hIndex = (hwIndex / wUnit);
                int wIndex = (hwIndex % wUnit);
                int iEpack = eTileNumber % ePack;
                int iETile = eTileNumber - iEpack;
                int ePackSegment = fmin(ePack, eTileReal - iETile);
                int ihIndex = hIndex * dstUnit - padY;
                int iwIndex = wIndex * dstUnit - padX;
                int ey    = ALIMIN(ihIndex + srcUnit, ih) - ihIndex;
                int sy    = ALIMAX(0, ihIndex) - ihIndex;
                int ex    = ALIMIN(iwIndex + srcUnit, iw) - iwIndex;
                int sx    = ALIMAX(0, iwIndex) - iwIndex;
                int count = pack_stride * (ex - sx);
                auto srcZ = srcOrigin + (iwIndex + ihIndex * iw + bIndex * ib_stride) * pack_stride + z * sourceZStep;
                auto dstZ = _srcOrigin + (iETile * ic_4 + z * ePackSegment + iEpack) * pack_stride;
                if (ex - sx == srcUnit && ey - sy == srcUnit) {

                    auto icMidBuffer1 = midBuffer1 + tId * IcBufferOffset;
                    mSourceUnrollTransform((const float*)srcZ, (float*)icMidBuffer1, iw * pack, pack, pack, pack * srcUnit);
                    mSourceUnrollTransform((const float*)icMidBuffer1, (float*)dstZ, srcUnit * pack, ICUnitStep, pack, ICUnitStep * srcUnit);
                } else {
                        // Extract

                    auto icMidBuffer1 = midBuffer1 + tId * IcBufferOffset;
                    auto icMidBuffer0 = midBuffer0 + tId * IcBufferOffset;
                    ::memset(icMidBuffer0, 0, mTransformMidBuffer->stride(1));
                    if (count > 0) {
                        for (int yy = sy; yy < ey; ++yy) {
                            auto dst_yy = icMidBuffer0 + (yy * srcUnit + sx) * pack_stride;
                            auto src_yy = srcZ + (iw * yy + sx) * pack_stride;
                            ::memcpy(dst_yy, src_yy, count);
                        }
                    }

                    mSourceUnrollTransform((const float*)icMidBuffer0, (float*)icMidBuffer1, srcUnit * pack, pack, pack, pack * srcUnit);
                    mSourceUnrollTransform((const float*)icMidBuffer1, (float*)dstZ, srcUnit * pack, ICUnitStep, pack, ICUnitStep * srcUnit);
                }

            }
        };

    auto parallelInnerPackFreeMultiplyFunction = [&](int tId, int tIndex) {

        int eTile = mConvPerfconfig.eTile;
        int hPackDynamic = mConvPerfconfig.hPack;

        int xIndex  = (int)tIndex * eTile;
        int xReamin = totalCount - xIndex;
        int eTileReal = xReamin > eTile ? eTile : xReamin;

        int tLast = eTileReal % ePack;
        int tBlock = eTileReal - tLast;
        const int oc_hpack = UP_DIV(oc, hPackDynamic);
        const int oc_pack_coeff = hPackDynamic / pack;
        const int weightStride = mResource->mWeight->stride(0);
        const int pack_stride = pack * bytes;

        auto threadParameters = Tile2MatMulParameters;
        auto threadParametersRemain = threadParameters;
        threadParameters[6] =  tBlock;
        threadParametersRemain[6] = tLast;
        threadParameters[3] = eTileReal * pack_stride;
        threadParametersRemain[3] = threadParameters[3];

        // copy pointer out
        auto MaxATileMatMulOC16Function = core->MNNPackedMatMulOC16Functions[ePack - 1];
        auto TailATileMatMulOC16Function = core->MNNPackedMatMulOC16Functions[tLast - 1];
        auto MaxATileMatMulOC32Function = core->MNNPackedMatMulOC32Functions[ePack - 1];
        auto TailATileMatMulOC32Function = core->MNNPackedMatMulOC32Functions[tLast - 1];
        auto MaxATileMatMulOC48Function = core->MNNPackedMatMulOC48Functions[ePack - 1];
        auto TailATileMatMulOC48Function = core->MNNPackedMatMulOC48Functions[tLast - 1];

        auto* _dstOrigin = _srcOrigin + eTileReal * srcUnit2 * ic_4 * pack * bytes;

        // srcUnit2, oc
        for (int i_oc_src = tId; i_oc_src < srcUnit2 * oc_hpack; i_oc_src += threadNumber) {
            int t_oc_mul = i_oc_src % oc_hpack;
            int i = i_oc_src / oc_hpack;

            int t_oc = t_oc_mul * oc_pack_coeff;
            int ocValidPack = ALIMIN(t_oc + oc_pack_coeff, dc_4) - t_oc;
            // calculate address
            auto srcTemp = (_srcOrigin + i * ic_4 * eTileReal * pack * bytes);
            auto _weightFloatPtr = (const float*)(weight + i * weightStride + (t_oc * ic_roundup * pack) * bytes);
            auto _dstFloatPtr = (_dstOrigin + (i * dc_4 + t_oc) * eTileReal * pack * bytes);

#ifdef PROFILE_DETAIL
            macs[tId] += eTileReal * (2 * ic) * (ocValidPack) * pack;
#endif

            if (tBlock) {
                switch (ocValidPack) {
                    case 1:
                        MaxATileMatMulOC16Function((float*)_dstFloatPtr, (const float*)srcTemp, _weightFloatPtr, threadParameters.data(), nullptr, nullptr);
                        break;
                    case 2:
                        MaxATileMatMulOC32Function((float*)_dstFloatPtr, (const float*)srcTemp, _weightFloatPtr, threadParameters.data(), nullptr, nullptr);
                        break;
                    case 3:
                        MaxATileMatMulOC48Function((float*)_dstFloatPtr, (const float*)srcTemp, _weightFloatPtr, threadParameters.data(), nullptr, nullptr);
                        break;
                }
                srcTemp += tBlock * ic_4 * pack * bytes;
                _dstFloatPtr += tBlock * pack * bytes;
            }
            if (tLast) {

                switch (ocValidPack) {
                    case 1:
                        TailATileMatMulOC16Function((float*)_dstFloatPtr, (const float*)srcTemp, _weightFloatPtr, threadParametersRemain.data(), nullptr, nullptr);
                        break;
                    case 2:
                        TailATileMatMulOC32Function((float*)_dstFloatPtr, (const float*)srcTemp, _weightFloatPtr, threadParametersRemain.data(), nullptr, nullptr);
                        break;
                    case 3:
                        TailATileMatMulOC48Function((float*)_dstFloatPtr, (const float*)srcTemp, _weightFloatPtr, threadParametersRemain.data(), nullptr, nullptr);
                        break;
                }
            }

        }
    };

        auto parallelInnerMultiplyFunction = [&](int tId, int tIndex) {
            int xIndex  = (int)tIndex * ePack;
            int xReamin = totalCount - xIndex;
            int xC      = xReamin > ePack ? ePack : xReamin;
            auto* _dstOrigin = _srcOrigin + xC * srcUnit2 * ic_4 * pack * bytes;

                /*Source Transform End*/
                // Multi
                int32_t info[4];
                info[0] = 1;
                info[1] = xC;
                info[2] = xC;
                info[3] = 1;
                int32_t el[4];
                el[0] = xC;
                el[1] = parameters[1];
                el[2] = 0;
                el[3] = 0;
                if (xC == ePackMax) {
                    for (int i = tId; i < srcUnit2; i+=threadNumber) {
                        auto srcTemp = (const float*)(_srcOrigin + i * ic_4 * pack * xC * bytes);
                        auto gemmBufferPtr = (const float*)(gemmBuffer + i * ePack * ic_roundup * bytes);
                        core->MNNPackC4ForMatMul_A((float*)gemmBufferPtr, &srcTemp, info, el);
                    }
                    for (int i = tId; i < srcUnit2; i+=threadNumber) {
                        auto _dstFloatPtr = (float*)(_dstOrigin + i * dc_4 * pack * xC * bytes);
                        auto _weightFloatPtr = (const float*)(weight + i * mResource->mWeight->stride(0));
                        auto gemmBufferPtr = (const float*)(gemmBuffer + i * ePack * ic_roundup * bytes);
                        core->MNNPackedMatMul(_dstFloatPtr, (float*)gemmBufferPtr, _weightFloatPtr, parameters.data(), nullptr, nullptr, nullptr, nullptr);
                    }
                } else {
                    for (int i = tId; i < srcUnit2; i+=threadNumber) {
                        auto srcTemp = (const float*)(_srcOrigin + i * ic_4 * pack * xC * bytes);
                        auto gemmBufferPtr = (const float*)(gemmBuffer + i * ePack * ic_roundup * bytes);
                        core->MNNPackC4ForMatMul_A((float*)gemmBufferPtr, &srcTemp, info, el);
                    }
                    for (int i = tId; i < srcUnit2; i+=threadNumber) {
                        auto _dstFloatPtr = (float*)(_dstOrigin + i * dc_4 * pack * xC * bytes);
                        auto _weightFloatPtr = (const float*)(weight + i * mResource->mWeight->stride(0));
                        auto gemmBufferPtr = (const float*)(gemmBuffer + i * ePack * ic_roundup * bytes);
                        core->MNNPackedMatMulRemain(_dstFloatPtr, (float*)gemmBufferPtr, _weightFloatPtr, xC, parametersRemain.data(), nullptr, nullptr, nullptr, nullptr);
                    }
                }
            };

            /* Dest Transform And Post Treat Begin */
        auto parallelInnerDestFunction = [&](int tId, int tIndex) {

            auto DestUnrollTransform = mDestUnrollTransform.get();
            int eTile = mConvPerfconfig.eTile;
            int hPackDynamic = mConvPerfconfig.hPack;
            int ic_pack = ROUND_UP(ic, pack);
            int xIndex  = (int)tIndex * eTile;
            int xReamin = totalCount - xIndex;
            int eTileReal = xReamin > eTile ? eTile : xReamin;
            const int pack_stride = pack * bytes;

            const int transb_stride = wUnit * hUnit;
            const int ob_stride = ow * oh;
            const int srcTransZStep = eTileReal * pack_stride;
            const int OCUnitStep = eTileReal * pack * dc_4;
            const int dstZStep = ob_stride * batch * pack_stride;
            const auto ocBufferOffset = mTransformMidBuffer->stride(0);
            const auto srcOriginSegment = _srcOrigin + eTileReal * srcUnit2 * ic_4 * pack_stride;

            for (int tile_k_z = tId; tile_k_z < dc_4 * eTileReal; tile_k_z += threadNumber) {
                int z = tile_k_z / eTileReal;
                int tile_k = (tile_k_z % eTileReal) + xIndex;
                int bIndex = tile_k / transb_stride;
                int hwIndex = tile_k % transb_stride;
                int hIndex = (hwIndex / wUnit);
                int wIndex = (hwIndex % wUnit);
                int ohIndex = hIndex * dstUnit;
                int owIndex = wIndex * dstUnit;
                const float* postParameters = mPostParameters.data();
                const float* biasFloatPtr = (const float*)(bias + z * pack_stride);
                int ey = ALIMIN(ohIndex + dstUnit, oh) - ohIndex;
                int ex = ALIMIN(owIndex + dstUnit, ow) - owIndex;
                auto dstStart = dstOrigin + (owIndex + ohIndex * ow + bIndex * ob_stride) * pack_stride;
                auto srcStart =  srcOriginSegment + (tile_k - xIndex) * pack_stride;
                int count = ex * pack_stride;
                if (ex == dstUnit) {
                    auto dstZAddr = dstStart + z * dstZStep;
                    auto srcZ     = srcStart + z * srcTransZStep;
                    auto ocMidBuffer0 = midBuffer0 + tId * ocBufferOffset;
                    DestUnrollTransform[srcUnit]((const float*)srcZ, (float*)ocMidBuffer0, nullptr, nullptr, OCUnitStep, dstUnit * pack, srcUnit * OCUnitStep, pack);
                    DestUnrollTransform[ey]((const float*)ocMidBuffer0, (float*)dstZAddr, biasFloatPtr, postParameters, pack, pack * ow, pack * dstUnit, pack);
                } else {
                    auto dstZAddr = dstStart + z * dstZStep;
                    auto srcZ     = srcStart + z * srcTransZStep;
                    auto ocMidBuffer0 = midBuffer0 + tId * ocBufferOffset;
                    auto ocMidBuffer1 = midBuffer1 + tId * ocBufferOffset;
                    DestUnrollTransform[srcUnit]((const float*)srcZ, (float*)ocMidBuffer0, nullptr, nullptr, OCUnitStep, dstUnit * pack, srcUnit * OCUnitStep, pack);
                    DestUnrollTransform[ey]((const float*)ocMidBuffer0, (float*)ocMidBuffer1, biasFloatPtr, postParameters, pack, pack * dstUnit, pack * dstUnit, pack);
                    for (int yy = 0; yy < ey; ++yy) {
                        auto dstYAddr = dstZAddr + yy * ow * pack_stride;
                        auto srcYAddr = ocMidBuffer1 + yy * dstUnit * pack_stride;
                        ::memcpy(dstYAddr, srcYAddr, count);
                    }
                }
            }
            /*Dest Transform And Post Treat End*/
        };

    auto parallelOuterPackFreeFunction = [&](int tId) {
        int eTile = mConvPerfconfig.eTile;
        int hPackDynamic = mConvPerfconfig.hPack;

        auto _srcOrigin = mTempBuffer->host<uint8_t>() + tId * mTempBuffer->stride(0);
        auto gemmBuffer = (mGemmMidBuffer->host<uint8_t>() + tId * mGemmMidBuffer->stride(0));
        auto midBuffer0 = mTransformMidBuffer->host<uint8_t>() + tId * mTransformMidBuffer->stride(0);
        auto midBuffer1 = midBuffer0 + midBuffer0Bytes;

        for (int tIndex = (int)tId; tIndex < tileCount; tIndex += threadNumber) {
            int xIndex  = (int)tIndex * eTile;
            int xReamin = totalCount - xIndex;
            int eTileReal = xReamin > eTile ? eTile : xReamin;

            /*Source Transform Begin*/
            const int bTransStride = wUnit * hUnit;
            const int ib_stride = iw * ih;
            const int pack_stride = pack * bytes;
            const int ICUnitStep    = ic_4 * eTileReal * pack;
            const int sourceZStep = iw * ih * batch * pack_stride;
            for (int z = 0; z < ic_4; z++) {
                for (int tile_k = xIndex; tile_k < xIndex + eTileReal; tile_k++) {
                    int bIndex = tile_k / bTransStride;
                    int hwIndex = tile_k % bTransStride;
                    int hIndex = (hwIndex / wUnit);
                    int wIndex = (hwIndex % wUnit);

                    int eTileNumber = tile_k - xIndex;
                    int iEpack = eTileNumber % ePack;
                    int iETile = eTileNumber - iEpack;
                    int ePackSegment = fmin(ePack, eTileReal - iETile);

                    int ihIndex = hIndex * dstUnit - padY;
                    int iwIndex = wIndex * dstUnit - padX;
                    int ey    = ALIMIN(ihIndex + srcUnit, ih) - ihIndex;
                    int sy    = ALIMAX(0, ihIndex) - ihIndex;
                    int ex    = ALIMIN(iwIndex + srcUnit, iw) - iwIndex;
                    int sx    = ALIMAX(0, iwIndex) - iwIndex;
                    int count = pack_stride * (ex - sx);

                    auto srcZ = srcOrigin + (iwIndex + ihIndex * iw + bIndex * ib_stride) * pack_stride + z * sourceZStep;
                    auto dstZ = _srcOrigin + (iETile * ic_4 + z * ePackSegment + iEpack) * pack_stride;

                    if (ex - sx == srcUnit && ey - sy == srcUnit) {

                        // Transform
                        mSourceUnrollTransform((const float*)srcZ, (float*)midBuffer1, iw * pack, pack, pack, pack * srcUnit);
                        mSourceUnrollTransform((const float*)midBuffer1, (float*)dstZ, srcUnit * pack, ICUnitStep, pack, ICUnitStep * srcUnit);

                    } else {
                        // Extract
                        ::memset(midBuffer0, 0, mTransformMidBuffer->stride(1));
                        if (count > 0) {
                            for (int yy = sy; yy < ey; ++yy) {
                                auto dst_yy = midBuffer0 + (yy * srcUnit + sx) * pack_stride;
                                auto src_yy = srcZ + (iw * yy + sx) * pack_stride;
                                ::memcpy(dst_yy, src_yy, count);
                            }
                        }

                        mSourceUnrollTransform((const float*)midBuffer0, (float*)midBuffer1, srcUnit * pack, pack, pack, pack * srcUnit);
                        mSourceUnrollTransform((const float*)midBuffer1, (float*)dstZ, srcUnit * pack, ICUnitStep, pack, ICUnitStep * srcUnit);
                    }
                }
            }
            /*Source Transform End*/
            //Multi
            int tLast = eTileReal % ePack;
            int tBlock = eTileReal - tLast;
            const int oc_hpack = UP_DIV(oc, hPackDynamic);
            const int oc_pack_coeff = hPackDynamic / pack;
            const int weightStride = mResource->mWeight->stride(0);

            auto threadParameters = Tile2MatMulParameters;
            auto threadParametersRemain = threadParameters;
            threadParameters[6] =  tBlock;
            threadParametersRemain[6] = tLast;
            threadParameters[3] = eTileReal * pack_stride;
            threadParametersRemain[3] = threadParameters[3];
            // copy pointer out
            auto MaxATileMatMulOC16Function = core->MNNPackedMatMulOC16Functions[ePack - 1];
            auto TailATileMatMulOC16Function = core->MNNPackedMatMulOC16Functions[tLast - 1];
            auto MaxATileMatMulOC32Function = core->MNNPackedMatMulOC32Functions[ePack - 1];
            auto TailATileMatMulOC32Function = core->MNNPackedMatMulOC32Functions[tLast - 1];
            auto MaxATileMatMulOC48Function = core->MNNPackedMatMulOC48Functions[ePack - 1];
            auto TailATileMatMulOC48Function = core->MNNPackedMatMulOC48Functions[tLast - 1];

            auto* _dstOrigin = _srcOrigin + eTileReal * srcUnit2 * ic_4 * pack * bytes;

            for (int i = 0; i < srcUnit2; ++i) {
                for (int t_oc_mul = 0; t_oc_mul < oc_hpack; ++t_oc_mul) {
                    int t_oc = t_oc_mul * oc_pack_coeff;
                    int ocValidPack = ALIMIN(t_oc + oc_pack_coeff, dc_4) - t_oc;

                    auto srcPtr = (_srcOrigin + i * ic_4 * eTileReal * pack * bytes);
                    auto _weightFloatPtr = (const float*)(weight + i * weightStride + (t_oc * ic_roundup * pack) * bytes);
                    auto _dstFloatPtr = (_dstOrigin + (i * dc_4 + t_oc) * eTileReal * pack * bytes);

#ifdef PROFILE_DETAIL
                    macs += eTileReal * (2 * ic) * (ocValidPack) * pack;
#endif

                    if (tBlock) {
                        switch (ocValidPack) {
                            case 1:
                                MaxATileMatMulOC16Function((float*)_dstFloatPtr, (const float*)srcPtr, _weightFloatPtr, threadParameters.data(), nullptr, nullptr);
                                break;
                            case 2:
                                MaxATileMatMulOC32Function((float*)_dstFloatPtr, (const float*)srcPtr, _weightFloatPtr, threadParameters.data(), nullptr, nullptr);
                                break;
                            case 3:
                                MaxATileMatMulOC48Function((float*)_dstFloatPtr, (const float*)srcPtr, _weightFloatPtr, threadParameters.data(), nullptr, nullptr);
                                break;
                        }
                        srcPtr += tBlock * ic_4 * pack * bytes;
                        _dstFloatPtr += tBlock * pack * bytes;
                    }
                    if (tLast) {

                        switch (ocValidPack) {
                            case 1:
                                TailATileMatMulOC16Function((float*)_dstFloatPtr, (const float*)srcPtr, _weightFloatPtr, threadParametersRemain.data(), nullptr, nullptr);
                                break;
                            case 2:
                                TailATileMatMulOC32Function((float*)_dstFloatPtr, (const float*)srcPtr, _weightFloatPtr, threadParametersRemain.data(), nullptr, nullptr);
                                break;
                            case 3:
                                TailATileMatMulOC48Function((float*)_dstFloatPtr, (const float*)srcPtr, _weightFloatPtr, threadParametersRemain.data(), nullptr, nullptr);
                                break;
                        }
                    }

                }
            }
            /* Dest Transform And Post Treat Begin */
            const int transb_stride = wUnit * hUnit;
            const int ob_stride = ow * oh;
            const int srcTransZStep =  eTileReal * pack_stride;
            const int OCUnitStep =  eTileReal * pack * dc_4;
            const int dstZStep = ob_stride * batch * pack_stride;
            const auto srcOriginSegment = _srcOrigin + eTileReal * srcUnit2 * ic_4 * pack_stride;
            const float* postParameters = mPostParameters.data();
            auto DestUnrollTransform = mDestUnrollTransform.get();
            for (int z = 0; z < dc_4; ++z) {
                const float* biasFloatPtr = (const float*)(bias + z * pack_stride);
                for (int tile_k = xIndex; tile_k < xIndex + eTileReal; tile_k++) {
                    int bIndex = tile_k / transb_stride;
                    int hwIndex = tile_k % transb_stride;
                    int hIndex = (hwIndex / wUnit);
                    int wIndex = (hwIndex % wUnit);
                    int ohIndex = hIndex * dstUnit;
                    int owIndex = wIndex * dstUnit;
                    int ey = ALIMIN(ohIndex + dstUnit, oh) - ohIndex;
                    int ex = ALIMIN(owIndex + dstUnit, ow) - owIndex;
                    auto dstZPtr = dstOrigin + (owIndex + ohIndex * ow + bIndex * ob_stride) * pack_stride + z * dstZStep;
                    auto srcZPtr =  srcOriginSegment + (tile_k - xIndex) * pack_stride + z * srcTransZStep;
                    int count = ex * pack_stride;

                    if (ex == dstUnit) {
                        DestUnrollTransform[srcUnit]((const float*)srcZPtr, (float*)midBuffer0, nullptr, nullptr, OCUnitStep, dstUnit * pack, srcUnit * OCUnitStep, pack);
                        DestUnrollTransform[ey]((const float*)midBuffer0, (float*)dstZPtr, biasFloatPtr, postParameters, pack, pack * ow, pack * dstUnit, pack);
                    } else {
                        DestUnrollTransform[srcUnit]((const float*)srcZPtr, (float*)midBuffer0, nullptr, nullptr, OCUnitStep, dstUnit * pack, srcUnit * OCUnitStep, pack);
                        DestUnrollTransform[ey]((const float*)midBuffer0, (float*)midBuffer1, biasFloatPtr, postParameters,  pack, pack * dstUnit, pack * dstUnit, pack);

                        for (int yy = 0; yy < ey; ++yy) {
                            auto dstYAddr = dstZPtr + yy * ow * pack_stride;
                            auto srcYAddr = midBuffer1 + yy * dstUnit * pack_stride;
                            ::memcpy(dstYAddr, srcYAddr, count);
                        }
                    }
                }
            }
            /*Dest Transform And Post Treat End*/
        }

#ifdef PROFILE_DETAIL
        double gflops = (double)macs / 1000.0 / durationMul;
        MNN_PRINT(
            "conv winograd. mParallelInner:%d, tId:%d, lastTile:%d, srcUnit: %d, inside measure: sourceTrans1:%lu us, "
            "sourceTrans2:%lu us, packATime:%lu us, durationMul:%lu us,  destTrans:%lu us, total:%lu us. %.3f GFLOPS, "
            "macs:%lu\n",
            mConvPerfconfig.isParallelInner, tId, tileCount % ePack, srcUnit, durationSourceTrans1,
            durationSourceTrans2, packATime, durationMul, durationDestTrans1,
            durationSourceTrans1 + durationSourceTrans2 + packATime + durationMul + durationDestTrans1, gflops, macs);
#endif
    };

    if (mConvPerfconfig.isParallelInner) {

        for (int tIndex = 0; tIndex < tileCount; tIndex += 1) {
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                parallelInnerSourceFunction((int)tId, tIndex);
            }
            MNN_CONCURRENCY_END();

            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                parallelInnerPackFreeMultiplyFunction((int)tId, tIndex);
            }
            MNN_CONCURRENCY_END();

            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                parallelInnerDestFunction((int)tId, tIndex);
            }
            MNN_CONCURRENCY_END();
        }

    } else {
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {

            parallelOuterPackFreeFunction(tId);
        }
        MNN_CONCURRENCY_END();
    }
    return NO_ERROR;
}

WinogradConfig ConvolutionPackFreeWinograd::bestWinogradUnit(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber, Backend* b, const PerfConfig& denseConfig) {

    WinogradConfig wconfig = updateBestWinogradUnit(common, inputTensor, outputTensor, threadNumber, b);
    if (wconfig.instructionCosts > denseConfig.instructionCosts) {
        wconfig.unit = 0;
    }
    return wconfig;
}


WinogradConfig ConvolutionPackFreeWinograd::updateBestWinogradUnit(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber, Backend* b) {
    auto core = static_cast<CPUBackend*>(b)->functions();
    int pack = core->pack, bytes = core->bytes;
    int ow      = outputTensor->width();
    int oh      = outputTensor->height();
    int oc      = outputTensor->channel();
    int batch   = outputTensor->batch();
    int ic      = inputTensor->channel();
    auto ic4 = UP_DIV(ic, pack);
    auto oc4 = UP_DIV(oc, pack);
    int ePackMax, hPack, lPack;
    core->MNNGetMatMulPackMode(&ePackMax, &lPack, &hPack);

    WinogradConfig bestConfig(0, false, 0, 0, 0, std::numeric_limits<float>().max());
    auto kernelSize  = common->kernelY();
    CoreFunctions::WinoUnrollDestTransFunc destTransform[CONVOLUTION_WINOGRAD_MAX_UNIT + 1];

    //In next major version: Would be read from microbenchmark result file.
    constexpr int roofLine = 20;
    constexpr int dynamicHPack = 32;
    constexpr int ePackUnit = 14;
    constexpr int InnerEPackCount = 8;
    constexpr int OuterEPackCount = 2;
    for (int ePack = ePackUnit; ePack <= ePackUnit; ePack += ePackUnit) {
        int unit2   = UP_DIV(batch * ow * oh, ePack);
        int maxUnit = (int)::sqrtf((float)unit2);
        maxUnit     = std::min(maxUnit, CONVOLUTION_WINOGRAD_MAX_UNIT);
        maxUnit     = std::max(maxUnit, CONVOLUTION_WINOGRAD_MIN_UNIT);
        std::set<int> supportSu{4, 6, 8};

        for (int u = CONVOLUTION_WINOGRAD_MIN_UNIT; u <= maxUnit; ++u) {
            auto dstUnit = u; // m
            auto srcUnit =  u + kernelSize - 1;

            if (supportSu.find(srcUnit) == supportSu.end()) {
                continue;
            }
            core->chooseWinoDestUnrollTransform(destTransform, CONVOLUTION_WINOGRAD_MAX_UNIT + 1, srcUnit, dstUnit);
            if (nullptr == destTransform[srcUnit]) {
                continue;
            }

            auto srcUnit2 = srcUnit * srcUnit;
            auto wUnit = UP_DIV(ow, dstUnit);
            auto hUnit = UP_DIV(oh, dstUnit);
            auto totalCount   = wUnit * hUnit * batch;

            WinogradConfig thisConfig(dstUnit, false, ePack * OuterEPackCount, ePack, dynamicHPack, -1);
            float outerFlops[4], innerFlops[4];
            float outerBandwidth[4], innerBandwidth[4], outer[4], inner[4], outerAcc = 0, innerAcc = 0;

            int eTile = ePack * OuterEPackCount;
            int tileCount = UP_DIV(totalCount, eTile);
            float tailCost = 0.0, lastTail = 0.0;
            if (totalCount % eTile == 0) {
                tailCost = 1.0f;
                lastTail = 1.0f;
            } else {
                bool moreThanOnetail = tileCount % threadNumber > 1;
                lastTail = (1.2f * (totalCount % eTile)) / eTile;
                tailCost = moreThanOnetail ? (std::max(1.0f, lastTail)) : lastTail;
            }

            float outerCoefficient = tailCost + ((tileCount - 1) / threadNumber);

            outerFlops[0] = outerCoefficient * (4 * srcUnit - 12) * srcUnit2 * ic4 * eTile * pack;
            outerFlops[1] = 0;
            outerFlops[2] = outerCoefficient * srcUnit2 * (2 * ic - 1) * eTile * oc4 * pack;
            outerFlops[3] = outerCoefficient * (srcUnit + dstUnit) * dstUnit * (2 * srcUnit - 6) * oc4 * ePack * pack;

            outerBandwidth[0] = outerCoefficient *  2 * 2 * srcUnit2 * ic4 * eTile * pack;
            outerBandwidth[1] = 0;
            outerBandwidth[2] = outerCoefficient * srcUnit2 * (eTile * ic + oc4 * pack * ic + eTile * oc4 * pack);

            outerBandwidth[3] = outerCoefficient * ((srcUnit + dstUnit) * 2 * 2 * dstUnit * oc4) * eTile * pack;

            eTile = ePack * InnerEPackCount;
            tileCount = UP_DIV(totalCount, eTile);
            if (totalCount % eTile == 0) {
                tailCost = 1.0f;
                lastTail = 1.0f;
            } else {
                bool moreThanOnetail = tileCount % threadNumber > 1;
                lastTail = (1.05f * (totalCount % eTile)) / eTile;
                tailCost = moreThanOnetail ? (std::max(1.0f, lastTail)) : lastTail;
            }
            float innerCoefficient = lastTail + ((totalCount - 1) / eTile);


            innerFlops[0] = innerCoefficient * UP_DIV(ic4 * eTile, threadNumber) * (4 * srcUnit - 12) * srcUnit2 * pack;
            innerFlops[1] = 0;
            innerFlops[2] = innerCoefficient * UP_DIV(srcUnit2 * UP_DIV(oc, dynamicHPack), threadNumber) * (2 * ic - 1) * eTile * UP_DIV(dynamicHPack, pack);
            innerFlops[3] = innerCoefficient * (srcUnit + dstUnit) * dstUnit * (2 * srcUnit - 6) * UP_DIV(oc4 * eTile, threadNumber) * pack;

            innerBandwidth[0] = innerCoefficient * UP_DIV(ic4 * eTile, threadNumber) * 2 * 2 * srcUnit2 * pack;
            innerBandwidth[1] = 0;
            innerBandwidth[2] = innerCoefficient * UP_DIV(srcUnit2 * UP_DIV(oc, dynamicHPack), threadNumber) * (eTile * ic + dynamicHPack * ic + eTile * dynamicHPack);
            innerBandwidth[3] = innerCoefficient * (srcUnit + dstUnit) * 2 * 2 * dstUnit * UP_DIV(oc4 * eTile, threadNumber) * pack;
            for (int i = 0; i < sizeof(outerFlops) / sizeof(float); i++) {
                 outer[i] = std::max(outerBandwidth[i] * roofLine, outerFlops[i]);
                 inner[i] = std::max(innerBandwidth[i] * roofLine, innerFlops[i]);
                 outerAcc += outer[i];
                 innerAcc += inner[i];
            }

            thisConfig.isParallelInner = outerAcc > innerAcc;
            thisConfig.instructionCosts = thisConfig.isParallelInner ? innerAcc : outerAcc;
            thisConfig.eTile = thisConfig.isParallelInner ? (ePack * InnerEPackCount) : (ePack * OuterEPackCount);


            if (bestConfig.instructionCosts > thisConfig.instructionCosts) {
                bestConfig = thisConfig;
            }
        }
    }


    return bestConfig;
}

bool ConvolutionPackFreeWinograd::updateWinogradBuffer(const Tensor* input, const Tensor* output) {

    auto core = static_cast<CPUBackend*>(backend())->functions();
    int pack = core->pack, bytes = core->bytes;
    MNN_ASSERT(mCommon->kernelX() == mCommon->kernelY());
    int threadNumber = ((CPUBackend *)backend())->threadNumber();

    int unit = mConvPerfconfig.unit;
    int ePack = mConvPerfconfig.ePack;
    int eTile = mConvPerfconfig.eTile;
    auto kernelSize = mCommon->kernelY();
    WinogradGenerater generator(unit, kernelSize, 1, true);
    int ePackMax, hPack, lPack;
    core->MNNGetMatMulPackMode(&ePackMax, &lPack, &hPack);

    int alpha        = unit + kernelSize - 1;
    int alpha2       = alpha * alpha;

    mSourceUnrollTransform =  core->chooseWinoSourceUnrollTransform(alpha, alpha);
    core->chooseWinoDestUnrollTransform(mDestUnrollTransform.get(), CONVOLUTION_WINOGRAD_MAX_UNIT + 1, alpha, unit);

    int srcCount                       = input->channel();
    int outputCount                    = output->channel();
    auto ic4 = UP_DIV(srcCount, pack);
    auto oc4 = UP_DIV(outputCount, pack);

    if (mConvPerfconfig.isParallelInner) {
        // pack-free multiply
        mTempBuffer.reset(Tensor::createDevice<uint8_t>({1, eTile, ic4 + oc4, pack * alpha2, bytes}));
        mTransformMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, 2, alpha2, pack, bytes}));
        mGemmMidBuffer.reset(Tensor::createDevice<uint8_t>({bytes}));
        hPack = mConvPerfconfig.hPack;

    } else {
        mTempBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, eTile, ic4 + oc4, pack * alpha2, bytes}));
        mTransformMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, 2, alpha2, pack, bytes}));
        mGemmMidBuffer.reset(Tensor::createDevice<uint8_t>({bytes}));
        hPack = mConvPerfconfig.hPack;

    }

    mA = generator.A();
    mB = generator.B();
    // Transform Kernel
    auto G = generator.G();
    // replace Tensor::createDevice by Tensor::create and allocTransformWeight's alloc=true to avoid malloc by onAcquireBuffer
    std::shared_ptr<Tensor> sourceWeight(Tensor::create<float>(
        std::vector<int>{outputCount, srcCount, kernelSize, kernelSize}, (void *)mOriginWeight, Tensor::CAFFE));
    auto tempWeight = generator.allocTransformWeight(sourceWeight.get(), lPack, hPack, true);

    auto shape = tempWeight->shape();
    shape.push_back(bytes);
    mResource->mWeight.reset(Tensor::createDevice<uint8_t>(shape));
    mValid = backend()->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
    if (!mValid) {
        return false;
    }
    generator.transformWeight(tempWeight.get(), sourceWeight.get(), true);
    if (bytes != 4) {
        core->MNNFp32ToLowp(tempWeight->host<float>(), mResource->mWeight->host<int16_t>(), tempWeight->elementSize());
    } else {
        ::memcpy(mResource->mWeight->host<float>(), tempWeight->host<float>(), tempWeight->size());
    }

    mPostParameters = getPostParameters();
    return true;
}

ErrorCode ConvolutionPackFreeWinograd::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    CPUConvolution::onResize(inputs, outputs);

    auto input   = inputs[0];
    auto output  = outputs[0];
    int threadNumber = std::max(((CPUBackend *)backend())->threadNumber(), 1);
    WinogradConfig bestConfig = updateBestWinogradUnit(mCommon, input, output, threadNumber, backend());
    if (bestConfig != mConvPerfconfig) {
        mConvPerfconfig = bestConfig;
        updateWinogradBuffer(input, output);
    }
    mConvPerfconfig.instructionCosts = bestConfig.instructionCosts;

    bool success = backend()->onAcquireBuffer(mTempBuffer.get(), Backend::DYNAMIC);
    success      = success && backend()->onAcquireBuffer(mGemmMidBuffer.get(), Backend::DYNAMIC);
    success      = success && (backend()->onAcquireBuffer(mTransformMidBuffer.get(), Backend::DYNAMIC));
    backend()->onReleaseBuffer(mTempBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTransformMidBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mGemmMidBuffer.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    return NO_ERROR;
}
} // namespace MNN
