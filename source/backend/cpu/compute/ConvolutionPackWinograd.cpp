//
//  ConvolutionPackWinograd.cpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvolutionPackWinograd.hpp"
#include <math.h>
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "math/WingoradGenerater.hpp"
#include <MNN/AutoTime.hpp>
#include "core/MemoryFormater.h"

constexpr int FULSE_THRESHHOLD_NUMERATOR = 10;
constexpr int FULSE_THRESHHOLD_DENOMINATOR = 10;

using namespace MNN::Math;

//#define MNN_WINOGRAD_PRINT_REDUCE_RATE
//#define MNN_WINO_TRANFORM_TEST_CLOSE
namespace MNN {
ConvolutionPackWinograd::ConvolutionPackWinograd(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output,
                                         Backend *b, const float *originWeight, size_t originWeightSize,
                                         const float *bias, size_t biasSize, WinogradConfig config)
    : ConvolutionWinogradImpl(convOp, b) {
    int unit = config.unit;
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int pack = core->pack, bytes = core->bytes;
    int weightBytes = bytes;
    if (0!=core->matmulBytes) {
        weightBytes = core->matmulBytes;
    }
    mResource.reset(new Resource);
    mResource->backend = b;

    mDestUnrollTransform.reset(new CoreFunctions::WinoUnrollDestTransFunc[CONVOLUTION_WINOGRAD_MAX_UNIT + 1],
        std::default_delete<CoreFunctions::WinoUnrollDestTransFunc[]>());

    if (!mResource->copyBiasAlign(bias, biasSize)) {
        MNN_ERROR("Not Enough Memory\n");
        mValid = false;
        return;
    }
    MNN_ASSERT(mCommon->kernelX() == mCommon->kernelY());

    int threadNumber = ((CPUBackend *)backend())->threadNumber();

    auto kernelSize = mCommon->kernelY();
    WinogradGenerater generator(unit, kernelSize, 1, true);

    int ePack, hPack, lPack;
    core->MNNGetMatMulPackMode(&ePack, &lPack, &hPack);

    int alpha        = unit + kernelSize - 1;
    int alpha2       = alpha * alpha;
    mSourceTransformPack = core->chooseWinoSourceTransformPack(alpha, alpha, ePack, lPack, pack);
    mSourceUnrollTransform =  core->chooseWinoSourceUnrollTransform(alpha, alpha);
    core->chooseWinoDestUnrollTransform(mDestUnrollTransform.get(), CONVOLUTION_WINOGRAD_MAX_UNIT + 1, alpha, unit);

    int srcCount                       = input->channel();
    int outputCount                    = output->channel();
    auto ic4 = UP_DIV(srcCount, pack);
    auto oc4 = UP_DIV(outputCount, pack);
    mTempBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, ePack, ic4 + oc4, pack * alpha2, bytes}));
    // mTransformMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, 2, alpha2, pack, bytes}));
    // mGemmMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, ePack * UP_DIV(srcCount, lPack) * lPack, bytes}));

    mTransformMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, (1 + ic4 * ePack), alpha2, pack, bytes})); // 1 means original small buffer of alpha2 * pack.
    mGemmMidBuffer.reset(Tensor::createDevice<uint8_t>({threadNumber, alpha, ePack * UP_DIV(srcCount, pack) * pack, bytes}));


    mA = generator.A();
    mB = generator.B();


    // Transform Kernel
    auto G = generator.G();
    // replace Tensor::createDevice by Tensor::create and allocTransformWeight's alloc=true to avoid malloc by onAcquireBuffer
    std::shared_ptr<Tensor> sourceWeight(Tensor::create<float>(
        std::vector<int>{outputCount, srcCount, kernelSize, kernelSize}, (void *)originWeight, Tensor::CAFFE));
    auto tempWeight = generator.allocTransformWeight(sourceWeight.get(), lPack, hPack, true);

    auto shape = tempWeight->shape();
    shape.push_back(weightBytes);
    mResource->mWeight.reset(Tensor::createDevice<uint8_t>(shape));
    mValid = backend()->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }
    generator.transformWeight(tempWeight.get(), sourceWeight.get(), true);
    if (weightBytes != 4) {
        core->MNNFp32ToLowp(tempWeight->host<float>(), mResource->mWeight->host<int16_t>(), tempWeight->elementSize());
    } else {
        ::memcpy(mResource->mWeight->host<float>(), tempWeight->host<float>(), tempWeight->size());
    }

    mPostParameters = getPostParameters();
}
ConvolutionPackWinograd::~ConvolutionPackWinograd() {
    // Do nothing
}
bool ConvolutionPackWinograd::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvolutionPackWinograd(mResource, op->main_as_Convolution2D()->common(), bn);
    dstExe->mA = mA;
    dstExe->mB = mB;
    dstExe->mTempBuffer.reset(Tensor::createDevice<uint8_t>(mTempBuffer->shape()));
    dstExe->mTransformMidBuffer.reset(Tensor::createDevice<uint8_t>(mTransformMidBuffer->shape()));
    dstExe->mGemmMidBuffer.reset(Tensor::createDevice<uint8_t>(mGemmMidBuffer->shape()));
    dstExe->mSourceTransformPack = mSourceTransformPack;
    dstExe->mSourceUnrollTransform = mSourceUnrollTransform;
    dstExe->mDestUnrollTransform = mDestUnrollTransform;
    dstExe->mPostParameters = mPostParameters;
    *dst = dstExe;
    return true;
}

ErrorCode ConvolutionPackWinograd::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_CONCURRENCY_BEGIN(tId, mMainFunction.first) {
        mMainFunction.second(tId, inputs[0]->host<uint8_t>(), outputs[0]->host<uint8_t>());
    };
    MNN_CONCURRENCY_END();

    MNN_CONCURRENCY_BEGIN(tId, mPostFunction.first) {
        mPostFunction.second(tId, outputs[0]->host<uint8_t>());
    };
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

WinogradConfig ConvolutionPackWinograd::bestWinogradUnit(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber, Backend* b, const PerfConfig& denseConfig) {

    // compare cost value
    WinogradConfig wconfig;


    auto core = static_cast<CPUBackend*>(b)->functions();
    auto winogradMemoryLevel = static_cast<CPUBackend*>(b)->getRuntime()->hint().winogradMemoryUsed;
    int multiBytes = static_cast<CPUBackend*>(b)->functions()->bytes;
    if (static_cast<CPUBackend*>(b)->functions()->matmulBytes != 0) {
        multiBytes = static_cast<CPUBackend*>(b)->functions()->matmulBytes;
    }
    int ow      = outputTensor->width();
    int oh      = outputTensor->height();
    int oc      = outputTensor->channel();
    int ePack, hPack, lPack;
    core->MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    int unit2   = UP_DIV(ow * oh, threadNumber);

    int maxUnit = (int)::sqrtf((float)unit2);
    maxUnit     = std::min(maxUnit, CONVOLUTION_WINOGRAD_MAX_UNIT);
    maxUnit     = std::max(maxUnit, CONVOLUTION_WINOGRAD_MIN_UNIT);
    if (winogradMemoryLevel != 3) {
       maxUnit = CONVOLUTION_WINOGRAD_MIN_UNIT;
    }

    int ic           = inputTensor->channel();
    auto kernelSize  = common->kernelY();
    int unit         = 0;
    float maxRate    = 0.0f;
    float originCost = (float)ow * oh * (2.0 * ic) * oc * kernelSize * kernelSize; // macs, with bias
    std::set<int> supportSu{4, 6, 8};
    if (multiBytes < 4) {
        supportSu = {4, 6};
    }
    CoreFunctions::WinoUnrollDestTransFunc destTransform[CONVOLUTION_WINOGRAD_MAX_UNIT + 1];
    for (int u = CONVOLUTION_WINOGRAD_MIN_UNIT; u <= maxUnit; ++u) {
        auto sui = u + kernelSize - 1;
        auto su = (float)sui;
        if (supportSu.find(sui) == supportSu.end()) {
            continue;
        }
        core->chooseWinoDestUnrollTransform(destTransform, CONVOLUTION_WINOGRAD_MAX_UNIT + 1, sui, u);
            if (nullptr == destTransform[sui]) {
            continue;
        }
        // /*Let F(6,3) be choosed when it can speed up from F(2,3) than 0.6*/

        // float penalty = (su * su) / (float)(kernelSize * kernelSize) * 0.12f;
        // float winogradCost =
        //     (2 * su * su * ic + su * su * ic * oc + (su + u) * u * oc) * 2 * (UP_DIV(ow, u) * UP_DIV(oh, u));
        // float reduceRate = originCost / winogradCost - penalty;

        // new metrics for winograd, only need to calculate absolute compute complexity.
        // add instructions are about (n - 2), multiply operations are (n - 4). as a result operations are (2n - 6).
        float winogradCost =
            ( (2 * su) * su * su * ic + 2 * su * su * ic * oc + ((su + u) * u * (2 * su) * oc)) * (UP_DIV(ow, u) * UP_DIV(oh, u));
        float reduceRate = originCost / winogradCost;

        // MNN_PRINT("ow=%d, oh=%d, winogradCost:%f, reduceRate:%f, winograd unit:%d\n", ow, oh, winogradCost, reduceRate, u);
        if (reduceRate > maxRate) {
            maxRate = reduceRate;
            unit    = u;
        }
    }
    if (maxRate < 1.0f) {
        wconfig.unit = 0;
        return wconfig;
    }
    wconfig.unit = unit;
    return wconfig;
}

ErrorCode ConvolutionPackWinograd::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    CPUConvolution::onResize(inputs, outputs);
    int threadNumber = ((CPUBackend*)(backend()))->threadNumber();
    mTempBuffer->setLength(0, threadNumber);
    mGemmMidBuffer->setLength(0, threadNumber);
    mTransformMidBuffer->setLength(0, threadNumber);
    // FUNC_PRINT(mA->length(1));
    bool success = backend()->onAcquireBuffer(mTempBuffer.get(), Backend::DYNAMIC);
    success      = success && backend()->onAcquireBuffer(mGemmMidBuffer.get(), Backend::DYNAMIC);
    success      = success && (backend()->onAcquireBuffer(mTransformMidBuffer.get(), Backend::DYNAMIC));
    backend()->onReleaseBuffer(mTempBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTransformMidBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mGemmMidBuffer.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int pack = core->pack, bytes = core->bytes;

    auto input   = inputs[0];
    auto output  = outputs[0];
    auto dstUnit = mA->length(1); // m
    auto srcUnit = mA->length(0); // n
    int ePack, lPack, hPack;
    core->MNNGetMatMulPackMode(&ePack, &lPack, &hPack);

    auto srcUnit2 = srcUnit * srcUnit;
    auto alphaXStride = srcUnit * ePack * pack;
    auto IC4alpha2Stride = srcUnit2 * ePack * pack;

    int ow   = output->width();
    int oh   = output->height();
    int iw   = input->width();
    int ih   = input->height();
    int ic_4 = UP_DIV(input->channel(), pack);
    int dc_4 = UP_DIV(output->channel(), pack);
    int batch = input->batch();
    // MNN_PRINT("%d, %d\n", srcUnit, dstUnit);

    int padY = mPadY;
    int padX = mPadX;

    auto wUnit = UP_DIV(ow, dstUnit); // ow / m
    auto hUnit = UP_DIV(oh, dstUnit); // oh / m

    auto totalCount   = wUnit * hUnit * batch;
    // MNN_PRINT("ow=%d, oh=%d\n", ow, oh);
    
    std::vector<int> divides(threadNumber+1);
    static_cast<const CPURuntime*>( static_cast<CPUBackend*>(backend())->getRuntime())->computeDivideSizes(totalCount, divides.data()+1);
    divides[0] = 0;
    auto midBuffer0Bytes = srcUnit2 * pack * bytes;
    bool allow_x86_bf16_winograd = true;
#ifdef MNN_USE_SSE
    allow_x86_bf16_winograd = bytes != 2; // only bf16 has length of 2 byte on x86. fp16 dosnot exist.
#endif

    auto weight    = mResource->mWeight->host<uint8_t>();
    auto bias      = mResource->mBias->host<uint8_t>();
    mMainFunction.first = threadNumber;
    mMainFunction.second = [=](int tId, const uint8_t* inputOrigin, uint8_t* dstOrigin) {
        int tSta = divides[tId];
        int tFin = divides[tId+1];
        if (tSta >= tFin) {
            return;
        }
        int eRemain = (tFin-tSta) % ePack;
        std::vector<size_t> parameters(6);
        parameters[1] = input->channel();
        parameters[2] = output->channel();
        parameters[4] = 0;
        parameters[5] = 0;
        parameters[0] = eRemain * bytes;
        parameters[3] = ePack * pack * bytes;

        std::vector<size_t> parametersRemain = parameters;
        parametersRemain[0] = eRemain * bytes;
        parametersRemain[3] = eRemain * pack * bytes;

        auto srcOrigin = inputOrigin;
        auto _srcOrigin = mTempBuffer->host<uint8_t>() + tId * mTempBuffer->stride(0);
        auto gemmBuffer = (mGemmMidBuffer->host<uint8_t>() + tId * mGemmMidBuffer->stride(0));
        auto midBuffer0 = mTransformMidBuffer->host<uint8_t>() + tId * mTransformMidBuffer->stride(0);
        auto midBufferStride1 = mTransformMidBuffer->stride(1);
        auto weightStride = mResource->mWeight->stride(0);
        auto midBuffer1 = midBuffer0 + midBuffer0Bytes;
        for (int xIndex = tSta; xIndex < tFin; xIndex+=ePack) {
            int xReamin = tFin - xIndex;
            int xC      = xReamin > ePack ? ePack : xReamin;

            const bool fuseTransformPack = (xC * FULSE_THRESHHOLD_DENOMINATOR >= FULSE_THRESHHOLD_NUMERATOR * ePack) && allow_x86_bf16_winograd && nullptr != mSourceTransformPack && core->matmulBytes == 0;
            /*Source Transform Begin*/
#ifndef MNN_WINO_TRANFORM_TEST_CLOSE
            {
                int sourceZStep = iw * ih * batch * pack;
                int oyBegin = xIndex / wUnit;
                int oxBegin = xIndex % wUnit;
                int oyEnd = (xIndex + xC-1) / wUnit;
                int remain = xC;
                int destSOffset = 0;
                if (fuseTransformPack) {
                    for (int hbIndex=oyBegin; hbIndex <= oyEnd; ++hbIndex) {
                        int hIndex = hbIndex % hUnit;
                        int bIndex = hbIndex / hUnit;
                        int step = ALIMIN(wUnit - oxBegin, remain);
                        int srcY  = hIndex * dstUnit - padY;
                        int ey    = ALIMIN(srcY + srcUnit, ih) - srcY;
                        int sy    = ALIMAX(0, srcY) - srcY;
                        auto srcStartY = srcOrigin + (srcY * iw + bIndex * iw * ih) * pack * bytes;

                        for (int si=0; si<step; ++si) {
                            auto wIndex = si + oxBegin;
                            int srcX  = wIndex * dstUnit - padX;
                            int sx    = ALIMAX(0, srcX) - srcX;
                            int ex    = ALIMIN(srcX + srcUnit, iw) - srcX;
                            int count = pack * (ex - sx);

                            auto srcStart = srcStartY + srcX * pack * bytes;
                            auto midBuffer1Offset = midBuffer1 + destSOffset;

                            if (ex - sx == srcUnit && ey - sy == srcUnit) {
                                for (int z = 0; z < ic_4; ++z) {
                                    auto srcZ = srcStart + z * sourceZStep * bytes;
                                    // Transform
                                    mSourceUnrollTransform((const float*)srcZ, (float*)midBuffer1Offset, iw * pack, ePack * pack, pack, alphaXStride);
                                    midBuffer1Offset += IC4alpha2Stride * bytes;
                                }
                            } else {
                                for (int z = 0; z < ic_4; ++z) {
                                    // Extract
                                    auto srcZ = srcStart + z * sourceZStep * bytes;
                                    ::memset(midBuffer0, 0, midBuffer0Bytes);
                                    if (count > 0) {
                                        for (int yy = sy; yy < ey; ++yy) {
                                            auto dst_yy = midBuffer0 + (yy * srcUnit + sx) * pack * bytes;
                                            auto src_yy = srcZ + (iw * yy + sx) * pack * bytes;
                                            ::memcpy(dst_yy, src_yy, count * bytes);
                                        }
                                    }

                                    mSourceUnrollTransform((const float*)midBuffer0, (float*)midBuffer1Offset, srcUnit * pack, ePack * pack, pack, alphaXStride);
                                    midBuffer1Offset += IC4alpha2Stride * bytes;
                                }
                            }
                            destSOffset += pack * bytes;
                        }
                        oxBegin = 0;
                        remain -= step;
                    }
                } else {
                    int dstZStep    = xC * pack;  // hUnit*wUnit * 4
                    int unitStep    = ic_4 * xC * pack; //  C/4 * hUnit*wUnit * 4
                    for (int hbIndex=oyBegin; hbIndex <= oyEnd; ++hbIndex) {
                        int hIndex = hbIndex % hUnit;
                        int bIndex = hbIndex / hUnit;
                        int step = ALIMIN(wUnit - oxBegin, remain);
                        int srcY  = hIndex * dstUnit - padY;
                        int ey    = ALIMIN(srcY + srcUnit, ih) - srcY; //h dim pack element length
                        int sy    = ALIMAX(0, srcY) - srcY;  // first y element
                        auto srcStartY = srcOrigin + (srcY * iw + bIndex * iw * ih) * pack * bytes;
                        for (int si=0; si<step; ++si) {
                            auto wIndex = si + oxBegin;
                            int srcX  = wIndex * dstUnit - padX;
                            int sx    = ALIMAX(0, srcX) - srcX;
                            int ex    = ALIMIN(srcX + srcUnit, iw) - srcX;
                            int count = pack * (ex - sx);

                            auto srcStart = srcStartY + srcX * pack * bytes;
                            auto dst_x = _srcOrigin + destSOffset;
                            if (ex - sx == srcUnit && ey - sy == srcUnit) {
                                for (int z = 0; z < ic_4; ++z) {
                                    auto srcZ = srcStart + z * sourceZStep * bytes;
                                    // Transform

                                    auto dstZ = dst_x + z * dstZStep * bytes;
                                    mSourceUnrollTransform((const float*)srcZ, (float*)midBuffer1, iw * pack, pack, pack, pack * srcUnit);
                                    mSourceUnrollTransform((const float*)midBuffer1, (float*)dstZ, srcUnit * pack, unitStep, pack, unitStep * srcUnit);
                                }
                            } else {
                                for (int z = 0; z < ic_4; ++z) {
                                    // Extract
                                    auto srcZ = srcStart + z * sourceZStep * bytes;
                                    ::memset(midBuffer0, 0, midBufferStride1);
                                    if (count > 0) {
                                        for (int yy = sy; yy < ey; ++yy) {
                                            auto dst_yy = midBuffer0 + (yy * srcUnit + sx) * pack * bytes;
                                            auto src_yy = srcZ + (iw * yy + sx) * pack * bytes;
                                            ::memcpy(dst_yy, src_yy, count * bytes);
                                        }
                                    }

                                    auto dstZ = dst_x + z * dstZStep * bytes;

                                    mSourceUnrollTransform((const float*)midBuffer0, (float*)midBuffer1, srcUnit * pack, pack, pack, pack * srcUnit);
                                    mSourceUnrollTransform((const float*)midBuffer1, (float*)dstZ, srcUnit * pack, unitStep, pack, unitStep * srcUnit);

                                }
                            }
                            destSOffset += pack * bytes;
                        }
                        oxBegin = 0;
                        remain -= step;
                    }
                }
            }

#endif
            auto* _dstOrigin = _srcOrigin;
            if (fuseTransformPack) {
                _dstOrigin += ePack * srcUnit2 * ic_4 * pack * bytes;
                if (xC != ePack) {
                    auto midTransformPtr = midBuffer1 + xC * pack * bytes;
                    for (int i = 0; i < ic_4 * srcUnit2; ++i) {
                        memset(midTransformPtr, 0, (ePack - xC) * pack * bytes);
                        midTransformPtr += ePack * pack * bytes;
                    }
                }
                for (int iNw = 0; iNw < srcUnit; ++iNw) { // i_Nw
                    auto midTransformPtr = midBuffer1 + iNw * alphaXStride * bytes;
                    auto unitsGemmbuffer = gemmBuffer;
                    for (int z = 0; z < ic_4; ++z) { // ic_4
                        mSourceTransformPack((float*)midTransformPtr, (float*)unitsGemmbuffer, ePack * pack * ic_4);
                        unitsGemmbuffer += ePack * pack * bytes;
                        midTransformPtr += IC4alpha2Stride * bytes;
                    }
                    // Previous tranform requires xC aligned with EPack, xC should be Epack;
                    for (int iNh = 0; iNh < srcUnit; ++iNh) { // i_Nh, gemm
                        auto unitsGemmbuffer = gemmBuffer + iNh * ic_4 * pack * ePack * bytes;
                        auto _dstFloatPtr = (float*)(_dstOrigin + (iNh * srcUnit + iNw) * dc_4 * pack * ePack * bytes);
                        auto _weightFloatPtr = (const float*)(weight + (iNh * srcUnit + iNw) * weightStride);
                        core->MNNPackedMatMul(_dstFloatPtr, (float*)unitsGemmbuffer, _weightFloatPtr, parameters.data(), nullptr, nullptr, nullptr, nullptr);
                    }
                }
            } else {
                /*Source Transform End*/
                // // Multi
                _dstOrigin += xC * srcUnit2 * ic_4 * pack * bytes;

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
                if (xC == ePack) {
                    for (int i = 0; i < srcUnit2; ++i) {
                        auto srcTemp = (const float*)(_srcOrigin + i * ic_4 * pack * xC * bytes);
                        auto _dstFloatPtr = (float*)(_dstOrigin + i * dc_4 * pack * xC * bytes);
                        auto _weightFloatPtr = (const float*)(weight + i * weightStride);
                        core->MNNPackC4ForMatMul_A((float*)gemmBuffer, &srcTemp, info, el);

                        core->MNNPackedMatMul(_dstFloatPtr, (float*)gemmBuffer, _weightFloatPtr, parameters.data(), nullptr, nullptr, nullptr, nullptr);
                    }
                } else {
                    for (int i = 0; i < srcUnit2; ++i) {
                        auto srcTemp = (const float*)(_srcOrigin + i * ic_4 * pack * xC * bytes);
                        auto _dstFloatPtr = (float*)(_dstOrigin + i * dc_4 * pack * xC * bytes);
                        auto _weightFloatPtr = (const float*)(weight + i * weightStride);
                        core->MNNPackC4ForMatMul_A((float*)gemmBuffer, &srcTemp, info, el);
                        core->MNNPackedMatMulRemain(_dstFloatPtr, (float*)gemmBuffer, _weightFloatPtr, xC, parametersRemain.data(), nullptr, nullptr, nullptr, nullptr);
                    }
                }
            }

#ifndef MNN_WINO_TRANFORM_TEST_CLOSE
            /* Dest Transform And Post Treat Begin */
            {
                auto DestUnrollTransform = mDestUnrollTransform.get();

                int srcZStep = (fuseTransformPack ? ePack : xC) * pack;
                int unitStep = (fuseTransformPack ? ePack : xC) * dc_4 * pack;
                int dstZStep = ow * oh * pack * batch;
                int oyBegin = xIndex / wUnit;
                int oxBegin = xIndex % wUnit;
                int oyEnd = (xIndex + xC-1) / wUnit;
                int remain = xC;
                auto dstS = _dstOrigin;
                for (int hbIndex=oyBegin; hbIndex <= oyEnd; ++hbIndex) {
                    int hIndex = hbIndex % hUnit;
                    int bIndex = hbIndex / hUnit;
                    int step = std::min(wUnit - oxBegin, remain);
                    int dstY = hIndex * dstUnit;
                    int ey = ALIMIN(dstY + dstUnit, oh) - dstY;
                    for (int si=0; si<step; ++si) {
                        auto wIndex = si + oxBegin;
                        auto srcXi = dstS + pack * si * bytes;
                        int dstX = wIndex * dstUnit;
                        auto dstStart = dstOrigin + (dstX + dstY * ow + bIndex * ow * oh) * pack * bytes;
                        int ex = ALIMIN(dstX + dstUnit, ow) - dstX;

                        int count = ex * pack;
                        if (ex == dstUnit) {
                            for (int z = 0; z < dc_4; ++z) {
                                auto dstZAddr = dstStart + z * dstZStep * bytes;
                                auto srcZ     = srcXi + z * srcZStep * bytes;

                                DestUnrollTransform[srcUnit]((const float*)srcZ, (float*)midBuffer0, nullptr, nullptr, unitStep, dstUnit * pack, srcUnit * unitStep, pack);
                                DestUnrollTransform[ey]((const float*)midBuffer0, (float*)dstZAddr,  nullptr, nullptr, pack, pack * ow, pack * dstUnit, pack);
                            }
                        } else {
                            for (int z = 0; z < dc_4; ++z) {
                                auto dstZAddr = dstStart + z * dstZStep * bytes;
                                auto srcZ     = srcXi + z * srcZStep * bytes;

                                DestUnrollTransform[srcUnit]((const float*)srcZ, (float*)midBuffer0, nullptr, nullptr, unitStep, dstUnit * pack, srcUnit * unitStep, pack);
                                DestUnrollTransform[ey]((const float*)midBuffer0, (float*)midBuffer1,  nullptr, nullptr, pack, pack * dstUnit, pack * dstUnit, pack);

                                for (int yy = 0; yy < ey; ++yy) {
                                    auto dstYAddr = dstZAddr + yy * pack * ow * bytes;
                                    auto srcYAddr = midBuffer1 + yy * pack * dstUnit * bytes;
                                    ::memcpy(dstYAddr, srcYAddr, count * bytes);
                                }
                            }
                        }
                    }
                    oxBegin = 0;
                    remain -= step;
                    dstS += pack * step * bytes;
                }
            }
#endif
            /*Dest Transform And Post Treat End*/
        }
    };
    std::vector<int> postDivides(threadNumber+1);
    static_cast<const CPURuntime*>( static_cast<CPUBackend*>(backend())->getRuntime())->computeDivideSizes(dc_4, postDivides.data()+1);
    postDivides[0] = 0;

    mPostFunction.first = threadNumber;
    mPostFunction.second = [=](int tId, uint8_t* outputOrigin) {
        auto dstOrigin = outputOrigin;
        int tSta = postDivides[tId];
        int tFin = postDivides[tId+1];
        for (int dy=tSta; dy < tFin; ++dy) {
            auto dataFloatPtr = (float*)(dstOrigin + ow * oh * batch * dy * pack * bytes);
            auto biasFloatPtr = (const float*)(bias + pack * dy * bytes);
            core->MNNAxByClampBroadcastUnit(dataFloatPtr, dataFloatPtr, biasFloatPtr, ow * oh * batch, 0, 0, 1,  mPostParameters.data());
        }
    };
    return NO_ERROR;
}
} // namespace MNN
